# Bug Fixes for Drone Go-to-Image Training

## Problem
Training loss was decreasing but evaluation results were always 0 (no successful episodes).

## Root Cause Analysis

The policy was trained and evaluated in **inconsistent coordinate spaces**, causing the learned energy function to be meaningless at inference time.

### Bug #1: Division by Zero in Spherical Conversion
**Location**: `fvf/utils/action_utils.py:38`

**Issue**: When action magnitude is zero, `arccos(z/r)` produces NaN.

**Fix**:
```python
# Before
theta = np.arccos(action[:, 2] / r)

# After
theta = np.arccos(np.clip(action[:, 2] / (r + 1e-8), -1.0, 1.0))
```

---

### Bug #2: Incorrect Coordinate Handling in Loss (CRITICAL)
**Location**: `fvf/policy/spherical_bessel_implicit_policy.py:170-177`

**Issue**: The code was unnormalizing theta and phi angles, which were never normalized in the first place. This caused the training to learn energy in a completely different coordinate space than intended.

**Before**:
```python
r = targets[:, :, 0, 0]  # normalized
theta = self.normalizer["action"].unnormalize(targets)[:, :, 0, 1]  # WRONG!
phi = self.normalizer["action"].unnormalize(targets)[:, :, 0, 2]    # WRONG!
```

**After**:
```python
# r is already normalized in targets
# theta and phi are in radians (never normalized), so don't unnormalize them
r = targets[:, :, 0, 0]
theta = targets[:, :, 0, 1]
phi = targets[:, :, 0, 2]
```

**Why this matters**:
- Dataset converts actions to spherical: (x,y,z) → (r, θ, φ)
- Only r gets normalized to [-1, 1]
- θ and φ remain in radians (θ ∈ [0, π], φ ∈ [0, 2π])
- Applying unnormalization to angles corrupted them completely

---

### Bug #3: Inconsistent Action Grid at Inference (CRITICAL)
**Location**: `fvf/policy/spherical_bessel_implicit_policy.py:51-131`

**Issue**: The action grid for inference was created in a different coordinate space than training:
- Training used: (r_normalized, θ_radians, φ_radians)
- Inference mixed: (r_normalized, θ_radians, φ_radians) but then tried to unnormalize everything

**Fix**: Made the coordinate spaces consistent throughout:

1. **Grid creation**: Use normalized radius
   ```python
   # Before
   r = torch.linspace(r_min + 0.01, r_max, sbh.n_k, ...)

   # After (with clear comments)
   r_normalized = torch.linspace(r_min + 0.01, r_max, sbh.n_k, ...)
   ```

2. **Action selection**: Properly handle coordinate unnormalization
   ```python
   # Create grid in (r_norm, theta_rad, phi_rad) - same as training
   actions_grid = torch.stack([R_NORM.flatten(), THETA.flatten(), PHI.flatten()])

   # After selecting best action, unnormalize only radius
   dummy = torch.zeros(B, 1, 3, device=device)
   dummy[:, 0, 0] = r_normalized_val.squeeze()
   dummy[:, 0, 1] = theta_best  # pass through, won't be changed
   dummy[:, 0, 2] = phi_best    # pass through, won't be changed

   unnorm = self.normalizer["action"].unnormalize(dummy)
   r_unnorm = unnorm[:, 0, 0]      # only this changes
   theta_unnorm = unnorm[:, 0, 1]  # unchanged
   phi_unnorm = unnorm[:, 0, 2]    # unchanged
   ```

---

## Impact

### Before Fixes
- Training: Energy learned in *wrong* coordinate space (r_norm, θ_corrupted, φ_corrupted)
- Inference: Querying in *different* coordinate space (r_norm, θ_rad, φ_rad)
- Result: Energy function was meaningless → random/zero actions → 0% success

### After Fixes
- Training: Energy learned in (r_norm, θ_rad, φ_rad)
- Inference: Querying in (r_norm, θ_rad, φ_rad)
- Result: Consistent coordinate space → energy function meaningful → policy should work

---

## Files Modified

1. `fvf/utils/action_utils.py` - Fixed division by zero
2. `fvf/policy/spherical_bessel_implicit_policy.py` - Fixed coordinate handling in both training and inference

---

## Testing Recommendations

1. **Verify debug script**:
   ```bash
   conda run -n g2s_new python debug_action_flow.py
   ```
   Should show no NaN warnings

2. **Restart training** from scratch:
   - Previous checkpoints learned the wrong energy function
   - Need fresh training with fixed coordinate handling

3. **Monitor metrics**:
   - Loss should still decrease (as before)
   - **NEW**: Success rate should now improve
   - Watch for `test_mean_score` > 0

4. **Check action outputs**:
   - Actions should be non-zero
   - Actions should vary based on observation
   - Energy visualization should show meaningful peaks

---

## Technical Details

### Coordinate System Used
- **Cartesian**: (x, y, z) - environment actions
- **Spherical**: (r, θ, φ) where:
  - r = √(x² + y² + z²)
  - θ = arccos(z/r) ∈ [0, π] (polar angle from +z axis)
  - φ = arctan2(y, x) + π ∈ [0, 2π] (azimuthal angle in xy-plane)

### Normalization Strategy
- **Radius (r)**: Normalized to [-1, 1] using LinearNormalizer
- **Angles (θ, φ)**: NOT normalized, kept in radians
- **Rationale**: Angles have natural bounded ranges; only magnitude needs normalization

### Energy Function Training
- **Method**: InfoNCE contrastive loss
- **Input**: (obs_features, spherical_actions)
- **Action space**: (r_norm ∈ [-1,1], θ ∈ [0,π], φ ∈ [0,2π])
- **Outputs**: Energy values (higher = better)

### Inference
- **Grid**: 100 radii × 20 theta × 40 phi ≈ 80K candidates
- **Selection**: Softmax over energies → sample or argmax
- **Output**: Cartesian action (x, y, z)
