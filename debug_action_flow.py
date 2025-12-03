"""
Debug script to trace action coordinate flow in drone training.
"""
import numpy as np
import sys
sys.path.insert(0, '/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.utils import action_utils

print("=" * 80)
print("DEBUGGING ACTION COORDINATE FLOW")
print("=" * 80)

# Simulate some sample Cartesian actions (x, y, z)
cartesian_actions = np.array([
    [0.1, 0.0, 0.05],  # Small movement in +x, +z
    [0.0, 0.1, 0.05],  # Small movement in +y, +z
    [0.0, 0.0, 0.1],   # Vertical movement
    [0.0, 0.0, 0.0],   # Zero action (CRITICAL TEST)
    [0.5, 0.5, 0.5],   # Larger diagonal movement
])

print("\n1. Original Cartesian Actions (x, y, z):")
print(cartesian_actions)

# Convert to spherical
spherical_actions = action_utils.convert_to_spherical(cartesian_actions)
print("\n2. Spherical Actions (r, theta, phi):")
print(spherical_actions)
print(f"   r range: [{spherical_actions[:, 0].min():.4f}, {spherical_actions[:, 0].max():.4f}]")
print(f"   theta range: [{spherical_actions[:, 1].min():.4f}, {spherical_actions[:, 1].max():.4f}] (0 to π)")
print(f"   phi range: [{spherical_actions[:, 2].min():.4f}, {spherical_actions[:, 2].max():.4f}] (0 to 2π)")

# Check for NaN values
if np.any(np.isnan(spherical_actions)):
    print("\n WARNING: NaN values detected in spherical conversion!")
    print(f"   NaN indices: {np.where(np.isnan(spherical_actions))}")

# Simulate normalization (what dataset does)
# Assume action normalizer normalizes to [-1, 1]
r_values = spherical_actions[:, 0:1]
r_min, r_max = r_values.min(), r_values.max()
r_normalized = 2 * (r_values - r_min) / (r_max - r_min + 1e-8) - 1

# Create "normalized" actions (only r is normalized)
normalized_actions = spherical_actions.copy()
normalized_actions[:, 0] = r_normalized.flatten()

print("\n3. 'Normalized' Actions (r_norm, theta, phi):")
print(normalized_actions)
print(f"   r_norm range: [{normalized_actions[:, 0].min():.4f}, {normalized_actions[:, 0].max():.4f}]")

# Now simulate what happens in compute_loss()
print("\n" + "=" * 80)
print("SIMULATING compute_loss() - CURRENT (BUGGY) BEHAVIOR")
print("=" * 80)

# Current buggy code does:
# r = targets[:, :, 0, 0]  # normalized
# theta = self.normalizer["action"].unnormalize(targets)[:, :, 0, 1]
# phi = self.normalizer["action"].unnormalize(targets)[:, :, 0, 2]

# Simulating unnormalize on angles (which is wrong!)
def fake_unnormalize(normalized_val, min_val, max_val):
    """Simulate unnormalization: x_unnorm = (x_norm + 1) / 2 * (max - min) + min"""
    return (normalized_val + 1) / 2 * (max_val - min_val) + min_val

# If unnormalize is applied to theta/phi (which are already in radians)
theta_wrongly_unnormalized = fake_unnormalize(normalized_actions[:, 1], r_min, r_max)
phi_wrongly_unnormalized = fake_unnormalize(normalized_actions[:, 2], r_min, r_max)

print("\nBUGGY: Theta after wrong unnormalization:")
print(f"   Original theta: {normalized_actions[:, 1]}")
print(f"   After unnorm: {theta_wrongly_unnormalized}")
print(f"   ⚠️  This is WRONG! Theta should stay in [0, π]")

print("\nBUGGY: Phi after wrong unnormalization:")
print(f"   Original phi: {normalized_actions[:, 2]}")
print(f"   After unnorm: {phi_wrongly_unnormalized}")
print(f"   ⚠️  This is WRONG! Phi should stay in [0, 2π]")

# Now simulate get_action()
print("\n" + "=" * 80)
print("SIMULATING get_action() - CURRENT (BUGGY) BEHAVIOR")
print("=" * 80)

# In get_action(), the grid is created with:
# r = torch.linspace(action_stats["min"][0], action_stats["max"][0], n_k)
# This means r is in NORMALIZED space [-1, 1]

# Then for reconstruction:
# r_unnorm = unnormalize(r_normalized)
# x = r_unnorm * sin(theta) * cos(phi)  <- theta, phi are from grid (in radians)

print("\nIn get_action():")
print("  - Grid r: in normalized space [-1, 1]")
print("  - Grid theta, phi: in radian space [0, π], [0, 2π]")
print("  - Action grid mixes NORMALIZED radius with RADIAN angles")
print("  - This is INCONSISTENT with training!")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("""
🔴 BUG #1: Inconsistent coordinate spaces
  Location: get_action() line 62-64
  Issue: Action grid uses normalized r but radian theta/phi

🔴 BUG #2: Wrong unnormalization in loss
  Location: compute_loss() lines 172-173
  Issue: Unnecessarily unnormalizes theta/phi (they're already in radians)

🔴 BUG #3: Division by zero
  Location: action_utils.py line 38
  Issue: arccos(z/r) fails when r=0, produces NaN

🔴 BUG #4: Likely cause of "always 0" results
  The inconsistent coordinate spaces mean:
  - Training learns energy in (r_norm, theta_rad, phi_rad) space
  - Inference queries in mixed coordinate space
  - Energy function cannot generalize properly
  - Policy outputs random/zero actions
""")

print("\n" + "=" * 80)
print("RECOMMENDED FIXES")
print("=" * 80)

print("""
✅ FIX #1: Make coordinate spaces consistent
  In compute_loss():
  - Don't unnormalize theta/phi (keep them in radians)
  - Use: sphere_act = torch.cat([r, theta, phi], dim=2)

✅ FIX #2: Fix action grid creation
  In get_action():
  - Create r in UNNORMALIZED space (actual distances)
  - Then normalize r before passing to energy_head
  - OR: Keep everything in normalized space consistently

✅ FIX #3: Handle zero radius
  In action_utils.py:
  - Add epsilon: theta = np.arccos(np.clip(action[:, 2] / (r + 1e-8), -1, 1))
""")
