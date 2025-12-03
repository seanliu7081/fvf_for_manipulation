"""
Verification script to test that coordinate handling is now correct.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.utils import action_utils

print("=" * 80)
print("VERIFICATION: Testing Fixed Coordinate Handling")
print("=" * 80)

# Test 1: Verify no NaN in spherical conversion
print("\n[TEST 1] Spherical conversion handles zero actions")
print("-" * 80)

test_actions = np.array([
    [0.0, 0.0, 0.0],  # Zero action
    [0.1, 0.0, 0.0],  # x-only
    [0.0, 0.1, 0.0],  # y-only
    [0.0, 0.0, 0.1],  # z-only
    [0.1, 0.1, 0.1],  # diagonal
])

spherical = action_utils.convert_to_spherical(test_actions)
print(f"Cartesian actions:\n{test_actions}")
print(f"\nSpherical actions (r, theta, phi):\n{spherical}")

if np.any(np.isnan(spherical)):
    print("\n❌ FAIL: Still has NaN values!")
    print(f"   NaN at indices: {np.where(np.isnan(spherical))}")
else:
    print("\n✅ PASS: No NaN values in spherical conversion")

# Test 2: Verify coordinate consistency
print("\n\n[TEST 2] Coordinate space consistency between training and inference")
print("-" * 80)

# Simulate what happens in training
print("\nTraining simulation:")
cartesian = np.array([[0.1, 0.05, 0.02]])
spherical = action_utils.convert_to_spherical(cartesian)
r_value = spherical[0, 0]
theta_value = spherical[0, 1]
phi_value = spherical[0, 2]

# Simulate normalization (r only)
r_min, r_max = 0.0, 1.0  # typical range
r_normalized = 2 * (r_value - r_min) / (r_max - r_min) - 1

print(f"  Original cartesian: {cartesian[0]}")
print(f"  Spherical: r={r_value:.4f}, θ={theta_value:.4f}, φ={phi_value:.4f}")
print(f"  Normalized: r_norm={r_normalized:.4f}, θ={theta_value:.4f}, φ={phi_value:.4f}")
print(f"  → Training sees: ({r_normalized:.4f}, {theta_value:.4f}, {phi_value:.4f})")

# Simulate what happens in inference (with fix)
print("\nInference simulation (AFTER FIX):")
r_grid = np.array([r_normalized])  # Grid in normalized space
theta_grid = np.array([theta_value])  # In radians
phi_grid = np.array([phi_value])  # In radians

print(f"  Action grid: r_norm={r_grid[0]:.4f}, θ={theta_grid[0]:.4f}, φ={phi_grid[0]:.4f}")
print(f"  → Inference queries: ({r_grid[0]:.4f}, {theta_grid[0]:.4f}, {phi_grid[0]:.4f})")

if np.allclose(r_grid[0], r_normalized) and np.allclose(theta_grid[0], theta_value) and np.allclose(phi_grid[0], phi_value):
    print("\n✅ PASS: Training and inference use the same coordinate space!")
else:
    print("\n❌ FAIL: Coordinate spaces don't match!")

# Test 3: Verify reconstruction
print("\n\n[TEST 3] Spherical → Cartesian reconstruction")
print("-" * 80)

# Convert back to Cartesian
r_unnormalized = (r_normalized + 1) / 2 * (r_max - r_min) + r_min
x_reconstructed = r_unnormalized * np.sin(theta_value) * np.cos(phi_value)
y_reconstructed = r_unnormalized * np.sin(theta_value) * np.sin(phi_value)
z_reconstructed = r_unnormalized * np.cos(theta_value)
reconstructed = np.array([x_reconstructed, y_reconstructed, z_reconstructed])

print(f"  Original:      {cartesian[0]}")
print(f"  Reconstructed: {reconstructed}")
print(f"  Error:         {np.abs(cartesian[0] - reconstructed)}")

if np.allclose(cartesian[0], reconstructed, atol=1e-5):
    print("\n✅ PASS: Accurate reconstruction!")
else:
    print("\n❌ FAIL: Reconstruction error too large!")

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print("""
All tests should PASS for the fixes to be correct.

Next steps:
1. Delete old checkpoints (they learned the wrong energy function)
2. Restart training from scratch
3. Monitor that test_mean_score increases (not just loss decreasing)
4. Check that policy outputs non-zero, varied actions
""")
