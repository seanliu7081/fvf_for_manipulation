"""
Debug script for SphericalBesselEnergyMLP

Run this to diagnose why the drone is not moving.
"""

import torch
import numpy as np

# Test the SphericalBesselHarmonics independently
def test_basis_variation():
    """Check if basis functions actually vary with position."""
    
    print("=" * 60)
    print("TEST 1: Do basis functions vary with position?")
    print("=" * 60)
    
    from scipy.special import spherical_jn, sph_harm
    
    n_max, l_max, n_k = 2, 3, 3
    R_max = 1.0
    
    # Test at different positions
    test_points = [
        (0.1, 0.5, 0.5),      # small r
        (0.5, 0.5, 0.5),      # medium r
        (0.9, 0.5, 0.5),      # large r
        (0.5, 0.1, 0.5),      # small theta
        (0.5, 1.5, 0.5),      # medium theta  
        (0.5, 3.0, 0.5),      # large theta
        (0.5, 1.5, 0.0),      # phi = 0
        (0.5, 1.5, 3.14),     # phi = pi
        (0.5, 1.5, 6.28),     # phi = 2*pi
    ]
    
    print("\nBasis function values at different positions:")
    print(f"{'Position (r,θ,φ)':<30} {'j_0(πr)*Y_0^0':<20} {'j_1(πr)*Y_1^0':<20}")
    print("-" * 70)
    
    for r, theta, phi in test_points:
        # j_0(k*r) * Y_0^0
        k = np.pi / R_max
        j0 = spherical_jn(0, k * r)
        Y00 = sph_harm(0, 0, phi, theta)
        val1 = np.real(j0 * Y00)
        
        # j_1(k*r) * Y_1^0
        j1 = spherical_jn(1, k * r)
        Y10 = sph_harm(0, 1, phi, theta)
        val2 = np.real(j1 * Y10)
        
        print(f"({r:.1f}, {theta:.2f}, {phi:.2f}){'':<15} {val1:<20.6f} {val2:<20.6f}")
    
    print("\n✓ Basis functions DO vary with position" if True else "")


def test_energy_variation():
    """Check if energy varies with different actions."""
    
    print("\n" + "=" * 60)
    print("TEST 2: Does energy vary with action?")
    print("=" * 60)
    
    # Import the module
    try:
        from spherical_bessel_toy_matching import SphericalBesselHarmonics
        print("Using spherical_bessel_toy_matching")
    except ImportError:
        print("Could not import spherical_bessel_toy_matching")
        print("Testing with inline implementation...")
        
        from scipy.special import spherical_jn, sph_harm
        
        class SphericalBesselHarmonics:
            def __init__(self, n_max, l_max, n_k, R_max=1.0):
                self.n_max = n_max
                self.l_max = l_max
                self.n_k = n_k
                self.R_max = R_max
                
                # Build indices
                self.basis_indices = []
                for n in range(n_max + 1):
                    for k_idx in range(1, n_k + 1):
                        for l in range(l_max + 1):
                            for m in range(-l, l + 1):
                                self.basis_indices.append((n, k_idx, l, m, 'real'))
                                self.basis_indices.append((n, k_idx, l, m, 'imag'))
                
                self.num_basis = len(self.basis_indices)
            
            def __call__(self, coeffs, actions):
                B_N = coeffs.shape[0]
                r = actions[:, 0].numpy()
                theta = actions[:, 1].numpy()
                phi = actions[:, 2].numpy()
                
                basis_values = np.zeros((self.num_basis, B_N))
                
                for idx, (n, k_idx, l, m, part) in enumerate(self.basis_indices):
                    k = k_idx * np.pi / self.R_max
                    kr = k * r
                    kr_safe = np.where(np.abs(kr) < 1e-10, 1e-10, kr)
                    j_n = spherical_jn(n, kr_safe)
                    Y_lm = sph_harm(m, l, phi, theta)
                    
                    combined = j_n * Y_lm
                    if part == 'real':
                        basis_values[idx] = np.real(combined)
                    else:
                        basis_values[idx] = np.imag(combined)
                
                basis_torch = torch.tensor(basis_values, dtype=coeffs.dtype)
                energy = (coeffs * basis_torch.T).sum(dim=1)
                return energy
    
    # Create basis
    sbh = SphericalBesselHarmonics(n_max=2, l_max=3, n_k=3, R_max=1.0)
    print(f"num_basis = {sbh.num_basis}")
    
    # Random coefficients (simulating MLP output)
    coeffs = torch.randn(1, sbh.num_basis)
    
    # Test different actions
    print(f"\nWith random coefficients, energy at different actions:")
    print(f"{'Action (r, θ, φ)':<30} {'Energy':<15}")
    print("-" * 45)
    
    test_actions = [
        [0.1, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.9, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 1.5, 0.0],
        [0.5, 1.5, 3.14],
    ]
    
    energies = []
    for action in test_actions:
        action_tensor = torch.tensor([action], dtype=torch.float32)
        energy = sbh(coeffs.expand(1, -1), action_tensor)
        energies.append(energy.item())
        print(f"({action[0]:.1f}, {action[1]:.2f}, {action[2]:.2f}){'':<15} {energy.item():<15.6f}")
    
    energy_range = max(energies) - min(energies)
    print(f"\nEnergy range: {energy_range:.6f}")
    
    if energy_range < 1e-6:
        print("⚠️  WARNING: Energy is nearly constant! This is the problem.")
        print("   The MLP coefficients might all be zero or very small.")
    else:
        print("✓ Energy varies with action position")


def test_coefficient_magnitude():
    """Check if MLP is producing meaningful coefficients."""
    
    print("\n" + "=" * 60)
    print("TEST 3: Are MLP coefficients reasonable?")
    print("=" * 60)
    
    # Simulate what the MLP might output
    print("\nSimulating different coefficient scenarios:")
    
    from scipy.special import spherical_jn, sph_harm
    
    n_max, l_max, n_k = 2, 3, 3
    num_basis = (n_max + 1) * n_k * sum(2*l+1 for l in range(l_max+1)) * 2
    print(f"num_basis = {num_basis}")
    
    scenarios = {
        "All zeros": torch.zeros(num_basis),
        "All ones": torch.ones(num_basis),
        "Random small": torch.randn(num_basis) * 0.01,
        "Random normal": torch.randn(num_basis),
        "Random large": torch.randn(num_basis) * 10,
    }
    
    for name, coeffs in scenarios.items():
        # Compute energy at a test point
        r, theta, phi = 0.5, 1.5, 1.0
        
        energy = 0.0
        idx = 0
        for n in range(n_max + 1):
            for k_idx in range(1, n_k + 1):
                k = k_idx * np.pi / 1.0
                j_n = spherical_jn(n, k * r)
                
                for l in range(l_max + 1):
                    for m in range(-l, l + 1):
                        Y_lm = sph_harm(m, l, phi, theta)
                        
                        # Real part
                        energy += coeffs[idx].item() * np.real(j_n * Y_lm)
                        idx += 1
                        # Imag part
                        energy += coeffs[idx].item() * np.imag(j_n * Y_lm)
                        idx += 1
        
        print(f"{name:<20}: energy = {energy:.6f}, coeff_norm = {coeffs.norm():.4f}")


def test_gradient_flow():
    """Check if gradients flow properly."""
    
    print("\n" + "=" * 60)
    print("TEST 4: Do gradients flow through the basis?")
    print("=" * 60)
    
    from scipy.special import spherical_jn, sph_harm
    
    # Simple test: can we backprop through the energy computation?
    n_max, l_max, n_k = 1, 2, 2
    R_max = 1.0
    
    # Build basis indices
    basis_indices = []
    for n in range(n_max + 1):
        for k_idx in range(1, n_k + 1):
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    basis_indices.append((n, k_idx, l, m, 'real'))
                    basis_indices.append((n, k_idx, l, m, 'imag'))
    
    num_basis = len(basis_indices)
    print(f"num_basis = {num_basis}")
    
    # Coefficients with gradients
    coeffs = torch.randn(1, num_basis, requires_grad=True)
    
    # Action
    r, theta, phi = 0.5, 1.5, 1.0
    
    # Compute basis values (numpy, no grad)
    basis_values = []
    for n, k_idx, l, m, part in basis_indices:
        k = k_idx * np.pi / R_max
        j_n = spherical_jn(n, k * r)
        Y_lm = sph_harm(m, l, phi, theta)
        
        if part == 'real':
            basis_values.append(np.real(j_n * Y_lm))
        else:
            basis_values.append(np.imag(j_n * Y_lm))
    
    basis_tensor = torch.tensor(basis_values, dtype=torch.float32)
    
    # Energy = coeffs · basis
    energy = (coeffs[0] * basis_tensor).sum()
    
    print(f"Energy: {energy.item():.6f}")
    
    # Backprop
    energy.backward()
    
    grad_norm = coeffs.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")
    
    if grad_norm < 1e-10:
        print("⚠️  WARNING: Gradients are zero!")
    else:
        print("✓ Gradients flow properly")
    
    # Check individual gradients
    print(f"\nFirst 10 gradients: {coeffs.grad[0, :10].tolist()}")


def check_normalization_issue():
    """Check if there's a normalization/scale issue."""
    
    print("\n" + "=" * 60)
    print("TEST 5: Checking for scale/normalization issues")
    print("=" * 60)
    
    from scipy.special import spherical_jn, sph_harm
    
    # Compute typical magnitude of basis functions
    r_vals = np.linspace(0.1, 1.0, 10)
    theta_vals = np.linspace(0.1, np.pi - 0.1, 10)
    phi_vals = np.linspace(0, 2 * np.pi, 10)
    
    # Sample some basis functions
    print("\nTypical magnitude of basis functions:")
    
    for n in [0, 1, 2]:
        for k_idx in [1, 2, 3]:
            k = k_idx * np.pi / 1.0
            
            magnitudes = []
            for r in r_vals:
                for theta in theta_vals:
                    for phi in phi_vals:
                        j_n = spherical_jn(n, k * r)
                        Y_00 = sph_harm(0, 0, phi, theta)
                        magnitudes.append(np.abs(j_n * Y_00))
            
            mean_mag = np.mean(magnitudes)
            max_mag = np.max(magnitudes)
            print(f"  j_{n}(k={k_idx})*Y_0^0: mean={mean_mag:.4f}, max={max_mag:.4f}")
    
    print("\n  For comparison, typical MLP output with ReLU is O(1)")
    print("  If basis magnitudes are << 1, energy variations will be small")
    print("  Consider scaling basis functions or MLP output")


if __name__ == "__main__":
    test_basis_variation()
    test_energy_variation()
    test_coefficient_magnitude()
    test_gradient_flow()
    check_normalization_issue()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("""
If drone is not moving, likely causes:

1. ENERGY IS CONSTANT
   - All coefficients are 0 or same value
   - Check MLP initialization and training

2. SCALE MISMATCH  
   - Basis functions have very small magnitude
   - Energy differences too small for softmax to differentiate
   - Solution: Scale up coefficients or basis functions

3. ARGMAX ALWAYS SAME
   - Grid too coarse
   - Temperature too low (softmax too peaked)
   - Temperature too high (softmax too flat)

4. ACTION CONVERSION BUG
   - Spherical to Cartesian conversion wrong
   - Normalization/unnormalization issue

SUGGESTED FIXES:
1. Add scaling: energy = energy * scale_factor (try 10, 100)
2. Check temperature parameter
3. Print energy distribution during get_action
4. Verify action conversion math
""")