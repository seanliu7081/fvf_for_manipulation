"""
SphericalBesselHarmonics - FAST version with fixes

Fixes:
1. Scale factor to make logits range larger
2. More radial points (n_k should be higher)
3. PRECOMPUTE basis functions for speed (no scipy in forward pass)
4. Use PyTorch for everything possible
"""

import torch
import numpy as np
from torch import nn
from scipy.special import spherical_jn, sph_harm
from typing import Optional, Dict, Tuple
import lie_learn.spaces.S2 as S2


class SphericalBesselHarmonics:
    """
    Fast Spherical Bessel Harmonics with precomputation.
    
    Key optimizations:
    1. Precompute basis functions on a grid
    2. Use interpolation for arbitrary points (or nearest neighbor)
    3. Scale output for better softmax differentiation
    """
    
    def __init__(
        self,
        n_max: int,
        l_max: int,
        n_k: int,
        R_max: float = 1.0,
        num_theta: int = 20,
        grid_type: str = "lie_learn",
        scale_factor: float = 10.0,  # Scale up energy
        n_r_grid: int = 20,  # Radial grid for precomputation
    ):
        self.n_max = n_max
        self.l_max = l_max
        self.L = l_max
        self.n_k = n_k
        self.R_max = R_max
        self.scale_factor = scale_factor
        self.n_r_grid = n_r_grid
        
        # Angular grid
        if grid_type == "lie_learn":
            self.grid = S2.meshgrid(num_theta, grid_type="Driscoll-Healy")
            theta, phi = self.grid
            self.num_theta = theta.shape[0]
            self.num_phi = theta.shape[1]
        
        # Build basis indices
        self.basis_indices = []
        for n in range(n_max + 1):
            for k_idx in range(1, n_k + 1):
                for l in range(l_max + 1):
                    for m in range(-l, l + 1):
                        self.basis_indices.append((n, k_idx, l, m, 'real'))
                        self.basis_indices.append((n, k_idx, l, m, 'imag'))
        
        self.num_basis = len(self.basis_indices)
        
        # Precompute basis on grid for FAST evaluation
        self._precompute_basis_grid()
        
        print(f"SphericalBesselHarmonicsFast initialized:")
        print(f"  n_max={n_max}, l_max={l_max}, n_k={n_k}")
        print(f"  num_basis={self.num_basis}")
        print(f"  scale_factor={scale_factor}")
        print(f"  Precomputed grid: {n_r_grid} x {self.num_theta} x {self.num_phi}")
    
    def _precompute_basis_grid(self):
        """Precompute all basis functions on a dense grid."""
        # Create dense grid
        r_grid = np.linspace(0.01, self.R_max, self.n_r_grid)  # Avoid r=0
        theta_grid = self.grid[0][:, 0]  # [num_theta]
        phi_grid = self.grid[1][0, :]    # [num_phi]
        
        self.r_grid_np = r_grid
        self.theta_grid_np = theta_grid
        self.phi_grid_np = phi_grid
        
        # Precompute radial parts: [num_basis // 2, n_r_grid]
        # (divide by 2 because real/imag share same radial part)
        n_radial_basis = (self.n_max + 1) * self.n_k
        self.radial_values = np.zeros((n_radial_basis, self.n_r_grid))
        
        radial_idx = 0
        for n in range(self.n_max + 1):
            for k_idx in range(1, self.n_k + 1):
                k = k_idx * np.pi / self.R_max
                for i, r in enumerate(r_grid):
                    kr = k * r
                    self.radial_values[radial_idx, i] = spherical_jn(n, kr)
                radial_idx += 1
        
        # Precompute angular parts: [num_lm * 2, num_theta, num_phi]
        num_lm = sum(2*l + 1 for l in range(self.l_max + 1))
        self.angular_real = np.zeros((num_lm, self.num_theta, self.num_phi))
        self.angular_imag = np.zeros((num_lm, self.num_theta, self.num_phi))
        
        lm_idx = 0
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                for i, theta in enumerate(theta_grid):
                    for j, phi in enumerate(phi_grid):
                        Y_lm = sph_harm(m, l, phi, theta)
                        self.angular_real[lm_idx, i, j] = np.real(Y_lm)
                        self.angular_imag[lm_idx, i, j] = np.imag(Y_lm)
                lm_idx += 1
        
        # Convert to torch tensors
        self.radial_torch = torch.tensor(self.radial_values, dtype=torch.float32)
        self.angular_real_torch = torch.tensor(self.angular_real, dtype=torch.float32)
        self.angular_imag_torch = torch.tensor(self.angular_imag, dtype=torch.float32)
        
        # Build mapping from basis_idx to (radial_idx, lm_idx, part)
        self.basis_to_components = []
        for n in range(self.n_max + 1):
            for k_idx in range(1, self.n_k + 1):
                radial_idx = n * self.n_k + (k_idx - 1)
                lm_idx = 0
                for l in range(self.l_max + 1):
                    for m in range(-l, l + 1):
                        self.basis_to_components.append((radial_idx, lm_idx, 'real'))
                        self.basis_to_components.append((radial_idx, lm_idx, 'imag'))
                        lm_idx += 1
    
    def _find_nearest_indices(self, r, theta, phi):
        """Find nearest grid indices for given coordinates."""
        # Radial index
        r_np = r.cpu().numpy()
        r_idx = np.argmin(np.abs(self.r_grid_np[:, None] - r_np[None, :]), axis=0)
        
        # Theta index
        theta_np = theta.cpu().numpy()
        theta_idx = np.argmin(np.abs(self.theta_grid_np[:, None] - theta_np[None, :]), axis=0)
        
        # Phi index  
        phi_np = phi.cpu().numpy()
        phi_idx = np.argmin(np.abs(self.phi_grid_np[:, None] - phi_np[None, :]), axis=0)
        
        return r_idx, theta_idx, phi_idx
    
    def forward(
        self,
        coeffs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast evaluation using precomputed basis.
        
        Args:
            coeffs: [B*N, num_basis]
            actions: [B*N, 3] as (r, θ, φ)
            
        Returns:
            [B*N] energy values
        """
        B_N = coeffs.shape[0]
        device = coeffs.device
        
        r = actions[:, 0]
        theta = actions[:, 1]
        phi = actions[:, 2]
        
        # Find nearest grid indices
        r_idx, theta_idx, phi_idx = self._find_nearest_indices(r, theta, phi)
        
        # Move precomputed tensors to device
        radial = self.radial_torch.to(device)
        angular_real = self.angular_real_torch.to(device)
        angular_imag = self.angular_imag_torch.to(device)
        
        # Compute basis values for each point: [B*N, num_basis]
        basis_values = torch.zeros(B_N, self.num_basis, device=device)
        
        for basis_idx, (radial_idx, lm_idx, part) in enumerate(self.basis_to_components):
            # Get radial values at the grid points nearest to query points
            rad_vals = radial[radial_idx, r_idx]  # [B*N]
            
            # Get angular values
            if part == 'real':
                ang_vals = angular_real[lm_idx, theta_idx, phi_idx]  # [B*N]
            else:
                ang_vals = angular_imag[lm_idx, theta_idx, phi_idx]
            
            ang_vals = torch.tensor(ang_vals, device=device, dtype=torch.float32)
            basis_values[:, basis_idx] = rad_vals * ang_vals
        
        # Energy = coeffs · basis_values
        energy = (coeffs * basis_values).sum(dim=1)
        
        # Scale up
        energy = energy * self.scale_factor
        
        return energy
    
    def __call__(self, coeffs, actions):
        return self.forward(coeffs, actions)


class SphericalBesselEnergyMLPFast(nn.Module):
    """
    Fast energy head with proper scaling.
    """
    
    def __init__(
        self,
        obs_feat_dim: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        spec_norm: bool,
        n_max: int = 1,
        l_max: int = 3,
        n_k: int = 5,  # Increased from 2!
        R_max: float = 1.0,
        num_theta: int = 20,
        scale_factor: float = 100.0,
        initialize: bool = True,
    ):
        super().__init__()
        
        from fvf.model.modules.layers import MLP
        
        self.n_max = n_max
        self.l_max = l_max
        self.n_k = n_k
        self.num_radii = n_k
        self.R_max = R_max
        
        self.sbh = SphericalBesselHarmonics(
            n_max=n_max,
            l_max=l_max,
            n_k=n_k,
            R_max=R_max,
            num_theta=num_theta,
            grid_type="lie_learn",
            scale_factor=scale_factor,
        )
        
        self.sh = self.sbh
        
        self.energy_mlp = MLP(
            [obs_feat_dim] + [mlp_dim] * num_layers + [self.sbh.num_basis],
            dropout=dropout,
            act_out=False,
            spec_norm=spec_norm,
        )
        
        print(f"SphericalBesselEnergyMLPFast:")
        print(f"  MLP output: {self.sbh.num_basis}")
        print(f"  scale_factor: {scale_factor}")
    
    def forward(
        self,
        obs_feat: torch.Tensor,
        actions: torch.Tensor,
        return_coeffs: bool = False,
    ):
        B = obs_feat.shape[0]
        
        if actions.dim() == 4:
            actions = actions.squeeze(2)
        
        B, N, D = actions.shape
        
        coeffs = self.energy_mlp(obs_feat)
        coeffs_expanded = coeffs.unsqueeze(1).expand(-1, N, -1)
        coeffs_flat = coeffs_expanded.reshape(B * N, -1)
        
        actions_flat = actions.reshape(B * N, 3)
        energy_flat = self.sbh(coeffs_flat, actions_flat)
        energy = energy_flat.view(B, N)
        
        if return_coeffs:
            return energy, coeffs
        else:
            return energy


# ============================================================
# SIMPLEST FIX: Just add scaling to existing implementation
# ============================================================

def patch_existing_sbh_forward(sbh_instance, scale_factor=100.0):
    """
    Monkey-patch an existing SphericalBesselHarmonics to add scaling.
    
    Usage:
        from patch import patch_existing_sbh_forward
        patch_existing_sbh_forward(self.energy_head.sbh, scale_factor=100.0)
    """
    original_forward = sbh_instance.forward
    
    def scaled_forward(coeffs, actions=None):
        energy = original_forward(coeffs, actions)
        return energy * scale_factor
    
    sbh_instance.forward = scaled_forward
    sbh_instance.__call__ = scaled_forward
    print(f"Patched SBH with scale_factor={scale_factor}")


if __name__ == "__main__":
    print("Testing SphericalBesselHarmonicsFast...")
    
    sbh = SphericalBesselHarmonics(
        n_max=1,
        l_max=3,
        n_k=5,
        R_max=1.0,
        num_theta=20,
        scale_factor=100.0,
    )
    
    # Test
    B_N = 3200
    coeffs = torch.randn(B_N, sbh.num_basis)
    actions = torch.rand(B_N, 3)
    actions[:, 0] = actions[:, 0] * 0.9 + 0.1  # r in [0.1, 1.0], avoid r=0
    actions[:, 1] *= np.pi
    actions[:, 2] *= 2 * np.pi
    
    import time
    start = time.time()
    energy = sbh(coeffs, actions)
    elapsed = time.time() - start
    
    print(f"\nOutput shape: {energy.shape}")
    print(f"Energy stats: mean={energy.mean():.4f}, std={energy.std():.4f}, "
          f"min={energy.min():.4f}, max={energy.max():.4f}")
    print(f"Energy range: {(energy.max() - energy.min()):.4f}")
    print(f"Time: {elapsed:.3f}s for {B_N} points")
    
    if energy.std() < 1.0:
        print("\n Energy variance is low - consider increasing scale_factor")
    else:
        print("\n Energy has good variance!")