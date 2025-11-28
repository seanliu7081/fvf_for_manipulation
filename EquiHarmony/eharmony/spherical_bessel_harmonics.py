import torch
import numpy as np
from torch import nn
from scipy.special import spherical_jn, sph_harm
from typing import Optional
import lie_learn.spaces.S2 as S2

from fvf.model.modules.layers import MLP
import math


# class SphericalBesselHarmonics:
#     """
#     Spherical Bessel Harmonics basis functions.
    
#     Basis: Ψ_{n,k,l,m}(r,θ,φ) = j_n(k·r) · Y_l^m(θ,φ)
    
#     where:
#         - j_n is spherical Bessel function of order n
#         - k = k_idx · π / R_max are radial frequencies
#         - Y_l^m are spherical harmonics
        
#     We store both real and imaginary parts of Y_l^m as separate basis functions.
    
#     Args:
#         n_max: Maximum order for Bessel function (n = 0, 1, ..., n_max)
#         l_max: Maximum degree for spherical harmonics (l = 0, 1, ..., l_max)
#         n_k: Number of radial frequencies (k_idx = 1, 2, ..., n_k)
#         R_max: Maximum radius for frequency calculation
#         num_theta: Angular grid resolution
#         grid_type: Type of angular grid
#         scale_factor: Scale factor for output energy
#         n_r_grid: Radial grid resolution for precomputation
#     """
    
#     def __init__(
#         self,
#         n_max: int,
#         l_max: int,
#         n_k: int,
#         R_max: float = 1.0,
#         num_theta: int = 20,
#         grid_type: str = "lie_learn",
#         scale_factor: float = 10.0,
#         n_r_grid: int = 20,
#     ):
#         self.n_max = n_max
#         self.l_max = l_max
#         self.L = l_max  # Alias for compatibility
#         self.n_k = n_k
#         self.R_max = R_max
#         self.scale_factor = scale_factor
#         self.n_r_grid = n_r_grid
        
#         # Angular grid (for get_action evaluation)
#         if grid_type == "lie_learn":
#             self.grid = S2.meshgrid(num_theta, grid_type="Driscoll-Healy")
#             theta, phi = self.grid
#             self.num_theta = theta.shape[0]
#             self.num_phi = theta.shape[1]
#         else:
#             theta = np.linspace(0, np.pi, num_theta)
#             phi = np.linspace(0, 2 * np.pi, num_theta * 2)
#             theta, phi = np.meshgrid(theta, phi, indexing='ij')
#             self.grid = (theta, phi)
#             self.num_theta = theta.shape[0]
#             self.num_phi = theta.shape[1]
        
#         # Build basis indices: (n, k_idx, l, m, part)
#         # part is 'real' or 'imag'
#         self.basis_indices = []
#         for n in range(n_max + 1):
#             for k_idx in range(1, n_k + 1):
#                 for l in range(l_max + 1):
#                     for m in range(-l, l + 1):
#                         self.basis_indices.append((n, k_idx, l, m, 'real'))
#                         self.basis_indices.append((n, k_idx, l, m, 'imag'))
        
#         self.num_basis = len(self.basis_indices)
        
#         # Precompute basis on grid for fast evaluation
#         self._precompute_basis_grid()
        
#         print(f"SphericalBesselHarmonics initialized:")
#         print(f"  n_max={n_max}, l_max={l_max}, n_k={n_k}")
#         print(f"  num_basis={self.num_basis}")
#         print(f"  scale_factor={scale_factor}")
#         print(f"  Grid: {n_r_grid} x {self.num_theta} x {self.num_phi}")
    
#     def _precompute_basis_grid(self):
#         """Precompute all basis functions on a dense grid for fast lookup."""
#         # Radial grid
#         r_grid = np.linspace(0.01, self.R_max, self.n_r_grid)  # Avoid r=0
#         theta_grid = self.grid[0][:, 0]  # [num_theta]
#         phi_grid = self.grid[1][0, :]    # [num_phi]
        
#         self.r_grid_np = r_grid
#         self.theta_grid_np = theta_grid
#         self.phi_grid_np = phi_grid
        
#         # Precompute radial parts: j_n(k·r)
#         # Shape: [n_radial_basis, n_r_grid] where n_radial_basis = (n_max+1) * n_k
#         n_radial_basis = (self.n_max + 1) * self.n_k
#         self.radial_values = np.zeros((n_radial_basis, self.n_r_grid))
        
#         radial_idx = 0
#         for n in range(self.n_max + 1):
#             for k_idx in range(1, self.n_k + 1):
#                 k = k_idx * np.pi / self.R_max
#                 for i, r in enumerate(r_grid):
#                     kr = k * r
#                     self.radial_values[radial_idx, i] = spherical_jn(n, kr)
#                 radial_idx += 1
        
#         # Precompute angular parts: Y_l^m(θ, φ)
#         # Shape: [num_lm, num_theta, num_phi] for real and imag separately
#         num_lm = sum(2*l + 1 for l in range(self.l_max + 1))
#         self.angular_real = np.zeros((num_lm, self.num_theta, self.num_phi))
#         self.angular_imag = np.zeros((num_lm, self.num_theta, self.num_phi))
        
#         lm_idx = 0
#         for l in range(self.l_max + 1):
#             for m in range(-l, l + 1):
#                 for i, theta in enumerate(theta_grid):
#                     for j, phi in enumerate(phi_grid):
#                         Y_lm = sph_harm(m, l, phi, theta)  # Note: sph_harm takes (m, l, phi, theta)
#                         self.angular_real[lm_idx, i, j] = np.real(Y_lm)
#                         self.angular_imag[lm_idx, i, j] = np.imag(Y_lm)
#                 lm_idx += 1
        
#         # Convert to torch tensors
#         self.radial_torch = torch.tensor(self.radial_values, dtype=torch.float32)
#         self.angular_real_torch = torch.tensor(self.angular_real, dtype=torch.float32)
#         self.angular_imag_torch = torch.tensor(self.angular_imag, dtype=torch.float32)
        
#         # Build mapping from basis_idx to (radial_idx, lm_idx, part)
#         self.basis_to_components = []
#         for n in range(self.n_max + 1):
#             for k_idx in range(1, self.n_k + 1):
#                 radial_idx = n * self.n_k + (k_idx - 1)
#                 lm_idx = 0
#                 for l in range(self.l_max + 1):
#                     for m in range(-l, l + 1):
#                         self.basis_to_components.append((radial_idx, lm_idx, 'real'))
#                         self.basis_to_components.append((radial_idx, lm_idx, 'imag'))
#                         lm_idx += 1
    
#     def _find_nearest_indices(self, r, theta, phi):
#         """Find nearest grid indices using PyTorch (stays on GPU)."""
#         device = r.device
        
#         # Move grids to device as tensors
#         r_grid = torch.tensor(self.r_grid_np, device=device, dtype=torch.float32)
#         theta_grid = torch.tensor(self.theta_grid_np, device=device, dtype=torch.float32)
#         phi_grid = torch.tensor(self.phi_grid_np, device=device, dtype=torch.float32)
        
#         # Find nearest indices (vectorized on GPU)
#         r_idx = torch.argmin(torch.abs(r_grid[:, None] - r[None, :]), dim=0)
#         theta_idx = torch.argmin(torch.abs(theta_grid[:, None] - theta[None, :]), dim=0)
#         phi_idx = torch.argmin(torch.abs(phi_grid[:, None] - phi[None, :]), dim=0)
        
#         return r_idx, theta_idx, phi_idx
    
#     def forward(self, coeffs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         """
#         Evaluate energy using VECTORIZED precomputed basis lookup.
        
#         Args:
#             coeffs: Basis coefficients, shape [B*N, num_basis]
#             actions: Spherical coordinates (r, θ, φ), shape [B*N, 3]
            
#         Returns:
#             Energy values, shape [B*N]
#         """
#         B_N = coeffs.shape[0]
#         device = coeffs.device
        
#         r = actions[:, 0]
#         theta = actions[:, 1]
#         phi = actions[:, 2]
        
#         # Find nearest grid indices (on GPU)
#         r_idx, theta_idx, phi_idx = self._find_nearest_indices(r, theta, phi)
        
#         # Move precomputed tensors to device (only once per forward)
#         if not hasattr(self, '_radial_device') or self._radial_device != device:
#             self._radial = self.radial_torch.to(device)
#             self._angular_real = self.angular_real_torch.to(device)
#             self._angular_imag = self.angular_imag_torch.to(device)
#             self._radial_device = device
        
#         radial = self._radial           # [n_radial_basis, n_r_grid]
#         angular_real = self._angular_real  # [num_lm, num_theta, num_phi]
#         angular_imag = self._angular_imag  # [num_lm, num_theta, num_phi]
        
#         # VECTORIZED basis computation
#         # radial[:, r_idx] -> [n_radial_basis, B*N]
#         # angular_real[:, theta_idx, phi_idx] -> [num_lm, B*N]
        
#         rad_vals = radial[:, r_idx]  # [n_radial_basis, B*N]
#         ang_real_vals = angular_real[:, theta_idx, phi_idx]  # [num_lm, B*N]
#         ang_imag_vals = angular_imag[:, theta_idx, phi_idx]  # [num_lm, B*N]
        
#         # Build full basis values: [num_basis, B*N]
#         # basis order: for each (n, k): for each (l, m): [real, imag]
#         n_radial_basis = (self.n_max + 1) * self.n_k
#         num_lm = ang_real_vals.shape[0]
        
#         # Efficient construction using einsum/broadcasting
#         # rad_vals: [n_radial_basis, B*N]
#         # ang_real_vals: [num_lm, B*N]
#         # We want: basis[radial_idx * num_lm * 2 + lm_idx * 2 + 0/1, :] = rad * ang
        
#         # Expand for outer product: [n_radial_basis, num_lm, B*N]
#         rad_expanded = rad_vals[:, None, :]  # [n_radial_basis, 1, B*N]
#         ang_real_expanded = ang_real_vals[None, :, :]  # [1, num_lm, B*N]
#         ang_imag_expanded = ang_imag_vals[None, :, :]  # [1, num_lm, B*N]
        
#         # Compute products: [n_radial_basis, num_lm, B*N]
#         basis_real = rad_expanded * ang_real_expanded
#         basis_imag = rad_expanded * ang_imag_expanded
        
#         # Interleave real and imag: [n_radial_basis, num_lm * 2, B*N]
#         # Stack along new dim then reshape
#         basis_interleaved = torch.stack([basis_real, basis_imag], dim=3)  # [n_radial, num_lm, B*N, 2]
#         basis_interleaved = basis_interleaved.permute(0, 1, 3, 2)  # [n_radial, num_lm, 2, B*N]
#         basis_interleaved = basis_interleaved.reshape(n_radial_basis * num_lm * 2, B_N)  # [num_basis, B*N]
        
#         # Transpose to [B*N, num_basis]
#         basis_values = basis_interleaved.T
        
#         # Energy = sum of coeffs * basis_values
#         energy = (coeffs * basis_values).sum(dim=1)
        
#         # Scale for better softmax differentiation
#         energy = energy * self.scale_factor
        
#         return energy
    
#     def __call__(self, coeffs, actions):
#         return self.forward(coeffs, actions)


class SphericalBesselHarmonics:
    """
    Spherical Bessel Harmonics with EXACT computation (no lookup table).
    
    Basis: Ψ_{n,k,l,m}(r,θ,φ) = j_n(k·r) · Y_l^m(θ,φ)
    
    All computations done analytically in PyTorch for:
    - Zero discretization error
    - Full GPU acceleration
    - Proper gradients
    """
    
    def __init__(
        self,
        n_max: int,
        l_max: int,
        n_k: int,
        R_max: float = 1.0,
        num_theta: int = 20,
        grid_type: str = "lie_learn",
        scale_factor: float = 10.0,
        n_r_grid: int = 20,  # kept for compatibility, not used
    ):
        self.n_max = n_max
        self.l_max = l_max
        self.L = l_max
        self.n_k = n_k
        self.R_max = R_max
        self.scale_factor = scale_factor
        
        # Angular grid (for get_action inference)
        if grid_type == "lie_learn":
            self.grid = S2.meshgrid(num_theta, grid_type="Driscoll-Healy")
            theta, phi = self.grid
            self.num_theta = theta.shape[0]
            self.num_phi = theta.shape[1]
        else:
            theta = np.linspace(0, np.pi, num_theta)
            phi = np.linspace(0, 2 * np.pi, num_theta * 2)
            theta, phi = np.meshgrid(theta, phi, indexing='ij')
            self.grid = (theta, phi)
            self.num_theta = theta.shape[0]
            self.num_phi = theta.shape[1]
        
        # Precompute radial frequencies k = k_idx * π / R_max
        self.k_values = torch.tensor(
            [k_idx * math.pi / R_max for k_idx in range(1, n_k + 1)],
            dtype=torch.float32
        )
        
        # Count basis functions
        # For each (n, k): (l_max+1)^2 angular * 2 (real/imag)
        n_radial = (n_max + 1) * n_k
        n_angular = (l_max + 1) ** 2
        self.num_basis = n_radial * n_angular * 2
        
        print(f"SphericalBesselHarmonics (EXACT) initialized:")
        print(f"  n_max={n_max}, l_max={l_max}, n_k={n_k}")
        print(f"  num_basis={self.num_basis}")
        print(f"  scale_factor={scale_factor}")
        print(f"  Mode: EXACT computation (no discretization error)")
    
    
    def spherical_bessel_j0(self, x: torch.Tensor) -> torch.Tensor:
        """j_0(x) = sin(x)/x = sinc(x/π) in PyTorch convention"""
        # Handle x=0 case: j_0(0) = 1
        return torch.sinc(x / math.pi)
    
    def spherical_bessel_j1(self, x: torch.Tensor) -> torch.Tensor:
        """j_1(x) = sin(x)/x² - cos(x)/x"""
        # Handle x→0: j_1(x) → x/3
        small = x.abs() < 1e-6
        x_safe = torch.where(small, torch.ones_like(x), x)
        
        result = torch.sin(x_safe) / (x_safe ** 2) - torch.cos(x_safe) / x_safe
        result = torch.where(small, x / 3.0, result)
        return result
    
    def spherical_bessel_jn(self, n: int, x: torch.Tensor) -> torch.Tensor:
        """Compute j_n(x) for n >= 0 using recurrence relation."""
        if n == 0:
            return self.spherical_bessel_j0(x)
        elif n == 1:
            return self.spherical_bessel_j1(x)
        else:
            # Upward recurrence: j_{n+1}(x) = (2n+1)/x * j_n(x) - j_{n-1}(x)
            # This can be unstable for large n, but we only need n <= 1 typically
            j_prev = self.spherical_bessel_j0(x)
            j_curr = self.spherical_bessel_j1(x)
            
            for l in range(1, n):
                # Handle x=0
                small = x.abs() < 1e-10
                x_safe = torch.where(small, torch.ones_like(x), x)
                
                j_next = (2*l + 1) / x_safe * j_curr - j_prev
                # j_n(0) = 0 for n > 0
                j_next = torch.where(small, torch.zeros_like(j_next), j_next)
                
                j_prev = j_curr
                j_curr = j_next
            
            return j_curr
    
    # =========================================================================
    # Spherical Harmonics (Exact PyTorch Implementation)
    # =========================================================================
    
    def compute_spherical_harmonics(
        self, 
        theta: torch.Tensor, 
        phi: torch.Tensor
    ) -> tuple:
        """
        Compute all Y_l^m(θ, φ) up to l_max.
        
        Returns:
            Y_real: [B, num_lm] - real parts
            Y_imag: [B, num_lm] - imaginary parts
        """
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        Y_real_list = []
        Y_imag_list = []
        
        # l = 0, m = 0
        c00 = 0.5 * math.sqrt(1.0 / math.pi)
        Y_real_list.append(torch.full_like(theta, c00))
        Y_imag_list.append(torch.zeros_like(theta))
        
        if self.l_max >= 1:
            # l = 1
            c1 = math.sqrt(3.0 / (4.0 * math.pi))
            # m = -1: Y_1^{-1} ∝ sin(θ) * e^{-iφ}
            Y_real_list.append(c1 * sin_t * torch.cos(-phi))
            Y_imag_list.append(c1 * sin_t * torch.sin(-phi))
            # m = 0: Y_1^0 ∝ cos(θ)
            Y_real_list.append(c1 * cos_t)
            Y_imag_list.append(torch.zeros_like(theta))
            # m = 1: Y_1^1 ∝ sin(θ) * e^{iφ}
            Y_real_list.append(c1 * sin_t * torch.cos(phi))
            Y_imag_list.append(c1 * sin_t * torch.sin(phi))
        
        if self.l_max >= 2:
            # l = 2
            sin2_t = sin_t ** 2
            cos2_t = cos_t ** 2
            
            c2_2 = 0.25 * math.sqrt(15.0 / math.pi)
            c2_1 = 0.5 * math.sqrt(15.0 / math.pi)
            c2_0 = 0.25 * math.sqrt(5.0 / math.pi)
            
            # m = -2
            Y_real_list.append(c2_2 * sin2_t * torch.cos(-2*phi))
            Y_imag_list.append(c2_2 * sin2_t * torch.sin(-2*phi))
            # m = -1
            Y_real_list.append(c2_1 * sin_t * cos_t * torch.cos(-phi))
            Y_imag_list.append(c2_1 * sin_t * cos_t * torch.sin(-phi))
            # m = 0
            Y_real_list.append(c2_0 * (3*cos2_t - 1))
            Y_imag_list.append(torch.zeros_like(theta))
            # m = 1
            Y_real_list.append(c2_1 * sin_t * cos_t * torch.cos(phi))
            Y_imag_list.append(c2_1 * sin_t * cos_t * torch.sin(phi))
            # m = 2
            Y_real_list.append(c2_2 * sin2_t * torch.cos(2*phi))
            Y_imag_list.append(c2_2 * sin2_t * torch.sin(2*phi))
        
        if self.l_max >= 3:
            # l = 3
            sin2_t = sin_t ** 2
            sin3_t = sin_t ** 3
            cos2_t = cos_t ** 2
            cos3_t = cos_t ** 3
            
            c3_3 = 0.25 * math.sqrt(35.0 / (2*math.pi))
            c3_2 = 0.25 * math.sqrt(105.0 / math.pi)
            c3_1 = 0.25 * math.sqrt(21.0 / (2*math.pi))
            c3_0 = 0.25 * math.sqrt(7.0 / math.pi)
            
            # m = -3
            Y_real_list.append(c3_3 * sin3_t * torch.cos(-3*phi))
            Y_imag_list.append(c3_3 * sin3_t * torch.sin(-3*phi))
            # m = -2
            Y_real_list.append(c3_2 * sin2_t * cos_t * torch.cos(-2*phi))
            Y_imag_list.append(c3_2 * sin2_t * cos_t * torch.sin(-2*phi))
            # m = -1
            Y_real_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.cos(-phi))
            Y_imag_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.sin(-phi))
            # m = 0
            Y_real_list.append(c3_0 * (5*cos3_t - 3*cos_t))
            Y_imag_list.append(torch.zeros_like(theta))
            # m = 1
            Y_real_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.cos(phi))
            Y_imag_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.sin(phi))
            # m = 2
            Y_real_list.append(c3_2 * sin2_t * cos_t * torch.cos(2*phi))
            Y_imag_list.append(c3_2 * sin2_t * cos_t * torch.sin(2*phi))
            # m = 3
            Y_real_list.append(c3_3 * sin3_t * torch.cos(3*phi))
            Y_imag_list.append(c3_3 * sin3_t * torch.sin(3*phi))
        
        Y_real = torch.stack(Y_real_list, dim=-1)  # [B, num_lm]
        Y_imag = torch.stack(Y_imag_list, dim=-1)  # [B, num_lm]
        
        return Y_real, Y_imag
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(self, coeffs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute energy using EXACT basis function evaluation.
        
        Args:
            coeffs: [B*N, num_basis] - MLP输出的系数
            actions: [B*N, 3] - (r, θ, φ) 球坐标
            
        Returns:
            energy: [B*N]
        """
        B_N = coeffs.shape[0]
        device = coeffs.device
        dtype = coeffs.dtype
        
        r = actions[:, 0]
        theta = actions[:, 1]
        phi = actions[:, 2]
        
        # Move k values to device
        k_values = self.k_values.to(device=device, dtype=dtype)  # [n_k]
        
        # =====================================================================
        # Compute radial part: j_n(k * r) for all (n, k) combinations
        # =====================================================================
        # k_values: [n_k], r: [B*N]
        # kr: [n_k, B*N]
        kr = k_values[:, None] * r[None, :]  # [n_k, B*N]
        
        # Compute j_n for each n
        radial_parts = []
        for n in range(self.n_max + 1):
            j_n = self.spherical_bessel_jn(n, kr)  # [n_k, B*N]
            radial_parts.append(j_n)
        
        # Stack: [n_max+1, n_k, B*N] -> reshape to [n_radial, B*N]
        radial = torch.cat(radial_parts, dim=0)  # [(n_max+1)*n_k, B*N]
        
        # =====================================================================
        # Compute angular part: Y_l^m(θ, φ)
        # =====================================================================
        Y_real, Y_imag = self.compute_spherical_harmonics(theta, phi)
        # Y_real, Y_imag: [B*N, num_lm]
        
        # =====================================================================
        # Combine: basis = radial * angular
        # =====================================================================
        # radial: [n_radial, B*N]
        # Y_real, Y_imag: [B*N, num_lm]
        
        n_radial = radial.shape[0]
        num_lm = Y_real.shape[1]
        
        # Outer product
        # radial: [n_radial, B*N] -> [n_radial, 1, B*N]
        # Y: [B*N, num_lm] -> [1, num_lm, B*N] (transposed)
        radial_exp = radial[:, None, :]  # [n_radial, 1, B*N]
        Y_real_exp = Y_real.T[None, :, :]  # [1, num_lm, B*N]
        Y_imag_exp = Y_imag.T[None, :, :]  # [1, num_lm, B*N]
        
        # basis_real[i,j,b] = radial[i,b] * Y_real[j,b]
        basis_real = radial_exp * Y_real_exp  # [n_radial, num_lm, B*N]
        basis_imag = radial_exp * Y_imag_exp  # [n_radial, num_lm, B*N]
        
        # Interleave real and imag
        # Stack: [n_radial, num_lm, B*N, 2]
        basis_interleaved = torch.stack([basis_real, basis_imag], dim=3)
        # Reshape: [n_radial * num_lm * 2, B*N]
        basis_interleaved = basis_interleaved.permute(0, 1, 3, 2)  # [n_radial, num_lm, 2, B*N]
        basis_values = basis_interleaved.reshape(n_radial * num_lm * 2, B_N)  # [num_basis, B*N]
        
        # Transpose: [B*N, num_basis]
        basis_values = basis_values.T
        
        # =====================================================================
        # Energy = sum of coeffs * basis_values
        # =====================================================================
        energy = (coeffs * basis_values).sum(dim=1)  # [B*N]
        
        # Scale
        energy = energy * self.scale_factor
        
        return energy
    
    def __call__(self, coeffs, actions):
        return self.forward(coeffs, actions)
