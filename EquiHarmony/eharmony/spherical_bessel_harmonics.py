import torch
import numpy as np
from torch import nn
from scipy.special import spherical_jn, sph_harm
from typing import Optional
import lie_learn.spaces.S2 as S2

from fvf.model.modules.layers import MLP
import math


class SphericalBesselHarmonics:
    """
    Spherical Bessel Harmonics with EXACT computation (no lookup table).

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
        
        # Angular grid
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
            coeffs: [B*N, num_basis]
            actions: [B*N, 3]
            
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


# class SphericalBesselHarmonics:
#     """
#     Supports both EXACT and Grid modes.
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
#         self.L = l_max
#         self.n_k = n_k
#         self.R_max = R_max
#         self.scale_factor = scale_factor
        
#         # Angular grid (for Grid mode inference)
#         if grid_type == "lie_learn":
#             import lie_learn.spaces.S2 as S2
#             self.grid = S2.meshgrid(num_theta, grid_type="Driscoll-Healy")
#             theta, phi = self.grid
#             self.num_theta = theta.shape[0]
#             self.num_phi = theta.shape[1]
#         else:
#             import numpy as np
#             theta = np.linspace(0, np.pi, num_theta)
#             phi = np.linspace(0, 2 * np.pi, num_theta * 2)
#             theta, phi = np.meshgrid(theta, phi, indexing='ij')
#             self.grid = (theta, phi)
#             self.num_theta = theta.shape[0]
#             self.num_phi = theta.shape[1]
        
#         # Precompute radial frequencies
#         self.k_values = torch.tensor(
#             [k_idx * math.pi / R_max for k_idx in range(1, n_k + 1)],
#             dtype=torch.float32
#         )
        
#         # Count basis functions
#         n_radial = (n_max + 1) * n_k
#         n_angular = (l_max + 1) ** 2
#         self.num_basis = n_radial * n_angular * 2
        
#         # =====================================================================
#         # precompute Y_l^m on the grid for Grid mode inference
#         # =====================================================================
#         self._precompute_grid_harmonics()
        
#         print(f"SphericalBesselHarmonics initialized:")
#         print(f"  n_max={n_max}, l_max={l_max}, n_k={n_k}")
#         print(f"  num_basis={self.num_basis}")
#         print(f"  Grid: num_theta={self.num_theta}, num_phi={self.num_phi}")
#         print(f"  Supports both EXACT and Grid modes")
    
#     def _precompute_grid_harmonics(self):
#         theta_grid, phi_grid = self.grid
        
#         # Flatten grid
#         theta_flat = torch.tensor(theta_grid.flatten(), dtype=torch.float32)
#         phi_flat = torch.tensor(phi_grid.flatten(), dtype=torch.float32)
        
#         # Compute Y_l^m on grid
#         Y_real, Y_imag = self.compute_spherical_harmonics(theta_flat, phi_flat)
        
#         # Store as buffers (will be moved to device when needed)
#         self.Y_grid_real = Y_real  # [num_theta * num_phi, num_lm]
#         self.Y_grid_imag = Y_imag  # [num_theta * num_phi, num_lm]
    
#     # =========================================================================
#     # Spherical Bessel Functions (unchanged)
#     # =========================================================================
    
#     def spherical_bessel_j0(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.sinc(x / math.pi)
    
#     def spherical_bessel_j1(self, x: torch.Tensor) -> torch.Tensor:
#         small = x.abs() < 1e-6
#         x_safe = torch.where(small, torch.ones_like(x), x)
#         result = torch.sin(x_safe) / (x_safe ** 2) - torch.cos(x_safe) / x_safe
#         result = torch.where(small, x / 3.0, result)
#         return result
    
#     def spherical_bessel_jn(self, n: int, x: torch.Tensor) -> torch.Tensor:
#         if n == 0:
#             return self.spherical_bessel_j0(x)
#         elif n == 1:
#             return self.spherical_bessel_j1(x)
#         else:
#             j_prev = self.spherical_bessel_j0(x)
#             j_curr = self.spherical_bessel_j1(x)
#             for l in range(1, n):
#                 small = x.abs() < 1e-10
#                 x_safe = torch.where(small, torch.ones_like(x), x)
#                 j_next = (2*l + 1) / x_safe * j_curr - j_prev
#                 j_next = torch.where(small, torch.zeros_like(j_next), j_next)
#                 j_prev = j_curr
#                 j_curr = j_next
#             return j_curr
    
#     # =========================================================================
#     # Spherical Harmonics (unchanged)
#     # =========================================================================
    
#     def compute_spherical_harmonics(
#         self, 
#         theta: torch.Tensor, 
#         phi: torch.Tensor
#     ) -> tuple:
#         """Compute all Y_l^m(θ, φ) up to l_max."""
#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)
        
#         Y_real_list = []
#         Y_imag_list = []
        
#         # l = 0, m = 0
#         c00 = 0.5 * math.sqrt(1.0 / math.pi)
#         Y_real_list.append(torch.full_like(theta, c00))
#         Y_imag_list.append(torch.zeros_like(theta))
        
#         if self.l_max >= 1:
#             c1 = math.sqrt(3.0 / (4.0 * math.pi))
#             Y_real_list.append(c1 * sin_t * torch.cos(-phi))
#             Y_imag_list.append(c1 * sin_t * torch.sin(-phi))
#             Y_real_list.append(c1 * cos_t)
#             Y_imag_list.append(torch.zeros_like(theta))
#             Y_real_list.append(c1 * sin_t * torch.cos(phi))
#             Y_imag_list.append(c1 * sin_t * torch.sin(phi))
        
#         if self.l_max >= 2:
#             sin2_t = sin_t ** 2
#             cos2_t = cos_t ** 2
#             c2_2 = 0.25 * math.sqrt(15.0 / math.pi)
#             c2_1 = 0.5 * math.sqrt(15.0 / math.pi)
#             c2_0 = 0.25 * math.sqrt(5.0 / math.pi)
#             Y_real_list.append(c2_2 * sin2_t * torch.cos(-2*phi))
#             Y_imag_list.append(c2_2 * sin2_t * torch.sin(-2*phi))
#             Y_real_list.append(c2_1 * sin_t * cos_t * torch.cos(-phi))
#             Y_imag_list.append(c2_1 * sin_t * cos_t * torch.sin(-phi))
#             Y_real_list.append(c2_0 * (3*cos2_t - 1))
#             Y_imag_list.append(torch.zeros_like(theta))
#             Y_real_list.append(c2_1 * sin_t * cos_t * torch.cos(phi))
#             Y_imag_list.append(c2_1 * sin_t * cos_t * torch.sin(phi))
#             Y_real_list.append(c2_2 * sin2_t * torch.cos(2*phi))
#             Y_imag_list.append(c2_2 * sin2_t * torch.sin(2*phi))
        
#         if self.l_max >= 3:
#             sin2_t = sin_t ** 2
#             sin3_t = sin_t ** 3
#             cos2_t = cos_t ** 2
#             cos3_t = cos_t ** 3
#             c3_3 = 0.25 * math.sqrt(35.0 / (2*math.pi))
#             c3_2 = 0.25 * math.sqrt(105.0 / math.pi)
#             c3_1 = 0.25 * math.sqrt(21.0 / (2*math.pi))
#             c3_0 = 0.25 * math.sqrt(7.0 / math.pi)
#             Y_real_list.append(c3_3 * sin3_t * torch.cos(-3*phi))
#             Y_imag_list.append(c3_3 * sin3_t * torch.sin(-3*phi))
#             Y_real_list.append(c3_2 * sin2_t * cos_t * torch.cos(-2*phi))
#             Y_imag_list.append(c3_2 * sin2_t * cos_t * torch.sin(-2*phi))
#             Y_real_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.cos(-phi))
#             Y_imag_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.sin(-phi))
#             Y_real_list.append(c3_0 * (5*cos3_t - 3*cos_t))
#             Y_imag_list.append(torch.zeros_like(theta))
#             Y_real_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.cos(phi))
#             Y_imag_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.sin(phi))
#             Y_real_list.append(c3_2 * sin2_t * cos_t * torch.cos(2*phi))
#             Y_imag_list.append(c3_2 * sin2_t * cos_t * torch.sin(2*phi))
#             Y_real_list.append(c3_3 * sin3_t * torch.cos(3*phi))
#             Y_imag_list.append(c3_3 * sin3_t * torch.sin(3*phi))
        
#         Y_real = torch.stack(Y_real_list, dim=-1)
#         Y_imag = torch.stack(Y_imag_list, dim=-1)
        
#         return Y_real, Y_imag
    
#     # =========================================================================
#     # Forward: EXACT
#     # =========================================================================
    
#     def forward_exact(self, coeffs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         """
#         EXACT
        
#         Args:
#             coeffs: [B*N, num_basis]
#             actions: [B*N, 3] - (r, θ, φ)
#         Returns:
#             energy: [B*N]
#         """
#         B_N = coeffs.shape[0]
#         device = coeffs.device
#         dtype = coeffs.dtype
        
#         r = actions[:, 0]
#         theta = actions[:, 1]
#         phi = actions[:, 2]
        
#         k_values = self.k_values.to(device=device, dtype=dtype)
        
#         # Radial part
#         kr = k_values[:, None] * r[None, :]
#         radial_parts = []
#         for n in range(self.n_max + 1):
#             j_n = self.spherical_bessel_jn(n, kr)
#             radial_parts.append(j_n)
#         radial = torch.cat(radial_parts, dim=0)
        
#         # Angular part
#         Y_real, Y_imag = self.compute_spherical_harmonics(theta, phi)
        
#         # Combine
#         n_radial = radial.shape[0]
#         num_lm = Y_real.shape[1]
        
#         radial_exp = radial[:, None, :]
#         Y_real_exp = Y_real.T[None, :, :]
#         Y_imag_exp = Y_imag.T[None, :, :]
        
#         basis_real = radial_exp * Y_real_exp
#         basis_imag = radial_exp * Y_imag_exp
        
#         basis_interleaved = torch.stack([basis_real, basis_imag], dim=3)
#         basis_interleaved = basis_interleaved.permute(0, 1, 3, 2)
#         basis_values = basis_interleaved.reshape(n_radial * num_lm * 2, B_N).T
        
#         energy = (coeffs * basis_values).sum(dim=1)
#         energy = energy * self.scale_factor
        
#         return energy
    
#     # =========================================================================
#     # Forward: Grid mode (输出整个角度网格)
#     # =========================================================================
    
#     def forward_grid(self, coeffs: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             coeffs: [B, num_basis]
#             r: [B, num_radii] or [B, num_radii, 1]
#         Returns:
#             energy: [B, num_radii, num_theta, num_phi]
#         """
#         device = coeffs.device
#         dtype = coeffs.dtype
        
#         # Handle r shape
#         if r.dim() == 3:
#             r = r.squeeze(-1)  # [B, num_radii, 1] -> [B, num_radii]
        
#         B, num_radii = r.shape
#         num_grid = self.num_theta * self.num_phi
        
#         k_values = self.k_values.to(device=device, dtype=dtype)
        
#         # Move precomputed grid harmonics to device
#         Y_grid_real = self.Y_grid_real.to(device=device, dtype=dtype)  # [num_grid, num_lm]
#         Y_grid_imag = self.Y_grid_imag.to(device=device, dtype=dtype)
        
#         # =====================================================================
#         # Compute radial part for all (batch, radius) combinations
#         # =====================================================================
#         # r: [B, num_radii] -> [B * num_radii]
#         r_flat = r.reshape(-1)  # [B * num_radii]
        
#         # kr: [n_k, B * num_radii]
#         kr = k_values[:, None] * r_flat[None, :]
        
#         radial_parts = []
#         for n in range(self.n_max + 1):
#             j_n = self.spherical_bessel_jn(n, kr)  # [n_k, B * num_radii]
#             radial_parts.append(j_n)
#         radial = torch.cat(radial_parts, dim=0)  # [n_radial, B * num_radii]
        
#         n_radial = radial.shape[0]
#         num_lm = Y_grid_real.shape[1]
        
#         # =====================================================================
#         # Combine radial and angular parts
#         # =====================================================================
#         # radial: [n_radial, B * num_radii]
#         # Y_grid: [num_grid, num_lm]
        
#         # Outer product for all combinations
#         # radial: [n_radial, B * num_radii] -> [n_radial, 1, B * num_radii]
#         # Y_grid: [num_grid, num_lm] -> [1, num_lm, num_grid]
        
#         radial_exp = radial[:, None, :]  # [n_radial, 1, B * num_radii]
#         Y_real_exp = Y_grid_real.T[None, :, :]  # [1, num_lm, num_grid]
#         Y_imag_exp = Y_grid_imag.T[None, :, :]
        
#         # For each (r_idx), compute all (theta, phi) combinations
#         # We need: basis[b, r_idx, theta_idx, phi_idx, basis_idx]
        
#         # Reshape radial: [n_radial, B, num_radii]
#         radial_reshaped = radial.view(n_radial, B, num_radii)
        
#         # Build basis values: [B, num_radii, num_grid, num_basis]
#         energy_all = torch.zeros(B, num_radii, num_grid, device=device, dtype=dtype)
        
#         # Efficient computation using einsum
#         # coeffs: [B, num_basis]
#         # We need to compute: sum over basis of coeffs * radial * angular
        
#         # Reshape coeffs to separate real/imag and (n, k, l, m)
#         # num_basis = n_radial * num_lm * 2
#         coeffs_reshaped = coeffs.view(B, n_radial, num_lm, 2)  # [B, n_radial, num_lm, 2]
#         coeffs_real = coeffs_reshaped[:, :, :, 0]  # [B, n_radial, num_lm]
#         coeffs_imag = coeffs_reshaped[:, :, :, 1]  # [B, n_radial, num_lm]
        
#         # radial_reshaped: [n_radial, B, num_radii]
#         # Y_grid_real: [num_grid, num_lm]
#         # coeffs_real: [B, n_radial, num_lm]
        
#         # Energy = sum_{n,k,l,m} c_real * j_n(kr) * Y_real + c_imag * j_n(kr) * Y_imag
        
#         # Step 1: c * j_n(kr) for each (b, n_radial, num_radii)
#         # radial_reshaped.permute: [B, n_radial, num_radii]
#         radial_perm = radial_reshaped.permute(1, 0, 2)  # [n_radial, B, num_radii]
        
#         # Compute weighted radial: coeffs * radial
#         # [B, n_radial, num_lm] * [n_radial, B, num_radii] -> need einsum
        
#         # For real part: sum_{n_radial, num_lm} c_real[b, nr, lm] * j[nr, b, r] * Y_real[g, lm]
#         # = einsum("bnl, nbr, gl -> brg", c_real, radial, Y_real)
        
#         energy_real = torch.einsum(
#             "bnl, nbr, gl -> brg", 
#             coeffs_real, 
#             radial_reshaped, 
#             Y_grid_real
#         )
        
#         energy_imag = torch.einsum(
#             "bnl, nbr, gl -> brg", 
#             coeffs_imag, 
#             radial_reshaped, 
#             Y_grid_imag
#         )
        
#         energy = (energy_real + energy_imag) * self.scale_factor
        
#         # Reshape to [B, num_radii, num_theta, num_phi]
#         energy = energy.view(B, num_radii, self.num_theta, self.num_phi)
        
#         return energy
    
#     # =========================================================================
#     # Main forward (auto-select mode)
#     # =========================================================================
    
#     def forward(self, coeffs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         """
#         - actions.shape[-1] == 3 EXACT
#         - actions.shape[-1] == 1 Grid
#         """
#         if actions.shape[-1] == 3:
#             return self.forward_exact(coeffs, actions)
#         else:
#             return self.forward_grid(coeffs, actions)
    
#     def __call__(self, coeffs, actions):
#         return self.forward(coeffs, actions)