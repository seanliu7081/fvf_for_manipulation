"""
so3_spherical_bessel_harmonics.py

SO3-compatible Spherical Bessel Harmonics for equivariant energy functions.

This module provides the basis functions for SO3SphericalBesselEnergyMLP.
The basis is: Ψ_{n,k,l,m}(r,θ,φ) = j_n(k·r) · Y_l^m(θ,φ)

Place at: eharmony/so3_spherical_bessel_harmonics.py
"""

import math
import torch
import numpy as np

import lie_learn.spaces.S2 as S2
from eharmony.harmonic_function import HarmonicFunction


class SO3SphericalBesselHarmonics(HarmonicFunction):
    """
    Spherical Bessel Harmonics with EXACT computation for SO3 equivariant networks.
    
    Basis: Ψ_{n,k,l,m}(r,θ,φ) = j_n(k·r) · Y_l^m(θ,φ)
    
    All computations done analytically in PyTorch for:
    - Zero discretization error
    - Full GPU acceleration  
    - Proper gradients
    
    Args:
        n_max: Maximum Bessel order (0, 1, 2, ...)
        l_max: Maximum spherical harmonic degree
        n_k: Number of radial frequencies
        R_max: Maximum radius for frequency normalization
        num_theta: Angular grid resolution for inference
        scale_factor: Output energy scaling
    """
    
    def __init__(
        self,
        n_max: int = 1,
        l_max: int = 3,
        n_k: int = 5,
        R_max: float = 1.0,
        num_theta: int = 20,
        grid_type: str = "lie_learn",
        scale_factor: float = 10.0,
    ):
        super().__init__()
        
        self.n_max = n_max
        self.l_max = l_max
        self.L = l_max  # Alias for compatibility
        self.n_k = n_k
        self.R_max = R_max
        self.scale_factor = scale_factor
        
        # Angular grid for inference
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
        
        # Radial frequencies: k = k_idx * π / R_max
        self.register_buffer(
            'k_values',
            torch.tensor([k_idx * math.pi / R_max for k_idx in range(1, n_k + 1)], dtype=torch.float32)
        )
        
        # Basis counts
        self.n_radial = (n_max + 1) * n_k
        self.num_angular = (l_max + 1) ** 2
        self.num_basis = self.n_radial * self.num_angular * 2  # *2 for real/imag
        
        print(f"SO3SphericalBesselHarmonics:")
        print(f"  n_max={n_max}, l_max={l_max}, n_k={n_k}")
        print(f"  n_radial={self.n_radial}, num_angular={self.num_angular}")
        print(f"  num_basis={self.num_basis}")
    
    # =========================================================================
    # Spherical Bessel Functions (Exact)
    # =========================================================================
    
    def spherical_bessel_j0(self, x: torch.Tensor) -> torch.Tensor:
        """j_0(x) = sin(x)/x = sinc(x/π)"""
        return torch.sinc(x / math.pi)
    
    def spherical_bessel_j1(self, x: torch.Tensor) -> torch.Tensor:
        """j_1(x) = sin(x)/x² - cos(x)/x"""
        small = x.abs() < 1e-6
        x_safe = torch.where(small, torch.ones_like(x), x)
        result = torch.sin(x_safe) / (x_safe ** 2) - torch.cos(x_safe) / x_safe
        result = torch.where(small, x / 3.0, result)
        return result
    
    def spherical_bessel_jn(self, n: int, x: torch.Tensor) -> torch.Tensor:
        """Compute j_n(x) using upward recurrence."""
        if n == 0:
            return self.spherical_bessel_j0(x)
        elif n == 1:
            return self.spherical_bessel_j1(x)
        else:
            j_prev = self.spherical_bessel_j0(x)
            j_curr = self.spherical_bessel_j1(x)
            for l in range(1, n):
                small = x.abs() < 1e-10
                x_safe = torch.where(small, torch.ones_like(x), x)
                j_next = (2*l + 1) / x_safe * j_curr - j_prev
                j_next = torch.where(small, torch.zeros_like(j_next), j_next)
                j_prev = j_curr
                j_curr = j_next
            return j_curr
    
    # =========================================================================
    # Spherical Harmonics (Complex)
    # =========================================================================
    
    def compute_spherical_harmonics(self, theta: torch.Tensor, phi: torch.Tensor):
        """
        Compute complex spherical harmonics and return real/imag parts.
        
        Args:
            theta: [B] polar angle
            phi: [B] azimuthal angle
            
        Returns:
            Y_real: [B, num_angular]
            Y_imag: [B, num_angular]
        """
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        Y_real_list = []
        Y_imag_list = []
        
        # l = 0
        c00 = 0.5 * math.sqrt(1.0 / math.pi)
        Y_real_list.append(torch.full_like(theta, c00))
        Y_imag_list.append(torch.zeros_like(theta))
        
        if self.l_max >= 1:
            c1 = math.sqrt(3.0 / (4.0 * math.pi))
            Y_real_list.append(c1 * sin_t * torch.cos(-phi))
            Y_imag_list.append(c1 * sin_t * torch.sin(-phi))
            Y_real_list.append(c1 * cos_t)
            Y_imag_list.append(torch.zeros_like(theta))
            Y_real_list.append(c1 * sin_t * torch.cos(phi))
            Y_imag_list.append(c1 * sin_t * torch.sin(phi))
        
        if self.l_max >= 2:
            sin2_t = sin_t ** 2
            cos2_t = cos_t ** 2
            c2_2 = 0.25 * math.sqrt(15.0 / math.pi)
            c2_1 = 0.5 * math.sqrt(15.0 / math.pi)
            c2_0 = 0.25 * math.sqrt(5.0 / math.pi)
            
            Y_real_list.append(c2_2 * sin2_t * torch.cos(-2*phi))
            Y_imag_list.append(c2_2 * sin2_t * torch.sin(-2*phi))
            Y_real_list.append(c2_1 * sin_t * cos_t * torch.cos(-phi))
            Y_imag_list.append(c2_1 * sin_t * cos_t * torch.sin(-phi))
            Y_real_list.append(c2_0 * (3*cos2_t - 1))
            Y_imag_list.append(torch.zeros_like(theta))
            Y_real_list.append(c2_1 * sin_t * cos_t * torch.cos(phi))
            Y_imag_list.append(c2_1 * sin_t * cos_t * torch.sin(phi))
            Y_real_list.append(c2_2 * sin2_t * torch.cos(2*phi))
            Y_imag_list.append(c2_2 * sin2_t * torch.sin(2*phi))
        
        if self.l_max >= 3:
            sin2_t = sin_t ** 2
            sin3_t = sin_t ** 3
            cos2_t = cos_t ** 2
            cos3_t = cos_t ** 3
            c3_3 = 0.25 * math.sqrt(35.0 / (2*math.pi))
            c3_2 = 0.25 * math.sqrt(105.0 / math.pi)
            c3_1 = 0.25 * math.sqrt(21.0 / (2*math.pi))
            c3_0 = 0.25 * math.sqrt(7.0 / math.pi)
            
            Y_real_list.append(c3_3 * sin3_t * torch.cos(-3*phi))
            Y_imag_list.append(c3_3 * sin3_t * torch.sin(-3*phi))
            Y_real_list.append(c3_2 * sin2_t * cos_t * torch.cos(-2*phi))
            Y_imag_list.append(c3_2 * sin2_t * cos_t * torch.sin(-2*phi))
            Y_real_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.cos(-phi))
            Y_imag_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.sin(-phi))
            Y_real_list.append(c3_0 * (5*cos3_t - 3*cos_t))
            Y_imag_list.append(torch.zeros_like(theta))
            Y_real_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.cos(phi))
            Y_imag_list.append(c3_1 * sin_t * (5*cos2_t - 1) * torch.sin(phi))
            Y_real_list.append(c3_2 * sin2_t * cos_t * torch.cos(2*phi))
            Y_imag_list.append(c3_2 * sin2_t * cos_t * torch.sin(2*phi))
            Y_real_list.append(c3_3 * sin3_t * torch.cos(3*phi))
            Y_imag_list.append(c3_3 * sin3_t * torch.sin(3*phi))
        
        Y_real = torch.stack(Y_real_list, dim=-1)
        Y_imag = torch.stack(Y_imag_list, dim=-1)
        return Y_real, Y_imag
    
    def forward(self, f: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        """
        Compute energy using Spherical Bessel Harmonics.
        
        Args:
            f: [B, num_basis] or [B*N, num_basis] - Fourier coefficients
            coords: [B, 3] or [B*N, 3] - (r, θ, φ) spherical coordinates
                    If None, evaluate on grid
        
        Returns:
            energy: [B] or [B*N] if coords provided
                    [B, num_theta, num_phi] if coords is None (grid mode)
        """
        if coords is not None:
            return self._forward_point(f, coords)
        else:
            return self._forward_grid(f)
    
    def _forward_point(self, coeffs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate at specific (r, θ, φ) points."""
        B_N = coeffs.shape[0]
        device = coeffs.device
        dtype = coeffs.dtype
        
        r = actions[:, 0]
        theta = actions[:, 1]
        phi = actions[:, 2]
        
        k_values = self.k_values.to(device=device, dtype=dtype)
        
        # Compute j_n(k*r) for all (n, k)
        kr = k_values[:, None] * r[None, :]  # [n_k, B*N]
        
        radial_parts = []
        for n in range(self.n_max + 1):
            j_n = self.spherical_bessel_jn(n, kr)  # [n_k, B*N]
            radial_parts.append(j_n)
        
        radial = torch.cat(radial_parts, dim=0)  # [n_radial, B*N]
        
        # Compute Y_l^m
        Y_real, Y_imag = self.compute_spherical_harmonics(theta, phi)
        
        # Build full basis via outer product
        n_radial = radial.shape[0]
        num_lm = Y_real.shape[1]
        
        radial_exp = radial[:, None, :]  # [n_radial, 1, B*N]
        Y_real_exp = Y_real.T[None, :, :]  # [1, num_lm, B*N]
        Y_imag_exp = Y_imag.T[None, :, :]
        
        basis_real = radial_exp * Y_real_exp  # [n_radial, num_lm, B*N]
        basis_imag = radial_exp * Y_imag_exp
        
        # Interleave real and imag
        basis_interleaved = torch.stack([basis_real, basis_imag], dim=3)
        basis_interleaved = basis_interleaved.permute(0, 1, 3, 2)
        basis_values = basis_interleaved.reshape(n_radial * num_lm * 2, B_N).T  # [B*N, num_basis]
        
        # Energy = sum(coeffs * basis)
        energy = (coeffs * basis_values).sum(dim=1)
        energy = energy * self.scale_factor
        
        return energy
    
    def _forward_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Evaluate on angular grid for all radii (for inference)."""
        B = coeffs.shape[0]
        device = coeffs.device
        dtype = coeffs.dtype
        
        # Get grid
        theta_grid, phi_grid = self.grid
        theta_flat = torch.tensor(theta_grid.flatten(), device=device, dtype=dtype)
        phi_flat = torch.tensor(phi_grid.flatten(), device=device, dtype=dtype)
        num_grid = len(theta_flat)
        
        # For grid mode, we evaluate at multiple radii
        # Use n_k radii uniformly spaced
        r_values = torch.linspace(0.1, self.R_max, self.n_k, device=device, dtype=dtype)
        
        # Output: [B, n_k, num_theta, num_phi]
        energies = []
        
        for r_idx, r_val in enumerate(r_values):
            r = torch.full((num_grid,), r_val, device=device, dtype=dtype)
            
            # Expand coeffs for grid
            coeffs_exp = coeffs.unsqueeze(1).expand(-1, num_grid, -1)  # [B, num_grid, num_basis]
            coeffs_flat = coeffs_exp.reshape(B * num_grid, -1)
            
            # Build coordinates
            coords = torch.stack([
                r.unsqueeze(0).expand(B, -1).reshape(-1),
                theta_flat.unsqueeze(0).expand(B, -1).reshape(-1),
                phi_flat.unsqueeze(0).expand(B, -1).reshape(-1),
            ], dim=-1)  # [B * num_grid, 3]
            
            energy = self._forward_point(coeffs_flat, coords)
            energy = energy.view(B, self.num_theta, self.num_phi)
            energies.append(energy)
        
        # Stack along radius dimension
        return torch.stack(energies, dim=1)  # [B, n_k, num_theta, num_phi]