""" polar_harmonics.py """

import math
import torch
from torch import nn

from eharmony.harmonic_function import HarmonicFunction
from eharmony import bessel, grid


def get_k(k_max: int) -> torch.Tensor:
    """Returns an array of k up to k_max.

    Args:
       k_max - Maximum k.
    """
    return torch.arange(k_max + 1)[1:]


def get_l(num_l: int) -> torch.Tensor:
    """Returns l for the 1D angular polar components.

    Args:
        num_l - Length of phi grid.
    """
    l = torch.arange(num_l + 1)
    return l


def get_zkl(xkl: torch.Tensor, r_max: float) -> torch.Tensor:
    """Returns the Fouirer Mode k components given the zeros and maximum radius

    Args:
        xkl - Location of zeros.
        Rmax: Maximum radius.
    """
    return xkl / r_max


def get_Nkl_zero(l: int, xkl: torch.Tensor, r_max: float) -> torch.Tensor:
    """Returns the normalization constant for zero-value boundaries.

    Args:
        l - Order.
        xkl - Location of zeros for zero-value boundaries.
        r_max - Maximum radius.
    """
    Nkl = (r_max**2.0 / 2.0) * torch.from_numpy(
        bessel.get_Jm(l + 1, xkl.numpy())
    ) ** 2.0
    return Nkl


def get_Nkl_deri(l: int, xkl: torch.Tensor, Rmax: float) -> torch.Tensor:
    """Returns the normalization constant for derivative boundaries.

    Args:
        l - Order
        xkl - Location of zeros for derivative boundaries.
        Rmax - Maximum radius.
    """
    return (
        (Rmax**2.0 / 2.0)
        * (1.0 - l**2.0 / xkl**2.0)
        * torch.from_numpy(bessel.get_Jm(l, xkl.numpy())) ** 2.0
    )


def get_Rkl(r: torch.Tensor, l: int, zkl: float, Nkl: float) -> torch.Tensor:
    """Radial component of the polar basis function.

    Args:
        r - Radial values.
        l - Order.
        zkl - Corresponding z Fourier mode for n and m.
        Nkl - Corresponding normalisation constant.
    """
    return (1.0 / math.sqrt(Nkl)) * torch.from_numpy(
        bessel.get_Jm(l, zkl * r.cpu().numpy())
    )


def get_Phi_l(l: int, phi: torch.Tensor) -> torch.Tensor:
    """Angular component of the polar basis function.

    Args:
        l - Order.
        phi - Angular values (radians).
    """
    if l == 0:
        Phi_l = torch.ones_like(phi) / math.sqrt(2 * torch.pi)
    else:
        Phi_l = torch.stack(
            [
                torch.cos(l * phi) / math.sqrt(2 * torch.pi),
                torch.sin(l * phi) / math.sqrt(2 * torch.pi),
            ]
        )

    return Phi_l


def get_Psi_kl(
    l: int, r: torch.Tensor, phi: torch.Tensor, zkl: float, Nkl: torch.Tensor
) -> torch.Tensor:
    """Polar radial basis function
    Args:
        l - Bessel order.
        r - Radius.
        phi - Angle.
        zkl - Corresponding z Fourier mode for k and l.
        Nkl - Corresponding normalisation constant.
    """
    Phi_l = get_Phi_l(l, phi)
    Rkl = get_Rkl(r, l, zkl, Nkl).to(r.device)
    Psi_kl = Phi_l * Rkl

    return Psi_kl


class PolarHarmonics(HarmonicFunction):
    """
    Torch module for computing a polar function using Fourier coefficients and polar
    harmonic basis functions. Pre-computes basis functions for a grid of values which can
    be used for faster evaluation.

    Args:
        K - Maximum radial frequency.
        L - Maximum angular frequency.
        min_radius - Minimum radius axis value
        max_radius - Maximum radius axis value
        num_radii - Number of elements on the radial axis.
        num_phi - Number of elements on the angular axis.
        boundary - Boundary type for radial basis functions: 'zero' or 'deri'.
    """

    def __init__(
        self,
        K: int,
        L: int,
        min_radius: float = 0.0,
        max_radius: float = 1.0,
        num_radii: int = 100,
        num_phi: int = 360,
        boundary: str = "zero",
    ):
        super().__init__()

        self.K = K
        self.L = L
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.num_phi = num_phi
        self.boundary = boundary

        self.init()

    def init(self) -> None:
        """Initialize the intermediate variables and basis functions."""
        self.r2d, self.p2d = grid.polar_grid(
            self.max_radius, self.num_radii, self.num_phi, r_min=self.min_radius
        )
        self.dr = self.r2d[0][1] - self.r2d[0][0]
        self.dphi = self.p2d[1][0] - self.p2d[0][0]
        self.l = get_l(self.L)
        self.k = get_k(self.K)
        self.l2d, self.k2d = torch.meshgrid(self.l, self.k, indexing="ij")

        self.xkl = torch.zeros(self.l2d.shape)
        self.zkl = torch.zeros(self.l2d.shape)
        self.Nkl = torch.zeros(self.l2d.shape)

        # Compute intermediate variables for Polar harmonics
        len_l = len(self.l2d)
        for i in range(len_l):
            lval = self.l[i].item()
            kval = self.k[-1].item()
            if self.boundary == "zero":
                xkl = torch.from_numpy(bessel.get_Jm_zeros(lval, kval))
                zkl = get_zkl(xkl, self.max_radius)
                Nkl = get_Nkl_zero(lval, xkl, self.max_radius)
            else:
                xkl = torch.from_numpy(bessel.get_dJm_zeros(lval, kval))
                zkl = get_zkl(xkl, self.max_radius)
                Nkl = get_Nkl_deri(lval, xkl, self.max_radius)

            self.xkl[i] = xkl
            self.zkl[i] = zkl
            self.Nkl[i] = Nkl

        self.l2d_flat = self.l2d.flatten()
        self.k2d_flat = self.k2d.flatten()
        self.xkl_flat = self.xkl.flatten()
        self.zkl_flat = self.zkl.flatten()
        self.Nkl_flat = self.Nkl.flatten()

        self.Psi = self.generate_basis_fns()

    def generate_basis_fns(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is not None:
            B = 1
            R = coords.size(0)
            D = 1
            r2d = coords[:, 0].view(R, 1)
            p2d = coords[:, 1].view(R, 1)
        else:
            B = 1
            R = self.num_radii
            D = self.num_phi
            r2d = self.r2d
            p2d = self.p2d

        Psi = torch.zeros(B, self.K, self.L * 2 + 1, R, D)
        for i in range(0, len(self.l2d_flat)):
            Psi_kl = get_Psi_kl(
                self.l2d_flat[i].item(),
                r2d,
                p2d,
                self.zkl_flat[i].item(),
                self.Nkl_flat[i],
            )

            if self.l2d_flat[i] == 0:
                Psi[:, self.k2d_flat[i] - 1, self.l2d_flat[i]] = Psi_kl
            else:
                li = self.l2d_flat[i] * 2 - 1
                Psi[:, self.k2d_flat[i] - 1, li] = Psi_kl[0]
                Psi[:, self.k2d_flat[i] - 1, li + 1] = Psi_kl[1]
        Psi = Psi.flatten(1, 2)

        return Psi

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        B = w.size(0)

        if coords is not None:
            Psi = self.generate_basis_fns(coords).permute(2, 1, 0, 3).to(w.device)
        else:
            Psi = self.Psi.repeat(B, 1, 1, 1).to(w.device)

        out = torch.einsum("bn,bnrp->bnrp", w, Psi).sum(1)

        return out
