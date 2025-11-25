""" cylindrical_harmonics.py """

import torch
from torch import nn

from eharmony.harmonic_function import HarmonicFunction
from eharmony import bessel, grid
from eharmony.polar_harmonics import (
    get_l,
    get_k,
    get_zkl,
    get_Nkl_zero,
    get_Nkl_deri,
    get_Rkl,
    get_Phi_l,
)


def get_m(num_m: int) -> torch.Tensor:
    """Returns m for the 1D axial cylindrical components.

    Args:
        num_m - Length of the n grid.
    """
    m = torch.arange(num_m + 1)[1:]
    return m


def get_Z_m(m: int, z: torch.Tensor) -> torch.Tensor:
    """Axial component of the cylinder basis function.

    Args:
        m - Order.
        z - Axial values.
    """
    return torch.sin(m * torch.pi * z)


def get_Psi_klm(
    l: int,
    m: int,
    r: torch.Tensor,
    phi: torch.Tensor,
    z: torch.Tensor,
    zkl: float,
    Nkl: torch.Tensor,
) -> torch.Tensor:
    """Cylinder radial basis function
    Args:
        k - Number of zeros.
        l - Bessel order.
        m - Axial order.
        r - Radius.
        phi - Angle.
        zkl - Corresponding z Fourier mode for k and l.
        Nkl - Corresponding normalisation constant.
    """
    Phi_l = get_Phi_l(l, phi)
    Rkl = get_Rkl(r, l, zkl, Nkl).to(r.device)
    Z_m = get_Z_m(m, z)
    Psi_klm = Phi_l * Rkl * Z_m

    return Psi_klm


class CylindricalHarmonics(HarmonicFunction):
    """Cylindrical Harmonics"""

    def __init__(
        self,
        K: int,
        L: int,
        M: int,
        min_radius: float = 0.0,
        max_radius: float = 1.0,
        max_height: float = 1.0,
        num_radii: int = 100,
        num_phi: int = 360,
        num_height: int = 100,
        boundary: str = "zero",
    ):
        super().__init__()

        self.K = K
        self.L = L
        self.M = M
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_height = max_height
        self.num_radii = num_radii
        self.num_phi = num_phi
        self.num_height = num_height
        self.boundary = boundary

        self.init()

    def init(self) -> None:
        """Initialize the intermediate variables and basis functions."""
        self.r2d, self.p2d, self.z2d = grid.cylinder_grid(
            self.max_radius,
            self.max_height,
            self.num_radii,
            self.num_phi,
            self.num_height,
            r_min=self.min_radius,
        )
        self.dr = self.r2d[0][1] - self.r2d[0][0]
        self.dphi = self.p2d[1][0] - self.p2d[0][0]
        self.dz = self.z2d[0][2] - self.z2d[0][0]

        self.k = get_k(self.K)
        self.l = get_l(self.L)
        self.m = get_m(self.M)
        self.l2d, self.k2d = torch.meshgrid(self.l, self.k, indexing="ij")

        self.xkl = torch.zeros(self.l2d.shape)
        self.zkl = torch.zeros(self.l2d.shape)
        self.Nkl = torch.zeros(self.l2d.shape)

        # Compute intermediate variables for Cylinder Basis Functions
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

        self.Psi = nn.Parameter(self.generate_basis_fns(), requires_grad=False)

    def generate_basis_fns(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is not None:
            B = 1
            R = coords.size(0)
            P = 1
            Z = 1
            r2d = coords[:, 0].view(R, 1)
            p2d = coords[:, 1].view(R, 1)
            z2d = coords[:, 2].view(R, 1)
        else:
            B = 1
            R = self.num_radii
            P = self.num_phi
            Z = self.num_height
            r2d = self.r2d
            p2d = self.p2d
            z2d = self.z2d

        Psi = torch.zeros(B, self.M, self.K, self.L * 2 + 1, R, P, Z)
        for m in range(0, self.m[-1]):
            for i in range(0, len(self.l2d_flat)):
                Psi_klm = get_Psi_klm(
                    self.l2d_flat[i].item(),
                    self.m[m].item(),
                    r2d,
                    p2d,
                    z2d,
                    self.zkl_flat[i].item(),
                    self.Nkl_flat[i],
                )
                if self.l2d_flat[i] == 0:
                    if coords is not None:
                        Psi_klm = Psi_klm.view(-1, 1, 1)
                    Psi[:, self.m[m] - 1, self.k2d_flat[i] - 1, self.l2d_flat[i]] = (
                        Psi_klm
                    )
                else:
                    li = self.l2d_flat[i] * 2 - 1
                    if coords is not None:
                        Psi_klm_0 = Psi_klm[0].view(-1, 1, 1)
                        Psi_klm_1 = Psi_klm[1].view(-1, 1, 1)
                    else:
                        Psi_klm_0 = Psi_klm[0]
                        Psi_klm_1 = Psi_klm[1]
                    Psi[:, self.m[m] - 1, self.k2d_flat[i] - 1, li] = Psi_klm_0
                    Psi[:, self.m[m] - 1, self.k2d_flat[i] - 1, li + 1] = Psi_klm_1
        Psi = Psi.flatten(1, 3)

        return Psi

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        B = w.size(0)

        if coords is not None:
            Psi = self.generate_basis_fns(coords)
            Psi = Psi.permute(2, 1, 0, 3, 4).to(w.device)
        else:
            Psi = self.Psi.repeat(B, 1, 1, 1, 1)

        out = torch.einsum("bn,bnrpz->bnrpz", w, Psi).sum(1)
        return out
