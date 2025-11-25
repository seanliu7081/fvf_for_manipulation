""" so3_harmoincs.py """

import numpy as np
import torch
from escnn.group.groups.so3_utils import _wigner_d_matrix, _grid

from eharmony.harmonic_function import HarmonicFunction


class SO3Harmonics(HarmonicFunction):
    """
    Torch module for computing a spherical function using Fourier coefficients and wigner-d basis functions.
    Pre-computes basis functions for a grid of values which can
    be used for faster evaluation.

    Args:
       L - Maximum frequency.
       num_lat - Number of elements on the latitudinal axis.
       num_lon - Number of elements on the longitudinal axis.
    """

    def __init__(self, L: int, num_grid_points: int = 100):
        super().__init__()

        self.L = L
        self.num_grid_points = num_grid_points

        self.D = self.generate_basis_fns()

    def generate_basis_fns(self, coords: torch.Tensor = None):
        if coords is None:
            grid = _grid("hopf", N=self.num_grid_points, parametrization="ZYZ")
            self.num_grid_points = grid.shape[0]
            self.grid = grid
        else:
            pass

        D = []
        for l in range(self.L + 1):
            d_l = []
            for e in grid:
                d_l.append(
                    torch.from_numpy(_wigner_d_matrix(e, l, param="ZYZ")).float()
                )
            D.append(torch.stack(d_l))

        return D

    def forward(self, f: torch.Tensor, R: torch.Tensor = None) -> torch.Tensor:
        B = f.size(0)
        if R is not None:
            F = torch.zeros((B)).to(f.device)
            li = 0
            for l in range(self.L + 1):
                ld = 2 * l + 1
                f_l = f[:, li : li + ld**2].view(B, ld, ld)
                d_l = []
                for e in R.cpu().numpy():
                    d_l.append(
                        torch.from_numpy(_wigner_d_matrix(e, l, param="ZYZ")).float()
                    )
                d_l = torch.stack(d_l).to(f_l.device)
                F += (
                    (2 * l + 1)
                    / (8 * torch.pi)
                    * torch.vmap(torch.trace)(f_l * d_l.transpose(2, 1))
                )
                li += ld**2
        else:
            F = torch.zeros((B * self.num_grid_points)).to(f.device)
            li = 0
            for l in range(self.L + 1):
                ld = 2 * l + 1
                d_l = self.D[l].unsqueeze(0).repeat(B, 1, 1, 1).to(f.device)
                f_l = f[:, li : li + ld**2].view(B, 1, ld, ld)
                f_l = f_l.repeat(1, self.num_grid_points, 1, 1)
                F += (
                    (2 * l + 1)
                    / (8 * torch.pi)
                    * torch.vmap(torch.trace)(
                        f_l.view(B * self.num_grid_points, ld, ld)
                        * d_l.view(B * self.num_grid_points, ld, ld).transpose(2, 1)
                    )
                )
                li += ld**2
            F = F.view(B, self.num_grid_points)

        return F
