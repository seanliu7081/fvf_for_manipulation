""" spherical_harmonics.py  """

import numpy as np
import torch
from torch import nn

from lie_learn.representations.SO3 import spherical_harmonics
from lie_learn.spaces import S2

from eharmony.harmonic_function import HarmonicFunction


class SphericalHarmonics(HarmonicFunction):
    """
    Torch module for computing a spherical function using Fourier coefficients and spherical
    harmonics basis functions. Pre-computes basis functions for a grid of values which can
    be used for faster evaluation.

    Args:
       L - Maximum angular frequency.
       num_lat - Number of elements on the latitudinal axis.
       num_lon - Number of elements on the longitudinal axis.
    """

    def __init__(
        self, L: int, grid_type="lie_learn", num_theta: int = 360, num_phi: int = 360
    ):
        super().__init__()

        self.L = L
        self.grid_type = grid_type
        self.num_theta = num_theta
        self.num_phi = num_phi

        self.mlp = nn.Linear(L, L)

        Y = self.generate_basis_fns()
        self.register_buffer("Y", Y, persistent=False)

    def generate_basis_fns(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is None:
            if self.grid_type == "lie_learn":
                self.grid = S2.meshgrid(self.num_theta, grid_type="Driscoll-Healy")
                self.num_theta = self.grid[0].shape[0]
                self.num_phi = self.grid[0].shape[1]

                theta, phi = self.grid
            else:
                self.grid = np.meshgrid(
                    np.linspace(0, np.pi, self.num_theta),
                    np.linspace(0, 2 * np.pi, self.num_phi),
                )

                theta, phi = self.grid
        else:
            theta = coords[:, 0].view(-1, 1)
            phi = coords[:, 1].view(-1, 1)

        irreps = np.arange(self.L + 1)
        ls = [[ls] * (2 * ls + 1) for ls in irreps]
        ls = np.array(
            [ll for sublist in ls for ll in sublist]
        )  # 0, 1, 1, 1, 2, 2, 2, 2, 2, ...

        ms = [list(range(-ls, ls + 1)) for ls in irreps]
        ms = np.array(
            [mm for sublist in ms for mm in sublist]
        )  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...

        Y = spherical_harmonics.sh(
            ls[:, None, None],
            ms[:, None, None],
            theta[None, :, :],
            phi[None, :, :],
            field="real",
            normalization="quantum",
            condon_shortley=True,
        )

        return torch.tensor(Y).float()

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        B = w.size(0)
        R = w.size(1)

        if coords is not None:
            Y = self.generate_basis_fns(coords.cpu())
            Y = Y.permute(1, 0, 2).unsqueeze(3)
            Y = Y.to(w.device)
        else:
            Y = self.Y.unsqueeze(0)
            Y = Y.expand(B, Y.size(1), Y.size(2), Y.size(3))

        out = torch.zeros((B, self.num_phi, self.num_theta, self.L, R), device=w.device)
        li = 0
        for l in range(self.L):
            if l == 0:
                w_l = w[:, 0:1]
                Y_l = Y[:, 0:1, :, :]
            else:
                w_l = w[:, li : (l + 1) ** 2]
                Y_l = Y[:, li : (l + 1) ** 2, :, :]
            out[:, :, :, L, :] = torch.einsum("brn,brncd->bcdr", w_l, Y_l)
            li = (l + 1) ** 2

        out = self.mlp(out.flatten(start_dim=-2))
        out = out.view(B, self.num_phi, self.num_theta, self.L).sum(-1)

        return out
