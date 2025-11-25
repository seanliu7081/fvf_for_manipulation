""" spherical_harmonics.py  """

import numpy as np
import torch
from torch import nn

from lie_learn.representations.SO3 import spherical_harmonics
from lie_learn.spaces import S2

from eharmony.harmonic_function import HarmonicFunction


class RadialSphericalHarmonics(HarmonicFunction):
    """
    Torch module for computing a spherical function using Fourier coefficients and radial spherical
    harmonics basis functions. Pre-computes basis functions for a grid of values which can
    be used for faster evaluation.

    Args:
       L - Maximum angular frequency.
       num_lat - Number of elements on the latitudinal axis.
       num_lon - Number of elements on the longitudinal axis.
    """

    def __init__(self, N: int, L: int, num_lat: int = 360, num_lon: int = 360):
        super().__init__()

        self.N = N
        self.L = L
        self.num_lat = num_lat
        self.num_lon = num_lon

        self.Y = self.generate_basis_fns()

    def generate_basis_fns(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is None:
            alpha, beta = np.meshgrid(
                np.linspace(0, 2 * np.pi, self.num_lon),
                np.linspace(0, 2 * np.pi, self.num_lat),
            )
            # beta, alpha = S2.meshgrid(self.num_lat)
        else:
            beta = coords[:, 1]
            alpha = coords[:, 0]

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
            beta[None, :, :],
            alpha[None, :, :],
            field="real",
            normalization="quantum",
            condon_shortley=True,
        )

        return torch.tensor(Y)

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        B = w.size(0)
        device = w.device()

        if coords is not None:
            Y = self.generate_basis_fns(coords).repeat(B, 1, 1, 1)
        else:
            Y = self.Y.repeat(B, 1, 1, 1)

        out = torch.einsum("bn,bncd->bncd", w.cpu(), Y).sum(1)

        return out.to(device)
