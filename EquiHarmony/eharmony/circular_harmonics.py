""" circular_harmonics.py  """

import numpy as np
import torch
from torch import nn
from eharmony.harmonic_function import HarmonicFunction


class CircularHarmonics(HarmonicFunction):
    """
    Torch module for computing a circular function using Fourier coefficients and circular
    harmonics basis functions. Pre-computes basis functions for a grid of values which can
    be used for faster evaluation.

    Args:
       L - Maximum angular frequency.
       num_phi - Number of elements on the angular axis.
    """

    def __init__(self, L: int, num_phi: int = 360):
        super().__init__()

        self.num_phi = num_phi
        self.L = L
        self.basis_fns = nn.Parameter(self.generate_basis_fns(), requires_grad=False)

    def generate_basis_fns(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is None:
            coords = torch.linspace(0, 2 * torch.pi, self.num_phi).view(-1, 1)
        basis_fns = [
            torch.tensor([1 / np.sqrt(2 * torch.pi)] * coords.size(0))
            .view(-1, 1)
            .to(coords.device)
        ]
        for l in range(1, self.L):
            basis_fns.append(torch.cos(l * coords) / np.sqrt(torch.pi))
            basis_fns.append(torch.sin(l * coords) / np.sqrt(torch.pi))

        return torch.stack(basis_fns).permute(1, 0, 2).float().squeeze().permute(1, 0)

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is not None:
            basis_fns = self.generate_basis_fns(coords).permute(1, 0)
            num_basis_fns = self.L * 2 - 1
            out = torch.bmm(
                w.view(-1, 1, num_basis_fns), basis_fns.view(-1, num_basis_fns, 1)
            )
        else:
            out = torch.mm(w, self.basis_fns)

        return out
