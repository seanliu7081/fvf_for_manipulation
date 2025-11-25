""" harmonic_function.py """

from abc import abstractmethod

import torch
from torch import nn


class HarmonicFunction(nn.Module):
    """Abstract class for harmonic functions in Torch."""

    @abstractmethod
    def generate_basis_fns(self) -> torch.Tensor:
        """Generate the basis functions for the pre-defined grid coordinates."""

    @abstractmethod
    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluate the function using the provided coefficients. Compute on the coordinates
        provided otherwise uses the pre-defined grid coordinates.
        Args:
            w: Harmonic cofficients.
            coords: Coordinates to evaluate the function at. Defaults to None.
        """
