""" grid.py """

from typing import Tuple
import numpy as np
import torch


def grid1D(
    boxsize: float, ngrid: int, origin: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the x coordinates of a cartesian grid.

    Args:
        boxsize : Box size.
        ngrid : Grid division along one axis.
        origin : Start point of the grid.
    """
    xedges = torch.linspace(0.0, boxsize, ngrid + 1) + origin
    x = 0.5 * (xedges[1:] + xedges[:-1])
    return xedges, x


def polar_grid(
    r_max: float, num_r: int, num_phi: int, r_min: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a 2D polar grid.

    Args:
        r_max - Maximum radius.
        Nr - Number of elements along the radial axis.
        Nphi- Number of elements along the angular axis.
        r_min - Minimum radius.
    """
    _, r = grid1D(r_max, num_r, origin=r_min)
    _, p = grid1D(2.0 * torch.pi, num_phi)
    r2d, p2d = torch.meshgrid(r, p, indexing="ij")
    return r2d, p2d


def reg_polar_grid(
    r_max: float, num_r: int, num_phi: int, r_min: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a 2D polar grid.

    Args:
        r_max - Maximum radius.
        Nr - Number of elements along the radial axis.
        Nphi- Number of elements along the angular axis.
        r_min - Minimum radius.
    """
    _, r = grid1D(r_max, num_r, origin=r_min)
    n = [num_phi * (i + 1) for i in range(num_r)]
    p = torch.concat([torch.linspace(0, 2 * np.pi, i + 1) for i in n])
    r2d, p2d = torch.meshgrid(r, p, indexing="ij")
    return r2d, p2d


def cylinder_grid(
    r_max: float,
    z_max: float,
    num_r: int,
    num_phi: int,
    num_z: int,
    r_min: float = 0.0,
    z_min: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a 3D cylinder grid.

    Args:
        r_max - Maximum radius.
        z_max - Maximum height.
        num_r - Number of elements along the radial axis.
        num_phi - Number of elements along the angular axis.
        num_z - Number of elements alongthe axial axis
        r_min - Minimum radius.
        z_min - Minimum height.
    """
    _, r = grid1D(r_max, num_r, origin=r_min)
    _, p = grid1D(2.0 * torch.pi, num_phi)
    _, z = grid1D(z_max, num_z, origin=z_min)
    r2d, p2d, z2d = torch.meshgrid(r, p, z, indexing="ij")
    return r2d, p2d, z2d


def spherical_healpix_grid(N_side: int) -> torch.Tensor:
    """Generate a grid of points on the 2-sphere using Healpix.

    Args:
        N_side - Number of elements.
    """
    north_points = []

    # north polar cap
    for i in range(1, N_side):
        for j in range(1, 4 * i + 1):
            cos_theta = 1.0 - i**2 / (3 * N_side**2)
            phi = np.pi / (2 * i) * (j - 0.5)

            sin_theta = np.sqrt(1.0 - cos_theta**2)

            north_points.append(
                np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
            )

    # north equatorial belt
    for i in range(N_side, 2 * N_side + 1):
        for j in range(1, 4 * N_side + 1):
            cos_theta = 4.0 / 3.0 - 2 * i / (3 * N_side)
            s = (i - N_side + 1) % 2
            phi = np.pi / (2 * N_side) * (j - s / 2.0)

            sin_theta = np.sqrt(1.0 - cos_theta**2)
            north_points.append(
                np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
            )

    points = []
    # add points on the south pole (symmetric to north pole wrt equator, i.e. z=0)
    for p in north_points:
        if p[2] > 0.0:
            sp = p.copy()
            sp[2] *= -1
            points.append(sp)
    points += north_points

    N_pix = 12 * N_side**2
    assert len(points) == N_pix, (len(points), N_side, N_pix)

    points = np.stack(points, axis=0)

    return points
