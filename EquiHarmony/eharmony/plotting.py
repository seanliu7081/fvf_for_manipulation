""" plotting.py """

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import cartopy
import cartopy.crs as ccrs

import numpy.typing as npt


def plot_circular_fn(
    data: npt.NDArray, fig: Figure = None, title: str = None, plot_neg: bool = False
):
    """Plot circular function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="polar")

    x = np.linspace(0, 2 * np.pi, data.shape[0])
    if plot_neg:
        pos_x = x[data > 0]
        pos = data[data > 0]
        neg_x = x[data < 0]
        neg = data[data < 0]
        ax.plot(pos_x, np.abs(pos), color="blue")
        ax.plot(neg_x, np.abs(neg), color="orange")
    else:
        ax.plot(x, data, color="blue")

    ax.set_rmax(np.max(data) + 0.2)
    ax.set_rticks([])

    ax.set_title(title, va="bottom")
    ax.grid(True)
    return ax


def plot_polar_fn(
    data: npt.NDArray,
    r: npt.NDArray = None,
    phi: npt.NDArray = None,
    vmin: float = None,
    vmax: float = None,
    fig: Figure = None,
    title: str = None,
):
    """Plot polar function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="polar")

    if r is None:
        r = np.linspace(0, 1, data.shape[0])
    if phi is None:
        phi = np.linspace(0, 2 * np.pi, data.shape[1])
    ax.pcolormesh(phi, r, data, vmin=vmin, vmax=vmax)

    ax.set_title(title, va="bottom")
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return ax


def plot_cylinder_fn(
    data: npt.NDArray, fig: Figure = None, title: str = None, vmin=None, vmax=None
):
    """Plot cylinder function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    r = np.linspace(0, 1, data.shape[0])
    p = np.linspace(0, 2 * np.pi, data.shape[1])
    z = np.linspace(0, 1, data.shape[2])

    P, R, zs = np.meshgrid(p, r, z)
    xs = (R * np.cos(P)).flatten()
    ys = (R * np.sin(P)).flatten()
    zs = zs.flatten()

    X = np.linspace(-1, 1, 50)
    Y = np.linspace(-1, 1, 50)
    Z = np.linspace(0, 1, 50)
    X, Y, Z = np.meshgrid(X, Y, Z)

    interp = LinearNDInterpolator(list(zip(xs, ys, zs)), data.flatten())
    idata = interp(X, Y, Z)
    idata[np.isnan(idata)] = 0
    mask = idata != 0.0

    plot = ax.scatter(
        X.flatten(),
        Y.flatten(),
        Z.flatten(),
        c=idata.flatten(),
        s=10.0 * mask,
        vmin=vmin,
        vmax=vmax,
        edgecolor="face",
        alpha=0.2,
        marker="o",
        cmap="magma",
        linewidth=0,
    )

    ax.set_title(title, va="bottom")

    return ax


def plot_cylinder_prob(
    data: npt.NDArray, fig: Figure = None, title: str = None, vmin=None, vmax=None
):
    """Plot cylinder function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    r = np.linspace(0, 1, data.shape[0])
    p = np.linspace(0, 2 * np.pi, data.shape[1])
    z = np.linspace(0, 1, data.shape[2])

    ps, rs, zs = np.meshgrid(p, r, z)
    xs = (rs * np.cos(ps)).flatten()
    ys = (rs * np.sin(ps)).flatten()
    zs = zs.flatten()

    data = data.numpy()
    data[np.isnan(data)] = 0
    mask = data > np.mean(data)

    plot = ax.scatter(
        xs.flatten(),
        ys.flatten(),
        zs.flatten(),
        c=data.flatten(),
        s=10.0 * mask,
        vmin=vmin,
        vmax=vmax,
        edgecolor="face",
        alpha=0.2,
        marker="o",
        cmap="magma",
        linewidth=0,
    )

    ax.set_title(title, va="bottom")

    return ax


def plot_spherical_fn(
    data: npt.NDArray,
    fig: Figure = None,
    title: str = None,
    central_longitude: int = 20,
    central_latitude: int = 20,
    vmin: float = None,
    vmax: float = None,
    colorbar: bool = True,
):
    if fig is None:
        fig = plt.figure()

    nlon = data.shape[-1]
    nlat = data.shape[-2]
    lon = np.linspace(0, 2 * np.pi, nlon)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    proj = ccrs.Orthographic(
        central_longitude=central_longitude, central_latitude=central_latitude
    )

    ax = fig.add_subplot(projection=proj)
    Lon = Lon * 180 / np.pi
    Lat = Lat * 180 / np.pi

    # contour data over the map.
    im = ax.pcolormesh(
        Lon,
        Lat,
        data,
        cmap="RdBu",
        transform=ccrs.PlateCarree(),
        antialiased=False,
        vmin=vmin,
        vmax=vmax,
    )

    # x_grid = np.arange(-180, 180, 20)
    # y_grid = np.arange(-180, 180, 20)
    # gl = ax.gridlines(
    #    crs=ccrs.PlateCarree(),
    #    draw_labels=False,
    #    linewidth=2,
    #    color="gray",
    #    alpha=0.5,
    #    linestyle="--",
    #    xlocs=x_grid,
    #    ylocs=y_grid,
    # )
    if False:
        ax.add_feature(
            cartopy.feature.COASTLINE,
            edgecolor="white",
            facecolor="none",
            linewidth=1.5,
        )
    if colorbar:
        plt.colorbar(im)
    ax.set_title(title, y=1.05, fontsize=30)

    return im


def plot_mollweide_spherical_fn(
    data: npt.NDArray,
    fig: Figure = None,
    title: str = None,
    vmin: float = None,
    vmax: float = None,
    colorbar: bool = True,
):
    if fig is None:
        fig = plt.figure()

    nlon = data.shape[-1]
    nlat = data.shape[-2]
    lon = np.linspace(-np.pi, np.pi, nlon)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    ax = fig.add_subplot(projection="mollweide")

    # contour data over the map.
    im = ax.pcolormesh(
        Lon,
        Lat,
        data,
        cmap="RdBu",
        vmin=vmin,
        vmax=vmax,
    )
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if colorbar:
        plt.colorbar(im)
    ax.set_title(title, y=1.05, fontsize=30)

    return im
