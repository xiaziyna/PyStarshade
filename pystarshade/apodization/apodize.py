import numpy as np
from scipy.ndimage import convolve

from pystarshade.diffraction.util import grid_points


def eval_hypergauss(N, dx):
    """
    Evaluate truncated Hypergaussian on a grid.

    Parameters
    ----------
    N : int
        Size of the grid in both dimensions (N x N).
    dx : float
        Pixel size of the grid.

    Returns
    -------
    np.ndarray
        A 2D array representing the evaluated truncated Hypergaussian apodization profile on the grid.
    """
    grid_xy = grid_points(N, N, dx=dx)
    (r_, th) = cart_to_pol(grid_xy[0], grid_xy[1])
    return hypergauss(r_, th)


def hypergauss(r, theta):
    """
    Truncated HyperGaussian Apodization: returns mask over polar grid (r, theta).

    Parameters
    ----------
    r : np.ndarray
        Radial coordinates in polar space.
    theta : np.ndarray
        Angular coordinates in polar space.

    Returns
    -------
    np.ndarray
        Binary mask over polar coordinates (r, theta).

    Notes
    -----
    Hidden parameters: (a, b, n). See Cash (2011).
    """

    a = b = 12.5
    R = 32
    n = 6.0

    mask_within_a = 1 * (r <= a)
    mask_between_a_and_R = (r > a) * np.exp(-(((r - a) / b) ** n)) * (r < R)

    return 1 - (mask_within_a + mask_between_a_and_R)


def cart_to_pol(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).

    Parameters
    ----------
    x : np.ndarray
        X-coordinates in Cartesian space.
    y : np.ndarray
        Y-coordinates in Cartesian space.

    Returns
    -------
    tuple of np.ndarray
        Polar coordinates (r, theta).
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta


def spher_to_cart(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

    Parameters
    ----------
    r : float
        Radius (> 0).
    theta : float
        Polar angle (0 to pi radians).
    phi : float
        Azimuthal angle (0 to 2*pi radians).

    Returns
    -------
    np.ndarray
        Cartesian coordinates (x, y, z).
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def cart_to_spher(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters
    ----------
    x : float
        X-coordinate in Cartesian space.
    y : float
        Y-coordinate in Cartesian space.
    z : float
        Z-coordinate in Cartesian space.

    Returns
    -------
    np.ndarray
        Spherical coordinates (r, theta, phi).
    """
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def pupil_func(x, y, r=3.0):
    """
    Return binary circle mask of radius r on Cartesian grid (x, y).

    Parameters
    ----------
    x : np.ndarray
        X-coordinates in Cartesian space.
    y : np.ndarray
        Y-coordinates in Cartesian space.
    r : float, optional
        Radius of the mask. Default is 3.0.

    Returns
    -------
    np.ndarray
        Pupil (circle) mask.
    """
    return np.hypot(x, y) <= r


def grey_pupil_func(Nx, dx=1, r=3.0):
    """
    Grey-pixel pupil function for spatial anti-aliasing using upsampling method.

    Parameters
    ----------
    Nx : int
        Number of points in each dimension.
    dx : float, optional
        Sample size. Default is 1.
    r : float, optional
        Radius of the pupil. Default is 3.0.

    Returns
    -------
    np.ndarray
        Grey-pixel pupil mask.
    """
    up_fac = 4
    bool_var = Nx % 2
    x_ = y_ = np.arange((up_fac * Nx / 2) + bool_var)
    xv, yv = np.meshgrid(x_, y_)
    bool_mask = (np.hypot(xv, yv) <= (up_fac * r / dx)).astype(float)
    grey_mask = (up_fac**-2) * convolve(
        bool_mask, np.ones((up_fac, up_fac))
    )[::up_fac, ::up_fac].astype(np.float32)
    full_mask = np.zeros((Nx, Nx))
    full_mask[: Nx // 2, Nx // 2 + bool_var :] = np.flipud(
        grey_mask[bool_var:, bool_var:]
    )
    full_mask[Nx // 2 :, Nx // 2 :] = grey_mask
    full_mask[Nx // 2 + bool_var :, : Nx // 2] = np.fliplr(
        grey_mask[bool_var:, bool_var:]
    )
    full_mask[: Nx // 2 + bool_var, : Nx // 2 + bool_var] = np.fliplr(
        np.flipud(grey_mask)
    )

    return full_mask


def qu_mask_to_full(grey_mask):
    """
    Take the negative quadrant of a starshade mask and return the full mask.

    Parameters
    ----------
    grey_mask : np.ndarray
        Negative quadrant of the mask.

    Returns
    -------
    np.ndarray
        Full mask made by flipping the negative quadrant.
    """
    Nx = Ny = len(grey_mask) - 1
    full_mask = np.zeros((2 * Nx + 1, 2 * Ny + 1), dtype=grey_mask.dtype)
    full_mask[:Nx, Ny + 1 :] = np.flipud(grey_mask[1:, 1:])
    full_mask[Nx:, Ny:] = grey_mask
    full_mask[Nx + 1 :, :Ny] = np.fliplr(grey_mask[1:, 1:])
    full_mask[: Nx + 1, : Ny + 1] = np.fliplr(np.flipud(grey_mask))
    return full_mask
