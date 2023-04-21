import numpy as np
from scipy.ndimage import convolve
from diffraction.util import grid_points

def eval_hypergauss(N, dx):
    """
    Evaluate truncated Hypergaussian on grid 

    Args:
        N : Size of the grid in both dimensions (N x N).
        dx : Pixel size of the grid.

    Returns:
        A 2D array representing the evaluated truncated Hypergaussian apodization profile on the grid.
    """
    grid_xy = grid_points(N, N, dx=dx)
    (r_, th) = cart_to_pol(grid_xy[0], grid_xy[1])
    return hypergauss(r_, th)

#hypergaussian apodization profile
def hypergauss(r, theta):
    """
    Truncated HyperGaussian Apodization: returns mask over polar grid (r, theta)

    Args:
    (r, theta): Polar coordinates.
    
    Hidden parameters: (a, b, n) See Cash. 2011

    Returns:
    Binary mask over polar coordinates (r, theta)
    """
    a=b=12.5
    R = 32
    n=6.
    
    mask_within_a = 1 * (r <= a)
    mask_between_a_and_R = (r > a) * np.exp(-((r - a) / b) ** n) * (r < R)

    return 1 - (mask_within_a + mask_between_a_and_R)

def cart_to_pol(x,y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).
    
    Args:
    (x, y): Cartesian coordinates. 

    Returns:
    (r, theta): Polar coordinates.

    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta

def spher_to_cart(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Parameters:
    (r, theta, phi): Spherical coordinates - radius (>0), polar angle(0 to pi rad), azimuthal angle (0 to 2pi rad). 
    
    Returns:
    (x, y, z):  Cartesian coordinates. 
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cart_to_spher(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    
    Arg:
    (x, y, z): Cartesian coordinates.

    Returns:
    (r, theta, phi): Spherical coordinates  - radius (>0), polar angle(0 to pi rad), azimuthal angle (0 to 2pi rad). 
    """
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])

def pupil_func(x, y, r = 3.):
    """
    Return binary circle mask of radius r on cartesian grid (x, y)
    
    Parameters:
    (x, y): Cartesian coordinates on which to evaluate mask.
    r: radius of mask.

    Returns:
    Pupil/circle mask 
    """
    return np.hypot(x, y) <= r


def grey_pupil_func(Nx, dx = 1, r = 3.):
    """
    Grey-pixel pupil function using upsampling method
    
    Args:
    Nx : Number of points
    dx : Sample size
    r : Radius of the pupil

    Returns:
    Grey-pixel pupil mask
 
    """
    up_fac = 4
    upsample = np.meshgrid(np.arange(-(up_fac * Nx / 2), (up_fac * Nx / 2) + 1), np.arange(-(up_fac * Nx / 2), (up_fac * Nx / 2) + 1))
    bool_mask = pupil_func(upsample[0], upsample[1], (up_fac * r / dx)).astype(float)
    grey_mask = convolve(bool_mask, np.ones((up_fac, up_fac)) )[(up_fac//2)::up_fac, (up_fac//2)::up_fac].astype(np.float32)
    return grey_mask
