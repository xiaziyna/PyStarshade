import numpy as np
from scipy.ndimage import convolve

#hypergaussian apodization profile
def t_MH(r, theta):
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

def grid_points(Nx, Ny, dx = 1):
    """
    Generate a grid of points with specified dimensions and sampling interval.

    Args:
    Nx : Number of points in the x-dimension
    Ny : Number of points in the y-dimension
    dx : Sampling interval

    Returns:
        list: Two 2D arrays, one for the x-coordinates and one for the y-coordinates.
    """
    x = np.arange(-(Nx // 2), (Nx // 2) + 1)[np.newaxis, :] * dx
    y = np.arange(-(Ny // 2), (Ny // 2) + 1)[:, np.newaxis] * dx
    return [x, y]


def grey_pupil_func(x, y, dx = 1, r = 3.):
    """
    Grey-pixel anti-aliased pupil function using upsampling.
    
    Args:
    (x, y) : Cartesian coordinates
    r : radius of the pupil

    Returns:
    Grey-pixel pupil mask
 
    """
    Nx = len(y)
    up_fac = 4
    upsample = np.meshgrid(np.arange(-(up_fac * Nx / 2), (up_fac * Nx / 2) + 1), np.arange(-(up_fac * Nx / 2), (up_fac * Nx / 2) + 1))
    bool_mask = pupil_func(upsample[0], upsample[1], (up_fac * r / dx)).astype(float)
    grey_mask = convolve(bool_mask, np.ones((up_fac, up_fac)) )[(up_fac//2)::up_fac, (up_fac//2)::up_fac].astype(np.float32)
    return grey_mask
