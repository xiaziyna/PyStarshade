import numpy as np
from bluestein_fft import zoom_fft_2d_mod

def zoom_fresnel_single_fft(field, d_x, z, wavelength, ZP, N_in, N_out):
    """
    Single FFT fresnel diffraction using Bluestein FFT

    Output on grid defined by (N_out/ZP*N_in)*wl_z/d_x, (N_out/ZP*N_in)*wl_z/d_x

    Args:
    field : 2D input field to be propagated. Field should be at least the size of (N_in + N_out - 1)
    d_x : Sampling interval of the input field [m]
    z : Propagation distance [m]
    wavelength : Wavelength of light [m]
    ZP : Zero-padding factor
    N_in :  Number of input samples in each dimension
    N_out : Number of output samples in each dimension

    Returns:
        tuple: The propagated output field and the output grid points
    """

    wl_z = wavelength * z
    max_freq =  1 / d_x
    k = 2 * np.pi / wavelength
    Ny, Nx = field.shape    
    in_xy = grid_points(Nx, Ny, dx = d_x)

    output_field = zoom_fft_2d_mod(ZP, N_in, N_out, field * np.exp(1j * (np.pi /wl_z) * (in_xy[0]**2 + in_xy[1]**2))) * (d_x**2)
    out_xy = grid_points(N_out, N_out, dx = (max_freq*wl_z / (ZP*N_in + 1)) )

    quad_out_fac = np.exp(1j * k * z) * np.exp(1j * k / (2 * z) * (out_xy[0]**2 + out_xy[1]**2)) / ( 1j * wl_z) 
    return quad_out_fac * output_field, out_xy

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
