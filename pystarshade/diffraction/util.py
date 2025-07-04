import numpy as np
import os
from importlib import resources
import h5py

#=== useful constants====
pc_to_meter = 3.08567782e16
au_to_meter = 149597870700.
mas_to_rad = 4.84814e-9 # preference
rad_to_mas = 206264806.2471
#========================

def data_file_path(file_name, *subfolders):
    """
    Constructs the full path to a file in the 'data' directory.

    Parameters
    ----------
    file_name : str
        The relative name of the file within the 'data' directory.
    subfolders : str
        Additional subfolders within the 'data' directory.

    Returns
    -------
    str
        The full path to the file.
    """
    package_path = resources.files('pystarshade')
    data_path = os.path.abspath(os.path.join(str(package_path), 'data'))
    os.environ['BASE_PATH'] = os.path.abspath(os.path.join(package_path, 'data'))
    base_path = os.environ.get('BASE_PATH')
    return os.path.join(base_path, *subfolders, file_name)

def ang_res(wl, D):
    """
    Calculate the angular resolution of lens with diameter D in milliarcseconds.

    Parameters
    ----------
    wl : float
        Wavelength of light (in meters).
    D : float
        Diameter of the aperture (in meters).

    Returns
    -------
    float
        Angular resolution in milliarcseconds.
    """
    return (wl/D)/mas_to_rad

def flux_to_mag(flux, pixel_size_mas=2):
    """
    Calculate the surface brightness in mag/arcsec^2 from flux per pixel.

    Parameters
    ----------
    flux : float
        Flux per pixel in Jansky (Jy/pixel).
    pixel_size_mas : float
        Pixel size in milliarcseconds (mas). Default is 2.0.

    Returns
    -------
    float
        Surface brightness in mag/arcsec^2.
    """
    pixels_per_arcsec2 = (1000 // pixel_size_mas)**2  # Pixels per arcsec^2
    return -2.5 * np.log10(flux * pixels_per_arcsec2 / 3640.0)

def fresnel_num(R, wl, dist_ss_t):
    """
    Calculate the Fresnel number.

    Parameters
    ----------
    R : float
        Radius of the aperture (in meters).
    wl : float
        Wavelength of light (in meters).
    dist_ss_t : float
        Distance between the starshade and the telescope (in meters).

    Returns
    -------
    float
        The Fresnel number.
    """
    return R**2 / (wl * dist_ss_t)

def flat_grid(N, negative=True):
    """
    Generate a 2D grid of points as a flat array.

    Parameters
    ----------
    N : int
        Number of points in each dimension of the grid.
    negative : bool, optional
        If True, the grid ranges from (-N/2 to N/2) including zero. If False, the grid ranges from (0 to N).
        Default is True.

    Returns
    -------
    np.ndarray
        A flattened array containing grid points.
    """
    if negative: xv, yv = np.meshgrid(np.arange(-(N//2), (N//2)+1), np.arange(-(N//2), (N//2)+1))
    else: xv, yv = np.meshgrid(np.arange(N), np.arange(N))
    return np.hstack((xv.flatten('F')[:,np.newaxis],yv.flatten('F')[:,np.newaxis]))


def trunc_2d(x, N_out):
    """
    Truncate a 2D array to a smaller square size centered around the original array's center.

    Parameters
    ----------
    x : np.ndarray
        The input 2D array to be truncated.
    N_out : int
        The size of the output square array (N_out x N_out).

    Returns
    -------
    np.ndarray
        The truncated square 2D array of size (N_out x N_out).
    """
    bit = N_out % 2
    trunc_x = x[(x.shape[0]//2) - (N_out//2) : (x.shape[0]//2) + (N_out//2) + bit, (x.shape[1]//2) - (N_out//2) : (x.shape[1]//2) + (N_out//2) + bit]
    return trunc_x


def grid_points(Nx, Ny, dx = 1):
    """
    Generate a grid of points with specified dimensions and sampling interval.

    Parameters
    ----------
    Nx : int
        Number of points in the x-dimension.
    Ny : int
        Number of points in the y-dimension.
    dx : float, optional
        Sampling interval. Default is 1.

    Returns
    -------
    list of np.ndarray
        Two 2D arrays: one for the x-coordinates and one for the y-coordinates.
    """
    x = np.arange(-(Nx // 2), (Nx // 2) + 1)[np.newaxis, :] * dx
    y = np.arange(-(Ny // 2), (Ny // 2) + 1)[:, np.newaxis] * dx
    return [x, y]

def N_in_2d(arr):
    """
    Find the largest number of non-zero values along the x and y axes of a 2D array.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array.

    Returns
    -------
    tuple of int
        A tuple containing two integers (max_x, max_y):
        - max_x is the largest number of non-zero values along the x-axis.
        - max_y is the largest number of non-zero values along the y-axis.
    """
    non_zero_x = np.count_nonzero(arr, axis=0)
    non_zero_y = np.count_nonzero(arr, axis=1)
    
    max_x = np.max(non_zero_x)
    max_y = np.max(non_zero_y)
    
    return max_x, max_y

def bluestein_pad(arr, N_in, N_out_x, N_out_y=None):
    """
    Pad a 2D array for a Bluestein FFT with non-zero elements and specified FFT samples.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array to be padded.
    N_in : int
        Number of non-zero elements in the input array.
    N_out_x : int
        Number of output elements of the Bluestein FFT in the x-dimension.
    N_out_y : int, optional
        Number of output elements of the Bluestein FFT in the y-dimension. If None, it is set to N_out_x.

    Returns
    -------
    np.ndarray
        The padded 2D array with zeros, centered around the original array.
    """
    if N_out_y == None: N_out_y = N_out_x
    zp_arr = np.zeros((N_in + N_out_x - 1, N_in + N_out_y - 1), dtype=np.complex128)
    half_zp_N_x = (N_in + N_out_x - 1) // 2
    half_zp_N_y = (N_in + N_out_y - 1) // 2
    half_arr_N = np.shape(arr)[0] // 2
    bit_arr = np.shape(arr)[0] % 2
    zp_arr[half_zp_N_x - N_in//2 : half_zp_N_x + N_in//2 + bit_arr, \
           half_zp_N_y - N_in//2 : half_zp_N_y + N_in//2 + bit_arr] \
            = arr[half_arr_N - N_in//2 : half_arr_N + N_in//2 + bit_arr, \
            half_arr_N - N_in//2 : half_arr_N + N_in//2 + bit_arr]
    return zp_arr

def pad_2d(arr, N_out):
    """
    Zero-pad a 2D array to a larger square size.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array to be padded.
    N_out : int
        The size of the output square array (N_out x N_out).

    Returns
    -------
    np.ndarray
        The padded 2D array with zeros, centered around the original array.
    """
    pad_arr = np.zeros((N_out, N_out), dtype=arr.dtype)
    half_N_in = arr.shape[0] // 2
    bit = arr.shape[0] % 2
    pad_arr[N_out//2 - half_N_in : N_out//2 + half_N_in + bit ,N_out//2 - half_N_in : N_out//2 + half_N_in + bit] = arr
    return pad_arr


def zero_pad(arr, N_in, ZP):
    """
    Zero-pad a 2D array to a size (N_in * ZP + 1, N_in * ZP + 1).

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array.
    N_in : int
        Number of non-zero input samples.
    ZP : int
        Zero-padding factor.

    Returns
    -------
    np.ndarray
        The zero-padded 2D array, centered around the original array.
    """
    zp_arr = np.zeros((N_in * ZP + 1, N_in * ZP + 1), dtype=np.complex128)
    half_zp_N = (N_in * ZP + 1) // 2
    half_arr_N = np.shape(arr)[0] // 2
    bit_arr = np.shape(arr)[0] % 2
    zp_arr[half_zp_N - half_arr_N : half_zp_N + half_arr_N + bit_arr,
           half_zp_N - half_arr_N : half_zp_N + half_arr_N + bit_arr] = arr
    return zp_arr


def h5_to_npz(src_h5: str, dst_npz: str | None = None) -> None:
    """
    Convert an arbitrary HDF5 file to a flat .npz archive and delete the source.

    Warning
    -------
    * All datasets are loaded fully into memory.  If the .h5 is larger than
      available RAM, this will crash.  Keep HDF5 for truly large files.
    * HDF5 attributes are *not* copied—npz has no native attribute support.
    """
    if dst_npz is None:
        dst_npz = os.path.splitext(src_h5)[0] + ".npz"

    arrays = {}

    with h5py.File(src_h5, "r") as f:
        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                arrays[name.lstrip("/")] = obj[...]
        f.visititems(_collect)

    np.savez_compressed(dst_npz, **arrays)
    os.remove(src_h5)
