import numpy as np

#=== useful constants====
pc_to_meter = 3.08567782e16
au_to_meter = 149597870700.
#========================

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

def N_in_2d(arr):
    """
    Find the largest number of non-zero values along the x and y axes of a 2D array.
    
    Args:
        arr : The 2D input array.

    Returns:
        A tuple containing two integers (max_x, max_y) where max_x is the largest
        number of non-zero values along the x-axis and max_y is the largest number
        of non-zero values along the y-axis.
    """
    non_zero_x = np.count_nonzero(arr, axis=0)
    non_zero_y = np.count_nonzero(arr, axis=1)
    
    max_x = np.max(non_zero_x)
    max_y = np.max(non_zero_y)
    
    return max_x, max_y


def bluestein_pad(arr, N_in, N_out):
    """
    Pad a 2D array to size (N_in + N_out - 1, N_in + N_out - 1) for computing a Bluestein FFT
    with N_in non-zero elements and N_out FFT samples needed. 
    The original array will be centered in the padded output.

    Args:
        arr: The 2D input array to be padded.
        N_in: The number of non-zero elements of the input arr
        N_out: The number of output elements of the Bluestein FFT.

    Returns:
        The (centered) padded 2D array with zeros.
    """
    zp_arr = np.zeros((N_in + N_out - 1, N_in + N_out - 1), dtype=np.complex128)
    half_zp_N = (N_in + N_out - 1) // 2
    half_arr_N = np.shape(arr)[0] // 2
    bit_arr = np.shape(arr)[0] % 2
    zp_arr[half_zp_N - half_arr_N : half_zp_N + half_arr_N + bit_arr,
           half_zp_N - half_arr_N : half_zp_N + half_arr_N + bit_arr] = arr
    return zp_arr

def zero_pad(arr, N_in, ZP):
    """
    Zero-pad arr to the size (N_in * ZP + 1, N_in * ZP + 1).
    The original array will be centered in the padded output.

    Args:
    arr : 2D input field.
    N_in : Number of non-zero input samples.
    ZP : Zero-padding factor.

    Returns:
        The (centered) padded 2D array with zeros.
    """
    zp_arr = np.zeros((N_in * ZP + 1, N_in * ZP + 1), dtype=np.complex128)
    half_zp_N = (N_in * ZP + 1) // 2
    half_arr_N = np.shape(arr)[0] // 2
    bit_arr = np.shape(arr)[0] % 2
    zp_arr[half_zp_N - half_arr_N : half_zp_N + half_arr_N + bit_arr,
           half_zp_N - half_arr_N : half_zp_N + half_arr_N + bit_arr] = arr
    return zp_arr
