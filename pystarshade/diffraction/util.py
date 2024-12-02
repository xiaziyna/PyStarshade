import numpy as np

#=== useful constants====
pc_to_meter = 3.08567782e16
au_to_meter = 149597870700.
mas_to_rad = mas_to_rad = 4.84814e-9 # preference
rad_to_mas = 206264806.2471
#========================

def ang_res(wl, D):
    return (wl/D)/mas_to_rad

def fresnel_num(R, wl, dist_ss_t):
    return R**2 / (wl * dist_ss_t)

def flat_grid(N, negative=True):
    """
    Generate a 2D grid of points into a flat array
    If negative is true, the grid will range from (-N/2 to N/2) including zero
    If negative is false, the grid will range from (0 to N)
    """
    if negative: xv, yv = np.meshgrid(np.arange(-(N//2), (N//2)+1), np.arange(-(N//2), (N//2)+1))
    else: xv, yv = np.meshgrid(np.arange(N), np.arange(N))
    return np.hstack((xv.flatten('F')[:,np.newaxis],yv.flatten('F')[:,np.newaxis]))


def trunc_2d(x, N_out):
    """
    Truncate a 2D array to a smaller square size centered around the original array's center.

    Args:
        x (numpy.ndarray): The input 2D array to be truncated.
        N_out (int): The size of the output square array (N_out x N_out).

    Returns:
        numpy.ndarray: The truncated square 2D array of size (N_out x N_out).
    """
    bit = N_out % 2
    trunc_x = x[(x.shape[0]//2) - (N_out//2) : (x.shape[0]//2) + (N_out//2) + bit, (x.shape[1]//2) - (N_out//2) : (x.shape[1]//2) + (N_out//2) + bit]
    return trunc_x


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

def bluestein_pad(arr, N_in, N_out_x, N_out_y=None):
    """
    Pad a 2D array to size (N_in + N_out_x - 1, N_in + N_out_y - 1) for computing a Bluestein FFT
    with N_in non-zero elements and N_out FFT samples needed. 
    The original array will be centered in the padded output.

    Args:
        arr: The 2D input array to be padded.
        N_in: The number of non-zero elements of the input arr
        N_out_x: The number of output elements of the Bluestein FFT.
        if N_out_y undefined, assume x and y out are the same
        
    Returns:
        The (centered) padded 2D array with zeros.
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
    pad_arr = np.zeros((N_out, N_out), dtype=arr.dtype)
    half_N_in = arr.shape[0] // 2
    bit = arr.shape[0] % 2
    pad_arr[N_out//2 - half_N_in : N_out//2 + half_N_in + bit ,N_out//2 - half_N_in : N_out//2 + half_N_in + bit] = arr
    return pad_arr


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
