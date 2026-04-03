import numpy as np
import os
import scipy.fft
from pystarshade.diffraction.util import bluestein_pad, trunc_2d
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chirp cache (v2 optimisation: dict-based, pre-allocated buffers)
# ---------------------------------------------------------------------------
_chirp_cache = {}


def _build_chirp_cache(N_x, N_out, N_X):
    """
    Build or retrieve a cached set of Bluestein chirp arrays.

    Precomputes the 2D outer products of the chirp vectors and a
    reusable zero-padded buffer, avoiding repeated allocation.

    Parameters
    ----------
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the desired output signal.
    N_X : float
        Phantom zero-padded length for the internal FFT.

    Returns
    -------
    cache : dict
        Keys: 'bb', 'hh' (2D chirp outer products), 'trunc_buf'
        (pre-allocated buffer), 'N_chirp', 'N_x', 'N_out', 'N_X'.
    """
    key = (N_x, N_out, N_X)
    if key in _chirp_cache:
        return _chirp_cache[key]

    N_chirp = N_x + N_out - 1
    bit_chirp = N_chirp % 2

    b = np.exp(-np.pi * (1.0 / N_X) * 1j
               * np.arange(-(N_chirp // 2), (N_chirp // 2) + bit_chirp) ** 2)
    h = np.exp(np.pi * (1.0 / N_X) * 1j
               * np.arange(-(N_out // 2) - (N_x // 2),
                           (N_out // 2) + (N_x // 2) + bit_chirp) ** 2)
    h = np.roll(h, (N_chirp // 2) + 1)
    ft_h = scipy.fft.fft(h, workers=-1)

    bb = np.outer(b, b)
    hh = np.outer(ft_h, ft_h)
    trunc_buf = np.zeros((N_chirp, N_chirp), dtype=np.complex128)

    cache = {'bb': bb, 'hh': hh, 'trunc_buf': trunc_buf,
             'N_chirp': N_chirp, 'N_x': N_x, 'N_out': N_out, 'N_X': N_X}
    _chirp_cache[key] = cache
    return cache


def _bluestein_pad_into(buf, arr, N_in, N_out):
    """
    Copy *arr* into the centre of a pre-allocated zero buffer (in-place).

    Parameters
    ----------
    buf : np.ndarray
        Pre-allocated buffer of shape (N_chirp, N_chirp).
    arr : np.ndarray
        Input array.
    N_in : int
        Number of non-zero input samples.
    N_out : int
        Number of output samples.
    """
    buf[:] = 0
    half_zp = (N_in + N_out - 1) // 2
    half_arr = arr.shape[0] // 2
    bit_arr = arr.shape[0] % 2
    s = slice(half_zp - N_in // 2, half_zp + N_in // 2 + bit_arr)
    sa = slice(half_arr - N_in // 2, half_arr + N_in // 2 + bit_arr)
    buf[s, s] = arr[sa, sa]

def get_cached_chirp_functions(N_x, N_out, N_X):
    """
    Retrieve cached chirp functions for the Bluestein zoom FFT.

    Now delegates to ``_build_chirp_cache`` (v2 optimisation) and returns
    the (bb, ft_h) pair for backward compatibility.

    Parameters
    ----------
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the desired output signal.
    N_X : int
        Zero-padded length used for the internal FFT.

    Returns
    -------
    b_outer : np.ndarray
        2D outer product of the Bluestein chirp vector.
    ft_h : np.ndarray
        1D FFT of the convolution kernel 'h'.
    """
    cache = _build_chirp_cache(N_x, N_out, N_X)
    # Return (bb, ft_h_1d) for backward compat with zoom_fft_2d_mod_cached
    N_chirp = cache['N_chirp']
    bit_chirp = N_chirp % 2
    b = np.exp(-np.pi * (1.0 / N_X) * 1j
               * np.arange(-(N_chirp // 2), (N_chirp // 2) + bit_chirp) ** 2)
    h = np.exp(np.pi * (1.0 / N_X) * 1j
               * np.arange(-(N_out // 2) - (N_x // 2),
                           (N_out // 2) + (N_x // 2) + bit_chirp) ** 2)
    h = np.roll(h, (N_chirp // 2) + 1)
    ft_h = scipy.fft.fft(h, workers=-1)
    return cache['bb'], ft_h

def zoom_fft_2d_mod_cached(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm WITH A CACHED CHIRP.

    Now delegates to ``zoom_fft_2d_mod`` which always uses the v2 cache.

    Parameters
    ----------
    x : np.ndarray
        Centered input signal (complex numpy array).
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the output signal.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : int, optional
        Zero-padded length of input signal (Z_pad * N_x + 1).

    Returns
    -------
    zoom_fft: np.ndarray
        Zoomed FFT of the input signal (complex numpy array).
    """
    return zoom_fft_2d_mod(x, N_x, N_out, Z_pad=Z_pad, N_X=N_X)

@lru_cache(maxsize=32, typed=True)
def get_cached_corr_out(phase_shift, N_out, N_X):
    """
    Retrieve cached outer-product phase correction for zoom_fft_2d_cached.

    Parameters
    ----------
    phase_shift : float
        Fractional pixel shift to apply in Fourier domain.
    N_out : int
        Output size of the zoomed FFT.
    N_X : int
        Total zero-padded size of the FFT input.

    Returns
    -------
    outer_prod_fac : np.ndarray
        Complex 2D array of shape (N_out+1, N_out+1) representing the
        outer product of the phase correction vector with itself.
    """
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    outer_prod_fac = np.outer(out_fac, out_fac)
    return outer_prod_fac


def zoom_fft_2d_cached(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm with cached chirps.

    Now delegates to ``zoom_fft_2d`` which always uses the v2 cache.

    Parameters
    ----------
    x : np.ndarray
        Centered input signal (complex numpy array).
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the output signal.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : int, optional
        Zero-padded length of input signal (Z_pad * N_x + 1).

    Returns
    -------
    np.ndarray
        Zoomed FFT of the input signal (complex numpy array).
    """
    return zoom_fft_2d(x, N_x, N_out, Z_pad=Z_pad, N_X=N_X)


def zoom_fft_2d_mod(x, N_x, N_out, Z_pad=None, N_X=None, _cache=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm.

    MODIFIED VERSION: computes the Bluestein FFT equivalent to
    fftshift(fft2(ifftshift(x_pad))) [N_X/2 - N_out/2: N_X/2 + N_out/2, ...]
    where x_pad is x zero-padded to length N_X.
    The input x is centered.

    v2 optimisations: pre-allocated buffer reuse, in-place multiplies,
    multithreaded scipy FFTs, dict-based chirp cache.

    Parameters
    ----------
    x : np.ndarray
        Centered input signal (complex numpy array).
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the output signal.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : int, optional
        Zero-padded length of input signal (Z_pad * N_x + 1).
    _cache : dict, optional
        Pre-built chirp cache from ``_build_chirp_cache``.

    Returns
    -------
    zoom_fft: np.ndarray
        Zoomed FFT of the input signal (complex numpy array).
    """
    if _cache is None:
        if (Z_pad is None and N_X is None) or (Z_pad is not None and N_X is not None):
            raise ValueError("You must provide exactly one of Z_pad or N_X.")
        if Z_pad is not None:
            N_X = Z_pad * N_x + 1
        _cache = _build_chirp_cache(N_x, N_out, N_X)

    bb = _cache['bb']
    hh = _cache['hh']
    trunc_buf = _cache['trunc_buf']
    N_chirp = _cache['N_chirp']
    bit_out = N_out % 2

    _bluestein_pad_into(trunc_buf, x, N_x, N_out)

    np.multiply(bb, trunc_buf, out=trunc_buf)
    tmp = scipy.fft.fft2(trunc_buf, workers=-1)
    np.multiply(tmp, hh, out=tmp)
    tmp = scipy.fft.ifft2(tmp, workers=-1)
    np.multiply(bb, tmp, out=tmp)

    s = slice((N_chirp // 2) - (N_out // 2),
              (N_chirp // 2) + (N_out // 2) + bit_out)
    return tmp[s, s].copy()

def zoom_fft_2d(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm with phase correction.
    The input x is centered.

    Parameters
    ----------
    x : np.ndarray
        Centered input signal (complex numpy array).
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the output signal.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : int, optional
        Zero-padded length of input signal (Z_pad * N_x + 1).

    Returns
    -------
    np.ndarray
        Zoomed FFT of the input signal (complex numpy array).
    """
    if (Z_pad is None and N_X is None) or (Z_pad is not None and N_X is not None):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")
    if Z_pad is not None:
        N_X = Z_pad * N_x + 1
        phase_shift = (N_x * Z_pad) // 2 + 1
    else:
        phase_shift = float((N_X - 1) // 2 + 1)

    uncorrected = zoom_fft_2d_mod(x, N_x, N_out, N_X=N_X)
    out_fac = np.exp(np.arange(-(N_out // 2), (N_out // 2) + 1)
                     * (1j * 2 * np.pi * phase_shift / N_X))
    return uncorrected * np.outer(out_fac, out_fac)

def four_chunked_zoom_fft_mod(x_file, N_x, N_out, N_X):
    """
    Cumulatively computes a 2D zoom FFT with four smaller Bluestein FFTs.
    Peak memory usage ~ N_x/4 + N_out.
    MODIFIED VERSION:
    fftshift(fft2(ifftshift(x_pad))) [N_X/2 - N_out/2: N_X/2 + N_out/2, N_X/2 - N_out/2: N_X/2 + N_out/2],
    where x_pad is the zero-padded x_file to length N_X.

    Parameters
    ----------
    x_file : np.memmap
        Input to FFT (a numpy memmap object of type np.complex128).
    N_x : int
        Size in one dimension of x_file.
    N_out : int
        Number of output points of FFT needed.
    N_X : int
        Phantom zero-padded length of input x_file for desired output sampling 
        (see the fresnel class to calculate this).

    Returns
    -------
    zoom_fft_out: np.ndarray
        Returns the 2D FFT over the chosen output region (np.complex128).
    """
    bit_x = N_x%2
    sec_N_x = N_x//2 + 1 # this is the (max) size of non-zero portion of segment, same for all 4
    shift_bit = 1 - sec_N_x%2
    sec_N_x += shift_bit # want this to be an odd number as it makes it easier to calculate shifts and center
    phase_shift =  sec_N_x//2
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    x_trunc = np.memmap(x_file, dtype=np.complex128, mode='r', shape=(N_x,N_x))
    for i in range(4):
        if i == 0: x = x_trunc[:N_x//2 + bit_x, :N_x//2 + bit_x] #upper left
        elif i == 1: x = x_trunc[:N_x//2 + bit_x, N_x//2 + bit_x:] #upper right
        elif i == 2: x = x_trunc[N_x//2 + bit_x:, N_x//2 + bit_x:] #lower right
        elif i == 3: x = x_trunc[N_x//2 + bit_x:, :N_x//2 + bit_x] #lower left

        if shift_bit:
            if i == 0: x = np.pad(x, ((0,1),(0,1)), 'constant')
            elif i == 1: x = np.pad(x, ((0,1),(0,2)), 'constant')
            elif i == 2: x = np.pad(x, ((0,2),(0,2)), 'constant')
            elif i == 3: x = np.pad(x, ((0,2),(0,1)), 'constant')
        else:
            if i == 1: x = np.pad(x, ((0,0),(0,1)), 'constant')
            elif i == 2: x = np.pad(x, ((0,1),(0,1)), 'constant')
            elif i == 3: x = np.pad(x, ((0,1),(0,0)), 'constant')
        x_cent = bluestein_pad(x, sec_N_x, N_out)
        zoom_ft_x = zoom_fft_2d_mod(x_cent, sec_N_x, N_out, N_X=N_X)
        if i == 0:
            ph_1 = phase_shift - shift_bit
            ph_2 = ph_1
        elif i == 1:
            ph_1 = phase_shift - shift_bit
            ph_2 = -(phase_shift + 1)
        elif i == 2:
            ph_1 = -(phase_shift + 1)
            ph_2 = -(phase_shift + 1)
        elif i == 3:
            ph_1 = -(phase_shift + 1)
            ph_2 = phase_shift - shift_bit

        out_fac_1 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * ph_1 * (1 / (N_X)) ) )
        out_fac_2 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * ph_2 * (1 / (N_X)) ) )
        zoom_fft_out += zoom_ft_x * np.outer(out_fac_1, out_fac_2)
    return zoom_fft_out

def four_chunked_zoom_fft(x_file, N_x, N_out, N_X):
    """
    Cumulatively computes a 2D zoom FFT with four smaller Bluestein FFTs.
    Peak memory usage ~ N_x/4 + N_out.

    Parameters
    ----------
    x_file : np.memmap
        Input to FFT (a numpy memmap object of type np.complex128).
    N_x : int
        Size in one dimension of x_file.
    N_out : int
        Number of output points of FFT needed.
    N_X : int
        Phantom zero-padded length of input x_file for desired output sampling 
        (see the fresnel class to calculate this).

    Returns
    -------
    np.ndarray
        The 2D FFT over the chosen output region (complex).
    """
    phase_shift = float((N_X - 1) //2 + 1)
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    uncorrected_output_field = four_chunked_zoom_fft_mod(x_file, N_x, N_out, N_X)
    return uncorrected_output_field*np.outer(out_fac, out_fac)


def zoom_fft_quad_out_mod(x, N_x, N_out, N_X, chunk=0):
    """
    Computes a quadrant of the output spectrum (upper left, upper right, lower left, or lower right)
    With N_out samples, and as if input was zero-padded to N_X, such that output sample size is
    d_f = 1/(N_X * dx).
    This version with the extension 'mod' computes this as if the input is ifftshift(x). 

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    N_x : int
        Size in one dimension of x_file.
    N_out : int
        Number of output points of FFT needed.
    N_X : int
        Phantom zero-padded length of input x_file for desired output sampling 
        (see the fresnel class to calculate this).
    chunk : int
        The chunk index is between {0 and 3} (UL, UR, LL, LR).

    Returns
    -------
    zoom_fft_out : np.ndarray
        The 2D FFT over the chosen output region (complex).
    """
    if chunk not in [0, 1, 2, 3]: raise ValueError("Invalid value for chunk, must be 0, 1, 2, or 3.")

    N_chirp = N_x + N_out - 1
    bit_chirp = N_chirp % 2
    bit_out = N_out % 2

    trunc_x = bluestein_pad(x, N_x, N_out)
    
    b = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange( -(N_chirp//2), (N_chirp//2) + bit_chirp)**2)
    
    if chunk == 0:
        h1 = h2 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( -(N_x//2) - N_out, (N_x//2) )**2)
        c1 = c2 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( -N_out, 0 )**2))
    elif chunk == 1:
        c1 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( -N_out, 0 )**2))
        c2 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( N_out )**2))
        h1 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( -(N_x//2) - N_out, (N_x//2) )**2)
        h2 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( - (N_x//2) , N_out + (N_x//2) )**2)
    elif chunk == 2:
        c1 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( N_out )**2))
        c2 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( -N_out, 0 )**2))
        h1 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( - (N_x//2) , N_out + (N_x//2) )**2)
        h2 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( -(N_x//2) - N_out, (N_x//2) )**2)
    elif chunk == 3:
        h1 = h2 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( - (N_x//2) , N_out + (N_x//2) )**2)
        c1 = c2 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( N_out )**2))

    h1 = np.roll(h1, (N_chirp//2) + 1)
    h2 = np.roll(h2, (N_chirp//2) + 1)
    ft_h1 = scipy.fft.fft(h1)
    ft_h2 = scipy.fft.fft(h2)

    zoom_fft =  (scipy.fft.ifft2( scipy.fft.fft2(np.outer(b, b) * trunc_x) * np.outer(ft_h1, ft_h2) ) )
    zoom_fft = zoom_fft[(N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + bit_out, 
                        (N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + bit_out]
    zoom_fft *= np.outer(c1, c2)
    return zoom_fft

def zoom_fft_quad_out(x, N_x, N_out, N_X, chunk=0):
    """
    Computes a quadrant of the output spectrum (upper left, upper right, lower left, or lower right)
    With N_out samples, and as if input was zero-padded to N_X, such that output sample size is
    d_f = 1/(N_X * dx).

    Note: Use this with the four chunked FFT, if you can't fit your full input on harddisk 
    (or if you want some quadrant region).

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    N_x : int
        Size in one dimension of x_file.
    N_out : int
        Number of output points of FFT needed.
    N_X : int
        Phantom zero-padded length of input x_file for desired output sampling 
        (see the fresnel class to calculate this).
    chunk : int
        The chunk index is between {0 and 3} (UL, UR, LL, LR).

    Returns
    -------
    zoom_fft_out : np.ndarray
        The 2D FFT over the chosen output region (complex).
    """
    if chunk not in [0, 1, 2, 3]: raise ValueError("Invalid value for chunk, must be 0, 1, 2, or 3.")
    phase_shift = float((N_X - 1) //2 + 1)
    if chunk == 0:
        out_fac1 = out_fac2 = np.exp ( np.arange(-N_out, 0) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    elif chunk == 1:
        out_fac1 = np.exp ( np.arange(-N_out, 0) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
        out_fac2 = np.exp ( np.arange(N_out) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    elif chunk == 2:
        out_fac1 = np.exp ( np.arange(N_out) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
        out_fac2 = np.exp ( np.arange(-N_out, 0) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    elif chunk == 3:
        out_fac1 = out_fac2 = np.exp ( np.arange(N_out) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    uncorrected_output_field = zoom_fft_quad_out_mod(x, N_x, N_out, N_X, chunk=chunk)
    return uncorrected_output_field*np.outer(out_fac1, out_fac2)

def single_chunked_zoom_fft_mod(x, N_x, N_out, N_X, i=0):
    # For computing the FFT over a single input chunk i = 0...3, and deciding which one
    bit_x = N_x%2
    sec_N_x = N_x//2 + 1 # this is the (max) size of non-zero portion of segment, same for all 4
    shift_bit = 1 - sec_N_x%2
    sec_N_x += shift_bit # want this to be an odd number as it makes it easier to calculate shifts and center
    phase_shift =  sec_N_x//2
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    if shift_bit:
        if i == 0: x = np.pad(x, ((0,1),(0,1)), 'constant')
        elif i == 1: x = np.pad(x, ((0,1),(0,2)), 'constant')
        elif i == 2: x = np.pad(x, ((0,2),(0,2)), 'constant')
        elif i == 3: x = np.pad(x, ((0,2),(0,1)), 'constant')
    else:
        if i == 1: x = np.pad(x, ((0,0),(0,1)), 'constant')
        elif i == 2: x = np.pad(x, ((0,1),(0,1)), 'constant')
        elif i == 3: x = np.pad(x, ((0,1),(0,0)), 'constant')
    x_cent = bluestein_pad(x, sec_N_x, N_out)
    zoom_ft_x = zoom_fft_2d_mod(x_cent, sec_N_x, N_out, N_X = N_X)
    if i == 0:
        ph_1 = phase_shift - shift_bit
        ph_2 = ph_1
    elif i == 1:
        ph_1 = phase_shift - shift_bit
        ph_2 = -(phase_shift + 1)
    elif i == 2:
        ph_1 = -(phase_shift + 1)
        ph_2 = -(phase_shift + 1)
    elif i == 3:
        ph_1 = -(phase_shift + 1)
        ph_2 = phase_shift - shift_bit

    out_fac_1 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * ph_1 * (1 / (N_X)) ) )
    out_fac_2 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * ph_2 * (1 / (N_X)) ) )
    zoom_fft_out = zoom_ft_x * np.outer(out_fac_1, out_fac_2)
    return zoom_fft_out

def chunk_out_zoom_fft_2d_mod(x, N_x, N_out_x, N_out_y, start_chunk_x, start_chunk_y, Z_pad=None, N_X=None):
    """
    Compute a non-centered output chunk of 2D FFT using the Bluestein algorithm. 
    fftshift(fft2(ifftshift(x)))[N_X//2 + start_chunk_x: N_X//2 + start_chunk_x + N_out_x,...]

    Parameters
    ----------
    x : np.ndarray
        Centered input signal (complex numpy array).
    N_x : int
        Length of the input signal.
    N_out_x : int
        Length of the output signal chunk in the x-dimension.
    N_out_y : int
        Length of the output signal chunk in the y-dimension.
    start_chunk_x : int
        First index (frequency sample) of the output chunk in the x-dimension.
    start_chunk_y : int
        First index (frequency sample) of the output chunk in the y-dimension.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : int, optional
        Zero-padded length of input signal (Z_pad * N_x + 1).

    Returns
    -------
    zoom_fft : np.ndarray
        Chunk of the zoomed FFT of the input signal (complex).
    """
    if (Z_pad is None and N_X is None) or (Z_pad is not None and N_X is not None):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")

    if Z_pad is not None: N_X = Z_pad*N_x + 1 #X before truncation

    N_chirp_x = N_x + N_out_x - 1
    N_chirp_y = N_x + N_out_y - 1

    bit_x = N_x % 2
    bit_chirp_x = N_chirp_x % 2
    bit_chirp_y = N_chirp_y % 2
    bit_out_x = N_out_x % 2
    bit_out_y = N_out_y % 2
    print (bit_x, bit_chirp_x, bit_chirp_y, bit_out_x, bit_out_y)

    trunc_x = bluestein_pad(x, N_x, N_out_x, N_out_y)
    h1 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( -(N_x//2) + start_chunk_x, (N_x//2) + start_chunk_x + N_out_x )**2)
    h2 = np.exp(   np.pi*(1/(N_X))*1j*np.arange( -(N_x//2) + start_chunk_y, (N_x//2) + start_chunk_y + N_out_y )**2)
    c1 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( start_chunk_x, start_chunk_x + N_out_x )**2))
    c2 = np.exp(-1*np.pi*(1/N_X)*1j* (np.arange( start_chunk_y, start_chunk_y + N_out_y )**2))
    b1 = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange(- (N_chirp_x//2), (N_chirp_x//2) + bit_chirp_x)**2)
    b2 = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange(- (N_chirp_y//2), (N_chirp_y//2) + bit_chirp_y)**2)

    h1 = np.roll(h1, (N_chirp_x//2) + bit_chirp_x)
    h2 = np.roll(h2, (N_chirp_y//2) + bit_chirp_y)
    ft_h1 = scipy.fft.fft(h1)
    ft_h2 = scipy.fft.fft(h2)

    zoom_fft =  (scipy.fft.ifft2( scipy.fft.fft2(np.outer(b1, b2) * trunc_x) * np.outer(ft_h1, ft_h2) ) )
    zoom_fft = zoom_fft[(N_chirp_x//2) - (N_out_x//2) : (N_chirp_x//2) + (N_out_x//2) + bit_out_x, 
                        (N_chirp_y//2) - (N_out_y//2) : (N_chirp_y//2) + (N_out_y//2) + bit_out_y]
    zoom_fft *= np.outer(c1, c2)
    return zoom_fft

def chunk_out_zoom_fft_2d(x, N_x, N_out_x, N_out_y, start_chunk_x, start_chunk_y, Z_pad=None, N_X=None):
    """
    Compute a non-centered output chunk of 2D FFT using the Bluestein algorithm. 
    fftshift(fft2(x))[N_X//2 + start_chunk_x: N_X//2 + start_chunk_x + N_out_x,...]

    Parameters
    ----------
    x : np.ndarray
        Centered input signal (complex numpy array).
    N_x : int
        Length of the input signal.
    N_out_x : int
        Length of the output signal chunk in the x-dimension.
    N_out_y : int
        Length of the output signal chunk in the y-dimension.
    start_chunk_x : int
        First index (frequency sample) of the output chunk in the x-dimension.
    start_chunk_y : int
        First index (frequency sample) of the output chunk in the y-dimension.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : int, optional
        Zero-padded length of input signal (Z_pad * N_x + 1).

    Returns
    -------
    np.ndarray
        Chunk of the zoomed FFT of the input signal (complex).
    """
    if (Z_pad is None and N_X is None) or (Z_pad is not None and N_X is not None):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")

    phase_shift = float((N_X - 1) //2 + 1)
    out_fac1 = np.exp ( np.arange(start_chunk_x, start_chunk_x + N_out_x ) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    out_fac2 = np.exp ( np.arange(start_chunk_y, start_chunk_y + N_out_y ) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    if Z_pad is not None: uncorrected_output_field = chunk_out_zoom_fft_2d_mod(x, N_x, N_out_x, N_out_y, start_chunk_x, start_chunk_y, Z_pad=Z_pad)
    else: uncorrected_output_field = chunk_out_zoom_fft_2d_mod(x, N_x, N_out_x, N_out_y, start_chunk_x, start_chunk_y, N_X = N_X)
    return uncorrected_output_field*np.outer(out_fac1, out_fac2)
    
def chunk_in_zoom_fft_2d_mod(x_file, N_x, N_out, N_X, N_chunk=4):
    """
    Compute a 2D FFT using the Bluestein algorithm. 
    Experimental chunked version - computed in chunks of the input.

    Define x_file as: 
    arr = np.memmap('x.dat', dtype=np.complex128,mode='w+',shape=(N_x, N_x))
    arr[:] = trunc_x
    arr.flush()

    Parameters
    ----------
    x_file : str
        Path to the input signal as a memmap object.
    N_x : int
        Size in one dimension of x_file.
    N_out : int
        Number of output points of FFT needed.
    N_X : int
        Phantom zero-padded length of input x_file for desired output sampling
        (see the fresnel class to calculate this).
    N_chunk : int, optional
        Number of chunks along one axis (default is 4).

    Returns
    -------
    zoom_fft_out : np.ndarray
        The 2D FFT over the output region (np.complex128).
    """
    x_vals = np.linspace(-(N_x//2), (N_x//2), N_x)
    chunk = (N_x//N_chunk) + 1 - ((N_x//N_chunk)%2)
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    x_trunc = np.memmap(x_file, dtype=np.complex128, mode='r', shape=(N_x, N_x))
    for i in range(N_chunk):
        for j in range(N_chunk):
            sec_N_x = sec_N_y = chunk
            if i == N_chunk-1:
                sec_N_x = N_x - i*sec_N_x
                sec_N_x += 1 - (sec_N_x%2)
            if j == N_chunk-1:
                sec_N_y = N_x - j*sec_N_y
                sec_N_y += 1 - (sec_N_y%2)
            x = x_trunc[i*chunk : min(i*chunk + sec_N_x, N_x), j*chunk : min(j*chunk + sec_N_y, N_x)]
            if sec_N_x != sec_N_y:
                sec_N_x = sec_N_y = max(sec_N_x, sec_N_y)
            x = np.pad(x, [(0, sec_N_x-np.shape(x)[0]), (0, sec_N_y-np.shape(x)[1])], mode='constant')
            ph1 = - x_vals[i*chunk + (sec_N_x//2)]
            ph2 = - x_vals[j*chunk + (sec_N_y//2)]
            out_fac_1 = np.exp ( np.arange(-(N_out//2), (N_out//2) + N_out%2) * (1j * 2 * np.pi * ph1 * (1 / (N_X)) ) )
            out_fac_2 = np.exp ( np.arange(-(N_out//2), (N_out//2) + N_out%2) * (1j * 2 * np.pi * ph2 * (1 / (N_X)) ) )
            ft_x = zoom_fft_2d_mod(x, sec_N_x, N_out, N_X = N_X)
            zoom_fft_out +=  ft_x * np.outer(out_fac_1, out_fac_2)
    return zoom_fft_out

def chunk_in_chirp_zoom_fft_2d_mod(x_file, wl_z, d_x, N_x, N_out, N_X, N_chunk=4):
    """
    Compute a chirp-modulated 2D zoom FFT in chunks of the input.

    This version is useful for Fresnel diffraction, when you need to multiply
    your input x_file by a chirp which depends on lambda*z, before computing the
    zoom FFT.

    v2 optimisations: chirp cache built once and reused for all same-size
    chunks, pre-allocated buffer, multithreaded scipy FFTs, in-place multiplies,
    explicit memory cleanup per chunk.

    Parameters
    ----------
    x_file : str
        Path to the input signal as a memmap file (float32).
    wl_z : float
        Wavelength times distance (lambda * z).
    d_x : float
        Sampling interval of the input field.
    N_x : int
        Size in one dimension of x_file.
    N_out : int
        Number of output points of FFT needed.
    N_X : float
        Phantom zero-padded length of input x_file for desired output sampling
        (see the FresnelSingle class to calculate this).
    N_chunk : int, optional
        Number of chunks along one axis (default is 4).

    Returns
    -------
    zoom_fft_out : np.ndarray
        The 2D FFT over the chosen output region (np.complex128).
    """
    x_vals = np.linspace(-(N_x // 2), (N_x // 2), N_x)
    chunk_sz = (N_x // N_chunk) + 1 - ((N_x // N_chunk) % 2)

    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    x_trunc = np.memmap(x_file, dtype=np.float32, mode='r', shape=(N_x, N_x))

    cache = _build_chirp_cache(chunk_sz, N_out, N_X)
    last_cache = None

    for i in range(N_chunk):
        for j in range(N_chunk):
            sec_N_x = sec_N_y = chunk_sz
            if i == N_chunk - 1:
                sec_N_x = N_x - i * chunk_sz
                sec_N_x += 1 - (sec_N_x % 2)
            if j == N_chunk - 1:
                sec_N_y = N_x - j * chunk_sz
                sec_N_y += 1 - (sec_N_y % 2)

            logger.debug("chunk (%d, %d)", i, j)

            x = x_trunc[i * chunk_sz: min(i * chunk_sz + sec_N_x, N_x),
                         j * chunk_sz: min(j * chunk_sz + sec_N_y, N_x)]

            index_x = np.arange(i * chunk_sz, min(i * chunk_sz + sec_N_x, N_x))
            index_y = np.arange(j * chunk_sz, min(j * chunk_sz + sec_N_y, N_x))
            xx = x_vals[index_x][:, np.newaxis] * d_x
            yy = x_vals[index_y][np.newaxis, :] * d_x

            if sec_N_x != sec_N_y:
                sec_N_x = sec_N_y = max(sec_N_x, sec_N_y)

            chirped = x.astype(np.complex128) * np.exp(
                1j * (np.pi / wl_z) * (xx ** 2 + yy ** 2))
            x_padded = np.pad(chirped,
                              [(0, sec_N_x - chirped.shape[0]),
                               (0, sec_N_y - chirped.shape[1])],
                              mode='constant')
            del chirped

            use_cache = cache if sec_N_x == chunk_sz else None
            if use_cache is None:
                if last_cache is not None and last_cache['N_x'] == sec_N_x:
                    use_cache = last_cache
                else:
                    last_cache = _build_chirp_cache(sec_N_x, N_out, N_X)
                    use_cache = last_cache

            ft_x = zoom_fft_2d_mod(x_padded, sec_N_x, N_out, N_X=N_X,
                                   _cache=use_cache)
            del x_padded

            ph1 = -x_vals[i * chunk_sz + (sec_N_x // 2)]
            ph2 = -x_vals[j * chunk_sz + (sec_N_y // 2)]
            out_fac_1 = np.exp(np.arange(-(N_out // 2), (N_out // 2) + N_out % 2)
                               * (1j * 2 * np.pi * ph1 / N_X))
            out_fac_2 = np.exp(np.arange(-(N_out // 2), (N_out // 2) + N_out % 2)
                               * (1j * 2 * np.pi * ph2 / N_X))

            zoom_fft_out += ft_x * np.outer(out_fac_1, out_fac_2)
            del ft_x

    return zoom_fft_out
