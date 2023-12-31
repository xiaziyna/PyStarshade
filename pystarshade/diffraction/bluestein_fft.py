import numpy as np
import os
from .util import bluestein_pad

def zoom_fft_2d_mod(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    MODIFIED VERSION: computes the Bluestein FFT on fftshift(x) (without explicitly shifting x).
    Compute a zoomed 2D FFT using the Bluestein algorithm. 
    The input x is centered. 
    
    Args
    x: Centered Input signal (complex numpy array).
    N_x: Length of the input signal.
    N_out: Length of the output signal.
    Z_pad: Zero-padding factor.
    N_X: Zero-padded length of input signal (Z_pad * N_x + 1).
    
    Returns
    Zoomed FFT of the input signal (complex numpy array).
    """
    if (Z_pad is None and N_X is None) or (Z_pad is not None and N_X is not None):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")

    if Z_pad is not None: N_X = Z_pad*N_x + 1 #X before truncation

    N_chirp = N_x + N_out - 1

    bit_x = N_x % 2
    bit_chirp = N_chirp % 2
    bit_out = N_out % 2

    trunc_x = bluestein_pad(x, N_x, N_out)
    
    b = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange(- (N_chirp//2), (N_chirp//2) + bit_chirp)**2)
    h = np.exp(   np.pi*(1/(N_X))*1j*np.arange(- (N_out//2) - (N_x//2) , (N_out//2) + (N_x//2) + bit_chirp)**2)
    h = np.roll(h, (N_chirp//2) + 1)
    ft_h = np.fft.fft(h)

    zoom_fft = np.outer(b, b) * (np.fft.ifft2( np.fft.fft2(np.outer(b, b) * trunc_x) * np.outer(ft_h, ft_h) ) )
    zoom_fft = zoom_fft[(N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + bit_out, 
                        (N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + bit_out]
    return zoom_fft

def zoom_fft_2d(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm.
    The input x is centered. 
    
    Args
    x: Centered Input signal (complex numpy array).
    N_x: Length of the input signal.
    N_out: Length of the output signal.
    Z_pad: Zero-padding factor.
    N_X: Zero-padded length of input signal (Z_pad * N_x + 1).

    Returns
    Zoomed FFT of the input signal (complex numpy array).
    """
    if (Z_pad is None and N_X is None) or (Z_pad is not None and N_X is not None):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")
    if Z_pad is not None:
        N_X = Z_pad*N_x + 1 #X before truncation
        phase_shift = (N_x*Z_pad)//2 + 1 
        uncorrected_output_field = zoom_fft_2d_mod(x, N_x, N_out, Z_pad=Z_pad)
    else:
        phase_shift = float((N_X - 1) //2 + 1)
        uncorrected_output_field = zoom_fft_2d_mod(x, N_x, N_out, N_X=N_X)
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    return uncorrected_output_field*np.outer(out_fac, out_fac)
