import numpy as np

def zoom_fft_2d_mod(x, N_x, N_out, Z_pad):
    """
    MODIFIED VERSION: computes the Bluestein FFT on fftshift(x) (without explicitly shifting x).
    Compute a zoomed 2D FFT using the Bluestein algorithm. 
    The input x is centered. 
    Args:
    Z_pad: Zero-padding factor.
    N_x: Length of the input signal.
    N_out: Length of the output signal.
    x: Centered Input signal (complex numpy array).
    Returns:
    Zoomed FFT of the input signal (complex numpy array).
    """
    N_chirp = N_x + N_out - 1
    N_X = Z_pad*N_x + 1 #X before truncation

    bit_x = N_x % 2
    bit_chirp = N_chirp % 2
    bit_out = N_out % 2

    trunc_x = x[(x.shape[0]//2) - (N_chirp//2) : (x.shape[0]//2) + (N_chirp//2) + bit_x, 
                (x.shape[1]//2) - (N_chirp//2) : (x.shape[1]//2) + (N_chirp//2) + bit_x]

    b = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange(- (N_chirp//2), (N_chirp//2) + bit_chirp)**2)
    h = np.exp(   np.pi*(1/(N_X))*1j*np.arange(- (N_out//2) - (N_x//2) , (N_out//2) + (N_x//2) + bit_chirp)**2)
    h = np.roll(h, (N_chirp//2) + 1)
    ft_h = np.fft.fft(h)

    zoom_fft = np.outer(b, b) * (np.fft.ifft2( np.fft.fft2(np.outer(b, b) * trunc_x) * np.outer(ft_h, ft_h) ) )
    zoom_fft = zoom_fft[(N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + bit_out, 
                        (N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + bit_out]
    return zoom_fft

def zoom_fft_2d(x, N_x, N_out, Z_pad):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm.
    The input x is centered. 
    Args:
    Z_pad: Zero-padding factor.
    N_x: Length of the input signal.
    N_out: Length of the output signal.
    x: Centered Input signal (complex numpy array).
    Returns:
    Zoomed FFT of the input signal (complex numpy array).
    """
    N_X = Z_pad*N_x + 1
    phase_shift = (N_x*Z_pad)//2 + 1 
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)) ) )
    uncorrected_output_field = zoom_fft_2d_mod(x, N_x, N_out, Z_pad)
    return uncorrected_output_field*np.outer(out_fac, out_fac)

