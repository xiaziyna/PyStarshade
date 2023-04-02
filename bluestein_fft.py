import numpy as np


def zoom_fft_1d(Z_pad, N_x, N_out, x):
    """
    Compute a zoomed FFT using the Bluestein algorithm.
    The input x is centered. The complexity of this method is O(N_chirp log N_chirp) for any zero-padding.

    Args:
    Z_pad: Zero-padding factor.
    N_x: Length of the input signal.
    N_out: Length of the output signal.
    x: Centered Input signal (complex numpy array).

    Returns:
    Zoomed FFT of the input signal (complex numpy array).
    """
    N_chirp = N_x + N_out - 1
    N_X = Z_pad*N_x+1 #X before truncation
    x_half = N_X // 2

    bit = N_out % 2
    bit_x = N_x % 2
    bit_chirp = N_chirp%2

    trunc_x = x[(len(x)//2) - (N_chirp//2) : (len(x)//2) + (N_chirp//2) + bit_chirp]
    phase_shift = (N_x*Z_pad)//2 + 1 
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1)*(1j * 2 * np.pi * phase_shift * (1 / (N_X))))

    b = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange(- (N_chirp//2), (N_chirp//2) + bit_chirp)**2)
    h = np.exp(   np.pi*(1/(N_X))*1j*np.arange(- (N_out//2) - (N_x//2) , (N_out//2) + (N_x//2) + bit)**2)
    h = np.roll(h, (N_chirp//2) + 1)

    zoom_fft = b * (np.fft.ifft( np.fft.fft(b * trunc_x) * np.fft.fft(h) ) )
    zoom_fft = zoom_fft[(N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + 1]
    return zoom_fft*out_fac

def zoom_fft_2d(Z_pad, N_x, N_out, x):
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
    N_chirp = N_x + N_out - 1
    N_X = Z_pad*N_x+1 #X before truncation
    x_half = N_X // 2

    bit = N_out % 2
    bit_x = N_x % 2
    bit_chirp = N_chirp%2

    trunc_x = x[(x.shape[0]//2) - (N_chirp//2) : (x.shape[0]//2) + (N_chirp//2) + bit_chirp, (x.shape[1]//2) - (N_chirp//2) : (x.shape[1]//2) + (N_chirp//2) + bit_chirp]
    phase_shift = (N_x*Z_pad)//2 + 1 
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1)*(1j * 2 * np.pi * phase_shift * (1 / (N_X))))

    b = np.exp(-1*np.pi*(1/(N_X))*1j*np.arange(- (N_chirp//2), (N_chirp//2) + bit_chirp)**2)
    h = np.exp(   np.pi*(1/(N_X))*1j*np.arange(- (N_out//2) - (N_x//2) , (N_out//2) + (N_x//2) + bit)**2)
    h = np.roll(h, (N_chirp//2) + 1)
    ft_h = np.fft.fft(h)

    zoom_fft = np.outer(b, b) * (np.fft.ifft2( np.fft.fft2(np.outer(b, b) * trunc_x) * np.outer(ft_h, ft_h) ) )
    zoom_fft = zoom_fft[(N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + 1, (N_chirp//2) - (N_out//2) : (N_chirp//2) + (N_out//2) + 1]
    return zoom_fft*np.outer(out_fac, out_fac)

# Example usage
Z_pad = 6
N_x = 7
N_out = 5
N_chirp = N_x + N_out - 1
N_X = Z_pad*N_x+1 #X before truncation
x = np.zeros(N_X, dtype = 'complex128')
x_half = Z_pad*N_x // 2
edge = x_half - (N_x//2) 
x[edge : edge + N_x ] = np.ones(N_x)
x[edge + (N_x//3)] = 2*1j
x[edge + (N_x//2)+1] = 0

#zoom and real produce the same output
zoom = zoom_fft_1d(Z_pad, N_x, N_out, x)
real = np.fft.fftshift(np.fft.fft(x))[(len(x)//2) - (N_out//2): (len(x)//2) + (N_out//2) + 1]

