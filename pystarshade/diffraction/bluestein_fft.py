
import numpy as np

from pystarshade.diffraction.util import bluestein_pad


def zoom_fft_2d_mod(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm.

    MODIFIED VERSION: computes the Bluestein FFT equivalent to
    fftshift(fft2(ifftshift(x_pad))) [N_X/2 - N_out/2: N_X/2 + N_out/2, N_X/2 - N_out/2: N_X/2 + N_out/2]
    where x_pad is x zero-padded to length N_X.
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
    zoom_fft: np.ndarray
        Zoomed FFT of the input signal (complex numpy array).
    """
    if (Z_pad is None and N_X is None) or (
        Z_pad is not None and N_X is not None
    ):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")

    if Z_pad is not None:
        N_X = Z_pad * N_x + 1  # X before truncation

    N_chirp = N_x + N_out - 1

    bit_x = N_x % 2
    bit_chirp = N_chirp % 2
    bit_out = N_out % 2

    trunc_x = bluestein_pad(x, N_x, N_out)

    b = np.exp(
        -1
        * np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(-(N_chirp // 2), (N_chirp // 2) + bit_chirp) ** 2
    )
    h = np.exp(
        np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(
            -(N_out // 2) - (N_x // 2), (N_out // 2) + (N_x // 2) + bit_chirp
        )
        ** 2
    )
    h = np.roll(h, (N_chirp // 2) + 1)
    ft_h = np.fft.fft(h)

    zoom_fft = np.outer(b, b) * (
        np.fft.ifft2(
            np.fft.fft2(np.outer(b, b) * trunc_x) * np.outer(ft_h, ft_h)
        )
    )
    zoom_fft = zoom_fft[
        (N_chirp // 2)
        - (N_out // 2) : (N_chirp // 2)
        + (N_out // 2)
        + bit_out,
        (N_chirp // 2)
        - (N_out // 2) : (N_chirp // 2)
        + (N_out // 2)
        + bit_out,
    ]
    return zoom_fft


def zoom_fft_2d(x, N_x, N_out, Z_pad=None, N_X=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm.
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
    out_fac: np.ndarray
        Zoomed FFT of the input signal (complex numpy array).
    """
    if (Z_pad is None and N_X is None) or (
        Z_pad is not None and N_X is not None
    ):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")
    if Z_pad is not None:
        N_X = Z_pad * N_x + 1  # X before truncation
        phase_shift = (N_x * Z_pad) // 2 + 1
        uncorrected_output_field = zoom_fft_2d_mod(x, N_x, N_out, Z_pad=Z_pad)
    else:
        phase_shift = float((N_X - 1) // 2 + 1)
        uncorrected_output_field = zoom_fft_2d_mod(x, N_x, N_out, N_X=N_X)
    out_fac = np.exp(
        np.arange(-(N_out // 2), (N_out // 2) + 1)
        * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
    )
    return uncorrected_output_field * np.outer(out_fac, out_fac)


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
    bit_x = N_x % 2
    sec_N_x = (
        N_x // 2 + 1
    )  # this is the (max) size of non-zero portion of segment, same for all 4
    shift_bit = 1 - sec_N_x % 2
    sec_N_x += shift_bit  # want this to be an odd number as it makes it easier to calculate shifts and center
    phase_shift = sec_N_x // 2
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    x_trunc = np.memmap(
        x_file, dtype=np.complex128, mode="r", shape=(N_x, N_x)
    )
    for i in range(4):
        if i == 0:
            x = x_trunc[: N_x // 2 + bit_x, : N_x // 2 + bit_x]  # upper left
        elif i == 1:
            x = x_trunc[: N_x // 2 + bit_x, N_x // 2 + bit_x :]  # upper right
        elif i == 2:
            x = x_trunc[N_x // 2 + bit_x :, N_x // 2 + bit_x :]  # lower right
        elif i == 3:
            x = x_trunc[N_x // 2 + bit_x :, : N_x // 2 + bit_x]  # lower left

        if shift_bit:
            if i == 0:
                x = np.pad(x, ((0, 1), (0, 1)), "constant")
            elif i == 1:
                x = np.pad(x, ((0, 1), (0, 2)), "constant")
            elif i == 2:
                x = np.pad(x, ((0, 2), (0, 2)), "constant")
            elif i == 3:
                x = np.pad(x, ((0, 2), (0, 1)), "constant")
        else:
            if i == 1:
                x = np.pad(x, ((0, 0), (0, 1)), "constant")
            elif i == 2:
                x = np.pad(x, ((0, 1), (0, 1)), "constant")
            elif i == 3:
                x = np.pad(x, ((0, 1), (0, 0)), "constant")
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

        out_fac_1 = np.exp(
            np.arange(-(N_out // 2), (N_out // 2) + 1)
            * (1j * 2 * np.pi * ph_1 * (1 / (N_X)))
        )
        out_fac_2 = np.exp(
            np.arange(-(N_out // 2), (N_out // 2) + 1)
            * (1j * 2 * np.pi * ph_2 * (1 / (N_X)))
        )
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
    phase_shift = float((N_X - 1) // 2 + 1)
    out_fac = np.exp(
        np.arange(-(N_out // 2), (N_out // 2) + 1)
        * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
    )
    uncorrected_output_field = four_chunked_zoom_fft_mod(
        x_file, N_x, N_out, N_X
    )
    return uncorrected_output_field * np.outer(out_fac, out_fac)


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
    if chunk not in [0, 1, 2, 3]:
        raise ValueError("Invalid value for chunk, must be 0, 1, 2, or 3.")

    N_chirp = N_x + N_out - 1
    bit_chirp = N_chirp % 2
    bit_out = N_out % 2

    trunc_x = bluestein_pad(x, N_x, N_out)

    b = np.exp(
        -1
        * np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(-(N_chirp // 2), (N_chirp // 2) + bit_chirp) ** 2
    )

    if chunk == 0:
        h1 = h2 = np.exp(
            np.pi
            * (1 / (N_X))
            * 1j
            * np.arange(-(N_x // 2) - N_out, (N_x // 2)) ** 2
        )
        c1 = c2 = np.exp(
            -1 * np.pi * (1 / N_X) * 1j * (np.arange(-N_out, 0) ** 2)
        )
    elif chunk == 1:
        c1 = np.exp(-1 * np.pi * (1 / N_X) * 1j * (np.arange(-N_out, 0) ** 2))
        c2 = np.exp(-1 * np.pi * (1 / N_X) * 1j * (np.arange(N_out) ** 2))
        h1 = np.exp(
            np.pi
            * (1 / (N_X))
            * 1j
            * np.arange(-(N_x // 2) - N_out, (N_x // 2)) ** 2
        )
        h2 = np.exp(
            np.pi
            * (1 / (N_X))
            * 1j
            * np.arange(-(N_x // 2), N_out + (N_x // 2)) ** 2
        )
    elif chunk == 2:
        c1 = np.exp(-1 * np.pi * (1 / N_X) * 1j * (np.arange(N_out) ** 2))
        c2 = np.exp(-1 * np.pi * (1 / N_X) * 1j * (np.arange(-N_out, 0) ** 2))
        h1 = np.exp(
            np.pi
            * (1 / (N_X))
            * 1j
            * np.arange(-(N_x // 2), N_out + (N_x // 2)) ** 2
        )
        h2 = np.exp(
            np.pi
            * (1 / (N_X))
            * 1j
            * np.arange(-(N_x // 2) - N_out, (N_x // 2)) ** 2
        )
    elif chunk == 3:
        h1 = h2 = np.exp(
            np.pi
            * (1 / (N_X))
            * 1j
            * np.arange(-(N_x // 2), N_out + (N_x // 2)) ** 2
        )
        c1 = c2 = np.exp(-1 * np.pi * (1 / N_X) * 1j * (np.arange(N_out) ** 2))

    h1 = np.roll(h1, (N_chirp // 2) + 1)
    h2 = np.roll(h2, (N_chirp // 2) + 1)
    ft_h1 = np.fft.fft(h1)
    ft_h2 = np.fft.fft(h2)

    zoom_fft = np.fft.ifft2(
        np.fft.fft2(np.outer(b, b) * trunc_x) * np.outer(ft_h1, ft_h2)
    )
    zoom_fft = zoom_fft[
        (N_chirp // 2)
        - (N_out // 2) : (N_chirp // 2)
        + (N_out // 2)
        + bit_out,
        (N_chirp // 2)
        - (N_out // 2) : (N_chirp // 2)
        + (N_out // 2)
        + bit_out,
    ]
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
    if chunk not in [0, 1, 2, 3]:
        raise ValueError("Invalid value for chunk, must be 0, 1, 2, or 3.")
    phase_shift = float((N_X - 1) // 2 + 1)
    if chunk == 0:
        out_fac1 = out_fac2 = np.exp(
            np.arange(-N_out, 0) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
        )
    elif chunk == 1:
        out_fac1 = np.exp(
            np.arange(-N_out, 0) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
        )
        out_fac2 = np.exp(
            np.arange(N_out) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
        )
    elif chunk == 2:
        out_fac1 = np.exp(
            np.arange(N_out) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
        )
        out_fac2 = np.exp(
            np.arange(-N_out, 0) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
        )
    elif chunk == 3:
        out_fac1 = out_fac2 = np.exp(
            np.arange(N_out) * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
        )
    uncorrected_output_field = zoom_fft_quad_out_mod(
        x, N_x, N_out, N_X, chunk=chunk
    )
    return uncorrected_output_field * np.outer(out_fac1, out_fac2)


def single_chunked_zoom_fft_mod(x, N_x, N_out, N_X, i=0):
    # For computing the FFT over a single input chunk i = 0...3, and deciding which one
    bit_x = N_x % 2
    sec_N_x = (
        N_x // 2 + 1
    )  # this is the (max) size of non-zero portion of segment, same for all 4
    shift_bit = 1 - sec_N_x % 2
    sec_N_x += shift_bit  # want this to be an odd number as it makes it easier to calculate shifts and center
    phase_shift = sec_N_x // 2
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    if shift_bit:
        if i == 0:
            x = np.pad(x, ((0, 1), (0, 1)), "constant")
        elif i == 1:
            x = np.pad(x, ((0, 1), (0, 2)), "constant")
        elif i == 2:
            x = np.pad(x, ((0, 2), (0, 2)), "constant")
        elif i == 3:
            x = np.pad(x, ((0, 2), (0, 1)), "constant")
    else:
        if i == 1:
            x = np.pad(x, ((0, 0), (0, 1)), "constant")
        elif i == 2:
            x = np.pad(x, ((0, 1), (0, 1)), "constant")
        elif i == 3:
            x = np.pad(x, ((0, 1), (0, 0)), "constant")
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

    out_fac_1 = np.exp(
        np.arange(-(N_out // 2), (N_out // 2) + 1)
        * (1j * 2 * np.pi * ph_1 * (1 / (N_X)))
    )
    out_fac_2 = np.exp(
        np.arange(-(N_out // 2), (N_out // 2) + 1)
        * (1j * 2 * np.pi * ph_2 * (1 / (N_X)))
    )
    zoom_fft_out = zoom_ft_x * np.outer(out_fac_1, out_fac_2)
    return zoom_fft_out


def chunk_out_zoom_fft_2d_mod(
    x,
    N_x,
    N_out_x,
    N_out_y,
    start_chunk_x,
    start_chunk_y,
    Z_pad=None,
    N_X=None,
):
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
    if (Z_pad is None and N_X is None) or (
        Z_pad is not None and N_X is not None
    ):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")

    if Z_pad is not None:
        N_X = Z_pad * N_x + 1  # X before truncation

    N_chirp_x = N_x + N_out_x - 1
    N_chirp_y = N_x + N_out_y - 1

    bit_x = N_x % 2
    bit_chirp_x = N_chirp_x % 2
    bit_chirp_y = N_chirp_y % 2
    bit_out_x = N_out_x % 2
    bit_out_y = N_out_y % 2
    print(bit_x, bit_chirp_x, bit_chirp_y, bit_out_x, bit_out_y)

    trunc_x = bluestein_pad(x, N_x, N_out_x, N_out_y)
    h1 = np.exp(
        np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(
            -(N_x // 2) + start_chunk_x, (N_x // 2) + start_chunk_x + N_out_x
        )
        ** 2
    )
    h2 = np.exp(
        np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(
            -(N_x // 2) + start_chunk_y, (N_x // 2) + start_chunk_y + N_out_y
        )
        ** 2
    )
    c1 = np.exp(
        -1
        * np.pi
        * (1 / N_X)
        * 1j
        * (np.arange(start_chunk_x, start_chunk_x + N_out_x) ** 2)
    )
    c2 = np.exp(
        -1
        * np.pi
        * (1 / N_X)
        * 1j
        * (np.arange(start_chunk_y, start_chunk_y + N_out_y) ** 2)
    )
    b1 = np.exp(
        -1
        * np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(-(N_chirp_x // 2), (N_chirp_x // 2) + bit_chirp_x) ** 2
    )
    b2 = np.exp(
        -1
        * np.pi
        * (1 / (N_X))
        * 1j
        * np.arange(-(N_chirp_y // 2), (N_chirp_y // 2) + bit_chirp_y) ** 2
    )

    h1 = np.roll(h1, (N_chirp_x // 2) + bit_chirp_x)
    h2 = np.roll(h2, (N_chirp_y // 2) + bit_chirp_y)
    ft_h1 = np.fft.fft(h1)
    ft_h2 = np.fft.fft(h2)

    zoom_fft = np.fft.ifft2(
        np.fft.fft2(np.outer(b1, b2) * trunc_x) * np.outer(ft_h1, ft_h2)
    )
    zoom_fft = zoom_fft[
        (N_chirp_x // 2)
        - (N_out_x // 2) : (N_chirp_x // 2)
        + (N_out_x // 2)
        + bit_out_x,
        (N_chirp_y // 2)
        - (N_out_y // 2) : (N_chirp_y // 2)
        + (N_out_y // 2)
        + bit_out_y,
    ]
    zoom_fft *= np.outer(c1, c2)
    return zoom_fft


def chunk_out_zoom_fft_2d(
    x,
    N_x,
    N_out_x,
    N_out_y,
    start_chunk_x,
    start_chunk_y,
    Z_pad=None,
    N_X=None,
):
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
    if (Z_pad is None and N_X is None) or (
        Z_pad is not None and N_X is not None
    ):
        raise ValueError("You must provide exactly one of Z_pad or N_X.")

    phase_shift = float((N_X - 1) // 2 + 1)
    out_fac1 = np.exp(
        np.arange(start_chunk_x, start_chunk_x + N_out_x)
        * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
    )
    out_fac2 = np.exp(
        np.arange(start_chunk_y, start_chunk_y + N_out_y)
        * (1j * 2 * np.pi * phase_shift * (1 / (N_X)))
    )
    if Z_pad is not None:
        uncorrected_output_field = chunk_out_zoom_fft_2d_mod(
            x, N_x, N_out_x, N_out_y, start_chunk_x, start_chunk_y, Z_pad=Z_pad
        )
    else:
        uncorrected_output_field = chunk_out_zoom_fft_2d_mod(
            x, N_x, N_out_x, N_out_y, start_chunk_x, start_chunk_y, N_X=N_X
        )
    return uncorrected_output_field * np.outer(out_fac1, out_fac2)


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
    x_vals = np.linspace(-(N_x // 2), (N_x // 2), N_x)
    chunk = (N_x // N_chunk) + 1 - ((N_x // N_chunk) % 2)
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    x_trunc = np.memmap(
        x_file, dtype=np.complex128, mode="r", shape=(N_x, N_x)
    )
    for i in range(N_chunk):
        for j in range(N_chunk):
            sec_N_x = sec_N_y = chunk
            if i == N_chunk - 1:
                sec_N_x = N_x - i * sec_N_x
                sec_N_x += 1 - (sec_N_x % 2)
            if j == N_chunk - 1:
                sec_N_y = N_x - j * sec_N_y
                sec_N_y += 1 - (sec_N_y % 2)
            x = x_trunc[
                i * chunk : min(i * chunk + sec_N_x, N_x),
                j * chunk : min(j * chunk + sec_N_y, N_x),
            ]
            if sec_N_x != sec_N_y:
                sec_N_x = sec_N_y = max(sec_N_x, sec_N_y)
            x = np.pad(
                x,
                [(0, sec_N_x - np.shape(x)[0]), (0, sec_N_y - np.shape(x)[1])],
                mode="constant",
            )
            ph1 = -x_vals[i * chunk + (sec_N_x // 2)]
            ph2 = -x_vals[j * chunk + (sec_N_y // 2)]
            out_fac_1 = np.exp(
                np.arange(-(N_out // 2), (N_out // 2) + N_out % 2)
                * (1j * 2 * np.pi * ph1 * (1 / (N_X)))
            )
            out_fac_2 = np.exp(
                np.arange(-(N_out // 2), (N_out // 2) + N_out % 2)
                * (1j * 2 * np.pi * ph2 * (1 / (N_X)))
            )
            ft_x = zoom_fft_2d_mod(x, sec_N_x, N_out, N_X=N_X)
            zoom_fft_out += ft_x * np.outer(out_fac_1, out_fac_2)
    return zoom_fft_out


def chunk_in_chirp_zoom_fft_2d_mod(
    x_file, wl_z, d_x, N_x, N_out, N_X, N_chunk=4
):
    """
    Compute a 2D FFT using the Bluestein algorithm over x_file for different wl_z.
    Experimental chunked version - computed in chunks of the input (x_file).

    This version is useful for Fresnel diffraction, when you need to multiply
    your input x_file by a chirp which depends on lambda*z, before computing the
    zoom FFT. This function is useful when you need to do this over different
    lambda*z (wl_z).

    Define x_file as:
    arr = np.memmap('x.dat', dtype=np.complex128,mode='w+',shape=(N_x, N_x))
    arr[:] = x
    arr.flush()

    Parameters
    ----------
    x_file : str
        Path to the input signal as a memmap object.
    wl_z : float
        Wavelength times distance (lambda * z).
    d_x : float
        Sampling interval of the input field.
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
        The 2D FFT over the chosen output region (np.complex128).
    """
    x_vals = np.linspace(-(N_x // 2), (N_x // 2), N_x)
    chunk = (N_x // N_chunk) + 1 - ((N_x // N_chunk) % 2)
    zoom_fft_out = np.zeros((N_out, N_out), dtype=np.complex128)
    x_trunc = np.memmap(x_file, dtype=np.float32, mode="r", shape=(N_x, N_x))
    for i in range(N_chunk):
        for j in range(N_chunk):
            sec_N_x = sec_N_y = chunk
            if i == N_chunk - 1:
                sec_N_x = N_x - i * sec_N_x
                sec_N_x += 1 - (sec_N_x % 2)
            if j == N_chunk - 1:
                sec_N_y = N_x - j * sec_N_y
                sec_N_y += 1 - (sec_N_y % 2)
            print(i, j)
            x = x_trunc[
                i * chunk : min(i * chunk + sec_N_x, N_x),
                j * chunk : min(j * chunk + sec_N_y, N_x),
            ]
            index_x = np.arange(i * chunk, min(i * chunk + sec_N_x, N_x))
            index_y = np.arange(j * chunk, min(j * chunk + sec_N_y, N_x))
            xx = x_vals[index_x][:, np.newaxis] * d_x
            yy = x_vals[index_y][np.newaxis, :] * d_x
            if sec_N_x != sec_N_y:
                sec_N_x = sec_N_y = max(sec_N_x, sec_N_y)
            x = np.pad(
                x.astype(np.complex128)
                * np.exp(1j * (np.pi / wl_z) * (xx**2 + yy**2)),
                [(0, sec_N_x - np.shape(x)[0]), (0, sec_N_y - np.shape(x)[1])],
                mode="constant",
            )
            ph1 = -x_vals[i * chunk + (sec_N_x // 2)]
            ph2 = -x_vals[j * chunk + (sec_N_y // 2)]
            out_fac_1 = np.exp(
                np.arange(-(N_out // 2), (N_out // 2) + N_out % 2)
                * (1j * 2 * np.pi * ph1 * (1 / (N_X)))
            )
            out_fac_2 = np.exp(
                np.arange(-(N_out // 2), (N_out // 2) + N_out % 2)
                * (1j * 2 * np.pi * ph2 * (1 / (N_X)))
            )
            ft_x = zoom_fft_2d_mod(x, sec_N_x, N_out, N_X=N_X)
            zoom_fft_out += ft_x * np.outer(out_fac_1, out_fac_2)
    return zoom_fft_out
