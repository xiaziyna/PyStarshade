"""
GPU-accelerated Bluestein FFT routines using PyTorch (cuFFT backend).

This module provides GPU-accelerated versions of the chunked Bluestein zoom FFT
for Fresnel diffraction propagation. It is an optional backend; if PyTorch with
CUDA support is not installed, all functions will be unavailable but the rest of
PyStarshade will work normally on CPU.

The chirp vectors are computed on CPU with NumPy (bit-identical to the CPU path),
then transferred to GPU. All FFT computation happens on CUDA via cuFFT.
Complex128 precision is used throughout to maintain phase accuracy.

Requires
--------
PyTorch with CUDA support. Install with: ``pip install pystarshade[gpu]``
"""

import logging
import numpy as np
import scipy.fft

try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_device():
    """Return the CUDA device string, or raise if unavailable."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA is not available. Install PyTorch with CUDA support "
            "or set use_gpu=False.")
    return 'cuda'


def _build_chirp_cache_gpu(N_x, N_out, N_X, device=None):
    """
    Precompute Bluestein chirp vectors on CPU and transfer to GPU.

    Parameters
    ----------
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the desired output signal.
    N_X : float
        Phantom zero-padded length for the internal FFT.
    device : str, optional
        CUDA device string. If None, auto-detected.

    Returns
    -------
    cache : dict
        Dictionary with keys 'bb', 'hh', 'trunc_buf' (GPU tensors),
        and 'N_chirp', 'N_x', 'N_out', 'N_X' (scalars).
    """
    if device is None:
        device = _get_device()

    N_chirp = N_x + N_out - 1
    bit_chirp = N_chirp % 2

    b_idx = np.arange(-(N_chirp // 2), (N_chirp // 2) + bit_chirp,
                      dtype=np.float64)
    b = np.exp(-np.pi * (1.0 / N_X) * 1j * b_idx ** 2).astype(np.complex128)

    h_idx = np.arange(-(N_out // 2) - (N_x // 2),
                      (N_out // 2) + (N_x // 2) + bit_chirp, dtype=np.float64)
    h = np.exp(np.pi * (1.0 / N_X) * 1j * h_idx ** 2).astype(np.complex128)
    h = np.roll(h, (N_chirp // 2) + 1)
    ft_h = scipy.fft.fft(h, workers=-1).astype(np.complex128)

    bb = torch.from_numpy(np.outer(b, b).astype(np.complex128)).to(device)
    hh = torch.from_numpy(np.outer(ft_h, ft_h).astype(np.complex128)).to(device)
    trunc_buf = torch.zeros((N_chirp, N_chirp), dtype=torch.complex128,
                            device=device)

    return {'bb': bb, 'hh': hh, 'trunc_buf': trunc_buf,
            'N_chirp': N_chirp, 'N_x': N_x, 'N_out': N_out, 'N_X': N_X}


def _bluestein_pad_into_gpu(buf, arr, N_in, N_out):
    """
    Copy arr into the centre of a pre-allocated zero buffer on GPU.

    Parameters
    ----------
    buf : torch.Tensor
        Pre-allocated GPU buffer of shape (N_chirp, N_chirp).
    arr : torch.Tensor
        Input array on GPU.
    N_in : int
        Number of non-zero input samples.
    N_out : int
        Number of output samples.
    """
    buf.zero_()
    half_zp = (N_in + N_out - 1) // 2
    half_arr = arr.shape[0] // 2
    bit_arr = arr.shape[0] % 2
    s = slice(half_zp - N_in // 2, half_zp + N_in // 2 + bit_arr)
    sa = slice(half_arr - N_in // 2, half_arr + N_in // 2 + bit_arr)
    buf[s, s] = arr[sa, sa]


def zoom_fft_2d_mod_gpu(x_gpu, N_x, N_out, Z_pad=None, N_X=None, _cache=None):
    """
    Compute a zoomed 2D FFT using the Bluestein algorithm on GPU.

    GPU equivalent of ``zoom_fft_2d_mod`` from ``bluestein_fft.py``.

    Parameters
    ----------
    x_gpu : torch.Tensor
        Centred input signal on GPU (complex128).
    N_x : int
        Length of the input signal.
    N_out : int
        Length of the output signal.
    Z_pad : float, optional
        Zero-padding factor.
    N_X : float, optional
        Phantom zero-padded length (Z_pad * N_x + 1).
    _cache : dict, optional
        Pre-built chirp cache from ``_build_chirp_cache_gpu``.

    Returns
    -------
    torch.Tensor
        Zoomed FFT of the input signal (complex128, on GPU).
    """
    if _cache is None:
        if (Z_pad is None and N_X is None) or \
                (Z_pad is not None and N_X is not None):
            raise ValueError("Provide exactly one of Z_pad or N_X.")
        if Z_pad is not None:
            N_X = Z_pad * N_x + 1
        _cache = _build_chirp_cache_gpu(N_x, N_out, N_X)

    bb = _cache['bb']
    hh = _cache['hh']
    trunc_buf = _cache['trunc_buf']
    N_chirp = _cache['N_chirp']
    bit_out = N_out % 2

    _bluestein_pad_into_gpu(trunc_buf, x_gpu, N_x, N_out)

    torch.mul(bb, trunc_buf, out=trunc_buf)
    tmp = torch.fft.fft2(trunc_buf)
    torch.mul(tmp, hh, out=tmp)
    tmp = torch.fft.ifft2(tmp)
    torch.mul(bb, tmp, out=tmp)

    s = slice((N_chirp // 2) - (N_out // 2),
              (N_chirp // 2) + (N_out // 2) + bit_out)
    return tmp[s, s].clone()


def chunk_in_chirp_zoom_fft_2d_mod_gpu(x_file, wl_z, d_x, N_x, N_out, N_X,
                                       N_chunk=16):
    """
    Compute a chirp-modulated 2D zoom FFT in chunks, using the GPU.

    GPU equivalent of ``chunk_in_chirp_zoom_fft_2d_mod`` from
    ``bluestein_fft.py``. Reads chunks from a float32 memory-mapped mask
    on CPU, streams each chunk to GPU for the Bluestein FFT, and
    accumulates the result on GPU before transferring back.

    Parameters
    ----------
    x_file : str
        Path to the input mask as a memory-mapped file (float32).
    wl_z : float
        Wavelength times propagation distance (lambda * z) [m^2].
    d_x : float
        Spatial sampling interval of the input field [m].
    N_x : int
        Size of the input mask along one dimension.
    N_out : int
        Number of output samples along one dimension.
    N_X : float
        Phantom zero-padded length for the Bluestein transform.
    N_chunk : int, optional
        Number of chunks along each axis. Default is 16.

    Returns
    -------
    np.ndarray
        The 2D zoom FFT result of shape (N_out, N_out), complex128.
    """
    import time
    device = _get_device()
    t_start = time.time()

    x_vals = np.linspace(-(N_x // 2), (N_x // 2), N_x)
    chunk_sz = (N_x // N_chunk) + 1 - ((N_x // N_chunk) % 2)

    zoom_fft_out = torch.zeros((N_out, N_out), dtype=torch.complex128,
                               device=device)
    x_trunc = np.memmap(x_file, dtype=np.float32, mode='r', shape=(N_x, N_x))

    cache = _build_chirp_cache_gpu(chunk_sz, N_out, N_X, device=device)
    last_cache = None

    # Precompute full 1D Fresnel chirp on GPU (separable)
    coords = (x_vals * d_x).astype(np.float64)
    chirp_1d = torch.from_numpy(
        np.exp(1j * (np.pi / wl_z) * coords ** 2).astype(np.complex128)
    ).to(device)

    # Precompute output phase index on GPU
    k_out = torch.from_numpy(
        np.arange(-(N_out // 2), (N_out // 2) + N_out % 2, dtype=np.float64)
    ).to(dtype=torch.complex128, device=device)

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

            # Read chunk from memmap, transfer to GPU
            x = x_trunc[i * chunk_sz: min(i * chunk_sz + sec_N_x, N_x),
                         j * chunk_sz: min(j * chunk_sz + sec_N_y, N_x)]
            x_gpu = torch.from_numpy(np.array(x)).to(
                dtype=torch.complex128, device=device)

            if sec_N_x != sec_N_y:
                sec_N_x = sec_N_y = max(sec_N_x, sec_N_y)

            # Apply Fresnel chirp on GPU (separable, precomputed)
            ix_start = i * chunk_sz
            iy_start = j * chunk_sz
            cx = chirp_1d[ix_start: min(ix_start + x_gpu.shape[0], N_x)]
            cy = chirp_1d[iy_start: min(iy_start + x_gpu.shape[1], N_x)]
            x_gpu *= cx[:, None] * cy[None, :]

            # Pad on GPU if needed
            if x_gpu.shape[0] < sec_N_x or x_gpu.shape[1] < sec_N_y:
                x_padded = torch.zeros((sec_N_x, sec_N_y),
                                       dtype=torch.complex128, device=device)
                x_padded[:x_gpu.shape[0], :x_gpu.shape[1]] = x_gpu
            else:
                x_padded = x_gpu
            del x_gpu

            # Select or build chirp cache for this chunk size
            use_cache = cache if sec_N_x == chunk_sz else None
            if use_cache is None:
                if last_cache is not None and last_cache['N_x'] == sec_N_x:
                    use_cache = last_cache
                else:
                    last_cache = _build_chirp_cache_gpu(sec_N_x, N_out, N_X,
                                                       device=device)
                    use_cache = last_cache

            ft_x = zoom_fft_2d_mod_gpu(x_padded, sec_N_x, N_out, N_X=N_X,
                                       _cache=use_cache)
            del x_padded

            # Output phase shift on GPU
            ph1 = float(-x_vals[i * chunk_sz + (sec_N_x // 2)])
            ph2 = float(-x_vals[j * chunk_sz + (sec_N_y // 2)])
            angle_1 = k_out * (2 * np.pi * ph1 / N_X)
            angle_2 = k_out * (2 * np.pi * ph2 / N_X)
            out_fac_1 = torch.exp(1j * angle_1)
            out_fac_2 = torch.exp(1j * angle_2)

            zoom_fft_out += ft_x * torch.outer(out_fac_1, out_fac_2)
            del ft_x

    result = zoom_fft_out.cpu().numpy()
    elapsed = time.time() - t_start
    logger.info("GPU chunked Bluestein: %.1fs (%dx%d chunks)",
                elapsed, N_chunk, N_chunk)
    return result
