"""
Tests for GPU-accelerated Bluestein FFT.

All GPU tests are skipped if CUDA is not available.
"""

import numpy as np
import pytest
import tempfile
import os

from pystarshade.diffraction import gpu_available
from pystarshade.diffraction.bluestein_fft import zoom_fft_2d_mod, chunk_in_chirp_zoom_fft_2d_mod

skip_no_gpu = pytest.mark.skipif(
    not gpu_available(), reason="CUDA not available")


def test_gpu_available_returns_bool():
    """gpu_available() should return a bool without error in any environment."""
    result = gpu_available()
    assert isinstance(result, bool)


@skip_no_gpu
def test_gpu_cpu_equivalence_zoom_fft_2d_mod():
    """GPU zoom_fft_2d_mod_gpu should match CPU zoom_fft_2d_mod."""
    import torch
    from pystarshade.diffraction.bluestein_fft_gpu import (
        zoom_fft_2d_mod_gpu, _build_chirp_cache_gpu)

    N_x, N_out, N_X = 200, 101, 50000
    np.random.seed(42)
    x = np.random.randn(200, 200) + 1j * np.random.randn(200, 200)

    result_cpu = zoom_fft_2d_mod(x, N_x, N_out, N_X=N_X)

    cache = _build_chirp_cache_gpu(N_x, N_out, N_X)
    x_gpu = torch.from_numpy(x).to('cuda')
    result_gpu = zoom_fft_2d_mod_gpu(x_gpu, N_x, N_out, N_X=N_X,
                                     _cache=cache).cpu().numpy()

    rel_err = np.max(np.abs(result_cpu - result_gpu)) / np.max(np.abs(result_cpu))
    assert rel_err < 1e-12, f"Relative error {rel_err:.2e} exceeds 1e-12"


@skip_no_gpu
def test_gpu_cpu_equivalence_chunked():
    """GPU chunked Bluestein should match CPU chunked Bluestein."""
    from pystarshade.diffraction.bluestein_fft_gpu import (
        chunk_in_chirp_zoom_fft_2d_mod_gpu)

    N_x = 6401
    N_out = 501
    d_x = 0.01
    wl_z = 7e-7 * 1.7e8
    N_X = (1 / d_x) * wl_z / 0.04
    N_chunk = 4

    f = tempfile.NamedTemporaryFile(suffix='.dat', delete=False)
    try:
        mm = np.memmap(f.name, dtype=np.float32, mode='w+',
                       shape=(N_x, N_x))
        c = N_x // 2
        y, x = np.ogrid[-c:N_x - c, -c:N_x - c]
        mm[:] = ((x ** 2 + y ** 2) < (N_x // 4) ** 2).astype(np.float32)
        mm.flush()
        del mm

        result_cpu = chunk_in_chirp_zoom_fft_2d_mod(
            f.name, wl_z, d_x, N_x, N_out, N_X, N_chunk=N_chunk)
        result_gpu = chunk_in_chirp_zoom_fft_2d_mod_gpu(
            f.name, wl_z, d_x, N_x, N_out, N_X, N_chunk=N_chunk)

        rel_err = np.max(np.abs(result_cpu - result_gpu)) / \
            np.max(np.abs(result_cpu))
        assert rel_err < 1e-10, f"Relative error {rel_err:.2e} exceeds 1e-10"
    finally:
        os.unlink(f.name)


@skip_no_gpu
def test_gpu_returns_numpy():
    """GPU chunked function should return a numpy array, not a torch tensor."""
    from pystarshade.diffraction.bluestein_fft_gpu import (
        chunk_in_chirp_zoom_fft_2d_mod_gpu)

    N_x = 201
    f = tempfile.NamedTemporaryFile(suffix='.dat', delete=False)
    try:
        mm = np.memmap(f.name, dtype=np.float32, mode='w+',
                       shape=(N_x, N_x))
        mm[:] = 1.0
        mm.flush()
        del mm

        result = chunk_in_chirp_zoom_fft_2d_mod_gpu(
            f.name, 0.119, 0.01, N_x, 51, 50000, N_chunk=2)
        assert isinstance(result, np.ndarray), \
            f"Expected np.ndarray, got {type(result)}"
        assert result.dtype == np.complex128
    finally:
        os.unlink(f.name)


def test_fresnel_single_use_gpu_false():
    """FresnelSingle(use_gpu=False) should not require CUDA."""
    from pystarshade.diffraction.diffract import FresnelSingle
    fs = FresnelSingle(0.01, 0.03, 100, 1e8, 7e-7, use_gpu=False)
    assert fs._use_gpu is False


def test_fresnel_single_use_gpu_none():
    """FresnelSingle(use_gpu=None) should not raise."""
    from pystarshade.diffraction.diffract import FresnelSingle
    fs = FresnelSingle(0.01, 0.03, 100, 1e8, 7e-7, use_gpu=None)
    assert isinstance(fs._use_gpu, bool)
