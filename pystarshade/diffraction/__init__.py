
def gpu_available():
    """
    Check whether GPU-accelerated Bluestein FFT is available.

    Returns
    -------
    bool
        True if PyTorch is installed and a CUDA device is detected.
    """
    try:
        from pystarshade.diffraction.bluestein_fft_gpu import (
            _TORCH_AVAILABLE, _CUDA_AVAILABLE)
        return _TORCH_AVAILABLE and _CUDA_AVAILABLE
    except ImportError:
        return False
