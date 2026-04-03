"""
Integration test: run the library's GPU chunked Bluestein on the real
hoee_metashade mask and compare against the CPU reference pupil field.

This tests that the PyStarshade-integrated GPU code produces the same
result as the standalone scripts version, at production scale.
"""

import sys
import os
import time
import numpy as np

# Use the library from this repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MASK_FILE = '/home/xiaziyna/pystar/data/mask/grey_hoee_metashade_36_mask_001m.dat'
CPU_REF = '/home/xiaziyna/pystar/data/fields/hoee_metashade_pupil_001m_700.npz'

# Real propagation parameters for hoee_metashade at 700nm
# (extracted from StarshadeProp(drm='hoee_metashade', d_wl=100))
N_x = 99401
d_x = 0.001
wl = 7e-7
z = 1.7e8
wl_z = wl * z                          # 119.0
d_t = 0.04120919
N_X = (1.0 / d_x) * wl_z / d_t         # ~2887705
N_chunk = 16
N_out = 4999                            # over_N_t from propagator

print(f"N_x={N_x}, N_out={N_out}, N_chunk={N_chunk}")
print(f"N_X={N_X:.0f}")
print(f"Mask: {MASK_FILE}")
print()

# --- GPU Bluestein via the library ---
from pystarshade.diffraction.bluestein_fft_gpu import chunk_in_chirp_zoom_fft_2d_mod_gpu
from pystarshade.diffraction import gpu_available

assert gpu_available(), "CUDA not available"

print("Running library GPU chunked Bluestein...")
t0 = time.time()
result_gpu = chunk_in_chirp_zoom_fft_2d_mod_gpu(
    MASK_FILE, wl_z, d_x, N_x, N_out, N_X, N_chunk=N_chunk)
t_gpu = time.time() - t0
print(f"GPU Bluestein: {t_gpu:.1f}s ({t_gpu/60:.1f} min)")

# Apply the Fresnel output phase factor (same as FresnelSingle does)
from pystarshade.diffraction.util import grid_points
k = 2 * np.pi / wl
max_freq = 1.0 / d_x
df = max_freq * wl_z / N_X
out_xy = grid_points(N_out, N_out, dx=df)
quad_out_fac = (np.exp(1j * k * z)
                * np.exp(1j * k / (2 * z) * (out_xy[0]**2 + out_xy[1]**2))
                / (1j * wl_z))
field_gpu = quad_out_fac * result_gpu * (d_x ** 2)

# Babinet: free-space - complement
# For comparison purposes, just compare the raw complement field
# (the CPU reference is the full Babinet result, so extract the complement)
print(f"GPU field shape: {field_gpu.shape}, dtype: {field_gpu.dtype}")
print()

# --- Compare against CPU reference ---
if os.path.exists(CPU_REF):
    print("Comparing against CPU reference...")
    cpu_data = np.load(CPU_REF)
    cpu_field = cpu_data['field']
    cpu_freesp = cpu_data['freesp_field']

    # CPU reference stores: field = freesp - complement
    # So complement = freesp - field
    cpu_complement = cpu_freesp - cpu_field

    rel_err = np.max(np.abs(cpu_complement - field_gpu)) / np.max(np.abs(cpu_complement))
    print(f"CPU complement shape: {cpu_complement.shape}")
    print(f"GPU complement shape: {field_gpu.shape}")
    print(f"Max relative error: {rel_err:.2e}")

    if rel_err < 1e-8:
        print("\nPASS: Library GPU matches CPU reference")
    else:
        print(f"\nWARN: Relative error {rel_err:.2e} exceeds 1e-8")
else:
    print(f"CPU reference not found at {CPU_REF}")
    print("Skipping comparison")

print(f"\nTotal wall time: {time.time() - t0:.1f}s")
