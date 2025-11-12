import argparse
import os
import math
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import h5py
import cv2

# Example usage:
#python3 /home/jtaaki2/test_pystarshade/new_code/tilted_starshade/rasterize_loci_grey.py  --loci /home/jtaaki2/mask/hoee_locus_48.h5  --dx 0.0015 --rmax 49.5 --angle 0.4 --tile 256 --supersample 16 --max-proc 30  --out-dir /home/jtaaki2/mask/out --prefix hoee_raster

def load_loci_and_params(h5_path):
    with h5py.File(h5_path, 'r') as f:
        loci = f['loci'][()]
        params = f['params'][()] if 'params' in f else None
        num_petals = int(params[0]) if params is not None and params.shape[0] >= 1 else 0
        r0 = float(params[1]) if params is not None and params.shape[0] >= 2 else np.nan
        r1 = float(params[2]) if params is not None and params.shape[0] >= 3 else np.nan
    return loci, num_petals, r0, r1


def simplify_loci(loci_xy, tol):
    if tol is None or tol <= 0:
        return loci_xy
    # Ramer–Douglas–Peucker in-place via shapely-like approach
    # Fallback to numpy-based DP if shapely not used
    def _rdp(points, epsilon):
        if points.shape[0] < 3:
            return points
        start = points[0]
        end = points[-1]
        vec = end - start
        norm = np.linalg.norm(vec) + 1e-16
        vec /= norm
        w = points[1:-1] - start
        dist = np.abs(w[:, 0] * vec[1] - w[:, 1] * vec[0])
        idx = np.argmax(dist)
        dmax = dist[idx]
        if dmax > epsilon:
            left = _rdp(points[: idx + 2], epsilon)
            right = _rdp(points[idx + 1 :], epsilon)
            return np.vstack((left[:-1], right))
        else:
            return np.vstack((start, end))

    closed = np.allclose(loci_xy[0], loci_xy[-1])
    pts = loci_xy if not closed else loci_xy[:-1]
    out = _rdp(pts, tol)
    if closed:
        out = np.vstack((out, out[0]))
    return out


def tile_generator(Nx, Ny, tile_w, tile_h):
    for i0 in range(0, Nx + 1, tile_w):
        for j0 in range(0, Ny + 1, tile_h):
            i1 = min(i0 + tile_w, Nx + 1)
            j1 = min(j0 + tile_h, Ny + 1)
            yield i0, i1, j0, j1


def worker_render(tile, shm_name, shape, dtype, dx, dy, s, loci_px, r0, r1, y_scale):
    i0, i1, j0, j1 = tile
    # Map base-pixel indices to metric coords for tile corners
    x0 = i0 * dx
    y0 = j0 * dy
    x1 = (i1 - 1) * dx
    y1 = (j1 - 1) * dy

    inv_y = 1.0 / y_scale if y_scale > 0 else 1e12
    d00 = math.hypot(x0, y0 * inv_y)
    d01 = math.hypot(x0, y1 * inv_y)
    d10 = math.hypot(x1, y0 * inv_y)
    d11 = math.hypot(x1, y1 * inv_y)
    dmin = min(d00, d01, d10, d11)
    dmax = max(d00, d01, d10, d11)

    shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Fully outside or inside culling
    if dmin >= (r1 + 1.0):
        out[i0:i1, j0:j1] = 0.0
        shm.close()
        return
    if dmax <= (r0 - 1.0):
        out[i0:i1, j0:j1] = 1.0
        shm.close()
        return

    # Boundary tile: supersample rasterization
    tw = i1 - i0
    th = j1 - j0
    H = max(th * s, 1)
    W = max(tw * s, 1)
    canvas = np.zeros((H, W), dtype=np.uint8)

    # Shift polygon to tile origin in supersample pixel units
    shift_x = -i0 * s
    shift_y = -j0 * s
    poly = loci_px.copy()
    poly[:, 0] += shift_x
    poly[:, 1] += shift_y
    poly_i32 = np.round(poly).astype(np.int32)
    poly_i32 = poly_i32.reshape((-1, 1, 2))

    cv2.fillPoly(canvas, [poly_i32], color=255, lineType=cv2.LINE_8, shift=0)

    # Average down to base resolution
    if s == 1:
        grey = (canvas.astype(np.float32) / 255.0)
    else:
        Ht = (H // s) * s
        Wt = (W // s) * s
        block = canvas[:Ht, :Wt].reshape(Ht // s, s, Wt // s, s)
        grey = block.mean(axis=(1, 3)).astype(np.float32) / 255.0

    out[i0:i1, j0:j1] = grey[:th, :tw].T
    shm.close()


parser = argparse.ArgumentParser(description='High-accuracy rasterization of grey masks from loci using supersampled tiles and OpenCV fillPoly')
parser.add_argument('--loci', required=True, help='Input HDF5 loci file (datasets: loci, params)')
parser.add_argument('--dx', type=float, required=True, help='Base pixel width (meters)')
parser.add_argument('--dy', type=float, default=None, help='Base pixel height (meters); defaults to dx')
parser.add_argument('--rmax', type=float, required=True, help='Outer radius (meters)')
parser.add_argument('--angle', type=float, default=0.0, help='Tilt angle (radians); y scaled by cos(angle)')
parser.add_argument('--simplify-tol', type=float, default=0.0, help='Optional Douglas–Peucker tolerance (meters) before rasterization')
parser.add_argument('--tile', type=int, default=256, help='Tile size in base pixels (square)')
parser.add_argument('--supersample', type=int, default=8, help='Supersample factor per axis for boundary tiles')
parser.add_argument('--max-proc', type=int, default=30, help='Max worker processes (<=30)')
parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
parser.add_argument('--prefix', type=str, default='raster', help='Output file prefix')
args = parser.parse_args()

dy = args.dy if args.dy is not None else args.dx
loci, num_petals, r0, r1 = load_loci_and_params(args.loci)
y_scale = math.cos(args.angle)

# Simplify loci in metric space (optional)
if args.simplify_tol and args.simplify_tol > 0:
    loci = simplify_loci(loci, args.simplify_tol)

# Apply tilt scaling in metric coords (y scaled by cos(angle))
loci_scaled = loci.copy()
loci_scaled[:, 1] = loci_scaled[:, 1] * y_scale

# Determine grid size for quarter-plane
margin = 0.06
Nx = int(((args.rmax + margin) / args.dx))
Ny = int((((args.rmax + margin) * y_scale) / dy))
if Nx % 2 == 1:
    Nx += 1
if Ny % 2 == 1:
    Ny += 1

# Convert scaled loci to supersample pixel coordinates once
sx = (loci_scaled[:, 0] / args.dx) * args.supersample
sy = (loci_scaled[:, 1] / dy) * args.supersample
loci_px = np.stack([sx, sy], axis=1)

# Shared output (Nx+1, Ny+1) float32
out_shape = (Nx + 1, Ny + 1)
shm = shared_memory.SharedMemory(create=True, size=np.prod(out_shape) * np.float32().nbytes)
out = np.ndarray(out_shape, dtype=np.float32, buffer=shm.buf)
out.fill(0.0)

tiles = list(tile_generator(Nx, Ny, args.tile, args.tile))

total_tiles = len(tiles)
print(f'Starting render of {total_tiles} tiles (tile={args.tile}, supersample={args.supersample})')
with mp.get_context('spawn').Pool(processes=min(args.max_proc, 30)) as pool:
    jobs = []
    for tile in tiles:
        jobs.append(pool.apply_async(worker_render, (tile, shm.name, out_shape, np.float32, args.dx, dy, args.supersample, loci_px, r0, r1, y_scale)))
    for idx, j in enumerate(jobs, 1):
        j.get()
        if idx == total_tiles or idx % max(1, total_tiles // 100) == 0:
            print(f'Progress: {idx}/{total_tiles} tiles completed')

# Save outputs
os.makedirs(args.out_dir, exist_ok=True)
npz_path = os.path.join(args.out_dir, f'{args.prefix}_grey_qu.npz')
np.savez_compressed(npz_path,
                    grey_mask=out.astype(np.float32),
                    dx=args.dx,
                    dy=dy,
                    angle=args.angle,
                    cos_angle=y_scale,
                    r0=r0,
                    r1=r1)

# Clean up shared memory
shm.close()
shm.unlink()



