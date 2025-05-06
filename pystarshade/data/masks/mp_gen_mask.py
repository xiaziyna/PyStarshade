"""
Generate starshade mask on specified grid from WFIRST starshade locus (collection of edge points). Can replace this with any other locus.
Boundary generated with Bezier curves with matplotlib.path
This version uses multiprocessing with tiling (for especially large grids)
dx, dy: Output sampling
Nx, Ny: Number of samples
"""

import pickle
import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt
import multiprocessing as mp
import ctypes
import scipy.io

#mat = scipy.io.loadmat('starshade_edge_files/NI2.mat') # Wfirst mask, find this in the SISTERS codebase
#r1 = mat['occulterDiameter'][0][0]/2      # upper apodization radius in meters
#r0 = r1 - mat['petalLength'][0][0]
#x_vals_ss  = mat['xVals'][0]
#y_vals_ss  = mat['yVals'][0]

(r0, r1, x_vals_ss, y_vals_ss) = pickle.load( open("starshade_edge_files/HWO_locus.p", "rb" ))
print ('Inner radius: ', r0, 'Outer radius: ', r1)
x_vals_ss = x_vals_ss[::4]
y_vals_ss = y_vals_ss[::4]

#If want to use this grid to compute grey-mask then use up_sample factor, if not set up_sample=1
#If mask is symmetric, only compute the positive quadrant, if not compute all 4 (or other). 

up_sample = 5
dx = dy = 0.1/up_sample
Nx = Ny = int(((r1 + .2)*up_sample)/0.1)

len_grid = (Nx+1)**2
starshade_locus = path.Path(np.stack((x_vals_ss, y_vals_ss), axis=-1))

def eval_chunk(semaphore, i, j, l_x, l_y, mask):
    with semaphore:
        x_=np.arange(i*l_mask_chunk, i*l_mask_chunk + l_x)*dx
        y_=np.arange(j*l_mask_chunk, j*l_mask_chunk + l_y)*dy
        xv,yv = np.meshgrid(x_ , y_ )
        grid_points = np.hstack((xv.flatten('F')[:,np.newaxis],yv.flatten('F')[:,np.newaxis]))
        flags = starshade_locus.contains_points(grid_points)
        mask = np.reshape( np.frombuffer( mask, dtype=bool ), -1 )
        for f in range(len(flags)):
            mask[f] = flags[f]

num_processes_dim = 5
max_processes_current = 4 # Set to number of CPU cores available
l_mask_chunk = (Nx+1)//num_processes_dim
l_mask_ghost = (Nx+1)%l_mask_chunk

processes = []
mask_ = []
semaphore = mp.Semaphore(max_processes_current)

for i in range(num_processes_dim):
    for j in range(num_processes_dim):
        l_x = l_y = l_mask_chunk
        if i == num_processes_dim - 1: l_x += l_mask_ghost
        if j == num_processes_dim - 1: l_y += l_mask_ghost
        l_mem = l_x * l_y
        mask_.append(mp.RawArray(ctypes.c_bool, l_mem))
        p = mp.Process(target=eval_chunk, args=(semaphore, i, j, l_x, l_y, mask_[i*num_processes_dim + j]))
        processes.append(p)

for p in processes:
    p.start()

for p in processes:
    p.join()

mask = np.zeros((Nx+1, Ny+1), dtype=bool)

for i in range(num_processes_dim):
    for j in range(num_processes_dim):
        l_x = l_y = l_mask_chunk
        if i == num_processes_dim - 1: l_x += l_mask_ghost
        if j == num_processes_dim - 1: l_y += l_mask_ghost
        l_mem = l_x * l_y
        mask_chunk = np.reshape( np.frombuffer( mask_[i*num_processes_dim + j], dtype=bool ), -1)
        mask_chunk = mask_chunk.reshape((l_x, l_y))
        mask[i*l_mask_chunk:i*l_mask_chunk + l_x, j*l_mask_chunk:j*l_mask_chunk + l_y] = mask_chunk

mask = mask.astype(np.float64)
grey_mask = convolve(mask, np.ones((up_sample, up_sample)) )[::up_sample, ::up_sample].astype(np.float64)
grey_mask /= up_sample**2
np.savez_compressed('starshade_masks/grey_hwo_24_mask_'+final_dx_str+'m_qu.npz', grey_mask=grey_mask)
