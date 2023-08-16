import pickle
import numpy as np
from matplotlib import path
import multiprocessing as mp
from scipy.ndimage import convolve

"""

Generate starshade mask on specified grid from WFIRST starshade locus. Can replace this with any other locus.
This code uses the starshade locus (edge points) and evaluates whether a given set of points lie within locus.
Uses matplotlib path - need to test accuracy. 
Can be used to compute a boolean mask (set up_fac to 1). Here compute grey-pixel average by oversampling by a factor of 5
and averaging over 5x5 pixels (up_fac set to 5). 

Obtain locus file NI2.mat from SISTERS codebase. 

dx, dy: Output sampling
Nx, Ny: Number of samples

"""

mat = scipy.io.loadmat('NI2.mat') # Wfirst
r1 = mat['occulterDiameter'][0][0]/2      # upper apodization radius in meters
r0 = r1 - mat['petalLength'][0][0]
x_vals_ss  = mat['xVals'][0]
y_vals_ss  = mat['yVals'][0]

print ('Inner radius: ', r0, 'Outer radius: ', r1)

# If mask is symmetric, only compute the positive quadrant, if not compute all 4 (or other). 

up_fac = 5
dx = dy = 0.01/up_fac
Nx = Ny = int(1320*up_fac)

x_ = np.arange(Nx + 1)*dx
y_ = np.arange(Ny + 1)*dy
starshade_locus = path.Path(np.stack((x_vals_ss, y_vals_ss), axis=-1))
xv,yv = np.meshgrid(x_ , y_ )

grid_points = np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]))

def test_(i, mask):
    l_p = (len(grid_points)//num_processes)
    if i< num_processes - 1:
        flags = starshade_locus.contains_points(grid_points[i*l_p: (i+1)*l_p])
    elif i == num_processes - 1 :
        flags = starshade_locus.contains_points(grid_points[i*l_p:])
    mask = np.reshape( np.frombuffer( mask, dtype=np.uint32 ), -1 )
    for f in range(len(flags)):
        mask[f] = flags[f]
    return

num_processes = 20
processes = []
mask_ = []
l_pr = len(grid_points)//num_processes

for i in range(num_processes):
    l_mask_chunk = l_pr
    if i == num_processes - 1 : l_mask_chunk += len(grid_points)%l_pr
    mask_.append(mp.RawArray('i', l_mask_chunk))
    p = mp.Process(target=test_,args=(i, mask_[i]))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

mask = np.zeros(len(grid_points), dtype=np.uint32)

for i in range(num_processes):
    mask_chunk = np.reshape( np.frombuffer( mask_[i], dtype=np.uint32 ), -1)
    mask[i*l_pr:i*l_pr + len(mask_chunk)] = mask_chunk #.astype('int')

mask = mask.reshape((Nx+1, Ny+1))

full_mask = np.zeros((2*Nx + 1, 2*Ny + 1))
full_mask[:Nx, Ny+1:] = np.flipud(mask[1:, 1:])
full_mask[Nx:, Ny:] = mask
full_mask[Nx+1:, :Ny] = np.fliplr(mask[1:, 1:])
full_mask[:Nx+1, :Ny+1] = np.fliplr(np.flipud(mask))

grey_mask = convolve(full_mask, np.ones((up_fac, up_fac)) )[::up_fac, ::up_fac].astype(np.float32)
