import numpy as np
import sys
from scipy.ndimage import convolve

# Generate a grey-pixel mask for spatial anti-aliasing using a grid-sampled WFIRST mask

mask = np.loadtxt('output/wfirst_mask_002m') #generated with 0.001m grid
print (np.shape(mask))

up_fac = 5
grey_mask = convolve(mask, np.ones((up_fac, up_fac)) )[::up_fac, ::up_fac].astype(np.float32)
grey_mask /= (up_fac*up_fac)

Nx = Ny = len(grey_mask) - 1
full_mask = np.zeros((2*Nx + 1, 2*Ny + 1))
full_mask[:Nx, Ny+1:] = np.flipud(grey_mask[1:, 1:])
full_mask[Nx:, Ny:] = grey_mask
full_mask[Nx+1:, :Ny] = np.fliplr(grey_mask[1:, 1:])
full_mask[:Nx+1, :Ny+1] = np.fliplr(np.flipud(grey_mask))

print (np.shape(full_mask))
np.savetxt('output/grey_wfirst_mask_01m', full_mask)

