import numpy as np


def qu_mask_to_full(grey_mask):
    """
    Take the negative quadrant of a starshade mask and return the full mask
    Args:
    grey_mask : Negative quadrant of mask

    Returns:
    full_mask : Full mask made of flipped negative quad
    """
    print(np.shape(grey_mask))
    Nx = Ny = len(grey_mask) - 1
    full_mask = np.zeros((2 * Nx + 1, 2 * Ny + 1), dtype=grey_mask.dtype)
    full_mask[:Nx, Ny + 1 :] = np.flipud(grey_mask[1:, 1:])
    full_mask[Nx:, Ny:] = grey_mask
    full_mask[Nx + 1 :, :Ny] = np.fliplr(grey_mask[1:, 1:])
    full_mask[: Nx + 1, : Ny + 1] = np.fliplr(np.flipud(grey_mask))
    return full_mask


# If only a quarter of the mask provided:
fname = "starshade_masks/grey_hwo_24_mask_04m"
# grey_mask_qu = np.load(fname+'_qu.npz')['grey_mask'].astype(np.float32)
# grey_mask = qu_mask_to_full(grey_mask_qu)


# If full mask:
grey_mask = np.load(fname + ".npz")["grey_mask"].astype(np.float32)

N_x = len(grey_mask)
arr = np.memmap(
    "%s.dat" % (fname), dtype=np.float32, mode="w+", shape=(N_x, N_x)
)
arr[:] = grey_mask[:]
arr.flush()
