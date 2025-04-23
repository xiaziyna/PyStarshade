from pystarshade.data.scenes.Scene import *
from pystarshade.diffraction.util import mas_to_rad, au_to_meter, pc_to_meter, flux_to_mag, data_file_path
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import os

'''
Requires running earth_sim.py first!
Loads a solar-system scene FITS file (inclination 0° & 60°) and corresponding precomputed starshade diffraction outputs for the HWO off-axis pupil, then plots them side-by-side with a colorbar and saves the figure. 
'''

#fname = '999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.60.fits' #solar system at 60 deg inclinati
fname = '999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.00.fits'
drm = 'hwo'
pupil_type = 'hwopupil_offaxis'
#pupil_type = 'hwopupil_onaxis'

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'pystarshade', 'data'))
file_path = os.path.join(data_dir, 'scenes', fname)
hdul = fits.open(file_path)
dist_xo_ss = hdul[4].header['dist'] * pc_to_meter
inclination = hdul[2].header['I-0']
print (hdul[2].header['I-0'], hdul[2].header['NZODIS-0'], hdul[2].header['R-0'])

ss0_path = data_file_path('ss_0_'+drm+'_'+pupil_type+'.npz', 'out')
ss0 = np.load(ss0_path)
focal_intensity_0 = ss0['field']

ss60_path = data_file_path('ss_60_'+drm+'_'+pupil_type+'.npz', 'out')
ss60 = np.load(ss60_path)
focal_intensity_60 = ss60['field']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
N = np.shape(focal_intensity_0)[0]
N = 250
# First image
#[1 + (N//2)-125: (N//2)+125, 1 + (N//2)-125: (N//2)+125]
im1 = axes[0].imshow(focal_intensity_0[1 + 500-125: 500+125, 1 + 500-125: 500+125], cmap='Blues_r', extent=(-N, N, -N, N))
axes[0].set_title(r'inclination: %s$^{\circ}$' % (str(0)), fontsize=16)
axes[0].text(-940, 900, r'$\lambda = $500 nm', color='white', fontsize=14)
axes[0].set_xlabel('x [mas]', fontsize=14)
axes[0].set_ylabel('y [mas]', fontsize=14)

# Second image
im2 = axes[1].imshow(focal_intensity_60[1 + 500-125: 500+125, 1 + 500-125: 500+125], cmap='Blues_r', extent=(-N, N, -N, N))
axes[1].set_title(r'inclination: %s$^{\circ}$' % (str(60)), fontsize=16)
axes[1].text(-940, 900, r'$\lambda = $500 nm', color='white', fontsize=14)
axes[1].set_xlabel('x [mas]', fontsize=14)
axes[1].set_ylabel('y [mas]', fontsize=14)

# Shared colorbar
fig.subplots_adjust(right=0.85)  # Make space on the right
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label(r'Intensity', fontsize=16)
#cbar.set_label(r'Surface brightness [mag$\cdot$arcsec$^{-2}$]', fontsize=16)

plt.tight_layout(rect=[0, 0, 0.88, 1])  # Adjust layout to accommodate colorbar
save_path = os.path.join(script_dir, 'ss_hwo_offaxis_close.png')
plt.savefig(save_path, dpi=400)
