import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from diffraction.util import pc_to_meter, au_to_meter
from simulate_field import source_field_to_ccd
from astropy.io import fits

# In this example we use a source-field from the Haystacks project
# See https://asd.gsfc.nasa.gov/projects/haystacks/downloads.html
# Note: starshade suppression will occur within a 1 AU radius for these parameter values, in this example we 
# increase the source pixel size from (0.03 AU to 0.3 AU) to achieve contrast to see mercury & venus. 
# Note: In this example we just propagate a region of (333, 333) source pixels to see the close in planets
# Note: The sun brightness is lowered so that contrast for imaging mercury & venus can be achieved. 
# See (Taaki et al. 2023) for more information. 

focal_length_lens = 30
radius_lens = 2.4
dist_xo_ss = 10*pc_to_meter
dist_ss_t = 63942090. 

haystacks_file = 'modern_cube_zodi1inc0dist10_0.70-0.87um.fits'
haystacks = fits.open(haystacks_file)
cubehdr = haystacks[0].header
ds = cubehdr['PIXSCALE']*au_to_meter
source_field = haystacks[1].data
source_field[source_field > 20] = 5.6e-6 # reduce solar brightness (too bright for achievable contrast)
wavelength = haystacks[1].header['WAVEL'] * 1e-6 # wavelength in m

source_field = source_field[(3333//2)-166: (3333//2) + 167, (3333//2)-166: (3333//2) + 167]
N_s, _ = np.shape(source_field)

star_exo_field = source_field_to_ccd(
    source_field, wavelength, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
    N_s = N_s, N_x = 6401, N_t = 1001, N_pix = 4001, ds = 0.3*au_to_meter, dx = 0.01, dt = 0.0116, dp=.5*1.9e-7
)

plt.figure()
plt.title('Log Source field : Haystacks scene, %s nm' % (wavelength))
plt.imshow(source_field, norm=LogNorm(vmin=2.2e-11,vmax=np.max(np.abs(source_field))))
plt.xticks(np.arange(0, 333, 55), np.round(np.arange(-166, 166, 55)*0.3, 2))
plt.yticks(np.arange(0, 333, 55), np.round(np.arange(-166, 166, 55)*0.3, 2))
plt.xlabel('AU')
plt.ylabel('AU')
plt.colorbar()
plt.show()

plt.figure()
plt.title('Log Output field (trunc HG, 2.4m ap). Haystacks scene')
plt.imshow(np.abs(star_exo_field), norm=LogNorm(), cmap='Blues_r')
plt.colorbar()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()

plt.figure()
plt.title('Output field (trunc HG, 2.4m ap). Haystacks scene')
plt.imshow(np.abs(star_exo_field), cmap='Blues_r')
plt.colorbar()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
