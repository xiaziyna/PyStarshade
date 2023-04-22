import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from diffraction.util import pc_to_meter, au_to_meter
from simulate_field import point_source_to_ccd
np.set_printoptions(suppress=True)

wl = 633e-9
focal_length_lens = 30
radius_lens = 2.4
dist_xo_ss = 10*pc_to_meter
dist_ss_t = 63942090. # Distance for hypergaussian mask

print ('Farfield: wl*d_1: ', (dist_xo_ss*wl))
print ('Nearfield: wl*d_2: ', (dist_ss_t*wl))
print ('Fresnel number for HyperGaussian ~ diam^2 / wl*z: ', ((25)**2)/(dist_ss_t*wl))

# For this example, we place an exoplanet at 1.8 AU from the on-axis host star, with a brightness contrast ratio of 10e-7
# List of point sources: their magnitudes and locations in the source field

mag_list = np.array([1, 10e-7])
loc_list = np.array([[0, 0, 10*pc_to_meter], [0, 1.8*au_to_meter, 10*pc_to_meter]])

star_exo_field = point_source_to_ccd(mag_list, loc_list, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens)

plt.figure()
plt.imshow(np.abs(star_exo_field)**2, norm=LogNorm(), cmap='Blues_r')
plt.colorbar()
plt.show()
