"""
Load a scene simulated with ExoVista and simulate imaging with starshade.
Uses 'Scene.py' available in ExoVista directory on Github:
github.com/alexrhowe/ExoVista

Scenes generated with ExoVista are used here
Each pixel is 1 mas spacing, with 1001x1001 scene
"""
from astropy.io import fits
from astropy import units as u
from . import download_exovista
download_exovista.exovista_scenes_file()
from pystarshade.data.scenes.Scene import *
from pystarshade.diffraction.util import mas_to_rad, au_to_meter, pc_to_meter, data_file_path
from pystarshade.propagator import StarshadeProp
import os

fname = '1215-HIP_16537-TYC_-mv_3.72-L_0.35-d_3.20-Teff_5048.07.fits'
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'pystarshade', 'data'))
file_path = os.path.join(data_dir, 'scenes', fname)
hdul = fits.open(file_path)
dist_xo_ss = hdul[4].header['dist'] * pc_to_meter

t = 0 # timestep
scene = Scene(file_path)
bg_scene = scene.getDiskImage()
star_xy = scene.getXYstar()
bg_scene[:,int(star_xy[0]), int(star_xy[1])] = scene.getStarSpec(time = t)

planet_xy = scene.getXYplanets(time = t).astype(int)
planet_spectra = scene.getPlanetSpec(time = t)
star_spectra = scene.getStarSpec(time = t)

ds_mas = scene.getPixScale()
stel_ang_diam = scene.getAngDiam()
ds = dist_xo_ss * ds_mas * mas_to_rad
ds_au = ds / au_to_meter
wl_range = scene.lambdas * 1e-6 

N_wl = len(wl_range)
N_dust = np.shape(bg_scene)[1]
N_dust_half = N_dust//2

N_s = 1001
extended_field = np.zeros((N_wl, N_s, N_s), dtype=np.complex128) 
half_N = N_s // 2
half_in_N = N_dust // 2
bit_arr = N_dust % 2
extended_field[:, half_N - half_in_N : half_N + half_in_N + bit_arr,
       half_N - half_in_N : half_N + half_in_N + bit_arr] = bg_scene

n_planet = 0
for i in range(1, scene.nplanets):
    if np.all(planet_xy[i] - N_dust_half + half_N  <= N_s-1) and np.all(planet_xy[i] - N_dust_half + half_N >= 0):
        extended_field[:, planet_xy[i, 1]- N_dust_half + half_N , planet_xy[i, 0]- N_dust_half + half_N] += planet_spectra[i]
        n_planet +=1

w_i = 154 # choose wavelength
source_field = extended_field[w_i]
wl = wl_range[w_i]

print ('Distance from exo-scene to starshade [pc]: ', dist_xo_ss)
print ('Number of planets included in scene: ', n_planet)
print ('Star brightness: ', star_spectra[w_i])
print ('Planet brightness: ', planet_spectra[1:,w_i])
print ('Plan/Star brightness: ', (planet_spectra[1:,w_i]/(star_spectra[w_i])))
print ('Semi major axis[au]: ', ds_au*np.hypot(planet_xy[1:, 0] - N_dust_half, planet_xy[1:, 1] - N_dust_half))
print ('wavelength [m]', wl)
print ('ZODIs: ', hdul[2].header['NZODIS-0'])
print ('Physical extent (AU)', N_s * ds * (1/au_to_meter))
print ( '1 source pixel is (AU): ', ds/au_to_meter)
print ('Distance of system (pc): ', dist_xo_ss/pc_to_meter)

drm = 'hwo'
pupil_type = 'hex'
hwo_starshade = StarshadeProp(drm = drm)
hwo_starshade.gen_pupil_field()
hwo_starshade.gen_psf_basis(pupil_type = pupil_type)
focal_intensity = hwo_starshade.gen_scene(pupil_type, source_field.astype(np.float32), 500e-9)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.figure()
plt.imshow(focal_intensity, norm=LogNorm(), cmap='jet')
plt.colorbar()
plt.show()


save_path = data_file_path('exo_scene_'+drm+'_'+pupil_type+'.npz', 'out')
print (save_path)
np.savez_compressed(save_path, field = focal_intensity)
