"""
Load a scene simulated with ExoVista and simulate imaging with starshade.
Uses 'Scene.py' available in ExoVista directory on Github:
github.com/alexrhowe/ExoVista

Scenes generated with ExoVista are used here
Each pixel is 2 mas spacing, with 1001x1001 scene
"""
from astropy.io import fits
from . import download_exovista
download_exovista.exovista_scenes_file()
from pystarshade.data.scenes.Scene import *
from pystarshade.diffraction.util import mas_to_rad, au_to_meter, pc_to_meter, data_file_path
from pystarshade.propagator import StarshadeProp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import os

def flux_to_mag(flux):
    # 2 mas pixels
    return - 2.5*np.log10((500**2)*flux /  3640	)

def ring_mean(image, center, radius, thickness=1.0):
    cy, cx = center
    ny, nx = image.shape
    
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    inner_radius = radius - thickness / 2
    outer_radius = radius + thickness / 2
    ring_mask = (rr >= inner_radius) & (rr < outer_radius)
    
    return image[ring_mask].mean(), ring_mask


#fname = '999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.60.fits' #solar system at 60 deg inclinati
fname = '999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.00.fits'

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'pystarshade', 'data'))
file_path = os.path.join(data_dir, 'scenes', fname)
hdul = fits.open(file_path)
dist_xo_ss = hdul[4].header['dist'] * pc_to_meter

inclination = hdul[2].header['I-0']
print (hdul[2].header['I-0'], hdul[2].header['NZODIS-0'], hdul[2].header['R-0'])
local_zodi_mag_v_arcsec2 = 22.5

t = 0 # timestep
scene = Scene(file_path)
bg_scene = scene.getDiskImage()
star_xy = scene.getXYstar()
bg_scene[:,int(star_xy[0]), int(star_xy[1])] += scene.getStarSpec(time = t)

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

N_s = N_dust
extended_field = np.zeros((N_wl, N_s, N_s), dtype=np.complex128) 
half_N = N_s // 2
half_in_N = N_dust // 2
bit_arr = N_dust % 2
extended_field[:, half_N - half_in_N : half_N + half_in_N + bit_arr,
       half_N - half_in_N : half_N + half_in_N + bit_arr] = bg_scene

lam_ref = 0.55 * 1e-6
ild = np.where(wl_range >= lam_ref)[0][0]-1
print (wl_range[ild], ild)

n_planet = 0

for i in range(scene.nplanets):
    if np.all(planet_xy[i] - N_dust_half + half_N  <= N_s-1) and np.all(planet_xy[i] - N_dust_half + half_N >= 0):
        extended_field[:, planet_xy[i, 1]- N_dust_half + half_N , planet_xy[i, 0]- N_dust_half + half_N] += planet_spectra[i]
        n_planet +=1

w_i = 154 # choose wavelength
#w_i = 184
source_field = extended_field[w_i]

print (np.shape(source_field))
print ( -2.5*np.log10((500**2)*source_field[int(star_xy[0]), int(star_xy[1])])+8.9)
print (source_field[int(star_xy[0])-65, int(star_xy[1])])
print ( - 2.5*np.log10((500**2)*source_field[int(star_xy[0]) - 50, int(star_xy[1])]) + 8.9)
val, ring_mask = ring_mean(source_field, (int(star_xy[0]), int(star_xy[1])), 50, thickness=5.0)
print (flux_to_mag(val), val*1.3)

#source_field[ring_mask] = 1

plt.figure()
plt.title('ExoVista input scene', fontsize=14)
plt.text(-940, 900, r'$\lambda = $500 nm', color='white', fontsize=12)
plt.imshow(np.abs(source_field),  norm=LogNorm(), cmap='jet', extent=(-N_s, N_s, -N_s, N_s))
plt.xlabel('x [mas]', fontsize=14)
plt.ylabel('y [mas]', fontsize=14)
cbar = plt.colorbar()
cbar.set_label('Contrast', fontsize=14)
plt.tight_layout()
plt.show()

#source_field = np.zeros((250, 250))
#source_field[220, 125] = 1

wl = wl_range[w_i]

#plt.figure()
#plt.plot(np.arange(150)*ds_au, flux_to_mag(source_field[N_s//2+1, N_s//2+1:N_s//2+151]))
#plt.ylabel('Surface brightness [mag/sq. arcsec.]')
#plt.xlabel('Separation [au]')
#plt.show()


print ('Distance from exo-scene to starshade [pc]: ', dist_xo_ss/pc_to_meter)
print ('Number of planets included in scene: ', n_planet)
print ('Star brightness: ', star_spectra[w_i])
print ('Planet brightness: ', planet_spectra[:,w_i])
print ('Plan/Star brightness: ', (planet_spectra[:,w_i]/(star_spectra[w_i])))
print ('Semi major axis[au]: ', np.sqrt(2)*ds_au*np.hypot(planet_xy[:, 0] - N_dust_half, planet_xy[:, 1] - N_dust_half))
print ('wavelength [m]', wl)
print ('ZODIs: ', hdul[2].header['NZODIS-0'])
print ('Physical extent (AU)', N_s * ds * (1/au_to_meter))
print ( '1 source pixel is (AU): ', ds/au_to_meter)
print ('Distance of system (pc): ', dist_xo_ss/pc_to_meter)

drm = 'hwo'
pupil_type = 'hex'
pupil_type = 'hwopupil_onaxis'
#pupil_type = 'hwopupil_offaxis'
hwo_starshade = StarshadeProp(drm = drm)
#hwo_starshade.gen_pupil_field()
hwo_starshade.N_wl = 1 # edit an internal variable so that only a single wavelength PSF basis is calculated
#hwo_starshade.gen_psf_basis(pupil_type = pupil_type, pupil_symmetry = True)
focal_intensity = hwo_starshade.gen_scene(pupil_type, source_field.astype(np.float32), 500e-9, pupil_symmetry = False)
#focal_intensity = hwo_starshade.gen_scene(pupil_type, source_field.astype(np.float32), 500e-9, pupil_symmetry = True)
#[1 + 500-250: 500+250, 1 + 500-250: 500+250,]

N = np.shape(focal_intensity)[0]

y_max, x_max = np.where(focal_intensity == np.max(focal_intensity))
x_max, y_max = x_max[0], y_max[0]
x, y = np.meshgrid(np.arange(N), np.arange(N))
print (hwo_starshade.ang_res_pix[0], x_max, y_max)
circ_mask = np.hypot(x-x_max, y-y_max) >= hwo_starshade.ang_res_pix[0]*2.4

#focal_intensity*=circ_mask
val, ring_mask = ring_mean(focal_intensity, (N//2, N//2), 30, thickness=1.0)
print (flux_to_mag(val))
#focal_intensity[ring_mask] = 1

# Add in a median local zodi mag of 23.02 V mag∕arc sec2

plt.figure()
plt.title(r'inclination: %s$^{\circ}$' % (str(int(inclination))), fontsize=14)
plt.text(-940, 900, r'$\lambda = $500 nm', color='white', fontsize=12)
plt.imshow(flux_to_mag(focal_intensity),  cmap='Blues', extent=(-N, N, -N, N))
plt.xlabel('x [mas]', fontsize=14)
plt.ylabel('y [mas]', fontsize=14)
cbar = plt.colorbar()
cbar.set_label(r'Surface brightness [mag$\cdot$arcsec$^{-2}$]', fontsize=14)
plt.tight_layout()
plt.show()

print (np.sum(focal_intensity))
print ( - 2.5*np.log10((500**2)*focal_intensity[(N//2) - 50, N//2]) + 8.9)

save_path = data_file_path('ss_0_'+drm+'_'+pupil_type+'.npz', 'out')
print ('file saved at: ', save_path)
np.savez_compressed(save_path, field = focal_intensity)
