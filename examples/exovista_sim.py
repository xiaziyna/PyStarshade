"""
Load a scene simulated with exovista and simulate imaging with starshade
Uses 'exovista_scene.py' available in exovista directory on Github, called 'scene.py'
"""
from astropy.io import fits
from input.exovista_scene_data.exovista_scene import *

#------------------------- load and construct exovista scene
# Each pixel is 2 mas spacing, with 250x250 scene
# scene extent is 125*2 mas ~ 250 mas

t = 0

direct_ext = 'input/exovista_scene_data/'
filenames = [direct_ext+'mine/1215-HIP_16537-TYC_-mv_3.72-L_0.35-d_3.20-Teff_5048.07.fits', direct_ext+'mine/894-HIP_12114-TYC_-mv_5.79-L_0.27-d_7.24-Teff_5005.89.fits']
fname = filenames[0]
hdul = fits.open(fname)
dist_xo_ss = hdul[4].header['dist'] * pc_to_meter

scene = Scene(fname)
dust_ = scene.getDiskImage()
bg_scene = np.copy(dust_)

star_xy = scene.getXYstar()
bg_scene[:,int(star_xy[0]), int(star_xy[1])] = scene.getStarSpec(time = t)

planet_xy = scene.getXYplanets(time = t).astype(int)
planet_spectra = scene.getPlanetSpec(time = t)
star_spectra = scene.getStarSpec(time = t)

ds_mas = scene.getPixScale()
stel_ang_diam = scene.getAngDiam() #mas
ds = dist_xo_ss * ds_mas * mas_to_rad
ds_au = ds / au_to_meter
wl_range = scene.lambdas * 1e-6 # wavelength in m

#if len(t) > 1: 

#N_s = 2*(100+np.max(np.abs(planet_xy[1:]-125))) + 1
#N_s = np.min((N_s, 251))
N_wl = len(wl_range)
N_dust = np.shape(bg_scene)[1]
N_dust_half = N_dust//2

N_s = 501
extended_field = np.zeros((N_wl, N_s, N_s), dtype=np.complex128) #*((np.min(source_field, axis=(1, 2)) + np.median(source_field, axis=(1, 2))*.1))[:, np.newaxis, np.newaxis]
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

print ('Number of planets included in scene: ', n_planet)

w_i = 100
w_i = 130
w_i = 250

source_field = extended_field[w_i] #**.5 #field is in units of inensity and planets in contrast, need to square root to get wave amplitude
wl = wl_range[w_i]
print (wl)
raise SystemExit

print ('Star brightness: ', star_spectra[w_i])
print ('Planet brightness: ', planet_spectra[1:,w_i])
print ('Plan/Star brightness: ', (planet_spectra[1:,w_i]/(star_spectra[w_i])))
print ('Semi major axis[au]: ', ds_au*np.hypot(planet_xy[1:, 0] - N_dust_half, planet_xy[1:, 1] - N_dust_half))

#============================

print ((1.22*habex_wl_band*1e-09/diameter_telescope_m)/mas_to_rad)
print ('wavelength [m]', wl)
print ('Angular resolution (MAS, AU)', ( wl/diameter_telescope_m)/mas_to_rad, wl*dist_xo_ss/diameter_telescope_m*(1/au_to_meter))


print ('ZODIs: ', hdul[2].header['NZODIS-0'])
print ('Physical extent (AU)', N_s * ds * (1/au_to_meter))
print ('f number (starshade): ', f_ss, ', 1/f: ', np.round(1/f_ss, 2))
print ('f number (telescope): ', f_telescope, ', 1/f: ', np.round(1/f_telescope, 8))
print ( 'ROI (AU): ', 2.2 * f_ss * (dist_xo_ss/dist_ss_t/au_to_meter )) #2.2 scaling from sisters project
#print ('FOV calculation (AU): ', (1/magnify) * (N_pix*dp) * (1/au_to_meter))
print ( '1 source pixel is (AU): ', ds/au_to_meter)
print ('Distance of system (pc): ', dist_xo_ss/pc_to_meter)
print ('lambda z ~ ', wl*dist_ss_t)
print ('IWA(MAS, au): ', (ss_radius/dist_ss_t)/mas_to_rad, ((ss_radius/dist_ss_t)/mas_to_rad) * (1/2) * ds_au )

hwo_starshade = StarshadeProp(drm = 'hwo')
hwo_starshade.gen_pupil_field()
hwo_starshade.gen_psf_basis('hex')

out = hwo_starshade.gen_scene('hex', source_field, 500e-9)
np.savez_compressed('test_field.npz', field = out)
