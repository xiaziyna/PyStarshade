import numpy as np
from .apodization.apodize import *
from .diffraction.util import *
from .diffraction.field import *
from .diffraction.diffract import *

def source_field_to_ccd(source_field, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
                        N_s = 333, N_x = 6401, N_t = 1001, N_pix = 4001, 
                        ds = 10*0.03*au_to_meter, dx = 0.01, dt = 0.0116, dp=.5*1.9e-7):
    """
    Propagate an (N_s * N_s) source field (far field) through a starshade to a CCD.    
    Field incident on starshade is calculated as sum of planar waves for each source pixel.
    Fresnel diffraction from starshade to telescope, followed by Fraunhofer diffraction from telescope to CCD. 
    Uses Babinets principle, alongside Bluestein FFTs for high-resolution diffraction. 
    Based on calculations (Taaki et al. 2023 in prep.).

    Args:
        source_field (float): N_s * N_s source field
        wl (float): Wavelength of light.
        dist_xo_ss (float): Distance between source plane and starshade.
        dist_ss_t (float): Distance between starshade and telescope.
        focal_length_lens (float): Focal length of the telescope lens.
        radius_lens (float): Radius of the telescope lens.
        N_s (int): Number of pixels in the source field. 
        N_x (int): Number of non-zero samples in starshade plane. Diameter of the starshade ~ N_x * dx. 
        N_t (int): Number of output samples in the telescope plane. Diameter of the telescope ~ N_t * dt.
        N_pix (int): Number of output pixels required.
        ds (float): Source field sampling
        dx (float): Input sampling. Default is 0.01.
        dt (float): Telescope sampling, must be less than (1/dx)*wl*dist_ss_t.
        dp (float): Pixel size sampling, depends on the telescope and the desired field-of-view. 

    Returns:
        The resulting field on the CCD plane of size (N_pix, N_pix).

    """

    source_prop = SourceField(ds, N_s, wl, dist_xo_ss, source_field)
    field_incident_ss = source_prop.farfield(dx, N_x, 0)

    ss_complement = 1-eval_hypergauss(N_x, dx)

    field_after_ss = field_incident_ss * ss_complement 
    field_after_ss = bluestein_pad(field_after_ss, N_x, N_t)

    fresnel = FresnelSingle(dx, dt, N_x, dist_ss_t, wl)
    test_new, dt = fresnel.zoom_fresnel_single_fft(field_after_ss, N_t)

    field_free_prop = source_prop.farfield(dt, N_t, dist_ss_t)

    pupil_mask = grey_pupil_func(N_t, dx = dt, r = radius_lens)
    N_xt, N_yt = N_in_2d(pupil_mask) # Number of non-zero samples in telescope pupil

    field_aperture_freesp = bluestein_pad(field_free_prop*pupil_mask, N_xt, N_pix)
    field_aperture_ss = bluestein_pad(test_new*pupil_mask, N_xt, N_pix)

    fraunhofer = Fraunhofer(dt, dp, N_xt, focal_length_lens, wl)
    out_field_free_space, dp = fraunhofer.zoom_fraunhofer(field_aperture_freesp, N_pix)
    out_field_compl_ss, dp = fraunhofer.zoom_fraunhofer(field_aperture_ss, N_pix)

    out_field_ss = out_field_free_space - out_field_compl_ss
    return out_field_ss


def point_source_to_ccd(mag_s, loc_s, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
                        N_x = 6401, N_t = 1001, N_pix = 1001,
                        dx = 0.01, dt = 0.0116, dp=1.9e-7):
    """
    Propagate a collection of point-sources located in the source plane through a starshade to a CCD.
    Sequentially performs Fresnel diffraction from starshade to telescope, followed by Fraunhofer diffraction from telescope to CCD. 
    Uses Babinets principle, alongside Bluestein FFTs for high-resolution diffraction. 
    Based on calculations (Taaki et al. 2023 in prep.).

    Args:
        mag_s (list): List of magnitudes for each point source.
        loc_s (list): List of locations (x, y, z) for each point source (z = 0 is starshade plane).
        wl (float): Wavelength of light.
        dist_xo_ss (float): Distance between source plane and starshade.
        dist_ss_t (float): Distance between starshade and telescope.
        focal_length_lens (float): Focal length of the telescope lens.
        radius_lens (float): Radius of the telescope lens.
        N_x (int): Number of non-zero samples in starshade plane. Diameter of the starshade ~ N_x * dx. 
        N_t (int): Number of output samples in the telescope plane. Diameter of the telescope ~ N_t * dt.
        N_pix (int): Number of output pixels required. 
        dx (float): Input sampling. Default is 0.01.
        dt (float): Telescope sampling, must be less than (1/dx)*wl*dist_ss_t.
        dp (float): Pixel size sampling, depends on the telescope and the desired field-of-view. 

    Returns:
        The resulting field on the CCD plane of size (N_pix, N_pix).

    """

    N_X = N_x + N_t - 1 # The number of input samples required to perform Fresnel propagation with a Bluestein FFT

    no_sources = len(mag_s)
    k_list = []
    pointsource_list = []
    field_incident_ss = np.zeros((N_X, N_X), dtype='complex128')
    for i in range(no_sources):
        ps = PointSource(dx, N_X, wl, *loc_s[i], mag_s[i])
        pointsource_list.append(ps)
        k_list.append(ps.wave_numbers())
        field_incident_ss += ps.plane_wave(k_list[i], 0)

# optional add Gaussian background dust disk term
#    gs = GaussianSource(dx, N_X, wl, 0, 0, dist_xo_ss, .75*10e-18, 5*au_to_meter)
#    gs_A, gs_inv_sigma = gs.far_field_gaussian_params()
#    field_incident_ss += gs.far_field_gaussian(gs_A, gs_inv_sigma, 0)

    ss_complement = 1-eval_hypergauss(N_X, dx) # Complement of starshade aperture

    field_after_ss = field_incident_ss * ss_complement

    fresnel = FresnelSingle(dx, dt, N_x, dist_ss_t, wl)
    field_ss_prop, dt = fresnel.zoom_fresnel_single_fft(field_after_ss, N_t)

    pupil_mask = grey_pupil_func(N_t, dx = dt, r = radius_lens)

    field_free_prop = np.zeros((N_t, N_t), dtype='complex128')
    for i in range(no_sources):
        pointsource_list[i].update(d_x = dt, N = N_t)
        field_free_prop += pointsource_list[i].plane_wave(k_list[i], dist_ss_t)
   
#    gs.update(d_x = dt, N = N_t)
#    field_free_prop += gs.far_field_gaussian(gs_A, gs_inv_sigma, dist_ss_t)

    N_xt, N_yt = N_in_2d(pupil_mask) # Number of non-zero samples in telescope pupil

    field_aperture_freesp = bluestein_pad(field_free_prop*pupil_mask, N_xt, N_pix)
    field_aperture_ss = bluestein_pad(field_ss_prop*pupil_mask, N_xt, N_pix)

    fraunhofer = Fraunhofer(dt, dp, N_xt, focal_length_lens, wl)
    out_field_free_space, dp = fraunhofer.zoom_fraunhofer(field_aperture_freesp, N_pix)
    out_field_compl_ss, dp = fraunhofer.zoom_fraunhofer(field_aperture_ss, N_pix)

    out_field_ss = out_field_free_space - out_field_compl_ss
    return out_field_ss
