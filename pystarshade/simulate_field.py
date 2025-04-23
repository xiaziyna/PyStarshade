import numpy as np
from pystarshade.apodization.apodize import *
from pystarshade.diffraction.util import *
from pystarshade.diffraction.field import *
from pystarshade.diffraction.diffract import *

def pupil_to_ccd(wl, focal_length_lens, pupil_field, pupil_mask, dt, dp,  N_t, N_pix):
    """
    Propagate a field from a pupil to the focal plane/CCD.

    Parameters
    ----------
    wl : float
        Wavelength.
    focal_length_lens : float
        Focal length of the telescope lens.
    pupil_field : np.ndarray of complex
        Field incident on the pupil.
    dp : float
        Focal plane sampling, depends on the telescope and the desired field-of-view.

    Returns
    -------
    out_field_ss : np.ndarray of complex
        Field in the focal plane.
    """
    field_aperture_ss = bluestein_pad(pupil_field*pupil_mask, N_t, N_pix)
    fraunhofer = Fraunhofer(dt, dp, N_t, focal_length_lens, wl)
    out_field_ss, dp = fraunhofer.zoom_fraunhofer(field_aperture_ss, N_pix)
    return out_field_ss

def source_field_to_pupil(ss_mask_fname, wl, dist_ss_t,  N_x = 6401, N_t = 1001, dx = 0.01, dt = 0.03, chunk = 1):
    """
    Propagate starshade mask to the pupil using chunking of the input mask for generating an incoherent PSF basis.

    Parameters
    ----------
    ss_mask_fname : str
        Starshade mask filename (stored in mask directory).
    wl : float
        Wavelength.
    dist_ss_t : float
        Distance between the starshade and the telescope.
    N_x : int
        Number of non-zero samples in the starshade plane. Diameter of the starshade ~ N_x * dx.
    N_t : int
        Number of output samples in the telescope plane. Diameter of the telescope ~ N_t * dt.
    ds : float
        Source field sampling.
    dx : float
        Input starshade mask sampling.
    dt : float
        Telescope sampling, must be less than `(1 / dx) * wl * dist_ss_t`.
    chunk : bool
        If True, use memory chunked FFT. If using chunk, a memmap file must be passed.

    Returns
    -------
    field_incident_telescope : np.ndarray of complex
        Field incident on the telescope pupil.
    field_free_prop : np.ndarray of complex
        Field incident on the pupil if no starshade is used.
    params : tuple of float
        Tuple containing `(wl, dist_ss_t, dt)`.
    """
    if chunk and not ss_mask_fname.endswith('.dat'):
        raise ValueError(f'starshade mask must be a memmap file')
    N_s=11
    ds=0.04*au_to_meter
    source_field = np.zeros((N_s, N_s))
    source_field[N_s//2, N_s//2] = 1
    dist_xo_ss = 10*pc_to_meter
    source_prop = SourceField(ds, N_s, wl, dist_xo_ss, source_field)
    field_free_prop = source_prop.farfield(dt, N_t, dist_ss_t)

    field_incident_telescope_compl = np.zeros((N_t, N_t), dtype=np.complex128)
    fresnel = FresnelSingle(dx, dt, N_x, dist_ss_t, wl)
    if chunk:
        field_incident_telescope_compl, dt = fresnel.nchunk_zoom_fresnel_single_fft(ss_mask_fname, N_t, N_chunk = 4)
    else:
        starshade_qu = np.load(ss_mask_fname)
        starshade = qu_mask_to_full(starshade_qu['grey_mask'])
        field_incident_telescope_compl, dt = fresnel.zoom_fresnel_single_fft(starshade, field_after_ss, N_t)

    field_incident_telescope = field_free_prop - field_incident_telescope_compl
    params = np.array([wl, dist_xo_ss, dt])
    return field_incident_telescope, field_free_prop, params


def chunk_source_field_to_pupil(source_field, wl, dist_xo_ss, dist_ss_t, ss_mask_fname, N_s=1, N_x = 6401, N_t = 1001, ds=0.1*au_to_meter, dx = 0.01, dt = 0.03):
    '''
    Experimental chunked version.
    '''
    ps = PointSource(dx, N_x, wl, 0, 0, dist_xo_ss, 1)
    k_vals = ps.wave_numbers()
    field_incident_telescope_compl = np.zeros((N_t, N_t), dtype=np.complex128)
    source_prop = SourceField(ds, N_s, wl, dist_xo_ss, source_field)
#    test_plane = ps.plane_wave(k_vals, 0)
    for chunk in range(4):
    # chunk 0: UL, 1: UR, 2: LR, 3:LL
#        field_after_ss = np.load('../mask/grey_wfirst_16_mask_%s_qu.npz' % (grey_mask_dx[mask_choice]))['data'].astype(np.complex128)
        field_after_ss = np.load(ss_mask_fname)['grey_mask'].astype(np.complex128)
        print (np.shape(field_after_ss))
        if chunk == 0:
            field_after_ss = np.fliplr(np.flipud(field_after_ss))
        elif chunk == 1:
            field_after_ss = np.flipud(field_after_ss[:, 1:])
        elif chunk == 2:
            field_after_ss = field_after_ss[1:, 1:]
        elif chunk == 3:
            field_after_ss = np.fliplr(field_after_ss[1:,])
        fresnel = FresnelSingle(dx, dt, N_x, dist_ss_t, wl)
        field_incident_telescope_compl_quad, dt = fresnel.one_chunk_zoom_fresnel_single_fft(self, field, N_out, chunk=0)(field_after_ss, N_t, chunk=chunk)
        field_incident_telescope_compl += field_incident_telescope_compl_quad
    #ss_qu = np.load(ss_mask_fname)['grey_mask'].astype(np.complex128)
    #full_mask = qu_mask_to_full(ss_qu)
    #test, dt = fresnel.zoom_fresnel_single_fft(full_mask, N_t)
    field_free_prop = source_prop.farfield2(dt, N_t, dist_ss_t)
    field_incident_telescope = field_free_prop - field_incident_telescope_compl
    np.savetxt('out/pupil_test', np.abs(field_incident_telescope))
    #np.savez_compressed('pupil_out/hwo_pupil_'+drm_params['grey_mask_dx'][mask_choice]+'_'+str(wl)+'.npz', field=field_incident_telescope)
    #print (np.allclose(test, field_incident_telescope_compl))


def source_field_to_ccd(source_field, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
                        N_s = 333, N_x = 6401, N_t = 1001, N_pix = 4001, 
                        ds = 10*0.03*au_to_meter, dx = 0.01, dt = 0.0116, dp=.5*1.9e-7):
    """
    Propagate an (N_s x N_s) source field (far field) through a starshade to a CCD/focal plane coherently.
    Field incident on starshade is calculated as sum of planar waves for each source pixel.
    Fresnel diffraction from starshade to telescope, followed by Fraunhofer diffraction from telescope to CCD/focal plane. 
    Uses Babinets principle, alongside Bluestein FFTs for high-resolution diffraction. 

    Parameters
    ----------
    source_field : float
        N_s x N_s source field.
    wl : float
        Wavelength of light.
    dist_xo_ss : float
        Distance between the source plane and the starshade.
    dist_ss_t : float
        Distance between the starshade and the telescope.
    focal_length_lens : float
        Focal length of the telescope lens.
    radius_lens : float
        Radius of the telescope lens.
    N_s : int
        Number of pixels in the source field.
    N_x : int
        Number of non-zero samples in the starshade plane. Diameter of the starshade ~ N_x * dx.
    N_t : int
        Number of output samples in the telescope plane. Diameter of the telescope ~ N_t * dt.
    N_pix : int
        Number of output pixels required.
    ds : float
        Source field sampling.
    dx : float, optional
        Input sampling. Default is 0.01.
    dt : float
        Telescope sampling, must be less than `(1 / dx) * wl * dist_ss_t`.
    dp : float
        Focal plane sampling, depends on the telescope and the desired field-of-view.

    Returns
    -------
    out_field_ss : np.ndarray of complex
        Resulting field at the focal plane of size (N_pix, N_pix).
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
    Propagate a collection of point-sources located in the source plane through a starshade to a CCD/focal plane coherently.
    Sequentially performs Fresnel diffraction from starshade to telescope, followed by Fraunhofer diffraction from telescope to CCD/focal plane. 
    Uses Babinets principle, alongside Bluestein FFTs for high-resolution diffraction. 

    Parameters
    ----------
    mag_s : list
        List of magnitudes for each point source.
    loc_s : list
        List of locations `(x, y, z)` for each point source (z = 0 is the starshade plane).
    wl : float
        Wavelength of light.
    dist_xo_ss : float
        Distance between the source plane and the starshade.
    dist_ss_t : float
        Distance between the starshade and the telescope.
    focal_length_lens : float
        Focal length of the telescope lens.
    radius_lens : float
        Radius of the telescope lens.
    N_x : int
        Number of non-zero samples in the starshade plane. Diameter of the starshade ~ N_x * dx.
    N_t : int
        Number of output samples in the telescope plane. Diameter of the telescope ~ N_t * dt.
    N_pix : int
        Number of output pixels required.
    dx : float, optional
        Input starshade sampling. Default is 0.01.
    dt : float
        Telescope sampling, must be less than `(1 / dx) * wl * dist_ss_t`.
    dp : float
        Focal plane sampling, depends on the telescope and the desired field-of-view.

    Returns
    -------
    out_field_ss : np.ndarray of complex
        Resulting field at the focal plane of size (N_pix, N_pix).
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
