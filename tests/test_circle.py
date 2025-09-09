"""
Test analytic circle solution vs numerical propagation.
Functions calculate_circle_solution, get_field, lommels_V and lommels_U
are adapted/taken from DIFFRAQ (Harness) see solution_util.py of DIFFRAQ
"""

import numpy as np
import pytest
from pystarshade.apodization.apodize import grey_pupil_func
from pystarshade.diffraction.diffract import FresnelSingle
from pystarshade.diffraction.field import pc_to_meter, PointSource
from scipy.ndimage import convolve
from scipy.special import jn

np.set_printoptions(suppress=True)

wl = 633e-9
R_lens = 2.4 
R_ss = 30
dist_xo_ss = 10*pc_to_meter
dist_ss_t = 63942090. # distance for hypergaussian mask
N_t = 257
N_x = 4001
dt = R_lens/(N_t//2)
dx = R_ss/(N_x//2)

test_data = [
    (wl, dist_xo_ss, dist_ss_t, R_ss, N_x, N_t, dx, dt)]

def calculate_circle_solution(wl, dist_xo_ss, dist_ss_t, R_ss, N_t = N_t, dt = dt, is_opaque=True):
    """Calculate analytic solution to circular disk over observation points ss."""

    ss = np.arange(-(N_t//2), (N_t//2) + N_t%2 )*dt
    ss = np.hypot(ss, ss[:,None])

    kk = 2.*np.pi/wl
    zeff = dist_ss_t * dist_xo_ss / (dist_ss_t + dist_xo_ss)

    #Lommel variables
    uu = kk*R_ss**2./zeff
    vv = kk*ss*R_ss/dist_ss_t

    #Get value of where to break geometric shadow
    vu_brk = 0.99
    vu_val = np.abs(vv/uu)

    #Build nominal map
    Emap = np.zeros_like(ss) + 0j

    #Calculate inner region (shadow for disk, illuminated for aperture)
    sv_inds = vu_val <= vu_brk

    Emap[sv_inds] = get_field(uu, vv, ss, sv_inds, kk, dist_ss_t, dist_xo_ss, \
        is_opaque=is_opaque, is_shadow=is_opaque)

    #Calculate outer region (illuminated for disk, shadow for aperture)
    sv_inds = ~sv_inds
    Emap[sv_inds] = get_field(uu, vv, ss, sv_inds, kk, dist_ss_t, dist_xo_ss, \
        is_opaque=is_opaque, is_shadow=not is_opaque)

    return Emap

def get_field(uu, vv, ss, sv_inds, kk, zz, z0, is_opaque=True, is_shadow=True):
    #Lommel terms
    n_lom = 50

    #Return empty if given empty
    if len(ss[sv_inds]) == 0:
        return np.array([])

    #Shadow or illumination? Disk or Aperture?
    if (is_shadow and is_opaque) or (not is_shadow and not is_opaque):
        AA, BB = lommels_V(uu, vv[sv_inds], nt=n_lom)
    else:
        BB, AA = lommels_U(uu, vv[sv_inds], nt=n_lom)

    #Flip sign for aperture
    if not is_opaque:
        AA *= -1.

    #Calculate field due to mask QPF phase term
    EE = np.exp(1j*uu/2.)*(AA + 1j*BB*[1.,-1.][int(is_shadow)])

    #Add illuminated beam
    if not is_shadow:
        EE += np.exp(-1j*vv[sv_inds]**2./(2.*uu))

    #Add final plane QPF phase terms
    EE *= np.exp(1j*kk*(ss[sv_inds]**2./(2.*zz) + zz))

    #Scale for diverging beam
    EE *= z0 / (zz + z0)

    return EE

def lommels_V(u,v,nt=10):
    VV_0 = 0.
    VV_1 = 0.
    for m in range(nt):
        VV_0 += (-1.)**m*(v/u)**(0+2.*m)*jn(0+2*m,v)
        VV_1 += (-1.)**m*(v/u)**(1+2.*m)*jn(1+2*m,v)
    return VV_0, VV_1

def lommels_U(u,v,nt=10):
    UU_1 = 0.
    UU_2 = 0.
    for m in range(nt):
        UU_1 += (-1.)**m*(u/v)**(1+2.*m)*jn(1+2*m,v)
        UU_2 += (-1.)**m*(u/v)**(2+2.*m)*jn(2+2*m,v)
    return UU_1, UU_2

def pystar_circle(wl, dist_xo_ss, dist_ss_t, R_ss, N_x = 8001, N_t = 257, \
                                                        dx = 0.0075, dt = 0.01875):
    """ Fresnel diffraction via PyStarshade for comparison to analytic circle mask diffraction"""

    ps = PointSource(dx, N_x, wl, 0, 0, dist_xo_ss, 1)
    k_vals = ps.wave_numbers()
    field_incident_ss = ps.plane_wave(k_vals, 0)

    ss_complement = grey_pupil_func(N_x, dx, R_ss)
    field_after_ss = field_incident_ss * ss_complement 

    fresnel = FresnelSingle(dx, dt, N_x, dist_ss_t, wl)
    field_aperture, dt = fresnel.zoom_fresnel_single_fft(field_after_ss, N_t)

    ps.update(d_x = dt, N = N_t)
    field_free_prop = ps.plane_wave(k_vals, dist_ss_t)

    field_circ = field_free_prop - field_aperture
    return field_circ

@pytest.mark.parametrize("wl, dist_xo_ss, dist_ss_t, R_ss, N_x, N_t, dx, dt", test_data)
def test_circle(wl, dist_xo_ss, dist_ss_t, R_ss, N_x, N_t, dx, dt):
    field_circ = pystar_circle(wl, dist_xo_ss, dist_ss_t, R_ss, N_x = N_x, N_t = N_t, dx = dx, dt = dt)
    analytic_circ = calculate_circle_solution(wl, dist_xo_ss, dist_ss_t, R_ss, N_t = N_t, dt = dt)
    assert(np.max(np.abs(np.abs(field_circ) - np.abs(analytic_circ)))**2 < 1e-6)
