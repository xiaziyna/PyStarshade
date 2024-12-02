from hcipy import *
import numpy as np

# From HCIPy tutorial
def custom_hex(normalized=True):
    pupil_diameter = 1
    gap_size = 1e-3 # m
    num_rings = 2
    segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
    return make_hexagonal_segmented_aperture(num_rings, segment_flat_to_flat, gap_size)

aperture_funcs = {
    'circ': make_circular_aperture,
    'ELT': make_elt_aperture,
    'GMT': make_gmt_aperture,
    'TMT': make_tmt_aperture,
    'Hale': make_hale_aperture,
    'LUVOIR-A': make_luvoir_a_aperture,
    'LUVOIR-B': make_luvoir_b_aperture,
    'Magellan': make_magellan_aperture,
    'VLT': make_vlt_aperture,
    'HiCAT': make_hicat_aperture,
    'HabEx': make_habex_aperture,
    'HST': make_hst_aperture,
    'JWST': make_jwst_aperture,
    'Keck': make_keck_aperture,
    'hex': custom_hex
}

def call_aperture_function(aperture_name, *args, **kwargs):
    func = aperture_funcs[aperture_name]
    return func(*args, **kwargs)

def make_pupil(N_t, pupil_type='circ'):
    '''
    Generate a pupil of a specified type, save it as a compressed file, and return its 2D array representation.
    Args:
    N_t : The number of grid points along one dimension for the pupil.
    pupil_type : The type of pupil aperture to create. Default is 'circ' (circular aperture).

    Returns:
    The generated pupil is saved to a compressed `.npz` file in the `data/pupils/` directory
    '''
    pupil_grid = make_pupil_grid(N_t)
    pupil = evaluate_supersampled(call_aperture_function(pupil_type, normalized=True), pupil_grid, 8)
    pupil = np.reshape(pupil, (N_t, N_t))
    np.savez_compressed('data/pupils/'+pupil_type+'_'+str(int(N_t))+'.npz', pupil=pupil)
