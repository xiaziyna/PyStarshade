from pystarshade.diffraction.util import mas_to_rad, au_to_meter, pc_to_meter, data_file_path
from pystarshade.propagator import StarshadeProp
import os
from pystarshade.config import OUTPUT_DIR
#Creates a blank 1001×1001 field with two delta-like sources, initializes an HWO starshade with a hex pupil, propagates those point sources through the starshade at 500 nm, and writes the output intensity to test_field.npz. 

hwo_starshade = StarshadeProp(drm = 'hwo')
hwo_starshade.gen_pupil_field()
hwo_starshade.gen_psf_basis('hex')
test_field = np.zeros((1001, 1001))
test_field[500, 500] = 1e11
test_field[800, 760] = 1
out = hwo_starshade.gen_scene('hex', test_field, 500e-9)
path = os.path.join(OUTPUT_DIR, 'test_field.npz')
np.savez_compressed(path, field = out)
