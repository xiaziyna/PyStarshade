import numpy as np

from pystarshade.propagator import StarshadeProp

hwo_starshade = StarshadeProp(drm="hwo")
hwo_starshade.gen_pupil_field()
hwo_starshade.gen_psf_basis("hex")
test_field = np.zeros((1001, 1001))
test_field[500, 500] = 1e11
test_field[800, 760] = 1
out = hwo_starshade.gen_scene("hex", test_field, 500e-9)
np.savez_compressed("test_field.npz", field=out)
