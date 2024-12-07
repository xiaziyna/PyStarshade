Usage
------

PyStarshade will compute fields and PSF files for a given starshade configuration and store these on disk for simulating imaging of new scenes.
To start pick a starshade configuration , some options provided are ('wfirst', 'hwo', 'habex'). The starshade configuration is termed a drm (design reference mission)

.. code-block:: python

    drm = 'hwo'

If you wanted to design your own drm (or modify one of these), go to the file 'data/drm.py' and change some the parameters. The drm parameters define the size of the starshade, it's flight distance, number of petals, wavelength bands, size of the telescope aperture and focal length. In the drm is a parameter called 'dx_' which is the pixel size of the starshade mask in meters, you need to have a starshade mask on file (pystarshade cannot generate these from scratch). Some masks are provided for the drm's listed (stored in data/masks). An example drm is:

.. code-block:: python

    'hwo': {
        'focal_length_lens': 10,
        'diameter_telescope_m': 6,
        'radius_lens': 3,
        'ss_radius': 30,
        'dist_ss_t': [9.52e+07],
        'num_pet': 24,
        'wl_bands': [[500, 600]],
        'dx_': [0.04, 0.001],
        'grey_mask_dx': ['04m', '001m'],
        'iwa': [30 / 9.52e+07 / mas_to_rad]  # R_ss / dist_ss_t in MAS
    }

The easiest way to interface with PyStarshade is via the StarshadeProp class. 

.. code-block:: python
    from pystarshade.propagator import StarshadeProp

    drm = 'hwo'
    hwo_starshade = StarshadeProp(drm = drm)
    hwo_starshade.gen_pupil_field()

After running this code, a field incident on the telescope aperture will be stored on disk (and can be propagated with various different telescope apertures). There are optional parameters to change the samplings at the pupil, or wavelength sampling.

Note: there is an important setting called 'chunk', if this is invoked a chunked FFT will be used to compute the fields using a memory mapped starshade mask. This avoids hitting RAM bottlenecks, but you will need to have enough disk memory for the memmap starshade mask of :math:`\left( \frac{diameter_{ss}}{dx} \right)^2` floats. By default, this chunked FFT will process the starshade mask of size (N * N) in chunks of size (N/N_chunk * N/N_chunk). To modify the N_chunk parameter change it inside 'pystarshade.simulate_field.source_field_to_pupil', by default N_chunk is set to 4.


.. code-block:: python

    pupil_type = 'hex'
    hwo_starshade.gen_psf_basis(pupil_type = pupil_type)
    focal_intensity = hwo_starshade.gen_scene(pupil_type, source_field.astype(np.float32), 500e-9)


