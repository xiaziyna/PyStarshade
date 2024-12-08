Usage
=========================

Overview
----------

PyStarshade will compute fields and PSF files for a given starshade configuration and store these on disk for simulating imaging of new scenes.
To start pick a starshade configuration, some options provided are ('wfirst', 'hwo', 'habex'). The starshade configuration is termed a drm (design reference mission).

.. code-block:: python

    drm = 'hwo'

If you wanted to design your own drm (or modify one of these), go to the file 'data/drm.py' and change some the instrument parameters. The drm parameters define the size of the starshade, it's flight distance, number of petals, wavelength bands, size of the telescope aperture and focal length. In the drm is a parameter called 'dx_' which is the pixel size of the starshade mask in meters. You need to have a starshade mask on file (pystarshade cannot generate these from scratch). Some masks are provided for the drm's listed (stored in data/masks). An example drm is:

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



Fourier propagator
-------------------

In PyStarshade, the classes and functions are generally designed to act on a spatial grid of size N * N, where N is an odd number and the origin is at (N/2, N/2). 

Data
----------

The data directory is structured like so:

.. code-block:: bash

    data
    ├── fields
    ├── masks
    │   ├── starshade_edge_files
    │   └── starshade_masks
    ├── out
    ├── psf
    ├── pupils
    └── scenes

If you have new masks for the starshade, or telescope aperture masks, place them in the correct folders (starshade_masks and pupils respectively). 


Usage
--------

The simplest way to use PyStarshade is by using the precomputed pupil fields and the StarshadeProp class as described. The StarshadeProp class is designed to abstract away sampling calculations, as well as pre-compute data products and interface with them.
However, if desired, one may more directly interface with the optical propagation classes (Fresnel and Fraunhofer), or at a slightly higher level, the functions 'pupil_to_ccd' and 'source_field_to_pupil'.

As an example, calling the function 'source_field_to_ccd', this function
takes as input a 2D source-field of size (N_s, N_s) and spatial sampling ds and returns the 2D output
field incident on a CCD of size (N_pix, N_pix) and pixel size dp. 

.. code-block:: python

    from pystarshade.simulate_field import source_field_to_ccd

    source_field_to_ccd(source_field, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
                            N_s = 333, N_x = 6401, N_t = 1001, N_pix = 4001, 
                            ds = 10*0.03*au_to_meter, dx = 0.01, dt = 0.0116, dp=.5*1.9e-7)

Please see the examples folder for detailed examples!

Input data
--------

PyStarshade can take as input any pixelized source-field such as Haystacks model or an ExoVista model, or analytic descriptions of sources
(so far a point source and Gaussian source). If you wish to perform propagation using analytic descriptions, please 
use 'pystarshade.simulate_field.point_source_to_ccd'. 

