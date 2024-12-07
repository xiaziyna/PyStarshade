.. PyStarshade documentation master file, created by
   sphinx-quickstart on Fri Dec  6 09:40:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyStarshade documentation
=========================

`PyStarshade` is a tool for simulating high-contrast imaging of exoplanets with starshades. 

What is a starshade? A starshade is a shaped mask flown in formation with a telescope to block starlight and image faint exoplanets.

Complex electric fields are propagated at three planes (starshade, telescope aperture and focal plane) using Fresnel or Fraunhofer diffraction, computed with Bluesteins FFT. `PyStarshade` allows for simulating imaging for a discretized exoplanetary scene of flux, varying starshade mask and telescope aperture mask (interfacing with HCIPy to generate mission telescope apertures).

Developed by Jamila Taaki (MIDAS).

Installation
--------

.. code-block:: bash

    pip install pystarshade

Usage
--------

The simplest way to use PyStarshade is by calling the function 'source_field_to_ccd', this function
takes as input a 2D source-field of size (N_s, N_s) and spatial sampling ds and returns the 2D output
field incident on a CCD of size (N_pix, N_pix) and pixel size dp. 

.. code-block:: python

    from pystarshade.simulate_field import source_field_to_ccd

    source_field_to_ccd(source_field, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
                        N_s = 333, N_x = 6401, N_t = 1001, N_pix = 4001, 
                        ds = 0.3*au_to_meter, dx = 0.01, dt = 0.0116, dp=.5*1.9e-7)```

Input data
--------

PyStarshade can take as input any pixelized source-field such as Haystacks model, or analytic descriptions of sources
(so far a point source and Gaussian source). If you wish to perform propagation using analytic descriptions, please 
use 'pystarshade.simulate_field.point_source_to_ccd'. 

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <content/fft>`_
documentation for details.

Contents
--------

.. toctree::
    content/fft
    content/test

