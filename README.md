# PyStarshade

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Developed by Jamila Taaki (UIUC).

PyStarshade is a Python library for Starshade (or any external occulter) simulations from star-planet system to CCD with Fresnel diffraction methods. This library efficiently calculates output fields using Bluestein FFTs.

What is a starshade? A starshade is a particular apodization (mask), which flown at Fresnel distances from a telescope achieves star-light suppression for imaging exoplanets in orbit around the star. 

What is a Bluestein FFT? The Bluestein Fast Fourier Transform (1968) is an algorithm that computes M equispaced samples of arbitrary size df, of the Discrete-Time Fourier Transform (DTFT) over an arbitrary frequency region between [0, 1/dx] for a compact input signal containing N non-zero samples, each with a size of dx. The computational complexity of this method in one dimension is O((N+M)log(N+M)). The Bluestein FFT is particularly advantageous when large zero-padding factors would be needed for performing optical propagation using FFT's.

This means that end-to-end simulation can be performed with arbitrary high-resolution sampling in each plane of propagation. 

This library is compatible with Python 3.6 and later versions. 


## Example
Log starlight supression with a truncated Hypergaussian apodization, sweeping star planet brightness ratios between (10e-8, 10e-3). Planet at a 2 AU separation and 10 pc distance from Earth. 
<p align="center">
  <img src="images/contrast_.gif" alt="Star planet brightness ratio range (10e-8, 10e-3)">
</p>

## Installation

You can install PyStarshade using pip:

```bash
pip install pystarshade
```

## Dependencies

Scipy, Numpy

## Quickstart
Detailed documentation for all PyStarshade utilities can be found within the code's docstrings.

### Use
The simplest way to use PyStarshade is by calling the function 'source_field_to_ccd', this function
takes as input a 2D source-field of size (N_s, N_s) and spatial sampling ds and returns the 2D output
field incident on a CCD of size (N_pix, N_pix) and pixel size dp. 

```bash
from pystarshade.simulate_field import source_field_to_ccd

source_field_to_ccd(source_field, wl, dist_xo_ss, dist_ss_t, focal_length_lens, radius_lens, 
                    N_s = 333, N_x = 6401, N_t = 1001, N_pix = 4001, 
                    ds = 0.3*au_to_meter, dx = 0.01, dt = 0.0116, dp=.5*1.9e-7)
```

### Input data

PyStarshade can take as input any pixelized source-field such as Haystacks model, or analytic descriptions of sources
(so far a point source and Gaussian source). If you wish to perform propagation using analytic descriptions, please 
use 'pystarshade.simulate_field.point_source_to_ccd'. 

### Parameters
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


### Worked examples

See examples folder for different simulation examples.

## Organization

<pre>
Pystarshade
├── examples
│   ├── haystacks_model.py
│   └── star_exo.py
├── images
│   └── contrast_.gif
├── pystarshade
│   ├── apodization
│   │   ├── apodize.py
│   │   └── __init__.py
│   ├── diffraction
│   │   ├── bluestein_fft.py
│   │   ├── diffract.py
│   │   ├── field.py
│   │   ├── __init__.py
│   │   └── util.py
│   ├── __init__.py
│   ├── simulate_field.py
│   └── version.py
├── README.md
└── setup.py

</pre>

## License

[PyStarshade] is released under the [GNU General Public License v3.0](LICENSE).
