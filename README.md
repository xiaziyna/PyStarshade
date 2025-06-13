# PyStarshade

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Developed by Jamila Taaki (MIDAS fellow).

PyStarshade is a Python library for Starshade (or any external occulter) simulations from star-planet system to CCD with Fresnel diffraction methods. This library efficiently calculates output fields using Bluestein FFTs.

What is a starshade? A starshade is a particular apodization (mask), which flown at Fresnel distances from a telescope achieves star-light suppression for imaging exoplanets in orbit around the star. 

Numerical diffraction calculations for a starshade must use a very small numerical resolution $d u$ of the starshade $s(u, v)$ in order to accurately calculate starlight suppression. Using a standard FFT to perform these calculations is inefficient as very large zero-padding factors are needed to sample the field at the telescope aperture. The Bluestein FFT is a technique to calculate arbitrary spectral samples of a propagated field, indirectly using FFTs and therefore benefiting from their efficiency. For an $N \cdot N$ starshade mask, and an $M \cdot M$ telescope aperture, the Bluestein FFT approach achieves a complexity of $O((N+M)^2 \log (M+N))$. This technique is utilized in multiple aspects of the optical train to efficiently propagate fields.


## Example
Simulated imaging of a synthetic exoscene (ExoVista) with three visible exoplanets at a
wavelength of 500 nm. A 60 m starshade configuration and a 6m segmented pupil was used for this
example.
<p align="center">
  <img src="images/exo_scene.png" alt="Three planets imaged with a HWO concept starshade and a 6m hexagonal segmented pupil." width="50%">
</p>

## Installation

For a barebones install, use pip:

```bash
pip install pystarshade
```

If you want to use pre-generated data instead install the package from source, in editable mode and use [git lfs](https://git-lfs.com).
This requires several GB of space:

```bash
$ git clone https://github.com/xiaziyna/PyStarshade.git PyStarshade
$ cd PyStarshade
$ git lfs pull
pip install -e .
```

## Dependencies

Scipy, Numpy, HCIPy, astropy, setuptools, pytest

## Quickstart

Detailed [documentation](https://pystarshade.readthedocs.io/en/latest/) for all PyStarshade utilities.

### Input data

PyStarshade can take as input any pixelized source-field such as Haystacks model, or analytic descriptions of sources
(so far a point source and Gaussian source). If you wish to perform propagation using analytic descriptions, please 
use 'pystarshade.simulate_field.point_source_to_ccd'.

The easiest way to interface with PyStarshade is via the StarshadeProp class. Generate fields/psf models for a chosen design reference mission (drm).
Simulate imaging for a 'source_field' with a default 2 mas sampling.

```python
import numpy as np
from pystarshade.propagator import StarshadeProp
import matplotlib.pyplot as plt

# 1. Initialize the Starshade Propagator with HWO (Habitable Worlds Observatory) configuration
starshade = StarshadeProp(drm='hwo')  # drm = design reference mission

# 2. Generate the pupil field (this handles the Fresnel diffraction from starshade to telescope)
starshade.gen_pupil_field()

# 3. Generate PSF basis for a hexagonal pupil (this handles the Fraunhofer diffraction to focal plane)
pupil_type = 'hex'
starshade.gen_psf_basis(pupil_type=pupil_type)

# 4. Create a simple point source (representing a star)
# Using 2 mas (milliarcsecond) sampling as default
nx = ny = 100  # image size
source_field = np.zeros((nx, ny), dtype=np.float32)
source_field[nx//2, ny//2] = 1.0  # place a point source in the center

# 5. Generate the final image
wavelength = 500e-9  # 500 nm
focal_intensity = starshade.gen_scene(pupil_type, source_field, wavelength)

# 6. Visualize the result
plt.figure(figsize=(8, 8))
plt.imshow(focal_intensity, norm=plt.LogNorm())
plt.colorbar()
plt.title('Simulated Starshade Image')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()
```


### Worked examples

See examples folder for different simulation examples.

## Contributing

Feel free to reach out if you'd like to discuss contributing or go ahead and submit a pull request!
Try to keep any pull requests limited in scope. 
If there is demand for extra functionality, I am happy to help add these in.
See [here](https://pystarshade.readthedocs.io/en/latest/content/contribute.html) for further instructions.

## License

[PyStarshade] is released under the [GNU General Public License v3.0](LICENSE).
