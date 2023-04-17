# PyStarshade

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Developed by Jamila Taaki (UIUC).

PyStarshade is a Python library for computing end-to-end Starshade simulations with Fresnel diffraction methods. This library efficiently calculates output fields using Bluestein FFTs.

What is a Bluestein FFT?
A Bluestein FFT computes M equispaced samples of a DTFT over any arbitrary frequency region between [0, 1/dx] of a compact input signal consisting of N non-zero samples of size dx. The complexity of this method in 1D is O((N+M)log(N+M)). This method is beneficial when large zero-padding factors would be required to perform this calculation with a direct FFT. 

This library is compatible with Python 3.6 and later versions. 

## Installation

You can install PyStarshade using pip:

```bash
pip install pystarshade
```

## Dependencies

Scipy, Numpy

## Quickstart
See simulate_field.py

Log starlight supression with a truncated Hypergaussian apodization, sweeping star planet brightness ratios between (10e-8, 10e-3). Planet at a 0.2 au separation and 10 pc distance from Earth. 
<p align="center">
  <img src="images/contrast_.gif" alt="Star planet brightness ratio range (10e-8, 10e-3)">
</p>


## License

[PyStarshade] is released under the [GNU General Public License v3.0](LICENSE).
