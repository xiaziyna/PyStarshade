# PyStarshade

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

PyStarshade is a Python library for computing end-to-end Starshade simulations with Fresnel diffraction. This library is compatible with Python 3.6 and later versions.

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
