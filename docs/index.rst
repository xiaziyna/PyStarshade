.. PyStarshade documentation master file, created by
   sphinx-quickstart on Fri Dec  6 09:40:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyStarshade documentation
=========================

`PyStarshade` is a tool for simulating high-contrast imaging of exoplanets with starshades. 

What is a starshade? A starshade is a shaped mask flown in formation with a telescope to block starlight and image faint exoplanets.

Complex electric fields are propagated at three planes (starshade, telescope aperture and focal plane) using Fresnel or Fraunhofer diffraction, computed with Bluesteins FFT. `PyStarshade` allows for simulating imaging for a discretized exoplanetary scene of flux, varying starshade mask and telescope aperture mask (interfacing with HCIPy to generate mission telescope apertures).

Developed by Jamila Taaki (U Mich Schmidt postdoctoral fellow).

License and Citation
--------------------

``PyStarshade`` is licensed under the GNU GPL V3. If you use this tool in your research, please cite:

    Taaki, Kamalabadi, Kemball. PyStarshade: simulating high-contrast imaging of exoplanets with starshades


Table of Contents
-----------------

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    PyStarshade Documentation Landing Page <https://pystarshade.readthedocs.io>
    content/install       Installation Guide
    content/background    Background and Theory
    content/usage         Usage Instructions

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   content/solar_system   Simulating the solar system (Jupyter notebook)

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   content/test           Core classes & functions reference

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   content/contribute     How to report issues, request features, or submit code

