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

``PyStarshade`` is an open-source tool for simulating high-contrast imaging of exoplanets using starshades. It is licensed under the GNU General Public License version 3 (GPLv3).

If you use ``PyStarshade`` in your research, please include the following citation:

    Taaki, Kamalabadi, Kemball. *PyStarshade: Simulating High-Contrast Imaging of Exoplanets with Starshades*.


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   PyStarshade Documentation Landing Page <https://pystarshade.readthedocs.io>
   content/install       
   content/background   
   content/usage        

.. toctree::
   :maxdepth: 1

   content/solar_system
   content/test
   content/contribute

