.. PyStarshade documentation master file, created by
   sphinx-quickstart on Fri Dec  6 09:40:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyStarshade documentation
=========================

`PyStarshade` is a tool for simulating high-contrast imaging of exoplanets with starshades. 

What is a starshade? A starshade is a shaped mask flown in formation with a telescope to block starlight and image faint exoplanets.

Complex electric fields are propagated at three planes (starshade, telescope aperture and focal plane) using Fresnel or Fraunhofer diffraction, computed with Bluesteins FFT. `PyStarshade` allows for simulating imaging for a discretized exoplanetary scene of flux, varying starshade mask and telescope aperture mask (interfacing with HCIPy to generate mission telescope apertures).

Developed by Jamila Taaki (U Mich MIDAS postdoctoral fellow).


Installation
------------

For a barebones install, use pip:

.. code-block:: bash

    pip install pystarshade

If you want to use the examples, including starshade masks, telescope apertures and other pre-computed data, it is recommended you do NOT use pip. 
Instead install the package from source, in editable mode and use `git lfs <https://git-lfs.com>`_:

.. warning::

    Downloading the pre-computed data requires several gigabytes of disk space. Ensure you have sufficient storage available before proceeding.

.. code-block:: bash

    git clone https://github.com/xiaziyna/PyStarshade.git PyStarshade
    cd PyStarshade
    git lfs pull
    pip install -e .


Contents
--------

.. toctree::
    content/background
    content/usage
    content/solar_system.ipynb

API
--------

.. toctree::
    content/test

License and Citation
--------------------

``PyStarshade`` is licensed under the GNU GPL V3. If you use this tool in your research, please cite:

    Taaki, Kamalabadi, Kemball. PyStarshade: simulating high-contrast imaging of exoplanets with starshades

Contact
-------

For questions or bug reports, contact Jamila Taaki at tjamila@umich.edu or open an issue on the `GitHub repository <https://github.com/xiaziyna/PyStarshade>`_.

Contributions are welcome!

