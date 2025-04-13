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

You can install PyStarshade using pip:

.. code-block:: bash

    pip install pystarshade

Or to use pre-computed data in the examples, use `git lfs <https://git-lfs.com>`_:

.. warning::

    Downloading the pre-computed data requires several gigabytes of disk space. Ensure you have sufficient storage available before proceeding.

.. code-block:: bash

    git clone https://github.com/xiaziyna/PyStarshade.git PyStarshade
    cd PyStarshade
    git lfs pull
    pip install -e .

License and Citation
--------------------

``PyStarshade`` is licensed under the MIT License. If you use this tool in your research, please cite:

    Taaki, Kamalabadi, Kemball. "PyStarshade: simulating high-contrast imaging of exoplanets with starshades"

Contact
-------

For questions or bug reports, contact Jamila Taaki at tjamila@umich.edu or open an issue on the `GitHub repository <https://github.com/xiaziyna/PyStarshade>`_.

Contributions are welcome!


Contents
--------

.. toctree::
    content/usage
    content/fft

API
--------

.. toctree::
    content/test

