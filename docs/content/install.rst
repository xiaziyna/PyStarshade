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


Testing
^^^^^^^

To run the tests, navigate to the root directory of the **pystarshade** repository and execute:

.. code-block:: bash

   pytest tests/

This will automatically discover and run all unit tests in the `tests/` directory.

If all tests pass, your installation is working correctly. If a test fails - please open an issue!
