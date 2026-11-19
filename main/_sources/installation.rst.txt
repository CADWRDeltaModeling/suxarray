.. highlight:: shell

============
Installation
============


Stable release
--------------

The stable version may be slow to use the latest features and bug fixes. If you want to use the latest features and bug fixes, you can install the development version from the source, following the instruction in the next section.

If you use conda, run the following to install ``suxarray``:

.. code-block:: console

    $ conda install -c conda-forge suxarray

To install suxarray using ``pip``, run the command below in your terminal:

.. code-block:: console

    $ pip install suxarray

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From the source
---------------

``suxarray`` requires Python >= 3.12 and ``uxarray==2026.10.1`` (along with ``h5netcdf[h5py]``, ``shapely>=2.0``, ``gsw>=3.6``, ``pyproj``, and ``Click>=7.0``).

The source code for ``suxarray`` can be downloaded from the `Github repo`_. Check out the branch you want to use and install it in editable mode with ``pip``:

.. code-block:: console

    $ git clone https://github.com/cadwrdeltamodeling/suxarray.git
    $ cd suxarray
    $ pip install -e .

To include optional VTK export support, install the ``vtk`` extra:

.. code-block:: console

    $ pip install -e ".[vtk]"

.. _Github repo: https://github.com/cadwrdeltamodeling/suxarray
.. _tarball: https://github.com/cadwrdeltamodeling/suxarray/tarball/main
