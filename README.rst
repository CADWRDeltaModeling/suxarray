========
Suxarray
========

``suxarray`` is an extension of uxarray for SCHISM grids.


SCHISM grid
-----------

The SCHISM grid is a horizontal unstructured grid with a layered vertical grid. It uses a mixture of triangular and quadrilateral horizontal elements. The vertical grid can be a hybrid sigma and z coordinate system (SZ grid) or a variable number of sigma layers (LSC2). The SCHISM grid is described in the SCHISM manual (https://schism-dev.github.io/schism/master/index.html). The SCHISM uses `UGRID convention <https://ugrid-conventions.github.io/ugrid-conventions>`_ in its NetCDF output files.


Goals
-----

``suxarray`` is intended to use ``xarray`` through ``uxarray`` while keeping and handling SCHISM grid-specific information. Also, it will provide useful operations for SCHISM grid data.

.. note::
    ```suxarray``` is actively being developed, and the API may change frequently.


Installation
------------

Please see the installation section in the documentation at https://cadwrdeltamodeling.github.io/suxarray/main/installation.html.


Documentation
-------------
The full documentation is at GitHub https://cadwrdeltamodeling.github.io/suxarray/.


License
-------

``suxarray`` is licensed under the MIT license. See the LICENSE file for more information.