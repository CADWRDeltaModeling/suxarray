=======
History
=======

v2026.11.0 (2026-11)
--------------------
This version migrates to ``uxarray`` v2026.10.1 and adds water-column integration, lazy coordinate handling, and performance improvements.

Added
^^^^^
* Added water-column prism volume, ocean mass, and generic nodal variable integration functions with Dask parallelization support.
* Added lazy face and edge z-coordinate calculations and graceful handling when ``zCoordinates`` is absent.
* Added example postprocessing scripts, ``pytest-cov`` coverage workflow, and JUnit CI checks.

Changed
^^^^^^^
* Migrated to ``uxarray`` v2026.10.1 (Python >= 3.12) with native planar and spherical coordinate support.
* Preferred ``h5netcdf`` engine across the package and preserved native dtypes for SCHISM outputs.
* Optimized and hardened depth-averaging and Dask task graph performance.

Fixed
^^^^^
* Prevented NaN propagation issues in prism layer integration.

v2025.6.0 (2025-08)
-------------------
This version adopts ``uxarray`` v2025.6.0.

Changed
^^^^^^^
* Adopted ``uxarray`` v2025.6.0. This changes the internals of ``suxarray``, but it should not affect the user-facing APIs.
* Use netcdf4 engine by default.
* Changed the versioning scheme to use the year and month of the release date, following ``uxarray``.

v2024.09.0 (2024-09)
--------------------
This version is a major release with many changes following the changes in ``uxarray`` since the last year. So, the new version does not keep the backward compatibility.

Changed
^^^^^^^
* Changed the versioning scheme to use the year and month of the release date, following ``uxarray``.
* Changed the internal structures and APIs following the changes in ``uxarray``.

Added
^^^^^
* Adopted a versioned documentation.


0.1.3 (2023-05-30)
------------------
Fixed
^^^^^
* Add a missing import in ``__init__.py``

0.1.2 (2023-05-24)
------------------
Remove some dependency version limits. No functional changes.

0.1.1 (2023-05-24)
------------------
First release trial on conda-forge. No functional changes.

0.1.0 (2023-05-24)
------------------
First release on PyPI.

Added
^^^^^
* Add the main codes
