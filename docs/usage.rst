=====
Usage
=====

To use ``suxarray`` in a project:

.. code-block:: python

    import suxarray as sx

Reading SCHISM Grids and Outputs
--------------------------------

``suxarray`` reads SCHISM UGRID NetCDF outputs (``out2d_*.nc`` and optional ``zCoordinates_*.nc``) as well as ``hgrid.gr3`` files, preferring the ``h5netcdf`` engine when available (falling back to ``netcdf4``):

.. code-block:: python

    # Open a 3D grid (with zCoordinates) or a 2D-only grid (zCoordinates omitted)
    grid = sx.open_grid("out2d_1.nc", "zCoordinates_1.nc")
    grid_2d = sx.open_grid("out2d_1.nc")

    # Open a time-series dataset and bind it to the grid
    ds_sal = sx.core.api.open_ts_dataset("salinity_1.nc")
    sxds = sx.read_schism_nc(grid, ds_sal)

Planar and Spherical Coordinates
--------------------------------

``suxarray.Grid`` automatically detects whether the mesh uses planar (Cartesian) or spherical (geographic) coordinates via ``grid.is_planar``:

- For planar grids, ``grid.face_areas`` and ``SxDataArray.integrate()`` compute exact planar polygon areas and integrals.
- For spherical grids, ``grid.face_areas``, ``SxDataArray.topological_mean()``, and ``SxDataArray.integrate()`` delegate to ``uxarray``'s spherical routines.

Vertical Operations and Prism Integration
-----------------------------------------

``SxDataArray.depth_average()`` computes depth-averaged values at nodes while preserving floating-point dtypes.

For 3D water-column integration and prism volumes (single-timestep inputs), ``suxarray.grid.integrate`` provides:

- ``integrate_prism_variable(sxda, return_per_layer=False)``
- ``calc_volumes_from_projected(sxda)``
- ``projected_volume_ufunc_kernel(...)`` for Dask-parallelized ``xarray.apply_ufunc`` workflows

Spatial Subsetting
------------------

Both ``Grid`` and ``SxDataArray`` support spatial subsetting via ``.subset.bounding_polygon(...)`` and ``.subset.bounding_box_xy(...)``.

Example Scripts
---------------

Example postprocessing CLI scripts are available in the ``scripts/`` directory of the repository (see ``scripts/README.md``), including:

- ``calc_da_daily.py`` and ``calc_da_multi_days.py`` for single-day and multi-day parallel depth-averaging
- ``calc_lsz_daily.py``, ``calc_lsz_temp_daily.py``, and ``calc_lsz_temp_turbidity.py`` for daily low-salinity-zone (LSZ) area calculations

Example Notebooks
-----------------

.. toctree::
    :maxdepth: 2

    notebooks/example_read_schism_nc
    notebooks/example_subset
