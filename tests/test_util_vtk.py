"""Test related to VTK routines"""

from shapely.geometry import Polygon
import suxarray.util_vtk
from .testfixtures import *


def test_vtk_grid(sxds_test_dask, tmp_path):
    polygon = Polygon(
        [
            (569866.8953279585, 4213065.012236144),
            (570131.3752643355, 4213065.012236144),
            (570131.3752643355, 4213262.690668024),
            (569866.8953279585, 4213262.690668024),
        ]
    )
    da_subset = sxds_test_dask["salinity"].subset.bounding_polygon(polygon)
    vtk_grid = suxarray.util_vtk.create_vtk_grid(da_subset.uxgrid)
    # Just create a file for now, no actual tests with the output
    suxarray.util_vtk.write_vtk_grid(vtk_grid, tmp_path / "test.vtu")
