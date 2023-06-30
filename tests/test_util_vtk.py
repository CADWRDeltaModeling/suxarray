""" Test related to VTK routines
"""

from shapely.geometry import Polygon
import suxarray.util_vtk
from .testfixtures import *


def test_vtk_grid(grid_test_dask, tmp_path):
    polygon = Polygon(([55830, -10401], [56001, -10401],
                      [56001, -10240], [55830, -10240]))
    grid_sub = grid_test_dask.subset(polygon)
    vtk_grid = suxarray.util_vtk.create_vtk_grid(grid_sub)
    # Just create a file for now, no actual tests with the output
    suxarray.util_vtk.write_vtk_grid(vtk_grid, tmp_path / 'test.vtu')
