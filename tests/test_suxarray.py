from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from shapely.geometry import Polygon, Point
import suxarray as sx
from .testfixtures import *


def test_read_grid():
    """Test opening a SCHISM grid i3nfo from a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"))
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"))
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    assert grid.n_node == 2639
    assert grid.n_face == 4636


def test_read_schism_nc():
    """Test opening a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"))
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"))
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    ds_data = xr.open_dataset(str(p_cur / "testdata/salinity_1.nc"))
    sxds = sx.read_schism_nc(grid, ds_data)
    assert sxds.sxgrid.n_node == 2639
    assert sxds.sxgrid.n_face == 4636
    assert sxds.time.size == 72


def test_read_hgrid_gr3():
    """Test read_hgrid_gr3"""
    # Test with a HelloSCHISM v5.10 hgrid.gr3 file
    p_cur = Path(__file__).parent.absolute()
    grid = sx.open_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    assert grid.n_node == 112
    assert grid.n_face == 135


def test_find_element_at_position(grid_test):
    """Test find_element_at_position"""
    # When a point is inside an element
    coords = (2.0, 1.0)
    tree = grid_test.get_strtree(elements="faces")

    elem_ind = tree.query(Point(coords), predicate="intersects")
    assert elem_ind[0] == 123

    elem_ind = grid_test.intersect(Point(coords))
    assert elem_ind[0] == 123

    # When a point is on a boundary of two elements
    elem_ind = grid_test.intersect(Point([0.0, 0.0]))
    assert np.all(elem_ind == np.array([39, 123]))


def test_find_nearest_node(grid_test):
    """Test find_nearest_node"""
    # When a point is inside an element
    coords = (2.0, 1.0)
    tree = grid_test.get_strtree(elements="nodes")
    node_ind = tree.nearest(Point(coords))
    assert node_ind == 52


def test_subset_bounding_polygon(sxds_test_dask):
    """Test subsetting dataarray with a polygon"""
    polygon = Polygon([(0.0, 0.0), (1000.0, 0.0), (1000.0, 10400.0), (0.0, 10400.0)])
    da_subset = sxds_test_dask["salinity"].subset.bounding_polygon(polygon)
    assert da_subset.uxgrid.n_node == 21


def test_subset_bounding_box(sxds_test_dask):
    """Test find_element_at"""
    da_subset = sxds_test_dask["salinity"].subset.bounding_box_xy(
        [0.0, 1000.0], [0.0, 10400.0]
    )
    assert da_subset.uxgrid.n_node == 21


def test_isel(sxds_test_dask):
    da = sxds_test_dask["salinity"].isel(time=range(2))
    assert da.uxgrid.n_node == 2639
    assert da.uxgrid.sgrid_info.time.size == 2
    # Slice again
    da = sxds_test_dask["salinity"].isel(time=range(4))
    assert da.uxgrid.sgrid_info.time.size == 4


def test_depth_average(sxds_test_dask):
    da = sxds_test_dask["salinity"].depth_average()
    assert da.sel(n_node=492).values[0] == pytest.approx(0.145977, abs=1e-6)


def test_slice_and_depth_average(sxds_test_dask):
    polygon = Polygon([(0.0, 0.0), (1000.0, 0.0), (1000.0, 10400.0), (0.0, 10400.0)])
    da_subset = sxds_test_dask["salinity"].subset.bounding_polygon(polygon)
    da_da = da_subset.depth_average()
    # Node 492 is now 1
    assert da_da.sel(n_node=1).values[0] == pytest.approx(0.145977, abs=1e-6)


# def test_interpolate_xy(sxds_test_dask):
#     da = sxds_test_dask["salinity"]
#     da.interpolate_xy(1.0, 1.0)
#     assert True
