from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from shapely.geometry import Polygon
import suxarray as sx
from .testfixtures import *


def test_open_grid():
    """Test opening a SCHISM grid i3nfo from a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    ds = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"))
    grid = sx.open_grid(ds)
    assert grid.n_node == 2639
    assert grid.n_face == 4636


def test_open_schism_nc_without_zcoords():
    """Test opening a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    sxds = sx.open_schism_nc(
        str(p_cur / "testdata/out2d_1.nc"), str(p_cur / "testdata/out2d_1.nc")
    )
    assert sxds.sxgrid.n_node == 2639
    assert sxds.sxgrid.n_face == 4636


def test_open_schism_nc_with_zcoords():
    """Test opening a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    data_paths = ["testdata/zCoordinates_1.nc", "testdata/salinity_1.nc"]
    data_paths = [str(p_cur / path) for path in data_paths]
    sxds = sx.open_schism_nc(str(p_cur / "testdata/out2d_1.nc"), data_paths)
    assert sxds.sxgrid.n_node == 2639
    assert sxds.sxgrid.n_face == 4636


@pytest.fixture
def grid_test():
    """Test mesh fixture"""
    p_cur = Path(__file__).parent.absolute()
    grid = sx.open_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    return grid


def test_read_hgrid_gr3():
    """Test read_hgrid_gr3"""
    # Test with a HelloSCHISM v5.10 hgrid.gr3 file
    p_cur = Path(__file__).parent.absolute()
    grid = sx.core.api.open_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    assert grid.n_node == 112
    assert grid.n_face == 135
    # assert grid.ds.dims['nSCHISM_hgrid_edge'] == 10416
    # assert grid.ds.dims['nSCHISM_hgrid_max_face_nodes'] == 3
    # assert grid.ds.dims['nSCHISM_hgrid_max_edge_nodes'] == 2


def test_find_element_at_position(grid_test):
    """Test find_element_at"""
    # When a point is inside an element
    coords = (2.0, 1.0)
    tree = grid_test.get_strtree()
    from shapely.geometry import Point

    elem_ind = tree.query(Point(coords), predicate="intersects")
    assert np.all(elem_ind == np.array([123]))
    # # When a point is on a boundary of two elements
    # elem_ind = grid_test.find_element_at(0.0, 0.0)
    # assert np.all(elem_ind == np.array([39, 123]))


def test_subset_bounding_box_xy(sxds_test_dask):
    """Test find_element_at"""
    # When a point is inside an element
    # elem_ind = grid_test_dask.find_element_at(1.0, -9999.0)
    # assert np.all(elem_ind == np.array([372]))
    # # When a point is on a boundary of two elements
    # elem_ind = grid_test_dask.find_element_at(56000.0, -10350.0)
    # elem_ind.sort()
    # assert np.all(elem_ind == np.array([0, 2, 3722]))
    da_subset = sxds_test_dask["salinity"].subset.bounding_box_xy(
        [0.0, 1000.0], [0.0, 10400.0]
    )
    assert da_subset.sxgrid.n_node == 21


def test_depth_average(sxds_test_dask):
    da = sxds_test_dask["salinity"].depth_average()

    assert da.sel(n_node=492).values[0] == pytest.approx(0.145977, abs=1e-6)
