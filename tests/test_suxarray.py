from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from shapely.geometry import Polygon
import suxarray.suxarray as sx
from .testfixtures import *


def test_suxarray_init_with_out2d():
    """ Test suxarray initialization with a SCHISM out2d file """
    # Test with a HelloSCHISM v5.10 out2d file
    p_cur = Path(__file__).parent.absolute()
    ds = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"),
                         mask_and_scale=False)
    ds = sx.coerce_mesh_name(ds)
    grid = sx.Grid(ds)
    assert grid.mesh_type == 'ugrid'
    assert grid.nMesh2_node == 2639


def test_get_topology_variable():
    """ Test get_topology_variable """
    # Test with a HelloSCHISM v5.10 out2d file
    p_cur = Path(__file__).parent.absolute()
    ds = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"))
    da = sx.get_topology_variable(ds)
    assert da.name == 'SCHISM_hgrid'


@pytest.fixture
def grid_test():
    """ Test mesh fixture """
    p_cur = Path(__file__).parent.absolute()
    grid = sx.read_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    return grid


def test_triangulate(grid_test):
    """ Test triangulate """
    grid_tri = sx.triangulate(grid_test)
    assert grid_tri.nMesh2_node == 112
    assert grid_tri.nMesh2_face == 168


def test_triangulate_dask(grid_test_dask):
    """ Test triangulate_dask """
    grid_tri = sx.triangulate(grid_test_dask)
    assert grid_tri.nMesh2_node == 2639
    assert grid_tri.nMesh2_face == 4636


def test_read_hgrid_gr3():
    """ Test read_hgrid_gr3 """
    # Test with a HelloSCHISM v5.10 hgrid.gr3 file
    p_cur = Path(__file__).parent.absolute()
    grid = sx.read_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    assert grid.mesh_type == 'ugrid'
    assert grid.nMesh2_node == 112
    assert grid.nMesh2_face == 135
    # assert grid.ds.dims['nSCHISM_hgrid_edge'] == 10416
    # assert grid.ds.dims['nSCHISM_hgrid_max_face_nodes'] == 3
    # assert grid.ds.dims['nSCHISM_hgrid_max_edge_nodes'] == 2


def test_find_element_at_position(grid_test):
    """ Test find_element_at """
    # When a point is inside an element
    elem_ind = grid_test.find_element_at(2., 1.)
    assert np.all(elem_ind == np.array([123]))
    # When a point is on a boundary of two elements
    elem_ind = grid_test.find_element_at(0., 0.)
    assert np.all(elem_ind == np.array([39, 123]))


def test_find_element_at_position_dask(grid_test_dask):
    """ Test find_element_at """
    # When a point is inside an element
    elem_ind = grid_test_dask.find_element_at(1., -9999.)
    assert np.all(elem_ind == np.array([372]))
    # When a point is on a boundary of two elements
    elem_ind = grid_test_dask.find_element_at(56000., -10350.)
    elem_ind.sort()
    assert np.all(elem_ind == np.array([0, 2, 3722]))


def test_subset(grid_test_dask):
    polygon = Polygon(([55830, -10401], [56001, -10401],
                      [56001, -10240], [55830, -10240]))
    grid_sub = grid_test_dask.subset(polygon)
    assert grid_sub.nMesh2_face == 6
    assert grid_sub.Mesh2_face_nodes.values[0, -1] == \
        grid_sub.Mesh2_face_nodes.attrs['_FillValue']


def test_depth_average(grid_test_dask):
    ds = sx.Dataset(grid_test_dask._ds)
    da = ds.depth_average('salinity')

    assert da.sel(nMesh2_node=492).values[0] == pytest.approx(
        0.145977, abs=1e-6)
