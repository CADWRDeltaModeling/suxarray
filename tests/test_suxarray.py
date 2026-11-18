from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from shapely.geometry import Polygon, Point
import suxarray as sx
from suxarray.core.api import _choose_netcdf_engine
from .testfixtures import *


def test_read_grid():
    """Test opening a SCHISM grid i3nfo from a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    engine = _choose_netcdf_engine()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"), engine=engine)
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"), engine=engine)
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    assert grid.n_node == 823
    assert grid.n_face == 791


def test_read_schism_nc():
    """Test opening a SCHISM netCDF file"""
    p_cur = Path(__file__).parent.absolute()
    engine = _choose_netcdf_engine()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"), engine=engine)
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"), engine=engine)
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    ds_data = xr.open_dataset(str(p_cur / "testdata/salinity_1.nc"), engine=engine)
    sxds = sx.read_schism_nc(grid, ds_data)
    assert sxds.sxgrid.n_node == 823
    assert sxds.sxgrid.n_face == 791
    assert sxds.time.size == 48


def test_read_schism_nc_preserves_attrs():
    """
    Test conservation of dataset-level attrs in read_schism_nc.
    """
    p_cur = Path(__file__).parent.absolute()
    engine = _choose_netcdf_engine()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"), engine=engine)
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"), engine=engine)
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    ds_data = xr.open_dataset(str(p_cur / "testdata/salinity_1.nc"), engine=engine)

    # Attach sample global attributes to the source dataset.
    sample_attrs = {
        "Conventions": "CF-1.0",
        "title": "Bay-Delta SCHISM salinity output",
        "source": "SCHISM",
    }
    ds_data.attrs = sample_attrs

    sxds = sx.read_schism_nc(grid, ds_data)

    # The global attributes must survive the round trip.
    for key, value in sample_attrs.items():
        assert sxds.attrs.get(key) == value

    # The data variable must still be intact.
    assert "salinity" in sxds


def test_open_ts_dataset():
    """Test opening a time series dataset with open_ts_dataset."""
    p_cur = Path(__file__).parent.absolute()
    ds = sx.core.api.open_ts_dataset(str(p_cur / "testdata/salinity_1.nc"))
    assert "salinity" in ds
    assert ds["salinity"].dtype == np.float32
    assert np.issubdtype(ds["time"].dtype, np.datetime64)
    assert ds.time.size == 48


def test_read_schism_nc_preserves_native_dtype():
    """Test that read_schism_nc preserves native float32 dtype without where-masking."""
    p_cur = Path(__file__).parent.absolute()
    engine = _choose_netcdf_engine()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"), engine=engine)
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"), engine=engine)
    grid = sx.read_grid(ds_out2d, ds_zcoords)

    # Open with chunks to ensure Dask arrays are used
    ds_data = xr.open_dataset(
        str(p_cur / "testdata/salinity_1.nc"),
        engine=engine,
        chunks={"time": 12},
    )
    sxds = sx.read_schism_nc(grid, ds_data)

    assert sxds["salinity"].dtype == np.float32
    # Verify no where tasks were injected into the Dask graph
    dask_keys = [str(k) for k in sxds["salinity"].data.dask.keys()]
    assert not any("where" in k for k in dask_keys)


def test_open_grid_preserves_native_dtype():
    """Test that open_grid preserves native float32 dtype on zCoordinates without where-masking."""
    p_cur = Path(__file__).parent.absolute()
    path_out2d = str(p_cur / "testdata/out2d_1.nc")
    path_zcoords = str(p_cur / "testdata/zCoordinates_1.nc")
    chunks = {"time": 12}
    grid = sx.core.api.open_grid(path_out2d, path_zcoords, chunks=chunks)

    z = grid.sgrid_info.zCoordinates
    assert z.dtype == np.float32
    dask_keys = [str(k) for k in z.data.dask.keys()]
    assert not any("where" in k for k in dask_keys)


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
    # polygon = Polygon([(0.0, 0.0), (1000.0, 0.0),
    #                   (1000.0, 10400.0), (0.0, 10400.0)])
    polygon = Polygon(
        [
            (569866.8953279585, 4213065.012236144),
            (570131.3752643355, 4213065.012236144),
            (570131.3752643355, 4213262.690668024),
            (569866.8953279585, 4213262.690668024),
        ]
    )
    da_subset = sxds_test_dask["salinity"].subset.bounding_polygon(polygon)
    assert da_subset.uxgrid.n_node == 15
    assert da_subset.uxgrid.n_face == 10


def test_subset_elements_bounding_polygon(sxds_element_test):
    """Test subsetting dataarray with data at elements with a polygon."""
    grid = sxds_element_test.uxgrid

    lon_bounds = (grid.node_lon.min().values, grid.node_lon.max().values)
    lat_bounds = (grid.node_lat.min().values, 0)

    vertVel = sxds_element_test["verticalVelAtElement"]

    da_subset = vertVel.subset.bounding_box(lon_bounds, lat_bounds, [['face']])
    assert da_subset.uxgrid.n_node == 1319
    assert da_subset.uxgrid.n_face == 2221


def test_subset_bounding_box(sxds_test_dask):
    """Test find_element_at"""
    da_subset = sxds_test_dask["salinity"].subset.bounding_box_xy(
        [569866.8953279585, 570131.3752643355],
        [4213065.012236144, 4213262.690668024]
    )
    assert da_subset.uxgrid.n_node == 15
    assert da_subset.uxgrid.n_face == 10


def test_isel(sxds_test_dask):
    da = sxds_test_dask["salinity"].isel(time=range(2))
    assert da.uxgrid.n_node == 823
    assert da.uxgrid.sgrid_info.time.size == 2
    # Slice again
    da = sxds_test_dask["salinity"].isel(time=range(4))
    assert da.uxgrid.sgrid_info.time.size == 4


def test_depth_average(sxds_test_dask):
    assert sxds_test_dask["salinity"].dtype == np.float32
    da = sxds_test_dask["salinity"].depth_average()
    assert da.dtype == np.float32
    # The value is calculated from VisIt
    assert da.isel(n_node=439, ignore_grid=True).values[0] == pytest.approx(11.961535, abs=1e-5)


def test_slice_and_depth_average(sxds_test_dask):
    polygon = Polygon(
        [
            (569866.8953279585, 4213065.012236144),
            (570131.3752643355, 4213065.012236144),
            (570131.3752643355, 4213262.690668024),
            (569866.8953279585, 4213262.690668024),
        ]
    )
    da_subset = sxds_test_dask["salinity"].subset.bounding_polygon(polygon)
    assert da_subset.dtype == np.float32
    da_da = da_subset.depth_average()
    assert da_da.dtype == np.float32
    # Node 186 is now 7
    assert da_da.isel(n_node=7, ignore_grid=True).values[0] == pytest.approx(10.334445, abs=1e-5)


def test_depth_average_edge_cases():
    from suxarray.utils.computing import _depth_average

    # 1 time, 8 nodes, 4 layers
    v = np.array(
        [
            [
                [10.0, 20.0, 30.0, 40.0],  # node 0: normal, k=1 -> avg of [20,30,40] over [-2,-1,0] = 30.0
                [10.0, 20.0, 30.0, 40.0],  # node 1: dry=1.0 -> NaN
                [10.0, 20.0, 30.0, 40.0],  # node 2: dry=NaN -> NaN
                [10.0, 20.0, 30.0, 40.0],  # node 3: k=-1 (out of range) -> NaN
                [10.0, 20.0, 30.0, 40.0],  # node 4: k=3 (n_layer - 1) -> NaN
                [10.0, 20.0, 30.0, 40.0],  # node 5: depth == 0.0 -> NaN
                [10.0, 20.0, 30.0, 40.0],  # node 6: depth < 0.0 -> NaN
                [10.0, 20.0, 30.0, 40.0],  # node 7: zs[k] is NaN -> NaN
            ]
        ],
        dtype=np.float32,
    )
    zs = np.array(
        [
            [
                [-3.0, -2.0, -1.0, 0.0],
                [-3.0, -2.0, -1.0, 0.0],
                [-3.0, -2.0, -1.0, 0.0],
                [-3.0, -2.0, -1.0, 0.0],
                [-3.0, -2.0, -1.0, 0.0],
                [-3.0, 0.0, 0.0, 0.0],
                [-3.0, 1.0, 0.5, 0.0],
                [-3.0, np.nan, -1.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    k = np.array([1, 1, 1, -1, 3, 1, 1, 1], dtype=np.int32)
    dry = np.array([[0.0, 1.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    res = _depth_average(v, zs, k, dry)
    assert res.shape == (1, 8)
    assert res.dtype == np.float32
    assert res[0, 0] == pytest.approx(30.0, abs=1e-5)
    assert np.all(np.isnan(res[0, 1:]))

    # Float k with NaN and integer v
    v_int = np.array([[[10, 20, 30, 40], [10, 20, 30, 40]]], dtype=np.int32)
    k_float = np.array([1.0, np.nan], dtype=np.float32)
    res_int = _depth_average(v_int, zs[:, :2], k_float, dry[:, :2])
    assert np.issubdtype(res_int.dtype, np.floating)
    assert res_int[0, 0] == pytest.approx(30.0, abs=1e-5)
    assert np.isnan(res_int[0, 1])


def test_depth_average_dimensions(sxds_test_dask):
    from suxarray.utils.computing import _depth_average

    # 1D and 2D direct kernel calls
    zs_1d = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    v_1d = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    res_1d = _depth_average(v_1d, zs_1d, np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32))
    assert res_1d.shape == ()
    assert float(res_1d) == pytest.approx(20.0, abs=1e-5)

    # SxDataArray sliced (2D) and transposed (3D)
    da_t0 = sxds_test_dask["salinity"].isel(time=0).depth_average()
    assert da_t0.dims == ("n_node",)
    assert da_t0.isel(n_node=439, ignore_grid=True).values == pytest.approx(11.961535, abs=1e-5)

    da_T = sxds_test_dask["salinity"].transpose("n_node", "time", "n_layer").depth_average()
    assert da_T.dims == ("n_node", "time")
    assert da_T.isel(n_node=439, time=0, ignore_grid=True).values == pytest.approx(11.961535, abs=1e-5)


# def test_interpolate_xy(sxds_test_dask):
#     da = sxds_test_dask["salinity"]
#     da.interpolate_xy(1.0, 1.0)
#     assert True
