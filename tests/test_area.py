import numpy as np
import xarray as xr
import pytest
import suxarray.grid.area
import suxarray.utils.computing
from .testfixtures import *


def test_integrate_nodal(sxds_test_dask):
    da = sxds_test_dask.salinity.depth_average()
    assert da.dtype == np.float32
    result = suxarray.grid.area._integrate_nodal(
        da.sxgrid.node_x.values,
        da.sxgrid.node_y.values,
        da.values,
        da.sxgrid.face_node_connectivity.values,
    )
    assert result.dtype == np.float32
    assert result[0, 64] == pytest.approx(41743.07605, abs=1e-1)

    result = xr.apply_ufunc(
        suxarray.grid.area._integrate_nodal,
        da.sxgrid.node_x,
        da.sxgrid.node_y,
        da,
        da.sxgrid.face_node_connectivity,
        input_core_dims=[
            [
                "n_node",
            ],
            [
                "n_node",
            ],
            [
                "n_node",
            ],
            [
                "n_face",
                "n_max_face_nodes",
            ],
        ],
        output_core_dims=[["n_face"]],
        output_dtypes=da.dtype,
        dask="parallelized",
    )
    assert result.dtype == np.float32
    assert result.isel(time=0, n_face=64).values[()] == pytest.approx(
        41743.07555287409, rel=1e-4
    )

    result = da.integrate()
    assert result.dtype == np.float32
    assert result.isel(time=0, n_face=64, ignore_grid=True).values[()] == pytest.approx(
        41743.07555287409, rel=1e-4
    )


def test_integrate_nodal_float32_and_float64_shapes():
    """Test 1D and 2D shapes and preservation of float32 and float64 dtypes."""
    node_x = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)

    # 1D float32 input -> 1D float32 output
    z_1d_f32 = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    res_1d_f32 = suxarray.grid.area._integrate_nodal(node_x, node_y, z_1d_f32, connectivity)
    assert res_1d_f32.dtype == np.float32
    assert res_1d_f32.shape == (1,)
    assert res_1d_f32[0] == pytest.approx(12.0, rel=1e-5)

    # 2D float32 input -> 2D float32 output
    z_2d_f32 = np.array([[4.0, 4.0, 4.0]], dtype=np.float32)
    res_2d_f32 = suxarray.grid.area._integrate_nodal(node_x, node_y, z_2d_f32, connectivity)
    assert res_2d_f32.dtype == np.float32
    assert res_2d_f32.shape == (1, 1)
    assert res_2d_f32[0, 0] == pytest.approx(12.0, rel=1e-5)

    # 1D float64 input -> 1D float64 output
    z_1d_f64 = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    res_1d_f64 = suxarray.grid.area._integrate_nodal(node_x, node_y, z_1d_f64, connectivity)
    assert res_1d_f64.dtype == np.float64
    assert res_1d_f64.shape == (1,)
    assert res_1d_f64[0] == pytest.approx(12.0, rel=1e-12)


def test_integrate_nodal_quad_float32():
    """Test quad integration with float32 values."""
    node_x = np.array([0.0, 2.0, 2.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 3.0, 3.0], dtype=np.float64)
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)

    z_f32 = np.array([[4.0, 4.0, 4.0, 4.0]], dtype=np.float32)
    res = suxarray.grid.area._integrate_nodal(node_x, node_y, z_f32, connectivity)
    assert res.dtype == np.float32
    assert res.shape == (1, 1)
    assert res[0, 0] == pytest.approx(24.0, rel=1e-5)


def test_integrate_nodal_mixed_tri_quad_float32():
    """Test nodal integration on a mixed mesh containing both triangles and quads in float32."""
    # Element 0: Triangle with nodes [0, 1, 2, -1], vertices (0,0), (2,0), (0,2). Area = 2.0
    # Element 1: Quad with nodes [1, 3, 4, 5], vertices (2,0), (4,0), (4,2), (2,2). Area = 4.0
    node_x = np.array([0.0, 2.0, 0.0, 4.0, 4.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 2.0, 0.0, 2.0, 2.0], dtype=np.float64)
    connectivity = np.array(
        [
            [0, 1, 2, -1],
            [1, 3, 4, 5],
        ],
        dtype=np.int64,
    )

    # 1D float32 constant field z = 3.0 -> tri: 6.0, quad: 12.0
    z_1d_f32 = np.full(6, 3.0, dtype=np.float32)
    res_1d = suxarray.grid.area._integrate_nodal(node_x, node_y, z_1d_f32, connectivity)
    assert res_1d.dtype == np.float32
    assert res_1d.shape == (2,)
    np.testing.assert_allclose(res_1d, [6.0, 12.0], rtol=1e-5)

    # 2D float32 field with 2 timesteps -> timestep 0: [6.0, 12.0], timestep 1: [10.0, 20.0]
    z_2d_f32 = np.array(
        [
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    res_2d = suxarray.grid.area._integrate_nodal(node_x, node_y, z_2d_f32, connectivity)
    assert res_2d.dtype == np.float32
    assert res_2d.shape == (2, 2)
    np.testing.assert_allclose(res_2d[0], [6.0, 12.0], rtol=1e-5)
    np.testing.assert_allclose(res_2d[1], [10.0, 20.0], rtol=1e-5)
