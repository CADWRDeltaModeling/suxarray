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


def test_planar_face_areas(sxds_test_dask):
    """Test planar face_areas calculation and integration via UxArray native routines."""
    grid = sxds_test_dask.sxgrid

    # Check planar face areas are non-trivial physical areas (m^2)
    face_areas = grid.face_areas
    assert face_areas.ndim == 1
    assert face_areas.shape[0] == grid.n_face
    assert np.all(face_areas.values > 0.0)
    assert grid.calculate_total_face_area() == pytest.approx(5429325.17, rel=1e-3)

    # Test UxDataArray.topological_mean returning SxDataArray
    da = sxds_test_dask.salinity.depth_average()
    da_face = da.topological_mean(destination="face")
    assert isinstance(da_face, sx.SxDataArray)
    assert da_face.shape[-1] == grid.n_face

    # Test UxDataArray.integrate on face-centered data
    integral = da_face.integrate()
    assert isinstance(integral, sx.SxDataArray)
    assert integral.shape == (da.shape[0],)  # time dimension
    valid_int = integral.values[~np.isnan(integral.values)]
    assert len(valid_int) > 0
    assert np.all(valid_int > 0.0)

    # Test UxDataArray.weighted_mean on face-centered data
    wmean = da_face.weighted_mean()
    assert wmean.shape == (da.shape[0],)
    valid_wmean = wmean.values[~np.isnan(wmean.values)]
    assert len(valid_wmean) > 0
    assert np.all(valid_wmean > 0.0)


def test_spherical_coordinate_handling():
    """Test that a spherical (geographic) grid is recognized and handled gracefully."""
    # Create a synthetic UGRID dataset in spherical coordinates (degrees)
    node_lon = np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float64)
    node_lat = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    face_nodes = np.array([[0, 1, 2, 3]], dtype=np.int64)

    ds_sph = xr.Dataset(
        data_vars={
            "SCHISM_hgrid": xr.DataArray(
                1,
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "SCHISM_hgrid_node_lon SCHISM_hgrid_node_lat",
                    "face_node_connectivity": "SCHISM_hgrid_face_nodes",
                },
            ),
            "SCHISM_hgrid_face_nodes": xr.DataArray(
                face_nodes,
                dims=["nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"],
                attrs={"start_index": 0, "_FillValue": -1},
            ),
            "SCHISM_hgrid_node_lon": xr.DataArray(
                node_lon,
                dims=["nSCHISM_hgrid_node"],
                attrs={"units": "degrees_east", "standard_name": "longitude"},
            ),
            "SCHISM_hgrid_node_lat": xr.DataArray(
                node_lat,
                dims=["nSCHISM_hgrid_node"],
                attrs={"units": "degrees_north", "standard_name": "latitude"},
            ),
        }
    )

    grid_sph = sx.read_grid(ds_sph)
    assert not grid_sph.is_planar
    # Spherical face_areas uses UxArray's spherical quadrature
    areas_sph = grid_sph.face_areas
    assert areas_sph.shape == (1,)
    assert areas_sph.values[0] > 0.0



