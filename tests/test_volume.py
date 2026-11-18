from pathlib import Path

import numpy as np
import pytest
import uxarray as ux
import xarray as xr

import suxarray as sx
from suxarray.core.api import _choose_netcdf_engine
from suxarray.grid.integrate import (
    _calculate_mass_in_ocean,
    _calculate_prism_volumes,
    _calc_prism_layer_values,
    calc_volumes_from_projected,
    integrate_prism_variable,
    projected_volume_ufunc_kernel,
)


def test_calculate_prism_volume_synthetic_shaved_cells():
    """Validate prism volumes on a tiny synthetic triangular mesh.

    The mesh has 3 triangular faces of equal area (2 m^2 each), and node-wise
    shaved-cell bottom indices that mask lower interfaces at selected nodes.
    """
    node_x = np.array([0.0, 2.0, 0.0, -2.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 2.0, 0.0, -2.0], dtype=np.float64)

    # 3 triangles with -1 fill values in the 4th slot.
    connectivity = np.array(
        [
            [0, 1, 2, -1],
            [0, 2, 3, -1],
            [0, 3, 4, -1],
        ],
        dtype=np.int64,
    )
    # 0-based node bottom indices. k > 0 masks interfaces [:k] at that node.
    bottom_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)

    # z shape: (n_node=5, n_layer=4) -> 3 interfaces.
    z = np.array(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    volumes = _calculate_prism_volumes(
        node_x=node_x,
        node_y=node_y,
        z=z,
        connectivity=connectivity,
        bottom_indices=bottom_indices,
    )

    # Expected shape: (n_interface, n_face)
    assert volumes.shape == (3, 3)

    # Make the geometric factor explicit for readers:
    # expected volume = triangle_area * face-mean(thickness after masking)
    triangle_area_m2 = 2.0
    expected_face_mean_thickness = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected = triangle_area_m2 * expected_face_mean_thickness

    np.testing.assert_allclose(volumes, expected, rtol=0.0, atol=1e-12)

    # Spot-check a couple of entries with pytest.approx for readability.
    assert volumes[0, 0] == pytest.approx(2.0 / 3.0)
    assert volumes[2, 2] == pytest.approx(2.0)


def test_calculate_prism_volume_synthetic_float32():
    """Validate that prism volume calculation preserves float32 precision on a mixed tri-quad mesh.

    Mesh consists of:
    - Face 0: Triangle with vertices (0,0), (2,0), (0,2). Area = 2 m^2.
    - Face 1: Quadrilateral with vertices (2,0), (4,0), (4,2), (2,2). Area = 4 m^2.
    """
    node_x = np.array([0.0, 2.0, 0.0, 4.0, 4.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 2.0, 0.0, 2.0, 2.0], dtype=np.float64)

    connectivity = np.array(
        [
            [0, 1, 2, -1],  # Triangle
            [1, 3, 4, 5],   # Quadrilateral
        ],
        dtype=np.int64,
    )
    # Shaved cells: node 0 and node 3 have bottom interface masked
    bottom_indices = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)

    # 4 layers -> 3 interfaces, all in float32
    z = np.tile(
        np.array([-3.0, -2.0, -1.0, 0.0], dtype=np.float32),
        (6, 1),
    )

    volumes = _calculate_prism_volumes(
        node_x=node_x,
        node_y=node_y,
        z=z,
        connectivity=connectivity,
        bottom_indices=bottom_indices,
    )

    assert volumes.dtype == np.float32
    assert volumes.shape == (3, 2)

    # Expected volumes per layer and face:
    # Layer 0:
    #   Tri [0,1,2]: thickness at nodes = [0, 1, 1], mean = 2/3 -> vol = 2.0 * (2/3) = 4/3
    #   Quad [1,3,4,5]: thickness at nodes = [1, 0, 1, 1], mean = 3/4 -> vol = 4.0 * 0.75 = 3.0
    # Layers 1 and 2:
    #   Tri: thickness = 1.0 -> vol = 2.0 * 1.0 = 2.0
    #   Quad: thickness = 1.0 -> vol = 4.0 * 1.0 = 4.0
    expected = np.array(
        [
            [4.0 / 3.0, 3.0],
            [2.0, 4.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(volumes, expected, rtol=1e-5, atol=1e-6)


def test_integrate_prism_variable_synthetic_with_nans_and_quads():
    """Validate integrate_prism_variable and calc_volumes_from_projected
    on a mesh containing both triangles and quadrilaterals, shaved cells,
    and float32 variables with NaNs below the seabed."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    face_nodes = np.array(
        [
            [0, 1, 4, -1],
            [0, 4, 3, -1],
            [1, 2, 5, 4],
        ],
        dtype=np.int64,
    )

    uxgrid = ux.Grid.from_topology(
        node_lon=node_x, node_lat=node_y, face_node_connectivity=face_nodes
    )
    uxgrid._ds["node_x"] = xr.DataArray(node_x, dims=["n_node"])
    uxgrid._ds["node_y"] = xr.DataArray(node_y, dims=["n_node"])

    bottom_index_1based = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
    z = np.array(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [np.nan, -2.0, -1.0, 0.0],
            [np.nan, np.nan, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [np.nan, -2.0, -1.0, 0.0],
            [np.nan, np.nan, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    ds_sgrid = xr.Dataset(
        {
            "bottom_index_node": ("n_node", bottom_index_1based),
            "zCoordinates": (("n_node", "n_layer"), z),
        }
    )

    grid = sx.Grid(uxgrid, ds_sgrid_info=ds_sgrid)

    # float32 variable with NaNs below bottom index
    var_vals = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],
            [np.nan, 20.0, 20.0, 20.0],
            [np.nan, np.nan, 30.0, 30.0],
            [10.0, 10.0, 10.0, 10.0],
            [np.nan, 20.0, 20.0, 20.0],
            [np.nan, np.nan, 30.0, 30.0],
        ],
        dtype=np.float32,
    )

    da = xr.DataArray(var_vals, dims=["n_node", "n_layer"], name="tracer")
    sxda = sx.SxDataArray(da, sxgrid=grid)

    res_face = integrate_prism_variable(sxda, return_per_layer=False)
    res_layers = integrate_prism_variable(sxda, return_per_layer=True)

    # Assert no NaNs leaked into results
    assert not np.isnan(res_face).any()
    assert not np.isnan(res_layers).any()
    assert res_face.shape == (3,)
    assert res_layers.shape == (3, 3)
    np.testing.assert_allclose(res_face, res_layers.sum(axis=0))

    # Also test calc_volumes_from_projected
    vols = calc_volumes_from_projected(sxda)
    assert not np.isnan(vols).any()
    assert vols.shape == (3, 3)
    assert np.all(vols >= 0.0)


def test_integrate_prism_variable_real_data():
    """Validate integrate_prism_variable with real SCHISM output containing NaNs."""
    p_cur = Path(__file__).parent.absolute()
    engine = _choose_netcdf_engine()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"), engine=engine)
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"), engine=engine)
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    ds_data = xr.open_dataset(str(p_cur / "testdata/salinity_1.nc"), engine=engine)
    sxds = sx.read_schism_nc(grid, ds_data)

    # Multi-timestep input raises ValueError
    with pytest.raises(ValueError, match="must contain exactly one timestep"):
        integrate_prism_variable(sxds.salinity)

    # Single-timestep selection
    sal_t0 = sxds.salinity.isel(time=0)
    assert sal_t0.dtype == np.float32

    res_total = integrate_prism_variable(sal_t0, return_per_layer=False)
    res_layers = integrate_prism_variable(sal_t0, return_per_layer=True)

    # Ensure native float32 dtype is preserved
    assert res_total.dtype == np.float32
    assert res_layers.dtype == np.float32

    # Ensure no NaN leakage
    assert not np.isnan(res_total).any()
    assert not np.isnan(res_layers).any()
    assert res_total.shape == (grid.n_face,)
    assert res_layers.shape == (grid.sgrid_info.n_layer.size - 1, grid.n_face)
    np.testing.assert_allclose(res_total, res_layers.sum(axis=0))

    # calc_volumes_from_projected
    volumes = calc_volumes_from_projected(sal_t0)
    assert not np.isnan(volumes).any()
    assert volumes.shape == (grid.sgrid_info.n_layer.size - 1, grid.n_face)
    assert np.all(volumes >= 0.0)


def test_calculate_mass_in_ocean_real_data():
    """Validate _calculate_mass_in_ocean with real SCHISM output containing NaNs."""
    p_cur = Path(__file__).parent.absolute()
    engine = _choose_netcdf_engine()
    ds_out2d = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"), engine=engine)
    ds_zcoords = xr.open_dataset(str(p_cur / "testdata/zCoordinates_1.nc"), engine=engine)
    grid = sx.read_grid(ds_out2d, ds_zcoords)
    ds_data = xr.open_dataset(str(p_cur / "testdata/salinity_1.nc"), engine=engine)
    sxds = sx.read_schism_nc(grid, ds_data)

    sal_t0 = sxds.salinity.isel(time=0)
    mass = _calculate_mass_in_ocean(sal_t0)

    assert not np.isnan(mass).any()
    assert mass.shape == (grid.sgrid_info.n_layer.size - 1, grid.n_face)
    assert np.all(mass >= 0.0)


def test_projected_volume_ufunc_kernel_apply_ufunc():
    """Validate projected_volume_ufunc_kernel using xarray.apply_ufunc."""
    node_x = np.array([0.0, 2.0, 0.0, -2.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 2.0, 0.0, -2.0], dtype=np.float64)
    connectivity = np.array([[0, 1, 2, -1], [0, 2, 3, -1], [0, 3, 4, -1]], dtype=np.int64)
    bottom_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)

    # 2 timesteps
    z = np.array([
        [[-3.0, -2.0, -1.0, 0.0] for _ in range(5)],
        [[-6.0, -4.0, -2.0, 0.0] for _ in range(5)],
    ], dtype=np.float64)
    z_da = xr.DataArray(z, dims=["time", "n_node", "n_layer"])

    res = xr.apply_ufunc(
        projected_volume_ufunc_kernel,
        z_da,
        node_x,
        node_y,
        connectivity,
        bottom_indices,
        input_core_dims=[
            ["n_node", "n_layer"],
            ["n_node"],
            ["n_node"],
            ["n_face", "n_node_per_face"],
            ["n_node"],
        ],
        output_core_dims=[["n_layer_minus_1", "n_face"]],
        vectorize=True,
    )
    assert res.shape == (2, 3, 3)
    assert not np.isnan(res.values).any()
    assert res.values[0, 0, 0] == pytest.approx(2.0 / 3.0)


def test_calc_prism_layer_values():
    """Unit test for the Numba kernel _calc_prism_layer_values."""
    # 2 nodes, 4 vertical levels (3 intervals)
    # Node 0 has bottom index 0 (all intervals active)
    # Node 1 has bottom index 2 (intervals 0 and 1 below bottom, with NaNs)
    var_arr = np.array(
        [
            [2.0, 4.0, 6.0, 8.0],
            [np.nan, np.nan, 10.0, 20.0],
        ],
        dtype=np.float32,
    )
    z = np.array(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [np.nan, -2.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    bottom_indices = np.array([0, 2], dtype=np.int64)

    values = _calc_prism_layer_values(var_arr, z, bottom_indices)

    assert values.shape == (3, 2)
    assert values.dtype == np.float32
    assert not np.isnan(values).any()

    # Node 0:
    # interval 0: 0.5 * (2 + 4) * (-2 - (-3)) = 3.0 * 1.0 = 3.0
    # interval 1: 0.5 * (4 + 6) * (-1 - (-2)) = 5.0 * 1.0 = 5.0
    # interval 2: 0.5 * (6 + 8) * (0 - (-1)) = 7.0 * 1.0 = 7.0
    expected_col0 = np.array([3.0, 5.0, 7.0], dtype=np.float32)

    # Node 1:
    # intervals 0 and 1 below bottom index 2 -> 0.0
    # interval 2: 0.5 * (10 + 20) * (0 - (-1)) = 15.0 * 1.0 = 15.0
    expected_col1 = np.array([0.0, 0.0, 15.0], dtype=np.float32)

    np.testing.assert_allclose(values[:, 0], expected_col0)
    np.testing.assert_allclose(values[:, 1], expected_col1)

