import numpy as np
import xarray as xr
import pytest
import suxarray.grid.area
import suxarray.utils.computing
from .testfixtures import *


def test_integrate_nodal(sxds_test_dask):
    da = sxds_test_dask.salinity.depth_average()
    result = suxarray.grid.area._integrate_nodal(
        da.sxgrid.node_x.values,
        da.sxgrid.node_y.values,
        da.values,
        da.sxgrid.face_node_connectivity.values,
    )
    assert result[0, 64] == pytest.approx(41743.07605, abs=1e-3)

    # da = da.astype(np.float64)
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
        output_dtypes=np.float64,
        dask="parallelized",
    )
    assert result.isel(time=0, n_face=64).values[()] == pytest.approx(
        41743.07555287409, rel=1e-5
    )

    result = da.integrate()
    assert result.isel(time=0, n_face=64, ignore_grid=True).values[()] == pytest.approx(
        41743.07555287409, rel=1e-5
    )
