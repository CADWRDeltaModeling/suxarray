from pathlib import Path
import numpy as np
import xarray as xr
import pytest
import suxarray as sx


@pytest.fixture
def grid_test() -> sx.Grid:
    """Test mesh fixture"""
    p_cur = Path(__file__).parent.absolute()
    grid = sx.open_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    return grid


@pytest.fixture
def sxds_test_dask():
    """Test out2d_dask fixture"""
    p_cur = Path(__file__).parent.absolute()
    path_out2d = [str(p_cur / "testdata/out2d_{}.nc".format(i)) for i in range(1, 3)]
    path_zcoords = [
        str(p_cur / "testdata/zCoordinates_{}.nc".format(i)) for i in range(1, 3)
    ]
    chunks = {"time": 12}
    sxgrid = sx.core.api.open_grid(path_out2d, path_zcoords, chunks=chunks)
    path_var = [str(p_cur / "testdata/salinity_{}.nc".format(i)) for i in range(1, 3)]
    ds_salinity = xr.open_mfdataset(
        path_var, mask_and_scale=False, data_vars="minimal"
    ).astype(np.float64)
    sxds = sx.read_schism_nc(sxgrid, ds_salinity)
    return sxds
