from pathlib import Path
import xarray as xr
import pytest
import suxarray.suxarray as sx


@pytest.fixture
def grid_test_dask():
    """ Test out2d_dask fixture """
    p_cur = Path(__file__).parent.absolute()
    path_out2d = [str(p_cur / "testdata/out2d_{}.nc".format(i))
                  for i in range(1, 3)]
    ds_out2d = xr.open_mfdataset(
        path_out2d, mask_and_scale=False, data_vars='minimal')
    path_zcoord = [str(p_cur / "testdata/zCoordinates_{}.nc".format(i))
                   for i in range(1, 3)]
    ds_zcoord = xr.open_mfdataset(
        path_zcoord, mask_and_scale=False, data_vars='minimal')
    path_var = [str(p_cur / "testdata/salinity_{}.nc".format(i))
                for i in range(1, 3)]
    ds_salinity = xr.open_mfdataset(
        path_var, mask_and_scale=False, data_vars='minimal')
    ds = xr.merge([ds_out2d, ds_zcoord, ds_salinity])
    ds = sx.coerce_mesh_name(ds)
    grid = sx.Grid(ds)
    yield grid
    ds.close()
