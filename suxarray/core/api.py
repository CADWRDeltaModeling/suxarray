from typing import Dict, Union, Optional
import os
import xarray as xr
import uxarray as ux
from suxarray.grid import Grid
from suxarray.core.dataset import SxDataset
from suxarray.io._schismgrid import _transform_coordinates


def read_schism_nc(grid: Grid, ds_data: xr.Dataset) -> SxDataset:
    ds_data = ux.core.utils._map_dims_to_ugrid(ds_data, grid._source_dims_dict, grid)
    if "nSCHISM_vgrid_layers" in ds_data.dims:
        ds_data = ds_data.swap_dims({"nSCHISM_vgrid_layers": "n_layer"})
    sxds = SxDataset(ds_data, sxgrid=grid)
    return sxds


def read_grid(ds_out2d: xr.Dataset, ds_zcoords: Optional[xr.Dataset] = None) -> Grid:
    """Read grid information from xarray datasets and return a suxarray grid

    Parameters
    ----------
    ds_out2d : xr.Dataset
        Dataset containing SCHISM out2d-like information
    ds_zcoords : xr.Dataset, optional
        Dataset containing SCHISM z-coordinate like information

    Returns
    -------
    `suxarray.Grid`
    """
    # Uxarray expects lat-lon coordinates. Convert coordinates into global
    # geographic coordinates.
    ds_out2d_latlon = _transform_coordinates(ds_out2d)
    sxgrid = Grid.from_dataset(ds_out2d_latlon, ds_zcoords)
    return sxgrid


def open_grid(
    files_out2d: Union[str, os.PathLike, list, tuple],
    files_zcoords: Optional[Union[str, os.PathLike, list, tuple]],
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Open SCHISM out2d and zCoordinates files and return a grid object

    Parameters
    ----------
    files_out2d : str, os.PathLike, list, tuple
        SCHISM out2d files
    files_zcoords : str, os.PathLike, list, tuple
        SCHISM zCoordinates files
    chunks : dict, optional
        Chunks for dask

    Returns
    -------
    `suxarray.Grid`
    """

    if chunks is None:
        chunks = {
            "time": 12,
        }
    ds_out2d = xr.open_mfdataset(
        files_out2d,
        engine="h5netcdf",
        mask_and_scale=False,
        chunks=chunks,
        preprocess=lambda ds: ds[
            [
                "SCHISM_hgrid",
                "SCHISM_hgrid_node_x",
                "SCHISM_hgrid_node_y",
                "SCHISM_hgrid_face_x",
                "SCHISM_hgrid_face_y",
                "SCHISM_hgrid_edge_x",
                "SCHISM_hgrid_edge_y",
                "bottom_index_node",
                "SCHISM_hgrid_face_nodes",
                "SCHISM_hgrid_edge_nodes",
                "dryFlagNode",
            ]
        ],
        # join="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        parallel=True,
    )

    ds_zcoords = xr.open_mfdataset(
        files_zcoords,
        engine="h5netcdf",
        chunks=chunks,
        parallel=True,
    )

    sxgrid = read_grid(ds_out2d, ds_zcoords)

    return sxgrid
