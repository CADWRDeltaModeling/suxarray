from typing import Any, Dict, Union, Optional
import os
from warnings import warn
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import uxarray as ux
from suxarray.grid import Grid
from suxarray.core.dataset import SxDataset


def _transform_coordinates(ds: xr.Dataset) -> xr.Dataset:
    # Hardcoded for now. Ideally we need to use the CRS information from the file.
    transformer = pyproj.Transformer.from_crs("EPSG:26910", "EPSG:4326", always_xy=True)
    node_lon, node_lat = transformer.transform(
        ds.SCHISM_hgrid_node_x.values,
        ds.SCHISM_hgrid_node_y.values,
    )
    ds = ds.assign(
        node_lon=xr.DataArray(
            node_lon,
            dims=["nSCHISM_hgrid_node"],
            attrs=ds.SCHISM_hgrid_node_x.attrs,
        ),
        node_lat=xr.DataArray(
            node_lat,
            dims=["nSCHISM_hgrid_node"],
            attrs=ds.SCHISM_hgrid_node_y.attrs,
        ),
    )

    if "SCHISM_hgrid_edge_x" in ds:
        edge_lon, edge_lat = transformer.transform(
            ds.SCHISM_hgrid_edge_x.values,
            ds.SCHISM_hgrid_edge_y.values,
        )
        ds = ds.assign(
            edge_lon=xr.DataArray(
                edge_lon,
                dims=["nSCHISM_hgrid_edge"],
                attrs=ds.SCHISM_hgrid_edge_x.attrs,
            ),
            edge_lat=xr.DataArray(
                edge_lat,
                dims=["nSCHISM_hgrid_edge"],
                attrs=ds.SCHISM_hgrid_edge_y.attrs,
            ),
        )
    if "SCHISM_hgrid_face_x" in ds:
        face_lon, face_lat = transformer.transform(
            ds.SCHISM_hgrid_face_x.values,
            ds.SCHISM_hgrid_face_y.values,
        )
        ds = ds.assign(
            face_lon=xr.DataArray(
                face_lon,
                dims=["nSCHISM_hgrid_face"],
                attrs=ds.SCHISM_hgrid_face_x.attrs,
            ),
            face_lat=xr.DataArray(
                face_lat,
                dims=["nSCHISM_hgrid_face"],
                attrs=ds.SCHISM_hgrid_face_y.attrs,
            ),
        )

    return ds


def open_grid(
    grid_filename_or_obj: Union[
        str, os.PathLike, xr.Dataset, np.ndarray, list, tuple, dict
    ],
    ds_sgrid_info: xr.Dataset = None,
    **kwargs: Dict[str, Any],
) -> Grid:
    """Open SCHISM grid file and return a suxarray Grid object

    Parameters
    ----------
    grid_filename_or_obj : str, os.PathLike, xr.Dataset, np.ndarray, list, tuple, dict
        The path to the SCHISM grid file or a dataset object
    ds_sgrid_info : xr.Dataset, optional
        Extra SCHISM grid information such as z-coordinate data

    Returns
    -------
    suxarray.Grid
        A suxarray Grid object
    """
    if isinstance(grid_filename_or_obj, xr.Dataset):
        # construct a grid from a dataset file
        grid_filename_or_obj = _transform_coordinates(grid_filename_or_obj)
        sxgrid = Grid.from_dataset(grid_filename_or_obj, ds_sgrid_info=ds_sgrid_info)

    elif isinstance(grid_filename_or_obj, dict):
        # unpack the dictionary and construct a grid from topology
        raise NotImplementedError("Dictionary input not yet supported.")

    elif isinstance(grid_filename_or_obj, (list, tuple, np.ndarray, xr.DataArray)):
        # construct Grid from face vertices
        raise NotImplementedError("Face vertices input not yet supported.")

    # attempt to use Xarray directly for remaining input types
    else:
        try:
            ds = xr.open_dataset(grid_filename_or_obj)
            ds = _transform_coordinates(ds)
            sxgrid = Grid.from_dataset(ds)
        except Exception as e:
            warn(
                f"Could not open grid file {grid_filename_or_obj} as an Xarray dataset."
            )
            raise e

    return sxgrid


def open_schism_nc(
    out2d_filename: Union[str, os.PathLike],
    data_filename: Union[str, os.PathLike, xr.Dataset],
) -> SxDataset:
    """Open SCHISM NetCDF output files and return a suxarray Dataset

    Parameters
    ----------
    out2d_filename : str, os.PathLike
        The path to the SCHISM out2d NetCDF file
    data_filename : str, os.PathLike, xr.Dataset
        The path to the SCHISM data NetCDF file or a dataset object
    """
    # If the data is an xarray dataset, use it as it is.
    if isinstance(data_filename, xr.Dataset):
        ds = data_filename
    # Otherwise, try to read the data file
    elif isinstance(data_filename, list):
        ds = xr.open_mfdataset(data_filename)
    else:
        ds = xr.open_dataset(data_filename)

    ds_out2d = xr.open_mfdataset(out2d_filename, mask_and_scale=False, parallel=True)

    # Take SCHISM grid variables for the uxgrid
    from suxarray.constants import SCHISM_GRID_VARIABLES

    ds_sgrid_info = xr.Dataset()
    for var in SCHISM_GRID_VARIABLES:
        if var in ds.variables:
            ds_sgrid_info[var] = ds[var]
    if ds_sgrid_info:
        sxgrid = open_grid(ds_out2d, ds_sgrid_info)
    else:
        sxgrid = open_grid(ds_out2d)

    # Remove the spatial coordinates to make uxarray happy
    for var in ds.variables:
        if var in [
            "SCHISM_hgrid_node_x",
            "SCHISM_hgrid_node_y",
            "SCHISM_hgrid_edge_x",
            "SCHISM_hgrid_edge_y",
            "SCHISM_hgrid_face_x",
            "SCHISM_hgrid_face_y",
        ]:
            ds = ds.drop_vars([var])

    ds = ux.core.utils._map_dims_to_ugrid(ds, sxgrid._source_dims_dict, sxgrid)

    sxds = SxDataset(ds, sxgrid=sxgrid, source_datasets=str(data_filename))
    return sxds


def open_hgrid_gr3(path_hgrid):
    """Read SCHISM hgrid.gr3 file and return a suxarray grid"""
    # read the header
    with open(path_hgrid, "r") as f:
        f.readline()
        n_faces, n_nodes = [int(x) for x in f.readline().strip().split()[:2]]
    # Read the node section. Read only up to the fourth column
    df_nodes = pd.read_csv(
        path_hgrid,
        skiprows=2,
        header=None,
        nrows=n_nodes,
        sep="\s+",
        usecols=range(4),
    )
    # Read the face section. Read only up to the sixth column. The last column
    # may exist or not.
    df_faces = pd.read_csv(
        path_hgrid,
        skiprows=2 + n_nodes,
        header=None,
        nrows=n_faces,
        sep="\s+",
        names=range(6),
    )
    # TODO Read boundary information, if any
    # Create suxarray grid
    ds = xr.Dataset()
    ds["SCHISM_hgrid_node_x"] = xr.DataArray(
        data=df_nodes[1].values, dims="nSCHISM_hgrid_node"
    )
    ds["SCHISM_hgrid_node_y"] = xr.DataArray(
        data=df_nodes[2].values, dims="nSCHISM_hgrid_node"
    )
    ds = _transform_coordinates(ds)

    # Replace NaN with -1
    df_faces = df_faces.fillna(0)
    ds["SCHISM_hgrid_face_nodes"] = xr.DataArray(
        data=df_faces[[2, 3, 4, 5]].astype(int).values - 1,
        dims=("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
        attrs={"start_index": 0, "cf_role": "face_node_connectivity", "_FillValue": -1},
    )
    ds["depth"] = df_nodes[3].values
    da_topology = xr.DataArray(name="SCHISM_hgrid")
    da_topology = da_topology.assign_attrs(
        long_name="Topology data of 2d unstructured mesh",
        topology_dimension=2,
        cf_role="mesh_topology",
        node_coordinates="SCHISM_hgrid_node_x SCHISM_hgrid_node_y",
        # edge_coordinates="SCHISM_hgrid_edge_x SCHISM_hgrid_edge_y",
        # face_coordinates="SCHISM_hgrid_face_x SCHISM_hgrid_face_y",
        # edge_node_connectivity="SCHISM_hgrid_edge_nodes",
        face_node_connectivity="SCHISM_hgrid_face_nodes",
    )
    ds["SCHISM_hgrid"] = da_topology

    grid = Grid.from_dataset(ds, source_grid_spec="UGRID")
    return grid
