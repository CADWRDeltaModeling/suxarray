from typing import Dict, Union, Optional
import os
import numpy as np
import pandas as pd
import xarray as xr
import uxarray as ux
from suxarray.grid import Grid
from suxarray.core.dataset import SxDataset
from suxarray.core.dataarray import SxDataArray
from suxarray.io._schismgrid import _transform_coordinates, _rename_dims


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
    files_zcoords: Optional[Union[str, os.PathLike, list, tuple]] = None,
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
                "depth",
                "dryFlagNode",
                "dryFlagElement",
                "dryFlagSide",
            ]
        ],
        # join="override",
        data_vars="minimal",
        compat="override",
        coords="minimal",
        parallel=True,
    )

    if files_zcoords is not None:
        ds_zcoords = xr.open_mfdataset(
            files_zcoords,
            engine="h5netcdf",
            chunks=chunks,
            parallel=True,
        )
    else:
        ds_zcoords = None

    sxgrid = read_grid(ds_out2d, ds_zcoords)

    return sxgrid


def open_hgrid_gr3(path_hgrid: Union[str, os.PathLike]) -> Grid:
    """Read SCHISM hgrid.gr3 file and return a suxarray grid

    Parameters
    """
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

    # Create suxarray grid
    ds = xr.Dataset()
    ds["SCHISM_hgrid_node_x"] = xr.DataArray(
        data=df_nodes[1],
        dims="nSCHISM_hgrid_node",
        attrs={"units": "m", "standard_name": "projection_x_coordinate"},
    )
    ds["SCHISM_hgrid_node_y"] = xr.DataArray(
        data=df_nodes[2],
        dims="nSCHISM_hgrid_node",
        attrs={"units": "m", "standard_name": "projection_y_coordinate"},
    )
    # ds = _transform_coordinates(ds)

    # Replace NaN with -1
    df_faces = df_faces.fillna(0)
    ds["SCHISM_hgrid_face_nodes"] = xr.DataArray(
        data=df_faces[[2, 3, 4, 5]].astype(int) - 1,
        dims=("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
        attrs={"start_index": 0, "cf_role": "face_node_connectivity", "_FillValue": -1},
    )
    # ds["depth"] = df_nodes[3].values
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

    # grid = Grid.from_dataset(ds, ds_zcoords=None)
    grid = read_grid(ds, None)
    return grid


def write_schism_grid(grid: Grid, file_gridout: Union[str, os.PathLike]) -> None:
    """Write a SCHISM grid to a NetCDF file

    The goal of this function is to write a SCHISM grid information to a NetCDF
    file that can be understood by SCHISM VisIt plugin.

    Parameters
    ----------
    file_gridout : str or Path
        Path to the output NetCDF file
    grid : Grid
        SCHISM grid object
    """
    ds = grid.to_xarray()

    # Update the face and edge connectivity to start from 1
    ds["face_node_connectivity"] = (
        xr.where(
            ds.face_node_connectivity == ds.face_node_connectivity._FillValue,
            -2,
            ds.face_node_connectivity,
            keep_attrs=True,
        ).astype("int32")
        + 1
    )
    ds["edge_node_connectivity"] = ds.edge_node_connectivity.astype("int32") + 1

    ds.face_node_connectivity.attrs["start_index"] = np.int32(1)
    ds.face_node_connectivity.attrs["_FillValue"] = np.int32(-1)
    ds.edge_node_connectivity.attrs["start_index"] = np.int32(1)

    ds["face_node_connectivity"] = xr.DataArray(
        face_con,
        dims=("n_face", "n_max_face_nodes"),
        attrs={
            "cf_role": "face_node_connectivity",
            "start_index": np.int32(1),
            "_FillValue": np.int32(-1),
        },
    )
    ds["edge_node_connectivity"] = xr.DataArray(
        edge_con,
        dims=("n_edge", "two"),
        attrs={
            "cf_role": "edge_node_connectivity",
            "start_index": np.int32(1),
            "_FillValue": np.int32(-1),
        },
    )

    # Rename variables to SCHISM specific names
    varnames_to_swap = {
        "node_x": "SCHISM_hgrid_node_x",
        "node_y": "SCHISM_hgrid_node_y",
        "edge_x": "SCHISM_hgrid_edge_x",
        "edge_y": "SCHISM_hgrid_edge_y",
        "face_x": "SCHISM_hgrid_face_x",
        "face_y": "SCHISM_hgrid_face_y",
        "face_node_connectivity": "SCHISM_hgrid_face_nodes",
        "edge_node_connectivity": "SCHISM_hgrid_edge_nodes",
    }
    ds = ds.rename(varnames_to_swap)

    # Bring back SCHISM specific information for out2d
    ds = xr.merge(
        [
            ds,
            grid.sgrid_info[
                ["dryFlagNode", "dryFlagElement", "dryFlagSide", "bottom_index_node"]
            ],
        ]
    )

    # Switch back to SCHISM dimension names
    dims_to_swap = {
        "n_node": "nSCHISM_hgrid_node",
        "n_face": "nSCHISM_hgrid_face",
        "n_edge": "nSCHISM_hgrid_edge",
        "n_max_face_nodes": "nMaxSCHISM_hgrid_face_nodes",
    }
    ds = ds.swap_dims(dims_to_swap)

    # Remove variables that we do not want.
    ds = ds.drop_vars(
        [
            "face_lon",
            "face_lat",
            "edge_lon",
            "edge_lat",
            "node_lon",
            "node_lat",
            "grid_topology",
        ]
    )

    # Add back the SCHISM grid topology information
    ds["SCHISM_hgrid"] = xr.DataArray(
        data="",
        name="SCHISM_hgrid",
        attrs={
            "long_name": "Topology data of 2d unstructured mesh",
            "topology_dimension": 2,
            "cf_role": "mesh_topology",
            "node_coordinates": "SCHISM_hgrid_node_x SCHISM_hgrid_node_y",
            "edge_coordinates": "SCHISM_hgrid_edge_x SCHISM_hgrid_edge_y",
            "face_coordinates": "SCHISM_hgrid_face_x SCHISM_hgrid_face_y",
            "edge_node_connectivity": "SCHISM_hgrid_edge_nodes",
            "face_node_connectivity": "SCHISM_hgrid_face_nodes",
        },
    )

    # If the grid dataset does not have n_layers dim, add a dummy variable to
    # keep it.
    if "n_layer" in grid.sgrid_info.dims:
        ds["dummy"] = xr.DataArray(
            data=grid.sgrid_info.n_layer.values, dims="nSCHISM_vgrid_layers"
        )
    else:
        ds["dummy"] = xr.DataArray(data=np.array([0]), dims="nSCHISM_vgrid_layers")

    ds.to_netcdf(file_gridout)


def write_schism_nc(
    sxda: SxDataArray, file_dataout: Optional[Union[str, os.PathLike]] = None
) -> None:
    """Write a SCHISM data array to NetCDF files

    NOTE: WIP. The API is not finalized.

    Parameters
    ----------
    sxda : SxDataset
        SCHISM data array
    file_dataout : str or Path, optional
        Path to the output NetCDF file
    """
    # TODO Hardcoded file name
    file_gridout = "out2d_1.nc"
    write_schism_grid(sxda.sxgrid, file_gridout)

    # zCoordinates
    if "zCoordinates" in sxda.sxgrid.sgrid_info:
        # TODO Hardcoded file name for now
        file_zcoords = "zCoordinates_1.nc"
        da = _rename_dims(sxda.sxgrid.sgrid_info.zCoordinates)
        da.to_netcdf(file_zcoords)

    # TODO Hardcoded file name for now
    file_dataout = f"{sxda.name}_1.nc"
    da = _rename_dims(sxda)
    da.to_netcdf(file_dataout)

    return


def write_gr3(
    sxda: SxDataArray,
    file_gr3: Union[str, os.PathLike],
    fill_value: Optional[float] = -9999.0,
) -> None:
    """Write a SCHISM grid to a gr3 file

    Parameters
    ----------
    sxda : SxDataArray
        SCHISM data array to write out
    file_gr3 : str or Path
        Path to the output gr3 file
    fill_value : float, optional
        Fill value to use for missing values
    """
    varname = sxda.name
    grid = sxda.sxgrid
    with open(file_gr3, "w") as f:
        f.write(f"{varname}\n{grid.face_node_connectivity.shape[0]} {grid.n_node}\n")

    # TODO Need to check the dimension...
    values = sxda.values
    values[np.isnan(values)] = fill_value
    pd.DataFrame(
        data={
            "node_index": np.arange(grid.n_node, dtype=np.int32) + 1,
            "x": grid.node_x.values,
            "y": grid.node_y.values,
            "z": values,
        }
    ).to_csv(file_gr3, sep=" ", header=None, index=None, mode="a")
    # Connectivity
    pd.DataFrame(
        data={
            "index": np.arange(grid.face_node_connectivity.shape[0], dtype=np.int32)
            + 1,
            "n_nodes": np.sum(grid.face_node_connectivity >= 0, axis=1, dtype=np.int32),
            "node1": grid.face_node_connectivity[:, 0] + 1,
            "node2": grid.face_node_connectivity[:, 1] + 1,
            "node3": grid.face_node_connectivity[:, 2] + 1,
            "node4": grid.face_node_connectivity[:, 3] + 1,
        }
    ).to_csv(file_gr3, sep=" ", header=None, index=None, mode="a")
