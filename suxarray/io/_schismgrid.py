from __future__ import annotations
from typing import Optional
import pyproj
import pandas as pd
import xarray as xr
from uxarray.conventions.ugrid import (
    CARTESIAN_NODE_COORDINATES,
    CARTESIAN_FACE_COORDINATES,
    CARTESIAN_EDGE_COORDINATES,
)
from uxarray.io._ugrid import _read_ugrid
import suxarray as sx
from suxarray.conventions.schism_grid import (
    SCHISM_GRID_TIME_VARIABLES,
    SCHISM_CARTESIAN_NODE_COORDINATES,
    SCHISM_CARTESIAN_FACE_COORDINATES,
    SCHISM_CARTESIAN_EDGE_COORDINATES,
)


def _read_schism_grid(
    ds_out2d: xr.Dataset, ds_zcoords: Optional[xr.Dataset] = None
) -> tuple[xr.Dataset, xr.Dataset]:
    """Read grid information from a SCHISM out2d and zCoordinates datasets

    This function is copied and modified from uxarray.io._ugrid._read_ugrid.
    It may need to be updated when the original function is updated.

    Parameters
    ----------
    ds_out2d : xr.Dataset
        Dataset containing SCHISM out2d-like information
    ds_zcoords : xr.Dataset, optional
        Dataset containing SCHISM z-coordinate like information

    Returns
    -------
    xr.Dataset
        SCHISM out2d dataset
    xr.Dataset
        SCHISM grid information dataset
    """
    # We want to move time dependent variables to ds_zcoords from ds_outd2
    # ds_out2d will contain only static information.

    for varname in SCHISM_GRID_TIME_VARIABLES:
        if varname in ds_out2d:
            if ds_zcoords is None:
                ds_zcoords = ds_out2d[[varname]]
            else:
                ds_zcoords[varname] = ds_out2d[varname]

    # Drop any variables that depend on time from ds_out2d
    if "time" in ds_out2d.dims:
        ds_out2d = ds_out2d.drop_dims("time")

    # Use uxarray _read_ugrid to remap the dimensions
    # Note that it does not update the grid_topology information though
    # variable names are updated.
    ds_out2d, dim_dict = _read_ugrid(ds_out2d)

    if SCHISM_CARTESIAN_NODE_COORDINATES[0] in ds_out2d:
        coord_dict = {
            SCHISM_CARTESIAN_NODE_COORDINATES[0]: CARTESIAN_NODE_COORDINATES[0],
            SCHISM_CARTESIAN_NODE_COORDINATES[1]: CARTESIAN_NODE_COORDINATES[1],
        }
        ds_out2d = ds_out2d.rename(coord_dict)
    if SCHISM_CARTESIAN_EDGE_COORDINATES[0] in ds_out2d:
        coord_dict = {
            SCHISM_CARTESIAN_EDGE_COORDINATES[0]: CARTESIAN_EDGE_COORDINATES[0],
            SCHISM_CARTESIAN_EDGE_COORDINATES[1]: CARTESIAN_EDGE_COORDINATES[1],
        }
        ds_out2d = ds_out2d.rename(coord_dict)
    if SCHISM_CARTESIAN_FACE_COORDINATES[0] in ds_out2d:
        coord_dict = {
            SCHISM_CARTESIAN_FACE_COORDINATES[0]: CARTESIAN_FACE_COORDINATES[0],
            SCHISM_CARTESIAN_FACE_COORDINATES[1]: CARTESIAN_FACE_COORDINATES[1],
        }
        ds_out2d = ds_out2d.rename(coord_dict)

    if ds_zcoords is not None:
        dim_dict_zcoords = {
            "nSCHISM_hgrid_node": "n_node",
            "nSCHISM_vgrid_layers": "n_layer",
        }
        ds_zcoords = ds_zcoords.swap_dims(dim_dict_zcoords)

    ds_sgrid_info = xr.Dataset()
    for varname in SCHISM_GRID_TIME_VARIABLES:
        if ds_zcoords is not None and varname in ds_zcoords.variables:
            ds_sgrid_info[varname] = ds_zcoords[varname]

    return ds_out2d, ds_sgrid_info


def write_ugrid(grid: Grid, file_gridout):
    """Write a SCHISM grid to a NetCDF file

    Parameters
    ----------
    file_gridout : str or Path
        Path to the output NetCDF file
    grid : Grid
        SCHISM grid object
    """
    grid._ds.to_netcdf(file_gridout)


def _find_grid_topology_varname(ds: xr.Dataset) -> str:
    """Find the UGRID grid topology variable name in the dataset

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing UGRID grid information

    Returns
    -------
    str
        Name of the UGRID grid topology variable name
    """
    grid_topology_name = list(ds.filter_by_attrs(cf_role="mesh_topology").keys())[0]
    return grid_topology_name


def _transform_coordinates(
    ds: xr.Dataset, epsg_source: Optional[str] = None
) -> xr.Dataset:
    """Transform coordinates from a local projection to global geographic coordinates.

    Coordinate names are found from the UGRID grid topology information. The new
    coordinates will be named node_lon, node_lat, edge_lon, edge_lat, face_lon,
    face_lat.

    Parameters
    ----------
    ds : xr.Dataset
        SCHISM grid dataset
    epsg_source : str, optional
        Source EPSG code. Default is None, which assumes UTM zone 10N

    Returns
    -------
    xr.Dataset
        SCHISM grid dataset with transformed coordinates
    """
    if epsg_source is None:
        # The default projection is UTM zone 10N
        epsg_source = "EPSG:26910"
    transformer = pyproj.Transformer.from_crs(epsg_source, "EPSG:4326", always_xy=True)

    var_grid_topology = _find_grid_topology_varname(ds)
    grid_topology = ds[var_grid_topology]
    node_x_name, node_y_name = grid_topology.node_coordinates.split()

    node_lon, node_lat = transformer.transform(
        ds[node_x_name].values,
        ds[node_y_name].values,
    )

    node_lon_attrs = ds[node_x_name].attrs.copy()
    node_lon_attrs["standard_name"] = "longitude"
    node_lon_attrs["units"] = "degrees_north"
    node_lat_attrs = ds[node_y_name].attrs.copy()
    node_lat_attrs["standard_name"] = "latitude"
    node_lat_attrs["units"] = "degrees_east"
    ds = ds.assign(
        SCHISM_hgrid_node_lon=xr.DataArray(
            node_lon,
            dims=["nSCHISM_hgrid_node"],
            attrs=node_lon_attrs,
        ),
        SCHISM_hgrid_node_lat=xr.DataArray(
            node_lat,
            dims=["nSCHISM_hgrid_node"],
            attrs=node_lat_attrs,
        ),
    )
    grid_topology_new = grid_topology.assign_attrs(
        node_coordinates="SCHISM_hgrid_node_lon SCHISM_hgrid_node_lat"
    )

    if grid_topology.attrs.get("edge_coordinates") is not None:
        edge_x_name, edge_y_name = grid_topology.edge_coordinates.split()
        if edge_x_name in ds:
            edge_lon, edge_lat = transformer.transform(
                ds[edge_x_name].values,
                ds[edge_y_name].values,
            )
            ds = ds.assign(
                SCHISM_hgrid_edge_lon=xr.DataArray(
                    edge_lon,
                    dims=["nSCHISM_hgrid_edge"],
                    attrs=ds[edge_x_name].attrs,
                ),
                SCHISM_hgrid_edge_lat=xr.DataArray(
                    edge_lat,
                    dims=["nSCHISM_hgrid_edge"],
                    attrs=ds[edge_y_name].attrs,
                ),
            )
            grid_topology_new = grid_topology_new.assign_attrs(
                edge_coordinates="SCHISM_hgrid_edge_lon SCHISM_hgrid_edge_lat"
            )

    if grid_topology.attrs.get("face_coordinates") is not None:
        face_x_name, face_y_name = grid_topology.face_coordinates.split()
        if face_x_name in ds:
            face_lon, face_lat = transformer.transform(
                ds[face_x_name].values,
                ds[face_y_name].values,
            )
            ds = ds.assign(
                SCHISM_hgrid_face_lon=xr.DataArray(
                    face_lon,
                    dims=["nSCHISM_hgrid_face"],
                    attrs=ds[face_x_name].attrs,
                ),
                SCHISM_hgrid_face_lat=xr.DataArray(
                    face_lat,
                    dims=["nSCHISM_hgrid_face"],
                    attrs=ds[face_y_name].attrs,
                ),
            )
            grid_topology_new = grid_topology_new.assign_attrs(
                face_coordinates="SCHISM_hgrid_face_lon SCHISM_hgrid_face_lat"
            )

    # Update the topology variable info
    ds[var_grid_topology] = grid_topology_new

    return ds


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
        data=df_nodes[1].values,
        dims="nSCHISM_hgrid_node",
        attrs={"units": "m", "standard_name": "projection_x_coordinate"},
    )
    ds["SCHISM_hgrid_node_y"] = xr.DataArray(
        data=df_nodes[2].values,
        dims="nSCHISM_hgrid_node",
        attrs={"units": "m", "standard_name": "projection_y_coordinate"},
    )
    # ds = _transform_coordinates(ds)

    # Replace NaN with -1
    df_faces = df_faces.fillna(0)
    ds["SCHISM_hgrid_face_nodes"] = xr.DataArray(
        data=df_faces[[2, 3, 4, 5]].astype(int).values - 1,
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
    grid = sx.core.api.read_grid(ds, None)
    return grid
