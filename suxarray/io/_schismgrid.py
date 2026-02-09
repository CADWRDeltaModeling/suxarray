from __future__ import annotations
from typing import Optional, Union
import pyproj
import xarray as xr
from uxarray.conventions.ugrid import (
    CARTESIAN_NODE_COORDINATES,
    CARTESIAN_FACE_COORDINATES,
    CARTESIAN_EDGE_COORDINATES,
)
from uxarray.io._ugrid import _read_ugrid
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

    # Note, as of uxarray==2025.9.0 _read_ugrid does not correctly rename the
    # n_edge dimension to nSCHISM_hgrid_edge, which causes issues with the isel
    # method. This is a temporary patch to rename the dimensions back to n_edge,
    # n_face, and n_node after reading the grid. This will need to be removed
    # once the issue is fixed in uxarray.
    ds_out2d, dim_dict = _read_ugrid(ds_out2d)

    ds_out2d = _rename_coords(ds_out2d)

    if ds_zcoords is not None:
        dim_dict_zcoords = {
            "nSCHISM_hgrid_node": "n_node",
            "nSCHISM_hgrid_edge": "n_edge",
            "nSCHISM_hgrid_face": "n_face",
        }
        ds_zcoords = ds_zcoords.swap_dims(dim_dict_zcoords)
        if "nSCHISM_vgrid_layers" in ds_zcoords.dims:
            ds_zcoords = ds_zcoords.swap_dims({"nSCHISM_vgrid_layers": "n_layer"})

    ds_sgrid_info = xr.Dataset()
    for varname in SCHISM_GRID_TIME_VARIABLES:
        if ds_zcoords is not None and varname in ds_zcoords.variables:
            ds_sgrid_info[varname] = ds_zcoords[varname]
    ds_sgrid_info = _rename_coords(ds_sgrid_info)

    return ds_out2d, ds_sgrid_info


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


def _rename_coords(
    data: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    if SCHISM_CARTESIAN_NODE_COORDINATES[0] in data:
        coord_dict = {
            SCHISM_CARTESIAN_NODE_COORDINATES[0]: CARTESIAN_NODE_COORDINATES[0],
            SCHISM_CARTESIAN_NODE_COORDINATES[1]: CARTESIAN_NODE_COORDINATES[1],
        }
        data = data.rename(coord_dict)
    if SCHISM_CARTESIAN_EDGE_COORDINATES[0] in data:
        coord_dict = {
            SCHISM_CARTESIAN_EDGE_COORDINATES[0]: CARTESIAN_EDGE_COORDINATES[0],
            SCHISM_CARTESIAN_EDGE_COORDINATES[1]: CARTESIAN_EDGE_COORDINATES[1],
        }
        data = data.rename(coord_dict)
    if SCHISM_CARTESIAN_FACE_COORDINATES[0] in data:
        coord_dict = {
            SCHISM_CARTESIAN_FACE_COORDINATES[0]: CARTESIAN_FACE_COORDINATES[0],
            SCHISM_CARTESIAN_FACE_COORDINATES[1]: CARTESIAN_FACE_COORDINATES[1],
        }
        data = data.rename(coord_dict)
    # Patch for dimension name renaming.
    if "nSCHISM_hgrid_edge" in data.dims:
        data = data.rename({"nSCHISM_hgrid_edge": "n_edge"})
    if "nSCHISM_hgrid_face" in data.dims:
        data = data.rename({"nSCHISM_hgrid_face": "n_face"})
    if "nSCHISM_hgrid_node" in data.dims:
        data = data.rename({"nSCHISM_hgrid_node": "n_node"})

    return data


def _rename_coords_back(
    data: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    if CARTESIAN_NODE_COORDINATES[0] in data.coords:
        coord_dict = {
            CARTESIAN_NODE_COORDINATES[0]: SCHISM_CARTESIAN_NODE_COORDINATES[0],
            CARTESIAN_NODE_COORDINATES[1]: SCHISM_CARTESIAN_NODE_COORDINATES[1],
        }
        data = data.rename(coord_dict)
    if CARTESIAN_EDGE_COORDINATES[0] in data.coords:
        coord_dict = {
            CARTESIAN_EDGE_COORDINATES[0]: SCHISM_CARTESIAN_EDGE_COORDINATES[0],
            CARTESIAN_EDGE_COORDINATES[1]: SCHISM_CARTESIAN_EDGE_COORDINATES[1],
        }
        data = data.rename(coord_dict)
    if CARTESIAN_FACE_COORDINATES[0] in data.coords:
        coord_dict = {
            CARTESIAN_FACE_COORDINATES[0]: SCHISM_CARTESIAN_FACE_COORDINATES[0],
            CARTESIAN_FACE_COORDINATES[1]: SCHISM_CARTESIAN_FACE_COORDINATES[1],
        }
        data = data.rename(coord_dict)
    return data


def _rename_dims_back(da: SxDataArray) -> SxDataArray:
    """Rename SCHISM grid dimensions to UGRID dimensions

    Parameters
    ----------
    da : SxDataArray
        SCHISM data array

    Returns
    -------
    SxDataArray
        SCHISM data array with renamed dimensions
    """
    dim_dict = {
        "n_node": "nSCHISM_hgrid_node",
        "n_edge": "nSCHISM_hgrid_edge",
        "n_face": "nSCHISM_hgrid_face",
        "n_layer": "nSCHISM_vgrid_layers",
        "nMesh2_node": "nSCHISM_hgrid_node",
    }
    for k, v in dim_dict.items():
        if k in da.dims:
            da = da.rename({k: v})
    return da
