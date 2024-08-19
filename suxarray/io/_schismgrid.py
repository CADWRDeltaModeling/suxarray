from __future__ import annotations
import uxarray.conventions.ugrid as ugrid
from uxarray.io._ugrid import _standardize_connectivity


def _read_schism_out2d(ds_out2d, ds_sgrid_info=None):
    """Read grid information from a SCHISM out2d NetCDF file

    This function is copied and modified from uxarray.io._ugrid._read_ugrid.
    It may need to be updated when the original function is updated."""
    if "time" in ds_out2d.dims:
        ds_out2d = ds_out2d.drop_dims("time")
    grid_topology_name = list(ds_out2d.filter_by_attrs(cf_role="mesh_topology").keys())[
        0
    ]
    ds_out2d = ds_out2d.rename({grid_topology_name: "grid_topology"})

    # Coordinates
    # get the names of node_lon and node_lat
    node_x_name, node_y_name = ds_out2d["grid_topology"].node_coordinates.split()
    coord_dict = {
        node_x_name: ugrid.CARTESIAN_NODE_COORDINATES[0],
        node_y_name: ugrid.CARTESIAN_NODE_COORDINATES[1],
    }

    if "edge_coordinates" in ds_out2d["grid_topology"].attrs:
        # get the names of edge_lon and edge_lat, if they exist
        edge_x_name, edge_y_name = ds_out2d["grid_topology"].edge_coordinates.split()
        coord_dict[edge_x_name] = ugrid.CARTESIAN_EDGE_COORDINATES[0]
        coord_dict[edge_y_name] = ugrid.CARTESIAN_EDGE_COORDINATES[1]

    if "face_coordinates" in ds_out2d["grid_topology"].attrs:
        # get the names of face_lon and face_lat, if they exist
        face_x_name, face_y_name = ds_out2d["grid_topology"].face_coordinates.split()
        coord_dict[face_x_name] = ugrid.CARTESIAN_FACE_COORDINATES[0]
        coord_dict[face_y_name] = ugrid.CARTESIAN_FACE_COORDINATES[1]

    ds_out2d = ds_out2d.rename(coord_dict)

    # Connectivity

    conn_dict = {}
    for conn_name in ugrid.CONNECTIVITY_NAMES:
        if conn_name in ds_out2d.grid_topology.attrs:
            orig_conn_name = ds_out2d.grid_topology.attrs[conn_name]
            conn_dict[orig_conn_name] = conn_name
        elif len(ds_out2d.filter_by_attrs(cf_role=conn_name).keys()):
            orig_conn_name = list(ds_out2d.filter_by_attrs(cf_role=conn_name).keys())[0]
            conn_dict[orig_conn_name] = conn_name

    ds_out2d = ds_out2d.rename(conn_dict)

    for conn_name in conn_dict.values():
        ds_out2d = _standardize_connectivity(ds_out2d, conn_name)

    dim_dict = {}
    dim_dict_zcoords = {}

    # Rename Core Dims (node, edge, face)
    if "node_dimension" in ds_out2d["grid_topology"]:
        dim_dict[ds_out2d["grid_topology"].node_dimension] = ugrid.NODE_DIM
    else:
        dim_ori = ds_out2d["node_x"].dims[0]
        dim_dict[dim_ori] = ugrid.NODE_DIM
        if ds_sgrid_info is not None and dim_ori in ds_sgrid_info.dims:
            dim_dict_zcoords[dim_ori] = ugrid.NODE_DIM

    if "face_dimension" in ds_out2d["grid_topology"]:
        dim_dict[ds_out2d["grid_topology"].face_dimension] = ugrid.FACE_DIM
    else:
        dim_ori = ds_out2d["face_node_connectivity"].dims[0]
        dim_dict[dim_ori] = ugrid.FACE_DIM
        if ds_sgrid_info is not None and dim_ori in ds_sgrid_info.dims:
            dim_dict_zcoords[dim_ori] = ugrid.FACE_DIM

    if "edge_dimension" in ds_out2d["grid_topology"]:
        # edge dimension is not always provided
        dim_dict[ds_out2d["grid_topology"].edge_dimension] = ugrid.EDGE_DIM
    else:
        if "edge_x" in ds_out2d:
            dim_ori = ds_out2d["edge_x"].dims[0]
            dim_dict[dim_ori] = ugrid.EDGE_DIM
            if ds_sgrid_info is not None and dim_ori in ds_sgrid_info.dims:
                dim_dict_zcoords[dim_ori] = ugrid.EDGE_DIM

    dim_dict[ds_out2d["face_node_connectivity"].dims[1]] = ugrid.N_MAX_FACE_NODES_DIM

    ds_out2d = ds_out2d.swap_dims(dim_dict)
    if ds_sgrid_info is not None:
        ds_sgrid_info = ds_sgrid_info.swap_dims(dim_dict_zcoords)

    return ds_out2d, dim_dict, ds_sgrid_info
