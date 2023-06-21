import xarray as xr


def read_schism_nc(path_nc, **kwargs):
    """Read SCHISM NetCDF output files and return a suxarray Grid

    This function reads SCHISM output file(s) using `xarray.open_mfdataset` and
    return a `xarray.Dataset` object. Note that the data will be read lazily
    and the files will be left open.

    Parameters
    ----------
    path_nc : str, Path, or list
        Path pattern or list of file paths to the SCHISM NetCDF output file(s)
    **kwargs : dict

    Return
    ------
    xarray.Dataset
    """
    # Read the data
    ds = xr.open_mfdataset(
       path_nc, data_vars="minimal", mask_and_scale=False, **kwargs
    )

    # Add extra information to make dataset CF compliant for convenience
    # Check if "Mesh2" variable exists
    if "Mesh2" not in ds:
        Mesh2 = xr.DataArray(
            data="",
            name="SCHISM_hgrid",
            attrs={
                "cf_role": "mesh_topology",
                "node_coordinates": "SCHISM_hgrid_node_x SCHISM_hgrid_node_y",
                "face_node_connectivity": "SCHISM_hgrid_face_nodes",
                "topology_dimension": 2,
            },
        )
        ds[Mesh2.name] = Mesh2

    if "units" not in ds.time.attrs:
        base_date = ds.time.attrs.get("base_date")
        if base_date is not None:
            # NOTE Do not use the UTC timezone for now
            # NOTE Ignore the hours for now
            tokens = base_date.strip().split()
            cf_time_str = "seconds since {:4d}-{:02d}-{:02d} 00:00:00".format(
                *[int(x) for x in tokens[:3]]
            )
            ds.time.attrs["units"] = cf_time_str

    # Add more attributes for cf-conventions
    if "SCHISM_hgrid_face_nodes" in ds:
        ds.SCHISM_hgrid_face_nodes.attrs["start_index"] = 1
        ds.SCHISM_hgrid_face_nodes.attrs["_FillValue"] = -1
        ds.SCHISM_hgrid_face_nodes.attrs["cf_role"] = "face_node_connectivity"

    # Do not mask to keep the original data types and values
    ds = xr.decode_cf(ds, mask_and_scale=False)

    return ds
