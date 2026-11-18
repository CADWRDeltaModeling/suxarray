"""Calculate daily depth-averaged SCHISM variables."""

import logging
import os
from pathlib import Path

# Disable HDF5 file locking to avoid hangs on shared NFS mounts
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import click
import numpy as np
import suxarray as sx
import xarray as xr


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@click.command()
@click.option(
    "--nc_filename",
    type=str,
    required=True,
    help="Variable NetCDF filename, such as salinity_1.nc.",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing raw SCHISM NetCDF output files.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for depth-averaged NetCDF files.",
)
@click.option(
    "--time-chunks",
    type=int,
    default=2,
    show_default=True,
    help="Chunk size along the time dimension for Dask.",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    show_default=True,
    help="Enable NetCDF4 zlib compression on output files.",
)
def calculate_depth_average(
    nc_filename, input_dir, output_dir, time_chunks=2, compress=True
):
    """Calculate node and face depth averages for one daily variable file."""
    logger = logging.getLogger(__name__)
    logger.info("Starting to calculate depth averaged values...")

    chunks = {"time": time_chunks}

    filename = Path(nc_filename).name
    varname, fnumber_with_extension = filename.split("_", maxsplit=1)
    fnumber = Path(fnumber_with_extension).stem

    out2d = input_dir / f"out2d_{fnumber}.nc"
    zcoord = input_dir / f"zCoordinates_{fnumber}.nc"
    var_nc = input_dir / filename

    for required_file in (out2d, zcoord, var_nc):
        if not required_file.exists():
            raise FileNotFoundError(f"Required input file not found: {required_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading SCHISM grid and variable files...")
    grid = sx.open_grid(out2d, zcoord, chunks=chunks)
    ds_var = xr.open_dataset(var_nc, chunks=chunks).astype(np.float64)
    sx_ds = sx.read_schism_nc(grid, ds_var)

    path_depth_averaged_var = output_dir / f"depth_averaged_{varname}_{fnumber}.nc"
    logger.info("Calculating depth-averaged %s at nodes...", varname)

    da_depth_averaged_var = sx_ds[varname].depth_average()
    da_depth_averaged_var.name = f"depth_averaged_{varname}"
    if "units" in sx_ds[varname].attrs:
        da_depth_averaged_var.attrs["units"] = sx_ds[varname].attrs["units"]
    da_depth_averaged_var.attrs["long_name"] = f"depth-averaged {varname}"
    da_depth_averaged_var.attrs["location"] = "node"

    encoding_node = {
        da_depth_averaged_var.name: (
            {"dtype": "float32", "zlib": True, "complevel": 1}
            if compress
            else {"dtype": "float32"}
        )
    }

    logger.info("Writing node depth-averaged variable to %s...", path_depth_averaged_var)
    xr.Dataset(
        {da_depth_averaged_var.name: da_depth_averaged_var.astype(np.float32)}
    ).to_netcdf(path_depth_averaged_var, encoding=encoding_node)

    logger.info("Calculating face average via uxarray integration...")
    # Calculate face areas by integrating a constant unit field
    da_ones = da_depth_averaged_var.isel(time=slice(0, 1)) * 0.0 + 1.0
    face_areas = da_ones.integrate().isel(time=0).compute()

    # Integrate the depth-averaged field over each face and divide by face area
    da_integral = da_depth_averaged_var.integrate()
    da_depth_averaged_var_at_face = (da_integral / face_areas).astype(np.float32)
    da_depth_averaged_var_at_face.name = f"depth_averaged_{varname}_at_face"
    if "units" in sx_ds[varname].attrs:
        da_depth_averaged_var_at_face.attrs["units"] = sx_ds[varname].attrs["units"]
    da_depth_averaged_var_at_face.attrs["long_name"] = (
        f"depth-averaged {varname} at face"
    )
    da_depth_averaged_var_at_face.attrs["location"] = "face"

    path_depth_averaged_var_at_face = (
        output_dir / f"depth_averaged_{varname}_at_face_{fnumber}.nc"
    )
    encoding_face = {
        da_depth_averaged_var_at_face.name: (
            {"dtype": "float32", "zlib": True, "complevel": 1}
            if compress
            else {"dtype": "float32"}
        )
    }

    logger.info(
        "Writing face depth-averaged variable to %s...",
        path_depth_averaged_var_at_face,
    )
    xr.Dataset(
        {da_depth_averaged_var_at_face.name: da_depth_averaged_var_at_face}
    ).to_netcdf(path_depth_averaged_var_at_face, encoding=encoding_face)

    logger.info("Finished calculating depth averaged values.")


if __name__ == "__main__":
    calculate_depth_average()