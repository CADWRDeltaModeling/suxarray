"""Calculate daily low salinity zone (LSZ) areas under thresholds."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# Disable HDF5 file locking to avoid hangs on shared NFS mounts
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import click
import numpy as np
import pandas as pd
import suxarray as sx
from suxarray.core.api import _choose_netcdf_engine
import xarray as xr


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_thresholds(val: list[float] | tuple | str | float) -> list[float]:
    """Parse thresholds from list, tuple, float, or comma-separated string."""
    if isinstance(val, (list, tuple)):
        result = []
        for item in val:
            if isinstance(item, str) and "," in item:
                result.extend([float(x.strip()) for x in item.split(",") if x.strip()])
            else:
                result.append(float(item))
        return result
    elif isinstance(val, str):
        return [float(x.strip()) for x in val.split(",") if x.strip()]
    elif isinstance(val, (int, float)):
        return [float(val)]
    return [6.0]


def shift_30min_up(da: xr.DataArray) -> xr.DataArray:
    """Shift time coordinate 30 minutes up to align daily resample bins with hour boundary."""
    return da.assign_coords(time=da.coords["time"] - pd.to_timedelta("30m"))


def calc_lsz_single(
    path_out2d: Path,
    subregion_fn: Path,
    path_salinity_nc: Path,
    path_zcoords: Path | None = None,
    output_dir: Path = Path("."),
    out_suffix: str = "",
    thresholds: list[float] | None = None,
    chunks: dict | None = None,
    compress: bool = True,
):
    """Calculate LSZ less than salinity thresholds."""
    if thresholds is None:
        thresholds = [6.0]
    if chunks is None:
        chunks = {"time": -1}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read processed depth-average salinity at face
    logger.info("Reading postprocessed depth-averaged salinity at face from %s...", path_salinity_nc)
    ds_salinity = xr.open_dataset(
        path_salinity_nc, engine=_choose_netcdf_engine(), chunks=chunks
    )
    da_depth_averaged_salinity_at_face = ds_salinity["depth_averaged_salinity_at_face"]

    # Identify face dimension in salinity array
    face_dim = next(d for d in da_depth_averaged_salinity_at_face.dims if d != "time")

    # 2. Read subregion masks
    logger.info("Reading subregion data from %s...", subregion_fn)
    ds_subregions = xr.open_dataset(subregion_fn, engine=_choose_netcdf_engine())
    subregion_face_dim = next(iter(ds_subregions.sizes.keys()))
    if subregion_face_dim != face_dim:
        ds_subregions = ds_subregions.rename({subregion_face_dim: face_dim})
    list_regions = list(ds_subregions.keys())

    # 3. Compute planar face areas using suxarray Grid.face_areas
    logger.info("Computing planar face areas for mesh geometry...")
    da_face_areas = sx.open_grid(path_out2d).face_areas
    if da_face_areas.dims[0] != face_dim:
        da_face_areas = da_face_areas.rename({da_face_areas.dims[0]: face_dim})

    # Precompute subregion masks and static subregion areas once
    region_masks = {region: (ds_subregions[region] == 1).values for region in list_regions}
    region_face_areas = {
        region: da_face_areas.isel({face_dim: mask})
        for region, mask in region_masks.items()
    }
    list_total_areas = [region_face_areas[region].sum() for region in list_regions]
    da_subareas = xr.concat(list_total_areas, pd.Index(list_regions, name="region"))
    da_subareas.name = "subarea"
    da_subareas.attrs["units"] = "m2"
    da_subareas.attrs["long_name"] = "Total subregion area"

    # 4. Interpolate short gaps and compute daily mean salinity once across all thresholds
    logger.info("Computing daily mean salinity...")
    if da_depth_averaged_salinity_at_face.chunks is not None:
        da_depth_averaged_salinity_at_face = da_depth_averaged_salinity_at_face.chunk(
            {"time": -1}
        )
    da_salinity_interp = da_depth_averaged_salinity_at_face.interpolate_na(
        dim="time", limit=4
    )
    da_salinity_daily = (
        da_salinity_interp.resample(time="1D", origin="start")
        .mean(skipna=False)
        .compute()
    )
    da_salinity_daily = shift_30min_up(da_salinity_daily)

    # 5. Calculate LSZ areas for each threshold
    for threshold in thresholds:
        thresh_str = str(threshold).replace(".", "p")
        logger.info("Calculating LSZ less than %s PSU...", thresh_str)

        da_salinity_less_than_threshold_daily = xr.where(
            da_salinity_daily < threshold, 1.0, 0.0
        )

        # Total mesh LSZ area
        da_lsz_total_threshold = xr.dot(
            da_salinity_less_than_threshold_daily, da_face_areas, dims=face_dim
        )
        da_lsz_total_threshold.name = f"lsz_less_than_{thresh_str}"
        da_lsz_total_threshold.attrs["units"] = "m2"
        da_lsz_total_threshold.attrs["long_name"] = f"Total mesh area with salinity < {threshold} PSU"

        path_total_nc = output_dir / f"lsz_less_than_{thresh_str}{out_suffix}.nc"
        path_total_csv = output_dir / f"lsz_less_than_{thresh_str}{out_suffix}.csv"

        encoding_total = {
            da_lsz_total_threshold.name: (
                {"zlib": True, "complevel": 1} if compress else {}
            )
        }
        da_lsz_total_threshold.to_netcdf(path_total_nc, encoding=encoding_total)
        da_lsz_total_threshold.to_dataframe().to_csv(path_total_csv)

        # Per-subregion LSZ area
        logger.info("Calculating LSZ area for each subregion under %s PSU...", thresh_str)
        list_areas = [
            xr.dot(
                region_face_areas[region],
                da_salinity_less_than_threshold_daily.isel({face_dim: region_masks[region]}),
                dims=face_dim,
            )
            for region in list_regions
        ]

        da_area_lsz_all = xr.concat(list_areas, pd.Index(list_regions, name="region"))
        da_area_lsz_all.name = "lsz_area"
        da_area_lsz_all.attrs["units"] = "m2"
        da_area_lsz_all.attrs["long_name"] = (
            f"LSZ area for each region under {thresh_str} PSU"
        )

        ds_area_lsz = xr.Dataset({"lsz_area": da_area_lsz_all, "subarea": da_subareas})

        path_out_nc = output_dir / f"lsz_area_less_than_{thresh_str}{out_suffix}.nc"
        path_out_csv = output_dir / f"lsz_area_less_than_{thresh_str}{out_suffix}.csv"

        encoding_subregion = {
            "lsz_area": {"zlib": True, "complevel": 1} if compress else {},
            "subarea": {"zlib": True, "complevel": 1} if compress else {},
        }
        ds_area_lsz.to_netcdf(path_out_nc, encoding=encoding_subregion)
        ds_area_lsz.to_dataframe().to_csv(path_out_csv)

    logger.info("Finished LSZ area calculations successfully.")


@click.command(
    help="Calculate LSZ less than salinity thresholds. Produces lsz_area_less_than_*.nc and lsz_area_less_than_*.csv files."
)
@click.option(
    "--path_out2d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the out2d file.",
)
@click.option(
    "--subregion_fn",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the subregion nc file.",
)
@click.option(
    "--path_salinity_nc",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the salinity NetCDF file (with depth-averaged salinity at face).",
)
@click.option(
    "--path_zcoords",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional path to zCoordinates file (retained for backward compatibility; not required for 2D face areas).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory to write the LSZ NetCDF and CSV outputs.",
)
@click.option(
    "--out_suffix",
    type=str,
    default="",
    show_default=True,
    help="Output suffix for the NetCDF and CSV files.",
)
@click.option(
    "--thresholds",
    multiple=True,
    default=(6.0,),
    show_default=True,
    help="Salinity thresholds in PSU. Can specify multiple times or comma-separated.",
)
@click.option(
    "--chunks",
    type=str,
    default='{"time": -1}',
    show_default=True,
    help="Chunks to use when reading the salinity NetCDF file, provided as a JSON string.",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    show_default=True,
    help="Enable NetCDF4 zlib compression on output files.",
)
def calc_lsz_cli(
    path_out2d,
    subregion_fn,
    path_salinity_nc,
    path_zcoords,
    output_dir,
    out_suffix,
    thresholds,
    chunks,
    compress,
):
    """CLI entrypoint to calculate daily LSZ areas."""
    if isinstance(chunks, str):
        chunks_dict = json.loads(chunks)
    else:
        chunks_dict = chunks

    parsed_thresholds = parse_thresholds(thresholds)

    calc_lsz_single(
        path_out2d=path_out2d,
        subregion_fn=subregion_fn,
        path_salinity_nc=path_salinity_nc,
        path_zcoords=path_zcoords,
        output_dir=output_dir,
        out_suffix=out_suffix,
        thresholds=parsed_thresholds,
        chunks=chunks_dict,
        compress=compress,
    )


if __name__ == "__main__":
    calc_lsz_cli()
