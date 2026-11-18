"""Calculate daily LSZ areas constrained by salinity and temperature thresholds."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# Disable HDF5 file locking to avoid hangs on shared NFS mounts
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import click
import pandas as pd
import suxarray as sx
import xarray as xr


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def shift_30min_up(da: xr.DataArray) -> xr.DataArray:
    """Shift time coordinate 30 minutes up to align daily resample bins with hour boundary."""
    return da.assign_coords(time=da.coords["time"] - pd.to_timedelta("30m"))


def calc_lsz_single(
    path_out2d: Path,
    subregion_fn: Path,
    path_salinity_nc: Path,
    path_temperature_nc: Path,
    path_zcoords: Path | None = None,
    output_dir: Path = Path("."),
    out_suffix: str = "",
    thresholds: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
    chunks: dict | None = None,
):
    """Calculate LSZ less than salinity and temperature thresholds."""
    if thresholds is None:
        thresholds = ((6.0, 22.0),)
    if chunks is None:
        chunks = {"time": -1}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read processed depth-average salinity and temperature at face
    logger.info("Reading postprocessed depth-averaged salinity from %s...", path_salinity_nc)
    ds_salinity = xr.open_dataset(path_salinity_nc, chunks=chunks)
    da_depth_averaged_salinity_at_face = ds_salinity["depth_averaged_salinity_at_face"]
    face_dim = next(d for d in da_depth_averaged_salinity_at_face.dims if d != "time")

    logger.info("Reading postprocessed depth-averaged temperature from %s...", path_temperature_nc)
    ds_temperature = xr.open_dataset(path_temperature_nc, chunks=chunks)
    da_depth_averaged_temperature_at_face = ds_temperature["depth_averaged_temperature_at_face"]
    temp_face_dim = next(d for d in da_depth_averaged_temperature_at_face.dims if d != "time")
    if temp_face_dim != face_dim:
        da_depth_averaged_temperature_at_face = da_depth_averaged_temperature_at_face.rename(
            {temp_face_dim: face_dim}
        )

    # 2. Read subregion masks
    logger.info("Reading subregion data from %s...", subregion_fn)
    ds_subregions = xr.open_dataset(subregion_fn)
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

    # 4. Interpolate short gaps and compute daily mean salinity and max temperature once
    logger.info("Computing daily mean salinity and daily max temperature...")
    if da_depth_averaged_salinity_at_face.chunks is not None:
        da_depth_averaged_salinity_at_face = da_depth_averaged_salinity_at_face.chunk(
            {"time": -1}
        )
    if da_depth_averaged_temperature_at_face.chunks is not None:
        da_depth_averaged_temperature_at_face = da_depth_averaged_temperature_at_face.chunk(
            {"time": -1}
        )

    da_salinity_daily = (
        da_depth_averaged_salinity_at_face.interpolate_na(dim="time", limit=4)
        .resample(time="1D", origin="start")
        .mean(skipna=False)
        .compute()
    )
    da_temperature_daily = (
        da_depth_averaged_temperature_at_face.interpolate_na(dim="time", limit=4)
        .resample(time="1D", origin="start")
        .max(skipna=False)
        .compute()
    )
    da_salinity_daily, da_temperature_daily = xr.align(
        da_salinity_daily, da_temperature_daily, join="inner"
    )

    # 5. Calculate LSZ areas for each threshold pair
    for sal_thresh, temp_thresh in thresholds:
        sal_str = str(sal_thresh).replace(".", "p")
        temp_str = str(temp_thresh).replace(".", "p")
        thresh_label = f"{sal_str}_{temp_str}"
        logger.info("Calculating LSZ less than %s PSU and %s C...", sal_str, temp_str)

        da_combined_daily = xr.where(
            (da_salinity_daily < sal_thresh) & (da_temperature_daily < temp_thresh),
            1.0,
            0.0,
        )
        da_combined_daily = shift_30min_up(da_combined_daily)

        da_lsz_total_threshold = xr.dot(
            da_combined_daily, da_face_areas, dims=face_dim
        )
        da_lsz_total_threshold.name = f"lsz_temp_less_than_{thresh_label}"
        da_lsz_total_threshold.attrs["units"] = "m2"
        da_lsz_total_threshold.to_dataframe().to_csv(
            output_dir / f"lsz_less_than_{thresh_label}{out_suffix}.csv"
        )

        logger.info("Calculating LSZ area for each subregion under %s...", thresh_label)
        list_areas = [
            xr.dot(
                region_face_areas[region],
                da_combined_daily.isel({face_dim: region_masks[region]}),
                dims=face_dim,
            )
            for region in list_regions
        ]

        da_area_lsz_all = xr.concat(list_areas, pd.Index(list_regions, name="region"))
        da_area_lsz_all.name = "lsz_area"
        da_area_lsz_all.attrs["units"] = "m2"
        da_area_lsz_all.attrs["long_name"] = (
            f"LSZ area for each region under {thresh_label}"
        )
        ds_area_lsz = xr.Dataset({"lsz_area": da_area_lsz_all, "subarea": da_subareas})

        path_out_csv = output_dir / f"lsz_area_less_than_{thresh_label}{out_suffix}.csv"
        ds_area_lsz.to_dataframe().to_csv(path_out_csv)

    logger.info("Finished LSZ+temperature area calculations successfully.")


@click.command(
    help="Calculate LSZ less than salinity and temperature thresholds. Produces lsz_area_less_than_*.csv and lsz_less_than_*.csv files."
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
    "--path_temperature_nc",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the temperature NetCDF file (with depth-averaged temperature at face).",
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
    help="Directory to write the LSZ CSV outputs.",
)
@click.option(
    "--out_suffix",
    type=str,
    default="",
    show_default=True,
    help="Output suffix for the CSV files.",
)
@click.option(
    "--thresholds",
    type=(float, float),
    multiple=True,
    default=((6.0, 22.0),),
    show_default=True,
    metavar="SALINITY TEMPERATURE",
    help="Salinity (PSU) and temperature (C) threshold pair. May be supplied more than once.",
)
@click.option(
    "--chunks",
    type=str,
    default='{"time": -1}',
    show_default=True,
    help="Chunks to use when reading the NetCDF files, provided as a JSON string.",
)
def calc_lsz_cli(
    path_out2d,
    subregion_fn,
    path_salinity_nc,
    path_temperature_nc,
    path_zcoords,
    output_dir,
    out_suffix,
    thresholds,
    chunks,
):
    """CLI entrypoint to calculate daily LSZ+temperature areas."""
    chunks_dict = json.loads(chunks) if isinstance(chunks, str) else chunks
    if not isinstance(chunks_dict, dict):
        raise click.BadParameter("must be a JSON object", param_hint="--chunks")

    calc_lsz_single(
        path_out2d=path_out2d,
        subregion_fn=subregion_fn,
        path_salinity_nc=path_salinity_nc,
        path_temperature_nc=path_temperature_nc,
        path_zcoords=path_zcoords,
        output_dir=output_dir,
        out_suffix=out_suffix,
        thresholds=thresholds,
        chunks=chunks_dict,
    )


if __name__ == "__main__":
    calc_lsz_cli()
