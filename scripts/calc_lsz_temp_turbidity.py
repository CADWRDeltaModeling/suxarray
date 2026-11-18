"""Calculate daily LSZ areas constrained by salinity, temperature, and turbidity."""

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
from scipy.sparse import coo_matrix
import suxarray as sx
import xarray as xr

COMMON_DIR = Path(__file__).resolve().parent / "common"
DEFAULT_TURBIDITY_PATH = COMMON_DIR / "ln_turbidity_quantile_95_fit.csv"
DEFAULT_WEIGHTS_PATH = COMMON_DIR / "weights_2019.csv"
DEFAULT_ELEMS_MATRIX_PATH = COMMON_DIR / "elems_matrix_2019.csv"
TURBIDITY_DATES = pd.date_range("2001-05-01", "2001-11-30", freq="D") # the year is arbitrary, only the month and day matter
TURBIDITY_THRESHOLD = np.log(12.0)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@click.command(
    help=(
        "Calculate daily LSZ where salinity and temperature are below their "
        "thresholds and log turbidity is greater than ln(12)."
    )
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
    help="Path to the subregion NetCDF file.",
)
@click.option(
    "--path_salinity_nc",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to depth-averaged salinity at face.",
)
@click.option(
    "--path_temperature_nc",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to depth-averaged temperature at face.",
)
@click.option(
    "--path_turbidity_csv",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_TURBIDITY_PATH,
    show_default=True,
    help="Path to the 214-row coarse-grid log-turbidity climatology.",
)
@click.option(
    "--path_weights_csv",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_WEIGHTS_PATH,
    show_default=True,
    help="Path to the coarse-to-fine interpolation weights.",
)
@click.option(
    "--path_elems_matrix_csv",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_ELEMS_MATRIX_PATH,
    show_default=True,
    help="Path to the 1-based fine-face/coarse-value index pairs.",
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
    help="Output suffix, such as _121 for one simulation day file.",
)
@click.option(
    "--thresholds",
    type=(float, float),
    multiple=True,
    default=((6.0, 22.0),),
    show_default=True,
    metavar="SALINITY TEMPERATURE",
    help="Salinity and temperature threshold pair. May be supplied more than once.",
)
@click.option(
    "--chunks",
    type=str,
    default='{"time": -1}',
    show_default=True,
    help="xarray chunks as a JSON object.",
)
def calc_lsz_cli(
    path_out2d,
    subregion_fn,
    path_salinity_nc,
    path_temperature_nc,
    path_turbidity_csv,
    path_weights_csv,
    path_elems_matrix_csv,
    path_zcoords,
    output_dir,
    out_suffix,
    thresholds,
    chunks,
):
    chunks_dict = json.loads(chunks)
    if not isinstance(chunks_dict, dict):
        raise click.BadParameter("must be a JSON object", param_hint="--chunks")

    calc_lsz_single(
        path_out2d=path_out2d,
        subregion_fn=subregion_fn,
        path_salinity_nc=path_salinity_nc,
        path_temperature_nc=path_temperature_nc,
        path_turbidity_csv=path_turbidity_csv,
        path_weights_csv=path_weights_csv,
        path_elems_matrix_csv=path_elems_matrix_csv,
        path_zcoords=path_zcoords,
        output_dir=output_dir,
        out_suffix=out_suffix,
        thresholds=thresholds,
        chunks=chunks_dict,
    )


def load_turbidity_threshold_mask(
    path_turbidity_csv,
    path_weights_csv,
    path_elems_matrix_csv,
    number_of_faces,
):
    turbidity = pd.read_csv(path_turbidity_csv, header=None).to_numpy(dtype=float)
    weights = pd.read_csv(path_weights_csv, header=None).to_numpy(dtype=float)
    mapping_values = pd.read_csv(path_elems_matrix_csv, header=None).to_numpy(dtype=float)

    if turbidity.shape[0] != len(TURBIDITY_DATES):
        raise ValueError(
            f"Expected {len(TURBIDITY_DATES)} turbidity rows for May 1-November 30, "
            f"but found {turbidity.shape[0]}."
        )
    if mapping_values.ndim != 2 or mapping_values.shape[1] != 2:
        raise ValueError("Element mapping must contain exactly two columns.")
    if weights.ndim != 2 or weights.shape[1] != 1:
        raise ValueError("Weights file must contain exactly one column.")
    if mapping_values.shape[0] != weights.shape[0]:
        raise ValueError("Element mapping and weights must contain the same number of rows.")
    if not np.isfinite(turbidity).all() or not np.isfinite(weights).all():
        raise ValueError("Turbidity and weights must contain only finite numeric values.")
    if not np.isfinite(mapping_values).all() or not np.equal(
        mapping_values, np.floor(mapping_values)
    ).all():
        raise ValueError("Element mapping indices must be finite integers.")

    mapping = mapping_values.astype(np.int64) - 1
    if (mapping < 0).any():
        raise ValueError("Element mapping indices must be positive and 1-based.")
    if mapping[:, 0].max() >= number_of_faces:
        raise ValueError("Element mapping references a fine face outside the SCHISM mesh.")
    if mapping[:, 1].max() >= turbidity.shape[1]:
        raise ValueError("Element mapping references a coarse value outside the turbidity CSV.")
    if mapping[:, 0].max() + 1 != number_of_faces:
        raise ValueError(
            "Element mapping fine-face count does not match the SCHISM mesh face count."
        )

    interpolation_matrix = coo_matrix(
        (weights[:, 0], (mapping[:, 0], mapping[:, 1])),
        shape=(number_of_faces, turbidity.shape[1]),
    ).tocsr()
    row_sums = np.asarray(interpolation_matrix.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=2e-4):
        raise ValueError("Coarse-to-fine interpolation weights do not sum to one per face.")

    threshold_mask = np.empty((len(TURBIDITY_DATES), number_of_faces), dtype=bool)
    for day_index, coarse_values in enumerate(turbidity):
        threshold_mask[day_index] = (
            interpolation_matrix @ coarse_values
        ) > TURBIDITY_THRESHOLD

    return threshold_mask


def turbidity_mask_for_times(
    threshold_mask, times, face_coordinate, face_dim: str = "n_face"
):
    times = pd.DatetimeIndex(times)
    climatology_lookup = {
        date.strftime("%m-%d"): index for index, date in enumerate(TURBIDITY_DATES)
    }
    climatology_indices = np.array(
        [climatology_lookup.get(timestamp.strftime("%m-%d"), -1) for timestamp in times]
    )
    for timestamp in times[climatology_indices < 0]:
        logging.info(
            "%s not processed since it is outside the May 1 to November 30 window.",
            timestamp.strftime("%Y-%m-%d"),
        )
    common_time_indices = np.flatnonzero(climatology_indices >= 0)

    return xr.DataArray(
        threshold_mask[climatology_indices[common_time_indices]],
        dims=("time", face_dim),
        coords={
            "time": times[common_time_indices],
            face_dim: face_coordinate,
        },
        name="log_turbidity_greater_than_ln12",
    )


def format_threshold(value):
    return f"{value:g}".replace(".", "p")


def calc_lsz_single(
    path_out2d,
    subregion_fn,
    path_salinity_nc,
    path_temperature_nc,
    path_turbidity_csv=DEFAULT_TURBIDITY_PATH,
    path_weights_csv=DEFAULT_WEIGHTS_PATH,
    path_elems_matrix_csv=DEFAULT_ELEMS_MATRIX_PATH,
    path_zcoords=None,
    output_dir=Path("."),
    out_suffix="",
    thresholds=((6.0, 22.0),),
    chunks=None,
):
    """Calculate daily LSZ constrained by salinity, temperature, and turbidity."""
    if chunks is None:
        chunks = {"time": -1}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading postprocessed depth-averaged salinity and temperature...")
    ds_salinity = xr.open_dataset(path_salinity_nc, chunks=chunks)
    da_salinity = ds_salinity["depth_averaged_salinity_at_face"]
    face_dim = next(d for d in da_salinity.dims if d != "time")

    ds_temperature = xr.open_dataset(path_temperature_nc, chunks=chunks)
    da_temperature = ds_temperature["depth_averaged_temperature_at_face"]
    temp_face_dim = next(d for d in da_temperature.dims if d != "time")
    if temp_face_dim != face_dim:
        da_temperature = da_temperature.rename({temp_face_dim: face_dim})

    logging.info("Computing planar face areas for mesh geometry...")
    da_face_areas = sx.open_grid(path_out2d).face_areas
    if da_face_areas.dims[0] != face_dim:
        da_face_areas = da_face_areas.rename({da_face_areas.dims[0]: face_dim})
    number_of_faces = da_face_areas.sizes[face_dim]

    logging.info("Interpolating log turbidity from the coarse grid to fine faces...")
    turbidity_threshold_mask = load_turbidity_threshold_mask(
        path_turbidity_csv,
        path_weights_csv,
        path_elems_matrix_csv,
        number_of_faces,
    )

    if da_salinity.chunks is not None:
        da_salinity = da_salinity.chunk({"time": -1})
    if da_temperature.chunks is not None:
        da_temperature = da_temperature.chunk({"time": -1})

    da_salinity_daily = (
        da_salinity.interpolate_na(dim="time", limit=4)
        .resample(time="1D", origin="start")
        .mean(skipna=False)
        .compute()
    )
    da_temperature_daily = (
        da_temperature.interpolate_na(dim="time", limit=4)
        .resample(time="1D", origin="start")
        .max(skipna=False)
        .compute()
    )
    da_salinity_daily, da_temperature_daily = xr.align(
        da_salinity_daily, da_temperature_daily, join="inner"
    )
    face_coord = da_salinity_daily.coords.get(
        face_dim, np.arange(da_salinity_daily.sizes[face_dim])
    )
    da_turbidity_threshold = turbidity_mask_for_times(
        turbidity_threshold_mask,
        da_salinity_daily.time.values,
        face_coord,
        face_dim=face_dim,
    )
    if da_turbidity_threshold.sizes["time"] == 0:
        logging.info("No dates fall within the May 1 to November 30 window. Exiting.")
        return
    da_salinity_daily, da_temperature_daily, da_turbidity_threshold = xr.align(
        da_salinity_daily,
        da_temperature_daily,
        da_turbidity_threshold,
        join="inner",
    )

    logging.info("Reading subregion data...")
    ds_subregions = xr.open_dataset(subregion_fn)
    subregion_face_dim = next(iter(ds_subregions.sizes.keys()))
    if subregion_face_dim != face_dim:
        ds_subregions = ds_subregions.rename({subregion_face_dim: face_dim})
    list_regions = list(ds_subregions.keys())

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

    for salinity_threshold, temperature_threshold in thresholds:
        threshold_label = (
            f"{format_threshold(salinity_threshold)}_"
            f"{format_threshold(temperature_threshold)}_gt_ln12"
        )
        logging.info("Calculating LSZ for thresholds %s...", threshold_label)
        da_combined_threshold = xr.where(
            (da_salinity_daily < salinity_threshold)
            & (da_temperature_daily < temperature_threshold)
            & da_turbidity_threshold,
            1.0,
            0.0,
        )
        da_combined_threshold = shift_30min_up(da_combined_threshold)

        da_lsz_total = xr.dot(
            da_combined_threshold, da_face_areas, dims=face_dim
        )
        da_lsz_total.name = f"lsz_temp_turbidity_less_than_{threshold_label}"
        da_lsz_total.attrs["units"] = "m2"
        da_lsz_total.to_dataframe().to_csv(
            output_dir / f"lsz_less_than_{threshold_label}{out_suffix}.csv"
        )

        logging.info("Calculating LSZ area for each region...")
        list_areas = [
            xr.dot(
                region_face_areas[region],
                da_combined_threshold.isel({face_dim: region_masks[region]}),
                dims=face_dim,
            )
            for region in list_regions
        ]

        da_area_lsz_all = xr.concat(list_areas, pd.Index(list_regions, name="region"))
        da_area_lsz_all.name = "lsz_area"
        da_area_lsz_all.attrs["units"] = "m2"
        da_area_lsz_all.attrs["long_name"] = (
            "LSZ area where salinity and temperature are below "
            f"{salinity_threshold:g} and {temperature_threshold:g}, and log turbidity "
            "is greater than ln(12)"
        )
        ds_area_lsz = xr.Dataset({"lsz_area": da_area_lsz_all, "subarea": da_subareas})
        ds_area_lsz.to_dataframe().to_csv(
            output_dir / f"lsz_area_less_than_{threshold_label}{out_suffix}.csv"
        )

    logging.info("Done")


def shift_30min_up(da):
    """Shift time coordinate 30 minutes up."""
    return da.assign_coords(time=da.coords["time"] - pd.to_timedelta("30m"))


if __name__ == "__main__":
    calc_lsz_cli()
