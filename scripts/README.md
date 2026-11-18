# Suxarray Example Scripts

This directory contains example command-line scripts that use `suxarray` to
postprocess SCHISM NetCDF output.

The scripts are intended as practical use cases for scientific analysis.
They can be run directly, adapted to a project-specific data layout, or used
as examples for building related processing applications.

## Contents

| Script | Purpose |
| --- | --- |
| [`calc_da_daily.py`](calc_da_daily.py) | Processes a single day of SCHISM output and computes depth-averaged values at mesh nodes and faces. |
| [`calc_da_multi_days.py`](calc_da_multi_days.py) | Processes one or more days of SCHISM output in parallel and computes depth-averaged values at mesh nodes and faces. |
| [`calc_lsz_daily.py`](calc_lsz_daily.py) | Calculates daily low-salinity-zone (LSZ) areas for one or more salinity thresholds, both over the full mesh and within named subregions. |
| [`calc_lsz_temp_daily.py`](calc_lsz_temp_daily.py) | Calculates daily LSZ areas constrained by both daily mean salinity and daily maximum temperature thresholds. |
| [`calc_lsz_temp_turbidity.py`](calc_lsz_temp_turbidity.py) | Calculates daily LSZ areas constrained by salinity, temperature, and log-turbidity (`> ln(12)`) thresholds over May 1–November 30. |

## Workflow

The scripts support a typical two-stage workflow:

1. Use `calc_da_daily.py` (for a single day) or `calc_da_multi_days.py` (for
   multiple days in parallel) to depth-average variables such as salinity and
   temperature.
2. Combine or provide the resulting face values to `calc_lsz_daily.py`,
   `calc_lsz_temp_daily.py`, or `calc_lsz_temp_turbidity.py` to calculate daily
   LSZ areas.

## Depth-Averaging a Single Day

`calc_da_daily.py` computes node- and face-centered depth averages for one
daily SCHISM variable file. For a given `--nc_filename` (such as
`salinity_1.nc`), the script reads:

- `out2d_<day>.nc`
- `zCoordinates_<day>.nc`
- `<variable>_<day>.nc`

It writes two NetCDF files to `--output-dir`:

- `depth_averaged_<variable>_<day>.nc` for node-centered values
- `depth_averaged_<variable>_at_face_<day>.nc` for face-centered values

Example:

```bash
python scripts/calc_da_daily.py \
  --nc_filename salinity_1.nc \
  --input-dir /data/schism \
  --output-dir /data/processed \
  --time-chunks 2
```

Use `--no-compress` to disable NetCDF4 zlib compression.

## Depth-Averaging Multiple Days

`calc_da_multi_days.py` uses Dask and process-based parallelism to process
several SCHISM output days. For each day, the script reads:

- `out2d_<day>.nc`
- `zCoordinates_<day>.nc`
- `<variable>_<day>.nc`

It writes two NetCDF files for each successful day:

- `depth_averaged_<variable>_<day>.nc` for node-centered values
- `depth_averaged_<variable>_at_face_<day>.nc` for face-centered values

Example:

```bash
python scripts/calc_da_multi_days.py \
  --varname salinity \
  --input-dir /data/schism \
  --output-dir /data/processed \
  --days 1-10 \
  --num-workers 8 \
  --threads-per-worker 2 \
  --time-chunks 4
```

Days can be selected with `--start-day`, `--end-day`, or a `--days` expression
such as `1,3,5-7`. If none is supplied, the script discovers available
`<variable>_<day>.nc` files in the input directory.

Existing output files are skipped by default. Use `--overwrite` to replace
them. Use `--no-compress` to disable NetCDF zlib compression.

## Calculating Daily LSZ Areas

`calc_lsz_daily.py` reads a postprocessed NetCDF variable containing
depth-averaged salinity at faces and calculates the area where salinity is
below each requested threshold.

For one threshold, the script produces:

- `lsz_less_than_<threshold>.nc` and `.csv`: total LSZ area for the full mesh
- `lsz_area_less_than_<threshold>.nc` and `.csv`: LSZ and total area for each
  subregion

Example:

```bash
python scripts/calc_lsz_daily.py \
  --path_out2d /data/schism/out2d_1.nc \
  --subregion_fn /data/masks/subregions.nc \
  --path_salinity_nc /data/processed/depth_averaged_salinity_at_face.nc \
  --path_zcoords /data/schism/zCoordinates_1.nc \
  --output-dir /data/lsz \
  --thresholds 2.0 \
  --thresholds 6.0
```

The script also accepts comma-separated thresholds, such as
`--thresholds 2.0,6.0`.

The subregion NetCDF file must contain face-indexed binary masks: each variable
is the mask for one named region and uses `1` for faces inside that region and
`0` otherwise. Face dimensions are aligned to the salinity data automatically
when possible.

The planar face areas are calculated from the `out2d` SCHISM mesh using
`suxarray` `Grid.face_areas`. The `--path_zcoords` option is optional and
retained for backward compatibility (it is not required for 2D face areas).

`--output-dir` defaults to the current directory (`.`). Use `--chunks` to set
the Dask chunking JSON for the salinity input (default `'{"time": -1}'`), and
`--no-compress` to disable NetCDF zlib compression.

## Calculating Daily LSZ + Temperature Areas

`calc_lsz_temp_daily.py` reads postprocessed face-centered depth-averaged
salinity (`depth_averaged_salinity_at_face`) and temperature
(`depth_averaged_temperature_at_face`) NetCDF files, computes daily mean
salinity and daily maximum temperature, and calculates the area where both are
below each requested `(SALINITY, TEMPERATURE)` threshold pair.

For each threshold pair (default `6.0 22.0`), the script writes CSV outputs:

- `lsz_less_than_<sal>_<temp>.csv`: total LSZ area for the full mesh
- `lsz_area_less_than_<sal>_<temp>.csv`: LSZ and total area for each subregion

Example:

```bash
python scripts/calc_lsz_temp_daily.py \
  --path_out2d /data/schism/out2d_1.nc \
  --subregion_fn /data/masks/subregions.nc \
  --path_salinity_nc /data/processed/depth_averaged_salinity_at_face.nc \
  --path_temperature_nc /data/processed/depth_averaged_temperature_at_face.nc \
  --output-dir /data/lsz_temp \
  --thresholds 6.0 22.0 \
  --thresholds 2.0 20.0
```

## Calculating Daily LSZ + Temperature + Turbidity Areas

`calc_lsz_temp_turbidity.py` extends the salinity-and-temperature LSZ
calculation with a daily log-turbidity constraint (`log turbidity > ln(12)`)
interpolated from a 214-row coarse-grid May 1–November 30 data to the
fine SCHISM faces. Dates outside May 1–November 30 are skipped.

By default, the turbidity data, interpolation weights, and element
mapping CSVs are loaded from `scripts/common/` (`ln_turbidity_quantile_95_fit.csv`,
`weights_2019.csv`, and `elems_matrix_2019.csv`), or can be overridden with
`--path_turbidity_csv`, `--path_weights_csv`, and `--path_elems_matrix_csv`.

For each `(SALINITY, TEMPERATURE)` threshold pair (default `6.0 22.0`), the
script writes CSV outputs:

- `lsz_less_than_<sal>_<temp>_gt_ln12.csv`: total LSZ area for the full mesh
- `lsz_area_less_than_<sal>_<temp>_gt_ln12.csv`: LSZ and total area for each
  subregion

Example:

```bash
python scripts/calc_lsz_temp_turbidity.py \
  --path_out2d /data/schism/out2d_1.nc \
  --subregion_fn /data/masks/subregions.nc \
  --path_salinity_nc /data/processed/depth_averaged_salinity_at_face.nc \
  --path_temperature_nc /data/processed/depth_averaged_temperature_at_face.nc \
  --output-dir /data/lsz_temp_turbidity \
  --thresholds 6.0 22.0
```

## Common Requirements

These scripts require the packages used by `suxarray`, plus:

- Click
- Dask
- NumPy
- pandas
- SciPy (for `calc_lsz_temp_turbidity.py`)
- xarray

