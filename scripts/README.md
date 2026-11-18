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

## Workflow

The scripts support a typical two-stage workflow:

1. Use `calc_da_daily.py` (for a single day) or `calc_da_multi_days.py` (for
   multiple days in parallel) to depth-average a variable such as salinity.
2. Combine or provide the resulting face values to `calc_lsz_daily.py`, then
   calculate daily LSZ areas.

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

## Common Requirements

These scripts require the packages used by `suxarray`, plus:

- Click
- Dask
- NumPy
- pandas
- xarray

Set `HDF5_USE_FILE_LOCKING=FALSE` if shared NFS storage causes HDF5 lock hangs.
The scripts set this environment variable automatically unless it is already
defined.
