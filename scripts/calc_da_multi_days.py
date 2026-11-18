"""Calculate daily depth-averaged SCHISM variables for multiple days using Dask.

This script scales SCHISM depth-averaging postprocessing across multiple days
in parallel using Dask multiprocessing, efficiently utilizing multi-core compute nodes.

Recommended Compute & Worker Settings
-------------------------------------
Total CPU cores utilized during computation follow the formula:
    total_cores = num_workers * threads_per_worker

Resource & Sizing Guidelines:
1. 24-Core Node (64 GB RAM) - Balanced (Default / Recommended):
   - --num-workers 8
   - --threads-per-worker 2
   - --time-chunks 4
   - Cores: 16 cores compute Numba JIT depth-averaging and face integration,
     leaving 8 cores free for NetCDF4 zlib compression and filesystem I/O.
   - Memory: ~4 to 5 GB peak RAM per worker process (~35-40 GB total), safely
     fitting within 64 GB nodes without swapping.
   - Storage I/O: Reads ~600 MB/s aggregate from /scratch, consuming ~25%
     of available InfiniBand bandwidth (~2.4 GB/s read).

2. 24-Core Node (128 GB+ RAM) - Maximum Throughput:
   - --num-workers 12
   - --threads-per-worker 2
   - --time-chunks 4
   - Cores: Fully saturates all 24 cores (12 workers * 2 threads).
   - Memory: ~50 to 60 GB peak RAM total.

3. 32-Core Node (128 GB+ RAM):
   - --num-workers 12 to 16
   - --threads-per-worker 2
   - --time-chunks 4

Storage I/O Performance:
- Measured raw InfiniBand /scratch throughput: ~2.4 GB/s read, ~1.3 GB/s write.
- NetCDF C-library parsing/decompression throughput: ~75 MB/s per worker.
- Running 8 to 12 workers draws ~600-900 MB/s, well within storage bandwidth.
"""

import logging
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

# Disable HDF5 file locking to avoid hangs on shared NFS mounts
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import click
import dask
from dask.diagnostics import ProgressBar
import numpy as np
import suxarray as sx
import xarray as xr


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("calc_da_multi_days")


def parse_days_argument(days_str: str) -> list[int]:
    """Parse comma-separated and hyphen-ranged day specifications into sorted integer list.

    Examples:
        "1,2,3" -> [1, 2, 3]
        "1-5" -> [1, 2, 3, 4, 5]
        "1,3,5-7" -> [1, 3, 5, 6, 7]
    """
    days: set[int] = set()
    for token in days_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", maxsplit=1)
            days.update(range(int(start_s.strip()), int(end_s.strip()) + 1))
        else:
            days.add(int(token))
    return sorted(days)


def discover_available_days(input_dir: Path, varname: str) -> list[int]:
    """Discover all integer day numbers matching {varname}_{fnumber}.nc in input_dir."""
    pattern = re.compile(rf"^{re.escape(varname)}_(\d+)\.nc$")
    found_days: list[int] = []
    for item in input_dir.iterdir():
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if match:
            found_days.append(int(match.group(1)))
    return sorted(found_days)


def process_single_day(
    fnumber: int,
    varname: str,
    input_dir: Path,
    output_dir: Path,
    time_chunks: int = 4,
    compress: bool = True,
    threads_per_worker: int = 2,
) -> tuple[int, bool, float, str]:
    """Process depth-averaging for a single SCHISM day.

    Returns:
        (fnumber, success, elapsed_seconds, message)
    """
    t_start = time.time()
    out2d = input_dir / f"out2d_{fnumber}.nc"
    zcoord = input_dir / f"zCoordinates_{fnumber}.nc"
    var_nc = input_dir / f"{varname}_{fnumber}.nc"

    path_node = output_dir / f"depth_averaged_{varname}_{fnumber}.nc"
    path_face = output_dir / f"depth_averaged_{varname}_at_face_{fnumber}.nc"

    # Enforce thread-level parallelism inside each worker process
    # to avoid nested multiprocessing deadlock
    with dask.config.set(scheduler="threads", num_workers=threads_per_worker):
        try:
            chunks = {"time": time_chunks}
            grid = sx.open_grid(out2d, zcoord, chunks=chunks)
            ds_var = xr.open_dataset(
                var_nc,
                engine=sx.core.api._choose_netcdf_engine(),
                chunks=chunks,
            ).astype(np.float64)
            sx_ds = sx.read_schism_nc(grid, ds_var)

            # Node depth-average
            da_node = sx_ds[varname].depth_average()
            da_node.name = f"depth_averaged_{varname}"
            if "units" in sx_ds[varname].attrs:
                da_node.attrs["units"] = sx_ds[varname].attrs["units"]
            da_node.attrs["long_name"] = f"depth-averaged {varname}"
            da_node.attrs["location"] = "node"

            encoding_node = {
                da_node.name: (
                    {"dtype": "float32", "zlib": True, "complevel": 1}
                    if compress
                    else {"dtype": "float32"}
                )
            }
            xr.Dataset({da_node.name: da_node.astype(np.float32)}).to_netcdf(
                path_node, encoding=encoding_node
            )

            # Face average via integrate() normalized by planar face area
            da_ones = da_node.isel(time=slice(0, 1)) * 0.0 + 1.0
            face_areas = da_ones.integrate().isel(time=0).compute()

            da_integral = da_node.integrate()
            da_face = (da_integral / face_areas).astype(np.float32)
            da_face.name = f"depth_averaged_{varname}_at_face"
            if "units" in sx_ds[varname].attrs:
                da_face.attrs["units"] = sx_ds[varname].attrs["units"]
            da_face.attrs["long_name"] = f"depth-averaged {varname} at face"
            da_face.attrs["location"] = "face"

            encoding_face = {
                da_face.name: (
                    {"dtype": "float32", "zlib": True, "complevel": 1}
                    if compress
                    else {"dtype": "float32"}
                )
            }
            xr.Dataset({da_face.name: da_face}).to_netcdf(
                path_face, encoding=encoding_face
            )

            elapsed = time.time() - t_start
            return fnumber, True, elapsed, "Success"

        except Exception as e:
            elapsed = time.time() - t_start
            return fnumber, False, elapsed, str(e)


@click.command(
    help="Calculate daily depth-averaged SCHISM variables for multiple days using Dask."
)
@click.option(
    "--varname",
    type=str,
    default="salinity",
    show_default=True,
    help="Variable name prefix, such as salinity or temperature.",
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
    "--start-day",
    type=int,
    default=None,
    help="First day index to process (inclusive).",
)
@click.option(
    "--end-day",
    type=int,
    default=None,
    help="Last day index to process (inclusive).",
)
@click.option(
    "--days",
    type=str,
    default=None,
    help="Comma-separated or ranged list of days to process (e.g. '1,2,5' or '1-10').",
)
@click.option(
    "--num-workers",
    "-n",
    type=int,
    default=None,
    help=(
        "Number of parallel worker processes. Default: 8 (recommended for 24-core/64GB node), "
        "or min(allocated_cores // threads_per_worker, 8, num_days). Use 12 on 24-core/128GB nodes "
        "for maximum throughput."
    ),
)
@click.option(
    "--threads-per-worker",
    type=int,
    default=2,
    show_default=True,
    help="Threads used for internal chunk computations in each worker process.",
)
@click.option(
    "--time-chunks",
    type=int,
    default=4,
    show_default=True,
    help="Chunk size along the time dimension for Dask within each day.",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    show_default=True,
    help="Enable NetCDF4 zlib compression on output files.",
)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    show_default=True,
    help="Overwrite existing output files instead of skipping them.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Display Dask progress bar during computation.",
)
def calculate_depth_average_multi_days(
    varname: str,
    input_dir: Path,
    output_dir: Path,
    start_day: int | None,
    end_day: int | None,
    days: str | None,
    num_workers: int | None,
    threads_per_worker: int,
    time_chunks: int,
    compress: bool,
    overwrite: bool,
    progress: bool,
):
    """Main CLI entrypoint to process multiple days of SCHISM depth averaging."""
    t_global_start = time.time()
    logger.info("Initializing multi-day depth-average calculation for %s...", varname)

    # 1. Resolve days to process
    if days is not None:
        target_days = parse_days_argument(days)
    elif start_day is not None and end_day is not None:
        target_days = list(range(start_day, end_day + 1))
    elif start_day is not None:
        target_days = [start_day]
    else:
        logger.info("No day range explicitly specified; discovering available files...")
        target_days = discover_available_days(input_dir, varname)

    if not target_days:
        logger.error("No matching days found to process.")
        sys.exit(1)

    logger.info(
        "Identified %d candidate day(s) to process: %s",
        len(target_days),
        target_days[:10],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check input file availability and skip existing outputs
    days_to_run: list[int] = []
    skipped_days: list[int] = []
    missing_input_days: list[int] = []

    for fnumber in target_days:
        out2d = input_dir / f"out2d_{fnumber}.nc"
        zcoord = input_dir / f"zCoordinates_{fnumber}.nc"
        var_nc = input_dir / f"{varname}_{fnumber}.nc"

        if not (out2d.exists() and zcoord.exists() and var_nc.exists()):
            missing_input_days.append(fnumber)
            logger.warning(
                "Skipping day %d: missing one or more required input files.", fnumber
            )
            continue

        path_node = output_dir / f"depth_averaged_{varname}_{fnumber}.nc"
        path_face = output_dir / f"depth_averaged_{varname}_at_face_{fnumber}.nc"

        if not overwrite and path_node.exists() and path_face.exists():
            skipped_days.append(fnumber)
            continue

        days_to_run.append(fnumber)

    logger.info(
        "Plan summary: %d day(s) to compute, %d day(s) already exist (skipped), %d day(s) missing inputs.",
        len(days_to_run),
        len(skipped_days),
        len(missing_input_days),
    )

    if not days_to_run:
        logger.info("All requested days are already up to date. Nothing to compute.")
        return

    # 3. Configure Dask worker count
    cpu_count = multiprocessing.cpu_count() or 4
    if num_workers is None:
        num_workers = min(cpu_count, 8, len(days_to_run))
    num_workers = max(1, num_workers)

    logger.info(
        "Launching Dask multiprocessing pool with %d worker(s) (threads_per_worker=%d, time_chunks=%d)...",
        num_workers,
        threads_per_worker,
        -1 if time_chunks is None else time_chunks,
    )

    # 4. Build delayed computation tasks
    delayed_process = dask.delayed(process_single_day)
    tasks = [
        delayed_process(
            fnumber=fnumber,
            varname=varname,
            input_dir=input_dir,
            output_dir=output_dir,
            time_chunks=time_chunks,
            compress=compress,
            threads_per_worker=threads_per_worker,
        )
        for fnumber in days_to_run
    ]

    # 5. Execute using Dask processes scheduler with chunksize=1 for uniform load balancing
    with dask.config.set(scheduler="processes", num_workers=num_workers, chunksize=1):
        if progress and sys.stdout.isatty():
            with ProgressBar():
                results = dask.compute(*tasks)
        else:
            results = dask.compute(*tasks)

    # 6. Report results summary
    total_elapsed = time.time() - t_global_start
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    logger.info("=" * 60)
    logger.info("MULTI-DAY PROCESSING SUMMARY")
    logger.info("  Variable:                 %s", varname)
    logger.info("  Total days processed:     %d", len(days_to_run))
    logger.info("  Successfully completed:   %d", len(successful))
    logger.info("  Failed:                   %d", len(failed))
    logger.info("  Already existing/skipped: %d", len(skipped_days))
    logger.info("  Total wall-clock time:    %.2f s", total_elapsed)
    if successful:
        avg_rate = total_elapsed / len(successful)
        logger.info("  Average throughput:       %.2f s/day", avg_rate)

    if failed:
        logger.error("Failures encountered:")
        for fnumber, _, _, err_msg in failed:
            logger.error("  Day %d: %s", fnumber, err_msg)
        sys.exit(1)

    logger.info("Finished depth-averaging for all requested days successfully.")


if __name__ == "__main__":
    calculate_depth_average_multi_days()
