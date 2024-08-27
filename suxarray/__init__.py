"""Top-level package for suxarray."""

__author__ = """California Department of Water Resources"""
__email__ = "knam@water.ca.gov"
__version__ = "0.2.0"

from .core.api import (
    open_grid,
    read_grid,
    read_schism_nc,
    open_hgrid_gr3,
    write_schism_grid,
)
from .core.dataset import SxDataset
from .core.dataarray import SxDataArray
from .grid import Grid

__all__ = (
    "SxDataset",
    "SxDataArray",
    "open_grid",
    "open_hgrid_gr3",
    "read_grid",
    "read_schism_nc",
    "write_schism_grid",
    "Grid",
)
