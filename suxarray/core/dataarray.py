from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr
from xarray.core.utils import UncachedAccessor
import uxarray
from shapely.geometry import Point
from suxarray.grid import Grid
from suxarray.subset import DataArraySubsetAccessor
from suxarray.utils.computing import _depth_average


class SxDataArray(uxarray.UxDataArray):
    __slots__ = ("_sxgrid",)

    def __init__(self, *args, sxgrid: Grid = None, **kwargs):
        self._sxgrid = None
        if sxgrid is not None and not isinstance(sxgrid, Grid):
            raise RuntimeError("sxgrid must be a Grid object")
        else:
            self._sxgrid = sxgrid
        super().__init__(*args, uxgrid=sxgrid, **kwargs)

    subset = UncachedAccessor(DataArraySubsetAccessor)

    @property
    def sxgrid(self) -> Grid:
        return self._sxgrid

    def _slice_from_grid(self, grid: Grid) -> SxDataArray:
        """Slice the data array based on the grid object

        Parameters
        ----------
        grid : Grid
            Grid object

        Returns
        -------
        SxDataArray
            Sliced data array
        """
        return SxDataArray(super()._slice_from_grid(grid), sxgrid=grid)

    def depth_average(self) -> SxDataArray:
        """Calculate depth-average of a variable

        This may need to be moved to a separate file.
        TODO: Currently this assumes nodal values. Need to support other `element`.
        TODO: Maybe this can be moved to another file.

        Parameters
        ----------
        None

        Returns
        -------
        da : SxDataArray
            Depth averaged variable
        """

        bottom_index_node = self.sxgrid.zcoords.bottom_index_node
        dry_flag_node = self.sxgrid.zcoords.dryFlagNode

        da_da = xr.apply_ufunc(
            _depth_average,
            self,
            self.sxgrid.zcoords["zCoordinates"],
            bottom_index_node - 1,
            dry_flag_node,
            input_core_dims=[
                [
                    "nSCHISM_vgrid_layers",
                ],
                [
                    "nSCHISM_vgrid_layers",
                ],
                [],
                [],
            ],
            dask="parallelized",
            output_dtypes=[float],
        )
        return SxDataArray(da_da, sxgrid=self.sxgrid)
