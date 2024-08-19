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
    __slots__ = ()

    def __init__(self, *args, sxgrid: Grid = None, **kwargs):
        if sxgrid is not None and not isinstance(sxgrid, Grid):
            raise RuntimeError("sxgrid must be a Grid object")

        super().__init__(*args, uxgrid=sxgrid, **kwargs)

    subset = UncachedAccessor(DataArraySubsetAccessor)

    @property
    def sxgrid(self) -> Grid:
        return self.uxgrid

    def isel(self, ignore_grid=False, *args, **kwargs):
        da_new = super().isel(ignore_grid=ignore_grid, *args, **kwargs)
        if not ignore_grid and self.uxgrid.sgrid_info is not None:
            da_new.uxgrid = da_new.uxgrid.sgrid_isel(**kwargs)
        return da_new


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

        bottom_index_node = self.sxgrid.sgrid_info.bottom_index_node
        dry_flag_node = self.sxgrid.sgrid_info.dryFlagNode
        zcoords = self.sxgrid.sgrid_info.zCoordinates

        da_da = xr.apply_ufunc(
            _depth_average,
            self,
            zcoords,
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
