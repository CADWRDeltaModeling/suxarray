from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr
from xarray.core.utils import UncachedAccessor
import uxarray
from ..grid import Grid
from ..subset import DataArraySubsetAccessor


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

        def _depth_average(v, zs, k, dry):
            # Select the values with the last index from the bottom index
            # array, k
            z_bottom = np.take_along_axis(zs, k.reshape((1, -1, 1)), axis=-1)
            depth = zs[..., -1] - z_bottom.squeeze(axis=-1)
            # Mask nan values
            v = np.ma.masked_invalid(v, copy=False)
            zs = np.ma.masked_invalid(zs, copy=False)
            result = np.trapz(v, x=zs, axis=-1) / depth
            result[dry == 1.0] = np.nan
            return result

        bottom_index_node = self.sxgrid.z_coords.bottom_index_node
        dry_flag_node = self.sxgrid.z_coords.dryFlagNode

        da_da = xr.apply_ufunc(
            _depth_average,
            self,
            self.sxgrid.z_coords["zCoordinates"],
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
