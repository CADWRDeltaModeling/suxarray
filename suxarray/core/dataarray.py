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
from suxarray.grid.area import _integrate_nodal


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
                    "n_layer",
                ],
                [
                    "n_layer",
                ],
                [],
                [],
            ],
            dask="parallelized",
            output_dtypes=[float],
        )
        name = f"depth_averaged_{self.name}"
        return SxDataArray(da_da, sxgrid=self.sxgrid, name=name)

    def integrate(
        self, quadrature_rule: Optional[str] = "triangular", order: Optional[int] = 2
    ) -> SxDataArray:
        if self.values.shape[-1] == self.sxgrid.n_face:
            return super().integrate(quadrature_rule=quadrature_rule, order=order)
        elif self.values.shape[-1] == self.sxgrid.n_edge:
            raise ValueError("Integrate is not supported for edge values")
        elif self.values.shape[-1] == self.sxgrid.n_node:
            # If there is a vertical dimension, n_layer, raise an error for now
            if "n_layer" in self.dims:
                raise ValueError("Integrate is not supported for 3D nodal values yet.")
            else:
                integral = xr.apply_ufunc(
                    _integrate_nodal,
                    self.sxgrid.node_x,
                    self.sxgrid.node_y,
                    self,
                    self.sxgrid.face_node_connectivity,
                    input_core_dims=[
                        [
                            "n_node",
                        ],
                        [
                            "n_node",
                        ],
                        [
                            "n_node",
                        ],
                        [
                            "n_face",
                            "n_max_face_nodes",
                        ],
                    ],
                    output_core_dims=[["n_face"]],
                    output_dtypes=self.dtype,
                    dask="parallelized",
                )
            dims = self.dims[:-1] + ("n_face",)
            sxda = SxDataArray(integral, sxgrid=self.sxgrid, dims=dims, name=self.name)
            return sxda
        else:
            raise ValueError("None of dimensions match the grid information")
