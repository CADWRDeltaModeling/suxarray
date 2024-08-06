from __future__ import annotations
from typing import TYPE_CHECKING, Union, Tuple, List, Optional
import numpy as np
from shapely.geometry import Polygon
import xarray as xr
import uxarray.subset

if TYPE_CHECKING:
    from suxarray.grid import Grid


class GridSubsetAccessor(uxarray.subset.GridSubsetAccessor):
    def __init__(self, sxgrid: Grid) -> None:
        self.sxgrid = sxgrid
        super().__init__(sxgrid)

    def bounding_polygon(
        self,
        polygon: Polygon,
        element: Optional[str] = "nodes",
        predicate: Optional[str] = None,
        **kwargs,
    ):
        """Subsets an unstructured grid between two latitude and longitude
        points which form a bounding box.

        Parameters
        ----------
        polygon: shapely.geometry.Polygon
            Polygon for use with `coords` comparison
        element: str
            Element for use with `coords` comparison, one of `nodes`,
            `face`, or `edge`
        predicate: str
            Predicate for use with ‘intersects’, ‘within’, ‘contains’, ‘overlaps’, ‘crosses’,’touches’, ‘covers’, ‘covered_by’, ‘contains_properly’, ‘dwithin’ form Shapely STRTree.query.
        """
        if predicate is None:
            predicate = "intersects"

        if element == "nodes":
            strtree = self.sxgrid.get_strtree(coordinates=element)
            node_ilocs = strtree.query(polygon, predicate=predicate)
            return self.sxgrid.isel(n_node=node_ilocs)
        else:
            raise ValueError("TODO")
