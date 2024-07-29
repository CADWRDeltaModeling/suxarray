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

    def bounding_box_xy(
        self,
        x_bounds: Union[Tuple, List, np.ndarray],
        y_bounds: Union[Tuple, List, np.ndarray],
        element: Optional[str] = "nodes",
        # method: Optional[str] = "coords",
        predicate: Optional[str] = None,
        **kwargs,
    ):
        """Subsets an unstructured grid between two latitude and longitude
        points which form a bounding box.

        Parameters
        ----------
        x_bounds: tuple, list, np.ndarray
            (x_left, x_right)
        y_bounds: tuple, list, np.ndarray
            (y_bottom, y_top)
        element: str
            Element for use with `coords` comparison, one of `nodes`,
            `face`, or `edge`
        predicate: str
            Predicate for use with ‘intersects’, ‘within’, ‘contains’, ‘overlaps’, ‘crosses’,’touches’, ‘covers’, ‘covered_by’, ‘contains_properly’, ‘dwithin’ form Shapely STRTree.query.
        """
        if predicate is None:
            predicate = "intersects"
        if x_bounds[0] > x_bounds[1]:
            raise ValueError("Bounding box must be given in ascending order.")
        if y_bounds[0] > y_bounds[1]:
            raise ValueError("Bounding box must be given in ascending order.")

        bbox = Polygon(
            [
                (x_bounds[0], y_bounds[0]),
                (x_bounds[1], y_bounds[0]),
                (x_bounds[1], y_bounds[1]),
                (x_bounds[0], y_bounds[1]),
                (x_bounds[0], y_bounds[0]),
            ]
        )

        if element == "nodes":
            strtree = self.sxgrid.get_strtree(coordinates=element)
            node_ilocs = strtree.query(bbox, predicate=predicate)
            return self.sxgrid.isel(n_node=node_ilocs)
        else:
            raise ValueError("TODO")
