from __future__ import annotations
from typing import TYPE_CHECKING, Union, Tuple, List, Optional
import numpy as np
from shapely.geometry import Polygon
import uxarray.subset
import suxarray as sx


class DataArraySubsetAccessor(uxarray.subset.DataArraySubsetAccessor):
    def __init__(self, sxda) -> None:
        super().__init__(sxda)

    @property
    def sxda(self) -> sx.SxDataArray:
        return self.uxda

    def __repr__(self):
        prefix = "<suxarray.SuxDataArray.subset>\n"
        methods_heading = "Supported Methods:\n"

        # methods_heading += "  * nearest_neighbor(center_coord, k, element, **kwargs)\n"
        # methods_heading += "  * bounding_circle(center_coord, r, element, **kwargs)\n"
        # methods_heading += (
        #     "  * bounding_box(lon_bounds, lat_bounds, element, method, **kwargs)\n"
        # )

        return prefix + methods_heading

    def bounding_polygon(
        self,
        polygon: Polygon,
        element: Optional[str] = "nodes",
        predicate: Optional[str] = "intersects",
        use_xy: Optional[bool] = False,
        **kwargs,
    ):
        """Subsets the data array using a Shapely polygon

        With a given polygon, the data array is subset. The grid indices will be
        reindexed.

        Parameters
        ----------
        polygon: shapely.geometry.Polygon
            Polygon to subset the data
        element: str
            Element type to subset
        predicate: str
            Predicate for use with ‘intersects’, ‘within’, ‘contains’, ‘overlaps’, ‘crosses’,’touches’, ‘covers’, ‘covered_by’, ‘contains_properly’, ‘dwithin’ form Shapely STRTree.query.
        use_xy: bool
            If True, the polygon is assumed to be in the x-y plane. Default is False.

        Returns
        -------
        SxDataArray
            Subsetted suxarray data array
        """
        grid = self.sxda.sxgrid.subset.bounding_polygon(
            polygon, element=element, predicate=predicate
        )
        return sx.SxDataArray(self.sxda._slice_from_grid(grid), sxgrid=grid)

    def bounding_box_xy(
        self,
        x_bounds: Union[Tuple, List, np.ndarray],
        y_bounds: Union[Tuple, List, np.ndarray],
        element: Optional[str] = "nodes",
        # method: Optional[str] = "coords",
        predicate: Optional[str] = "intersects",
        **kwargs,
    ):
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

        return self.bounding_polygon(bbox, element=element, predicate=predicate)
