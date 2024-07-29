from __future__ import annotations
from typing import TYPE_CHECKING, Union, Tuple, List, Optional
import numpy as np
import uxarray.subset
import suxarray as sx


class DataArraySubsetAccessor(uxarray.subset.DataArraySubsetAccessor):
    def __init__(self, sxda) -> None:
        self.sxda = sxda
        super().__init__(sxda)

    def __repr__(self):
        prefix = "<suxarray.SuxDataArray.subset>\n"
        methods_heading = "Supported Methods:\n"

        # methods_heading += "  * nearest_neighbor(center_coord, k, element, **kwargs)\n"
        # methods_heading += "  * bounding_circle(center_coord, r, element, **kwargs)\n"
        # methods_heading += (
        #     "  * bounding_box(lon_bounds, lat_bounds, element, method, **kwargs)\n"
        # )

        return prefix + methods_heading

    def bounding_box_xy(
        self,
        x_bounds: Union[Tuple, List, np.ndarray],
        y_bounds: Union[Tuple, List, np.ndarray],
        element: Optional[str] = "nodes",
        # method: Optional[str] = "coords",
        predicate: Optional[str] = "intersects",
        **kwargs,
    ):
        grid = self.sxda.sxgrid.subset.bounding_box_xy(
            x_bounds, y_bounds, element=element, predicate=predicate
        )
        return self.sxda._slice_from_grid(grid)
