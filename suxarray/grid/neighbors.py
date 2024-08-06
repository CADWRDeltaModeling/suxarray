from typing import Optional, Union

from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree as SHSTRtree

import xarray as xr


class STRTree:

    def __init__(self, grid, elements: Optional[str] = "nodes"):
        self._grid = grid
        self._elements = elements
        self._strtree = None
        self._nodes = None
        self._faces = None

        if elements not in ["nodes", "faces", "edges"]:
            raise ValueError(
                f"Coordinates must be one of 'nodes', 'faces', or 'edges'."
            )
        if elements == "nodes":
            self._strtree = SHSTRtree(self.points)
        elif elements == "faces":
            self._strtree = SHSTRtree(self.faces)
        elif elements == "edges":
            raise NotImplementedError("TODO")
        else:
            raise RuntimeError("This should never happen")

    @property
    def points(self):
        if self._nodes is None:
            p_x = self._grid.node_x
            p_y = self._grid.node_y
            self._nodes = xr.apply_ufunc(
                lambda x, y: Point(x, y),
                p_x,
                p_y,
                vectorize=True,
                dask="parallelized",
            )
        return self._nodes

    @property
    def faces(self):
        if self._faces is None:
            # If the spatial tree is not built yet, build it
            p_x = self._grid.node_x
            p_y = self._grid.node_y
            fill_value = self._grid.face_node_connectivity.attrs["_FillValue"]

            def create_polygon(node_indices):
                # The node indices are 1-based
                valid = node_indices != fill_value
                ind = node_indices[valid]
                # Assuming the indices are positional
                return Polygon(zip(p_x[ind], p_y[ind]))

            # NOTE Assuming the second dimension of Mesh2_face_nodes is not safe.
            self._faces = xr.apply_ufunc(
                create_polygon,
                self._grid.face_node_connectivity,
                input_core_dims=((self._grid.face_node_connectivity.dims[1],),),
                dask="parallelized",
                output_dtypes=[object],
                vectorize=True,
            )
        return self._faces

    def query(self, geometry, predicate=None, distance=None):
        return self._strtree.query(geometry, predicate=predicate, distance=distance)
