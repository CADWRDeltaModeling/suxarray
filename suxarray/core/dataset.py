""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""

from typing import Optional
import numpy as np
import pandas as pd

# from numba import njit
import xarray as xr
from xarray.core.utils import UncachedAccessor
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import uxarray as ux
from suxarray.grid import Grid
from suxarray.core.dataarray import SxDataArray


class SxDataset(ux.UxDataset):
    __slots__ = ("_sxgrid",)

    def __init__(
        self,
        *args,
        sxgrid: Grid = None,
        source_datasets: Optional[str] = None,
        **kwargs,
    ):
        self._sxgrid = None
        self._sxgrid = sxgrid
        super().__init__(
            *args, uxgrid=sxgrid, source_datasets=source_datasets, **kwargs
        )

    def __getitem__(self, key):
        value = super().__getitem__(key)

        if isinstance(value, ux.UxDataArray):
            value = SxDataArray._from_uxdataarray(value)

        return value

    # def face_average(self, dataarray: xr.DataArray) -> xr.DataArray:
    #     """Calculate face average of a variable

    #     Parameters
    #     ----------
    #     dataarray: xr.DataArray, required
    #         Input variable

    #     Returns
    #     -------
    #     da : xr.DataArray
    #         Face averaged variable
    #     """
    #     da_result = xr.apply_ufunc(
    #         face_average,
    #         dataarray,
    #         self.uxgrid.Mesh2_face_nodes,
    #         self.uxgrid.nNodes_per_face,
    #         exclude_dims=set(["nMesh2_node", "nMesh2_face", "nMaxMesh2_face_nodes"]),
    #         input_core_dims=[
    #             [
    #                 "nMesh2_node",
    #             ],
    #             ["nMesh2_face", "nMaxMesh2_face_nodes"],
    #             [
    #                 "nMesh2_face",
    #             ],
    #         ],
    #         output_core_dims=[
    #             [
    #                 "nMesh2_face",
    #             ],
    #         ],
    #         output_dtypes=[float],
    #         dask_gufunc_kwargs={
    #             "output_sizes": {"nMesh2_face": self.uxgrid.nMesh2_face}
    #         },
    #         dask="parallelized",
    #     )
    #     return da_result

    @property
    def sxgrid(self):
        return self._sxgrid


# def add_topology_variable(ds, varname="Mesh2"):
#     """Add a dummy mesh_topology variable to a SCHISM out2d dataset

#     Parameters
#     ----------
#     ds : xarray.Dataset, required
#         Input SCHISM out2d xarray.Dataset
#     varname : str, optional
#         Name of the dummy topology variable. Default is "Mesh2"
#     """
#     ds = ds.assign({varname: 1})
#     ds[varname].attrs["cf_role"] = "mesh_topology"
#     ds[varname].attrs["topology_dimension"] = 2
#     ds[varname].attrs["node_coordinates"] = "Mesh2_node_x Mesh2_node_y"
#     ds[varname].attrs["face_node_connectivity"] = "Mesh2_face_nodes"
#     ds[varname].attrs["edge_node_connectivity"] = "Mesh2_edge_nodes"
#     ds[varname].attrs["Mesh2_layers"] = "zCoordinates"
#     ds[varname].attrs["start_index"] = 0

#     return ds


# # TODO separate utility functions to another file
# def renumber_nodes(a, fill_value: int = None):
#     if fill_value is None:
#         return _renumber(a)

#     valid = a != fill_value
#     renumbered = np.full_like(a, fill_value)
#     renumbered[valid] = _renumber(a[valid])
#     return renumbered


# def _renumber(a):
#     # Taken from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737
#     # (scipy is BSD-3-Clause License)
#     arr = np.ravel(np.asarray(a))
#     sorter = np.argsort(arr, kind="quicksort")
#     inv = np.empty(sorter.size, dtype=int)
#     inv[sorter] = np.arange(sorter.size, dtype=int)
#     arr = arr[sorter]
#     obs = np.r_[True, arr[1:] != arr[:-1]]
#     dense = obs.cumsum()[inv]
#     return dense.reshape(a.shape)


# def get_topology_variable(dataset: xr.Dataset) -> Optional[str]:
#     """Get the topology xarray.DataArray

#     Parameters
#     ----------
#     dataset : xarray.Dataset, required
#         Input xarray.Dataset

#     Returns
#     -------
#     da : xarray.DataArray
#         Topology Xarray.DataArray
#     """
#     ds = dataset.filter_by_attrs(cf_role="mesh_topology")
#     if len(ds) == 0:
#         return None
#     elif len(ds) > 1:
#         raise ValueError("Multiple mesh_topology variables found")
#     else:
#         return ds[list(ds.keys())[0]]


# def triangulate(grid):
#     """Triangulate a suxarray grid

#     This function split quadrilateral elements into two triangles and returns a
#     new suxarray grid. It does not consider the quality of the resulting
#     triangles. The main use case of this function is to visualize the grid
#     with holoviews or other plotting tools that require a triangulated grid.

#     Parameters
#     ----------
#     grid : Grid, required
#         Grid object to triangulate

#     Returns
#     -------
#     grid : Grid
#         Triangulated grid
#     """
#     mesh_name = grid.grid_var_names["Mesh2"]
#     face_nodes = grid.Mesh2_face_nodes

#     n_face = grid.nMesh2_face
#     fill_value = face_nodes.attrs["_FillValue"]
#     valid = face_nodes != fill_value
#     n_per_row = valid.sum(axis=1)
#     n_triangle_per_row = n_per_row - 2
#     face_ori = np.repeat(np.arange(n_face), n_per_row)
#     node_ori = face_nodes.values.ravel()[valid.values.ravel()]
#     if face_nodes.attrs["start_index"] == 1:
#         node_ori -= 1

#     def _triangulate(
#         face_ori: np.ndarray, node_ori: np.ndarray, n_triangle_per_row: xr.DataArray
#     ) -> np.ndarray:
#         n_triangle = n_triangle_per_row.sum().compute().item()
#         n_face = len(face_ori)
#         index_first = np.argwhere(np.diff(face_ori, prepend=-1) != 0)
#         index_second = index_first + 1
#         index_last = np.argwhere(np.diff(face_ori, append=-1) != 0)

#         first = np.full(n_face, False)
#         first[index_first] = True
#         second = np.full(n_face, True) & ~first
#         second[index_last] = False
#         third = np.full(n_face, True) & ~first
#         third[index_second] = False

#         triangles = np.empty((n_triangle, 3), np.int32)
#         triangles[:, 0] = np.repeat(node_ori[first], n_triangle_per_row)
#         triangles[:, 1] = node_ori[second]
#         triangles[:, 2] = node_ori[third]
#         return triangles

#     triangles = _triangulate(face_ori, node_ori, n_triangle_per_row)

#     triangle_original_ind = np.repeat(np.arange(n_face), repeats=n_triangle_per_row)

#     # Copy the data from the original grid
#     ds_tri = grid._ds.copy()
#     # Drop the original face_nodes variable
#     # TODO dryFlagElement is needed to be updated not dropped, but let's drop
#     # it for now.
#     varnames_to_drop = [
#         f"{mesh_name}_face_nodes",
#         "dryFlagElement",
#         f"{mesh_name}_face_x",
#         f"{mesh_name}_face_y",
#         "nNodes_per_face",
#     ]
#     for varname in varnames_to_drop:
#         if varname in ds_tri:
#             ds_tri = ds_tri.drop_vars(varname)
#     da_face_nodes = xr.DataArray(
#         data=triangles,
#         dims=(f"n{mesh_name}_face", "three"),
#         name=f"{mesh_name}_face_nodes",
#     )
#     da_face_nodes.attrs["start_index"] = 0
#     da_face_nodes.attrs["cf_role"] = "face_node_connectivity"
#     ds_tri[da_face_nodes.name] = da_face_nodes
#     da_elem_ind = xr.DataArray(
#         data=triangle_original_ind,
#         dims=(f"n{mesh_name}_face"),
#         name=f"{mesh_name}_face_original",
#     )
#     ds_tri[da_elem_ind.name] = da_elem_ind
#     grid_tri = Grid(ds_tri, islation=False, mesh_type="ugrid")
#     grid_tri.Mesh2.attrs["start_index"] = 0
#     return grid_tri


# def face_average(val, face_nodes, n_nodes_per_face):
#     """Calculate face average of a variable

#     Calculate average of a variable at each face. No weighting is applied.

#     Parameters
#     ----------
#     val: ndarray, required
#         values to process. The last dimension must be for the nodes.
#     face_nodes: ndarray, required
#         Face-node connectivity array
#     n_nodes_per_face: ndarray, required
#         Number of nodes per face

#     Returns
#     -------
#     xarray.DataArray
#         Face averaged variable
#     """
#     n_face, _ = face_nodes.shape

#     # set initial area of each face to 0
#     result = np.zeros(val.shape[:-1] + (n_face,))

#     for face_idx, max_nodes in enumerate(n_nodes_per_face):
#         avg = val[..., face_nodes[face_idx, 0:max_nodes]].sum(axis=-1) / max_nodes
#         result[..., face_idx] = avg
#     return result
