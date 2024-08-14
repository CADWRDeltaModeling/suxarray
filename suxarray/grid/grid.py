from __future__ import annotations
from typing import Optional, Union
import pandas as pd
import xarray as xr
import uxarray as ux
from xarray.core.utils import UncachedAccessor
from suxarray.io._schismgrid import _read_schism_out2d
from suxarray.subset import GridSubsetAccessor
from suxarray.grid.neighbors import STRTree


class Grid(ux.Grid):
    """uxarray Grid class for SCHISM
    See uxarray Grid documentation for the details of the methods and properties

    Examples
    --------
    TBD
    """

    def __init__(
        self,
        grid_obj: Union[xr.Dataset, ux.Grid],
        ds_zcoords: Optional[xr.Dataset] = None,
        source_grid_spec: str = "UGRID",
        source_dims_dict: Optional[dict] = {},
        **kwargs,
    ):
        """Initialize a Grid object

        Parameters
        ----------
        grid_odj : xarray.Dataset or uxarray.Grid

        Other Parameters
        ----------------
        source_grid_spec: str, optional
            Specifies gridspec
        source_dims_dict: dict, optional
            source dim dict
        """
        # Initialize the super class
        source_grid_spec = "UGRID"
        self._face_polygons = None
        self._face_strtree = None
        self._node_points = None
        self._node_strtree = None
        self._edge_strtree = None
        self._zcoords = ds_zcoords

        if isinstance(grid_obj, ux.Grid):
            super().__init__(
                grid_obj._ds,
                source_grid_spec=source_grid_spec,
                source_dims_dict=source_dims_dict,
            )
        elif isinstance(grid_obj, xr.Dataset):
            super().__init__(
                grid_obj,
                source_grid_spec=source_grid_spec,
                source_dims_dict=source_dims_dict,
            )
        else:
            raise ValueError("grid_ds must be xarray.Dataset or suxarray.Grid")

    subset = UncachedAccessor(GridSubsetAccessor)

    @classmethod
    def from_dataset(
        cls, ds_out2d: xr.Dataset, ds_zcoords: Optional[xr.Dataset] = None, **kwargs
    ):
        """Create a Grid object from a SCHISM output 2D dataset and a z-coordinate dataset

        Parameters
        ----------
        ds_out2d : xr.Dataset
            SCHISM output 2D dataset
        ds_zcoords : xr.Dataset
            z-coordinate dataset

        Returns
        -------
        Grid
        """
        source_grid_spec = "UGRID"
        grid_ds, source_dims_dict, ds_zcoords = _read_schism_out2d(ds_out2d, ds_zcoords)
        return cls(grid_ds, ds_zcoords, source_grid_spec, source_dims_dict)

    @classmethod
    def _from_uxgrid(cls, grid: ux.Grid):
        grid.__class__ = cls
        return grid

    @property
    def zcoords(self) -> xr.Dataset:
        return self._zcoords

    def get_strtree(self, elements: Optional[str] = "nodes"):
        if elements == "nodes":
            if self._node_strtree is None:
                self._node_strtree = STRTree(self, elements="nodes")
            return self._node_strtree
        elif elements == "faces":
            if self._face_strtree is None:
                self._face_strtree = STRTree(self, elements="faces")
            return self._face_strtree
        elif elements == "edges":
            if self._edge_strtree is None:
                self._edge_strtree = STRTree(self, elements="edges")
            return self._edge_strtree
        else:
            raise ValueError(
                f"Coordinates must be one of 'nodes', 'faces', or 'edges'."
            )

    def isel(self, **dim_kwargs):
        """Indexes an unstructured grid along a given dimension (``n_node``,
        ``n_edge``, or ``n_face``) and returns a new grid.

        Currently only supports inclusive selection, meaning that for cases where node or edge indices are provided,
        any face that contains that element is included in the resulting subset. This means that additional elements
        beyond those that were initially provided in the indices will be included. Support for more methods, such as
        exclusive and clipped indexing is in the works.

        Parameters
        **dims_kwargs: kwargs
            Dimension to index, one of ['n_node', 'n_edge', 'n_face']


        Example
        -------`
        >> grid = ux.open_grid(grid_path)
        >> grid.isel(n_face = [1,2,3,4])
        """
        # First, slice the grid information
        grid_new = self._from_uxgrid(super().isel(**dim_kwargs))

        # Need to subset (or slice) zCoords and other variables
        grid_new._zcoords = self.zcoords.isel(
            n_node=grid_new._ds["subgrid_node_indices"]
        )

        return grid_new

    def intersect(
        self,
        geometry: shapely.geometry.BaseGeometry,
        element: Optional[str] = "faces",
        **kwargs,
    ):
        """Find intersecting elements with a Shapely geometry

        Parameters
        ----------
        geometry: shapely.geometry.Geometry
        element: str, optional

        Returns
        -------
        int
            Indices of intersecting elements
        """
        if element == "faces":
            strtree = self.get_strtree(elements=element)
            face_ilocs = strtree.query(geometry, predicate="intersects")
            return face_ilocs
        else:
            raise ValueError("TODO")

    # def read_vgrid(self, path_vgrid):
    #     """Read a SCHISM vgrid file"""
    #     with open(path_vgrid, "r") as f:
    #         ivcor = int(f.readline().strip())
    #         if ivcor != 1:
    #             raise NotImplementedError("Only ivcor=1 is implemented")
    #         if ivcor == 1:
    #             nvrt = int(f.readline().strip())
    #     if ivcor == 1:
    #         n_nodes = self.Mesh2_node_x.size
    #         widths = np.full(n_nodes, 11, dtype=np.int32)
    #         widths[0] = 10
    #         df_nvrt = pd.read_fwf(
    #             path_vgrid,
    #             header=None,
    #             skiprows=2,
    #             nrows=1,
    #             widths=widths,
    #             dtype=np.int32,
    #         )
    #         self.ds["nvrt"] = xr.DataArray(
    #             df_nvrt.values.squeeze(), dims=("nMesh2_node",)
    #         )
    #         widths = np.full(n_nodes + 1, 15, dtype=np.int32)
    #         widths[0] = 10
    #         df_vgrid = pd.read_fwf(
    #             path_vgrid,
    #             header=None,
    #             skiprows=3,
    #             nrows=nvrt,
    #             widths=[10] + [15] * n_nodes,
    #             na_values=-9.0,
    #             dtype=np.float32,
    #         )
    #         self.ds["vgrid"] = xr.DataArray(
    #             df_vgrid.iloc[:, 1:].values + 1.0,
    #             dims=(
    #                 "nSCHISM_vgrid_layers",
    #                 "nMesh2_node",
    #             ),
    #         )
