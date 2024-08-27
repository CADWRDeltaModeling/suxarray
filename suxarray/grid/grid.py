from __future__ import annotations
from typing import Optional, Union
import xarray as xr
import uxarray as ux
from xarray.core.utils import UncachedAccessor
from suxarray.io._schismgrid import _read_schism_grid
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
        ds_sgrid_info: Optional[xr.Dataset] = None,
        source_grid_spec: str = "UGRID",
        source_dims_dict: Optional[dict] = {},
        **kwargs,
    ):
        """Initialize a Grid object

        Parameters
        ----------
        grid_odj : xarray.Dataset or uxarray.Grid
            Grid dataset
        ds_sgrid_info : xarray.Dataset, optional
            SCHISM grid information dataset

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
        self._sgrid_info = ds_sgrid_info

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
        cls, ds_out2d: xr.Dataset, ds_zcoords: xr.Dataset, **kwargs
    ):
        """Create a Grid object from a SCHISM output 2D dataset and a z-coordinate dataset

        Parameters
        ----------
        ds_out2d : xr.Dataset
            SCHISM output 2D dataset
        ds_sgrid_info : xr.Dataset, optional
            Extra SCHISM grid information

        Returns
        -------
        Grid
        """
        source_grid_spec = "UGRID"
        ds_grid, ds_sgrid_info = _read_schism_grid(
            ds_out2d, ds_zcoords
        )
        return cls(ds_grid, ds_sgrid_info, source_grid_spec)

    @property
    def sgrid_info(self) -> xr.Dataset:
        return self._sgrid_info

    @sgrid_info.setter
    def sgrid_info(self, sgrid_info: xr.Dataset):
        self._sgrid_info = sgrid_info

    def isel(self, **dim_kwargs):
        # Because uxarray Grid calls external methods, isel below will not return
        # Suxarray Grid object. It needs to be cast.
        grid_new = super().isel(**dim_kwargs)
        sgrid_new = Grid(grid_new, ds_sgrid_info=self.sgrid_info)
        if self.sgrid_info is not None:
            sgrid_new = sgrid_new.sgrid_isel(**dim_kwargs)
        return sgrid_new

    def sgrid_isel(self, **kwargs):
        from uxarray.constants import GRID_DIMS

        kwargs_no_grid_dim = {k: v for k, v in kwargs.items() if k not in GRID_DIMS}
        if kwargs_no_grid_dim:
            sgrid_info = self.sgrid_info.copy().isel(kwargs_no_grid_dim)
        else:
            sgrid_info = self.sgrid_info.copy()

        # Need to subset (or slice) sgrid_info
        if "subgrid_node_indices" in self._ds:
            sgrid_info = sgrid_info.isel(
                n_node=self._ds["subgrid_node_indices"].values,
                n_face=self._ds["subgrid_face_indices"].values,
                n_edge=self._ds["subgrid_edge_indices"].values,
            )

        return Grid(self, ds_sgrid_info=sgrid_info)

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
