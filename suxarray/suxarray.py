""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""
from typing import Optional
import numpy as np
import pandas as pd
from numba import njit
import xarray as xr
from xarray.core.utils import UncachedAccessor
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import uxarray as ux
import suxarray.hvplot


class Grid(ux.Grid):
    """ uxarray Grid class for SCHISM
    """
    _face_polygons = None
    _face_strtree = None
    _node_points = None
    _node_strtree = None

    def __init__(self, dataset, **kwargs):
        """ Initialize a Grid object

        Parameters
        ----------
        dataset : xarray.Dataset, ndarray, list, tuple, required
            Input xarray.Dataset or vertex coordinates that form one face.

        Other Parameters
        ----------------
        islatlon : bool, optional
            Specify if the grid is lat/lon based
        concave: bool, optional
            Specify if this grid has concave elements (internal checks for this
            are possible)
        gridspec: bool, optional
            Specifies gridspec
        mesh_type: str, optional
            Specify the mesh file type, eg. exo, ugrid, shp, etc
        """
        # Adjust the 1-based node indices to 0-based
        # FIXME Hardwired the SCHISM grid name
        if 'start_index' in dataset.Mesh2_face_nodes.attrs:
            start_index = dataset.Mesh2_face_nodes.attrs['start_index']
            if start_index == 1:
                original_attrs = dataset.Mesh2_face_nodes.attrs
                dataset.update({"Mesh2_face_nodes":
                                dataset.Mesh2_face_nodes - 1})
                dataset.Mesh2_face_nodes.attrs.update(original_attrs)
                dataset.Mesh2_face_nodes.attrs['start_index'] = 0
                fill_value = dataset.Mesh2_face_nodes.attrs['_FillValue']
                dataset.Mesh2_face_nodes.attrs['_FillValue'] = \
                    fill_value - 1
            elif start_index != 0:
                raise ValueError("start_index must be 0 or 1")
        # Initialize the super class
        super().__init__(dataset, **kwargs)

    def __init_grid_var_names__(self):
        super().__init_grid_var_names__()
        # uxarray 2023.06 does not have this variable mapping.
        self.grid_var_names['Mesh2_edge_nodes'] = 'Mesh2_edge_nodes'

    @property
    def face_polygons(self):
        if self._face_polygons is None:
            # If the spatial tree is not built yet, build it
            node_x = self.Mesh2_node_x.values
            node_y = self.Mesh2_node_y.values
            fill_value = self.Mesh2_face_nodes.attrs['_FillValue']

            def create_polygon(node_indices):
                # The node indices are 1-based
                valid = node_indices != fill_value
                ind = node_indices[valid]
                # Assuming the indices are positional
                return Polygon(zip(node_x[ind], node_y[ind]))

            # NOTE Assuming the second dimension of Mesh2_face_nodes is not safe.
            self._face_polygons = xr.apply_ufunc(create_polygon,
                                                 self.Mesh2_face_nodes,
                                                 input_core_dims=(
                                                     (self.Mesh2_face_nodes.dims[1],),),
                                                 dask="parallelized",
                                                 output_dtypes=[object],
                                                 vectorize=True)
        return self._face_polygons

    @property
    def node_points(self):
        if self._node_points is None:
            self._node_points = xr.apply_ufunc(lambda x, y: Point(x, y),
                                               self.Mesh2_node_x,
                                               self.Mesh2_node_y,
                                               vectorize=True,
                                               dask='parallelized')
        return self._node_points

    @property
    def node_strtree(self):
        if self._node_strtree is None:
            self._node_strtree = STRtree(self.node_points)
        return self._node_strtree

    @property
    def elem_strtree(self):
        if self._face_strtree is None:
            self._face_strtree = STRtree(self.face_polygons.values)
        return self._face_strtree

    def find_element_at(self, x, y, predicate='intersects'):
        """ Find the element that contains the point (x, y)

        Parameters
        ----------
        x : float, required
            x coordinate
        y : float, required
            y coordinate
        predicate : str, optional
            Predicate to use for the spatial query by shapely STRtree.
            Default is 'contains'

        Returns
        -------
        elem : array_like
            Element indices
        """
        point = Point(x, y)
        return self.elem_strtree.query(point, predicate=predicate)

    def subset(self, Polygon: Polygon):
        """ Subset the grid to the given polygon

        This function is copied and modified from xugrid topology_subset.

        Parameters
        ----------
        face_index: 1d array of integers or bool Edges of the subset.

        Returns
        -------
        subset: suxarray.Grid
        Parameters
        ----------
        Polygon : shapely.Polygon, required
            Polygon to subset the grid to

        Returns
        -------
        grid : suxarray.Grid
            Subgrid
        """
        # Find the elements that intersect the polygon
        elem_ilocs = self.elem_strtree.query(Polygon, predicate='contains')

        face_subset = self.Mesh2_face_nodes[elem_ilocs, :]
        fill_value = self.Mesh2_face_nodes.attrs['_FillValue']
        node_subset = np.unique(face_subset.where(
            face_subset >= 0, drop=True).values).astype(int)
        # Find edges in the subset
        # If the two nodes in an edge are in the node_subset, then the edge
        # is in the subset
        # Select the edges that are in the subset
        mesh2_edge_nodes = self.Mesh2_edge_nodes.values
        edge_subset_mask = (np.isin(mesh2_edge_nodes[:, 0], node_subset) &
                            np.isin(mesh2_edge_nodes[:, 1], node_subset))
        edge_subset = mesh2_edge_nodes[edge_subset_mask, :]

        ds = self._ds.sel(nMesh2_node=node_subset,
                          nMesh2_face=elem_ilocs,
                          nMesh2_edge=edge_subset_mask)
        # Rename dimensions
        # XXX Continue to work, but this will be a problem again...
        ds = ds.rename_dims(
            {"nMesh2_node": "nMesh2_node_subset"})
        new_face_nodes = renumber_nodes(face_subset.values, fill_value)
        da_new_face_nodes = xr.DataArray(new_face_nodes,
                                         dims=('nMesh2_face',
                                               'nMaxMesh2_face_nodes'),
                                         attrs=self.Mesh2_face_nodes.attrs)
        # Update the face-nodes connectivity variable
        ds.update({self.grid_var_names['Mesh2_face_nodes']: da_new_face_nodes})

        # Update the edge-nodes connectivity variable
        # Replace node indices in the edge connectivity with a node index
        # dictionary
        node_dict = dict(zip(node_subset, np.arange(len(node_subset))),
                         dtype=np.int32)
        node_dict[-1] = -1
        new_edge_nodes = np.array([[node_dict[n] for n in edge]
                                   for edge in edge_subset],
                                  dtype=np.int32)
        da_new_edge_nodes = xr.DataArray(new_edge_nodes,
                                         dims=('nMesh2_edge', 'two'),
                                         attrs=self.Mesh2_edge_nodes.attrs)
        ds.update({self.grid_var_names['Mesh2_edge_nodes']: da_new_edge_nodes})

        # Add the original node numbers as a variable
        da_original_node_indices = xr.DataArray(node_subset,
                                                dims=('nMesh2_node',),
                                                attrs={'long_name': 'Original node indices',
                                                       'start_index': 0})
        ds['Mesh2_node_indices'] = da_original_node_indices

        # Add a history
        ds.attrs['history'] = "Subset by suxarray"

        # Remove the face dimension
        if "nNodes_per_face" in ds:
            ds = ds.drop_vars("nNodes_per_face")

        # Create a suxarray grid and return
        grid_subset = Grid(ds)
        return grid_subset

    def read_vgrid(self, path_vgrid):
        """ Read a SCHISM vgrid file """
        with open(path_vgrid, "r") as f:
            ivcor = int(f.readline().strip())
            if ivcor != 1:
                raise NotImplementedError("Only ivcor=1 is implemented")
            if ivcor == 1:
                nvrt = int(f.readline().strip())
        if ivcor == 1:
            n_nodes = self.Mesh2_node_x.size
            widths = np.full(n_nodes, 11, dtype=np.int32)
            widths[0] = 10
            df_nvrt = pd.read_fwf(path_vgrid,
                                  header=None, skiprows=2, nrows=1,
                                  widths=widths,
                                  dtype=np.int32)
            self.ds['nvrt'] = xr.DataArray(df_nvrt.values.squeeze(),
                                           dims=('nMesh2_node',))
            widths = np.full(n_nodes + 1, 15, dtype=np.int32)
            widths[0] = 10
            df_vgrid = pd.read_fwf(path_vgrid,
                                   header=None, skiprows=3, nrows=nvrt,
                                   widths=[10] + [15] * n_nodes,
                                   na_values=-9.,
                                   dtype=np.float32)
            self.ds['vgrid'] = xr.DataArray(df_vgrid.iloc[:, 1:].values + 1.0,
                                            dims=('nSCHISM_vgrid_layers',
                                                  'nMesh2_node',))

    def compute_face_areas(self):
        """ Compute face areas

        Though uxarray has its own area calculation, it does not work at the
        moment for a hybrid grid. This function builds Shapley polygons for
        faces (elements) and calculates their areas using Shaplely, overriding
        the uxarray's area calculation.

        Returns
        -------
        da_face_areas : xr.DataArray
            Face areas
        """
        ret = xr.apply_ufunc(lambda v: np.vectorize(lambda x: x.area)(
            v), self.face_polygons, dask="parallelized", output_dtypes=float)
        return ret

    # Add plot methods
    hvplot = UncachedAccessor(suxarray.hvplot.PlotMethods)


class Dataset(ux.UxDataset):
    __slots__ = ("_sxgrid",)

    def __init__(self,
                 *args,
                 sxgrid: Grid = None,
                 source_datasets: Optional[str] = None,
                 **kwargs):
        self._sxgrid = sxgrid
        super().__init__(*args,
                         uxgrid=sxgrid,
                         source_datasets=source_datasets,
                         **kwargs)

    def depth_average(self, var_name):
        """ Calculate depth-average of a variable

        Parameters
        ----------
        var_name : str, required
            Variable name

        Returns
        -------
        da : xr.DataArray
            Depth averaged variable
        """
        def _depth_average(v, zs, k, dry):
            # Select the values with the last index from the bottom index
            # array, k
            z_bottom = np.take_along_axis(zs, k.reshape(1, -1, 1), axis=-1)
            depth = zs[:, :, -1] - z_bottom.squeeze(axis=-1)
            # Mask nan values
            v = np.ma.masked_invalid(v, copy=False)
            zs = np.ma.masked_invalid(zs, copy=False)
            result = np.trapz(v, x=zs, axis=-1) / depth
            result[dry == 1.] = np.nan
            return result

        da_da = xr.apply_ufunc(_depth_average,
                               self[var_name],
                               self.zCoordinates,
                               self.bottom_index_node - 1,
                               self.dryFlagNode,
                               input_core_dims=[["nSCHISM_vgrid_layers",],
                                                ["nSCHISM_vgrid_layers",],
                                                [],
                                                []],
                               dask='parallelized',
                               output_dtypes=[float])
        return da_da

    def face_average(self, dataarray: xr.DataArray) -> xr.DataArray:
        """Calculate face average of a variable

        Parameters
        ----------
        dataarray: xr.DataArray, required
            Input variable

        Returns
        -------
        da : xr.DataArray
            Face averaged variable
        """
        da_result = xr.apply_ufunc(
            face_average,
            dataarray,
            self.uxgrid.Mesh2_face_nodes,
            self.uxgrid.nNodes_per_face,
            exclude_dims=set(
                ["nMesh2_node",
                 "nMesh2_face",
                 "nMaxMesh2_face_nodes"]
            ),
            input_core_dims=[
                ["nMesh2_node", ],
                ["nMesh2_face", "nMaxMesh2_face_nodes"],
                ["nMesh2_face", ]],
            output_core_dims=[["nMesh2_face", ], ],
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {
                "nMesh2_face": self.uxgrid.nMesh2_face}},
            dask="parallelized"
        )
        return da_result

    @property
    def sxgrid(self):
        return self._sxgrid


def add_topology_variable(ds, varname="Mesh2"):
    """ Add a dummy mesh_topology variable to a SCHISM out2d dataset

    Parameters
    ----------
    ds : xarray.Dataset, required
        Input SCHISM out2d xarray.Dataset
    varname : str, optional
        Name of the dummy topology variable. Default is "Mesh2"
    """
    ds = ds.assign({varname: 1})
    ds[varname].attrs['cf_role'] = 'mesh_topology'
    ds[varname].attrs['topology_dimension'] = 2
    ds[varname].attrs['node_coordinates'] = "Mesh2_node_x Mesh2_node_y"
    ds[varname].attrs['face_node_connectivity'] = "Mesh2_face_nodes"
    ds[varname].attrs['edge_node_connectivity'] = "Mesh2_edge_nodes"
    ds[varname].attrs['Mesh2_layers'] = "zCoordinates"
    ds[varname].attrs['start_index'] = 0

    return ds


def coerce_mesh_name(ds):
    """Coerce the mesh name to Mesh2

    As of uxarray 2023.06, it is better to use the default mesh name, Mesh2,
    instead of a customized name. So, it is decide to use Mesh2 till this
    issue is resolved in uxarary side.
    """
    da_topo = get_topology_variable(ds)
    if da_topo is None:
        raise ValueError("No mesh_topology variable found")
    if da_topo.name == 'Mesh2':
        return ds
    name_org = da_topo.name
    # Rename the topology metadata variable to Mesh2
    ds = ds.rename_vars({name_org: 'Mesh2'})
    da_topo = ds['Mesh2']
    # if an attr value has the customized mesh name, replace it with Mesh2
    for k in da_topo.attrs:
        # If the value is str, replace it
        if isinstance(da_topo.attrs[k], str):
            da_topo.attrs[k] = da_topo.attrs[k].replace(name_org, 'Mesh2')
    # Rename dimensions
    # First, collect dimensions to rename
    dims_to_rename = {k: k.replace(name_org, 'Mesh2')
                      for k in ds.dims if name_org in k}
    ds = ds.rename_dims(dims_to_rename)
    # Rename coordinates
    # First, collect coordinates to rename
    coords_to_rename = {k: k.replace(name_org, 'Mesh2')
                        for k in ds.coords if name_org in k}
    ds = ds.rename(coords_to_rename)
    # Rename variables
    # First, collect variables to rename
    vars_to_rename = {k: k.replace(name_org, 'Mesh2')
                      for k in ds.data_vars if name_org in k}
    ds = ds.rename(vars_to_rename)
    return ds


# TODO separate utility functions to another file
def renumber_nodes(a, fill_value: int = None):
    if fill_value is None:
        return _renumber(a)

    valid = a != fill_value
    renumbered = np.full_like(a, fill_value)
    renumbered[valid] = _renumber(a[valid])
    return renumbered


def _renumber(a):
    # Taken from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737
    # (scipy is BSD-3-Clause License)
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=int)
    inv[sorter] = np.arange(sorter.size, dtype=int)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    return dense.reshape(a.shape)


def get_topology_variable(dataset: xr.Dataset) -> Optional[str]:
    """ Get the topology xarray.DataArray

    Parameters
    ----------
    dataset : xarray.Dataset, required
        Input xarray.Dataset

    Returns
    -------
    da : xarray.DataArray
        Topology Xarray.DataArray
    """
    ds = dataset.filter_by_attrs(cf_role="mesh_topology")
    if len(ds) == 0:
        return None
    elif len(ds) > 1:
        raise ValueError("Multiple mesh_topology variables found")
    else:
        return ds[list(ds.keys())[0]]


def triangulate(grid):
    """ Triangulate a suxarray grid

    This function split quadrilateral elements into two triangles and returns a
    new suxarray grid. It does not consider the quality of the resulting
    triangles. The main use case of this function is to visualize the grid
    with holoviews or other plotting tools that require a triangulated grid.

    Parameters
    ----------
    grid : Grid, required
        Grid object to triangulate

    Returns
    -------
    grid : Grid
        Triangulated grid
    """
    mesh_name = grid.grid_var_names['Mesh2']
    face_nodes = grid.Mesh2_face_nodes

    n_face = grid.nMesh2_face
    fill_value = face_nodes.attrs["_FillValue"]
    valid = face_nodes != fill_value
    n_per_row = valid.sum(axis=1)
    n_triangle_per_row = n_per_row - 2
    face_ori = np.repeat(np.arange(n_face), n_per_row)
    node_ori = face_nodes.values.ravel()[valid.values.ravel()]
    if face_nodes.attrs['start_index'] == 1:
        node_ori -= 1

    def _triangulate(face_ori: np.ndarray, node_ori: np.ndarray,
                     n_triangle_per_row: xr.DataArray) -> np.ndarray:
        n_triangle = n_triangle_per_row.sum().compute().item()
        n_face = len(face_ori)
        index_first = np.argwhere(np.diff(face_ori, prepend=-1) != 0)
        index_second = index_first + 1
        index_last = np.argwhere(np.diff(face_ori, append=-1) != 0)

        first = np.full(n_face, False)
        first[index_first] = True
        second = np.full(n_face, True) & ~first
        second[index_last] = False
        third = np.full(n_face, True) & ~first
        third[index_second] = False

        triangles = np.empty((n_triangle, 3), np.int32)
        triangles[:, 0] = np.repeat(node_ori[first], n_triangle_per_row)
        triangles[:, 1] = node_ori[second]
        triangles[:, 2] = node_ori[third]
        return triangles

    triangles = _triangulate(face_ori, node_ori, n_triangle_per_row)

    triangle_original_ind = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row)

    # Copy the data from the original grid
    ds_tri = grid._ds.copy()
    # Drop the original face_nodes variable
    # TODO dryFlagElement is needed to be updated not dropped, but let's drop
    # it for now.
    varnames_to_drop = [f"{mesh_name}_face_nodes", "dryFlagElement",
                        f"{mesh_name}_face_x", f"{mesh_name}_face_y",
                        "nNodes_per_face"]
    for varname in varnames_to_drop:
        if varname in ds_tri:
            ds_tri = ds_tri.drop_vars(varname)
    da_face_nodes = xr.DataArray(data=triangles,
                                 dims=(f"n{mesh_name}_face", "three"),
                                 name=f"{mesh_name}_face_nodes")
    da_face_nodes.attrs['start_index'] = 0
    da_face_nodes.attrs['cf_role'] = 'face_node_connectivity'
    ds_tri[da_face_nodes.name] = da_face_nodes
    da_elem_ind = xr.DataArray(data=triangle_original_ind,
                               dims=(f"n{mesh_name}_face"),
                               name=f"{mesh_name}_face_original")
    ds_tri[da_elem_ind.name] = da_elem_ind
    grid_tri = Grid(ds_tri, islation=False, mesh_type="ugrid")
    grid_tri.Mesh2.attrs['start_index'] = 0
    return grid_tri


def face_average(val, face_nodes, n_nodes_per_face):
    """Calculate face average of a variable

    Calculate average of a variable at each face. No weighting is applied.

    Parameters
    ----------
    val: ndarray, required
        values to process. The last dimension must be for the nodes.
    face_nodes: ndarray, required
        Face-node connectivity array
    n_nodes_per_face: ndarray, required
        Number of nodes per face

    Returns
    -------
    xarray.DataArray
        Face averaged variable
    """
    n_face, _ = face_nodes.shape

    # set initial area of each face to 0
    result = np.zeros(val.shape[:-1] + (n_face, ))

    for face_idx, max_nodes in enumerate(n_nodes_per_face):
        avg = (val[..., face_nodes[face_idx, 0:max_nodes]].sum(axis=-1)
               / max_nodes)
        result[..., face_idx] = avg
    return result


def read_hgrid_gr3(path_hgrid):
    """ Read SCHISM hgrid.gr3 file and return suxarray grid """
    # read the header
    with open(path_hgrid, "r") as f:
        f.readline()
        n_faces, n_nodes = [int(x) for x in f.readline().strip().split()[:2]]
    # Read the node section. Read only up to the fourth column
    df_nodes = pd.read_csv(path_hgrid, skiprows=2, header=None,
                           nrows=n_nodes, delim_whitespace=True, usecols=range(4))
    # Read the face section. Read only up to the sixth column. The last column
    # may exist or not.
    df_faces = pd.read_csv(path_hgrid, skiprows=2 + n_nodes, header=None,
                           nrows=n_faces, delim_whitespace=True, names=range(6))
    # TODO Read boundary information, if any
    # Create suxarray grid
    ds = xr.Dataset()
    ds['Mesh2_node_x'] = xr.DataArray(
        data=df_nodes[1].values, dims="nMesh2_node")
    ds['Mesh2_node_y'] = xr.DataArray(
        data=df_nodes[2].values, dims="nMesh2_node")
    # Replace NaN with -1
    df_faces = df_faces.fillna(0)
    ds['Mesh2_face_nodes'] = xr.DataArray(
        data=df_faces[[2, 3, 4, 5]].astype(int).values - 1,
        dims=("nMesh2_face", "nMaxMesh2_face_nodes"),
        attrs={"start_index": 0,
               "cf_role": "face_node_connectivity",
               "_FillValue": -1}
    )
    ds['depth'] = df_nodes[3].values
    # Add dummy mesh_topology variable
    ds = add_topology_variable(ds)
    ds = coerce_mesh_name(ds)

    grid = Grid(ds, islation=False, mesh_type="ugrid")
    return grid
