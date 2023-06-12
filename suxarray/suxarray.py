""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""
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
        # Add a topology dummy variable if it doesn't exist
        # The current SCHISM out2d does not have this variable.
        if get_topology_variable(dataset) is None:
            dataset = self.add_topology_variable(dataset)
        # Adjust the 1-based node indices to 0-based
        if 'start_index' in dataset.SCHISM_hgrid_face_nodes.attrs:
            start_index = dataset.SCHISM_hgrid_face_nodes.attrs['start_index']
            if start_index == 1:
                dataset.update({"SCHISM_hgrid_face_nodes":
                                dataset.SCHISM_hgrid_face_nodes - 1})
                dataset.SCHISM_hgrid_face_nodes.attrs['start_index'] = 0
            elif start_index != 0:
                raise ValueError("start_index must be 0 or 1")
        # Initialize the super class
        super().__init__(dataset, **kwargs)
        # Add an optional edge node connectivity variable name

    def __init_ds_var_names__(self):
        super().__init_ds_var_names__()
        self.ds_var_names['Mesh2_edge_nodes'] = 'SCHISM_hgrid_edge_nodes'

    @staticmethod
    def add_topology_variable(ds, varname="SCHISM_hgrid"):
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
        ds[varname].attrs['node_coordinates'] = "SCHISM_hgrid_node_x SCHISM_hgrid_node_y"
        ds[varname].attrs['face_node_connectivity'] = "SCHISM_hgrid_face_nodes"
        ds[varname].attrs['edge_node_connectivity'] = "SCHISM_hgrid_edge_nodes"
        ds[varname].attrs['Mesh2_layers'] = "zCoordinates"
        ds[varname].attrs['start_index'] = 1

        return ds

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
                ind = node_indices[valid] - 1
                # Assuming the indices are positional
                return Polygon(zip(node_x[ind], node_y[ind]))

            self._face_polygons = xr.apply_ufunc(create_polygon,
                                                 self.Mesh2_face_nodes,
                                                 input_core_dims=(
                                                     (self.ds_var_names['nMaxMesh2_face_nodes'],),),
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
            face_subset > 0, drop=True).values).astype(int) - 1
        # Find edges in the subset
        # If the two nodes in an edge are in the node_subset, then the edge
        # is in the subset
        # Select the edges that are in the subset
        mesh2_edge_nodes = self.ds.SCHISM_hgrid_edge_nodes.values - 1
        edge_subset_mask = (np.isin(mesh2_edge_nodes[:, 0], node_subset) &
                            np.isin(mesh2_edge_nodes[:, 1], node_subset))
        edge_subset = mesh2_edge_nodes[edge_subset_mask, :]

        # TODO Need to slice the edge variable as well.
        ds = self.ds.sel(nSCHISM_hgrid_node=node_subset,
                         nSCHISM_hgrid_face=elem_ilocs,
                         nSCHISM_hgrid_edge=edge_subset_mask)
        new_face_nodes = renumber_nodes(face_subset.values, fill_value)
        da_new_face_nodes = xr.DataArray(new_face_nodes,
                                         dims=('nSCHISM_hgrid_face',
                                               'nMaxSCHISM_hgrid_face_nodes'),
                                         attrs=self.Mesh2_face_nodes.attrs)
        # Update the face-nodes connectivity variable
        ds.update({self.ds_var_names['Mesh2_face_nodes']: da_new_face_nodes})

        # Update the edge-nodes connectivity variable
        # Replace node indices in the edge connectivity with a node index
        # dictionary
        node_dict = dict(zip(node_subset, np.arange(1, len(node_subset) + 1)),
                         dtype=np.int32)
        node_dict[-1] = -1
        new_edge_nodes = np.array([[node_dict[n] for n in edge]
                                   for edge in edge_subset],
                                  dtype=np.int32)
        da_new_edge_nodes = xr.DataArray(new_edge_nodes,
                                         dims=('nSCHISM_hgrid_edge', 'two'),
                                         attrs=self.ds.SCHISM_hgrid_edge_nodes.attrs)
        ds.update({self.ds_var_names['Mesh2_edge_nodes']: da_new_edge_nodes})

        # Add the original node numbers as a variable
        da_original_node_indices = xr.DataArray(node_subset + 1,
                                                dims=('nSCHISM_hgrid_node',),
                                                attrs={'long_name': 'Original node indices',
                                                       'start_index': 1})
        ds['SCHISM_hgrid_node_indices'] = da_original_node_indices

        # Add a history
        ds.attrs['history'] = "Subset by suxarray"

        # Remove the face dimension
        ds = ds.drop_vars("Mesh2_face_dimension")

        # Create a suxarray grid and return
        grid_subset = Grid(ds)
        return grid_subset

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
            dataarray.load(),
            self.Mesh2_face_nodes,
            self.Mesh2_face_dimension,
            exclude_dims=set(
                ["nSCHISM_hgrid_node",
                 "nSCHISM_hgrid_face",
                 "nMaxSCHISM_hgrid_face_nodes"]
            ),
            input_core_dims=[
                ["nSCHISM_hgrid_node", ],
                ["nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"],
                ["nMesh2_face",]],
            output_core_dims=[["nMesh2_face",]],
            dask="parallelized"
        )
        return da_result

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
            return np.trapz(v, x=zs, axis=-1) / depth

        da_da = xr.apply_ufunc(_depth_average,
                               self.ds[var_name],
                               self.ds.zCoordinates,
                               self.ds.bottom_index_node - 1,
                               self.ds.dryFlagNode,
                               input_core_dims=[["nSCHISM_vgrid_layers",],
                                                ["nSCHISM_vgrid_layers",],
                                                [],
                                                []],
                               dask='parallelized',
                               output_dtypes=[float])
        return da_da

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
                                           dims=('nSCHISM_hgrid_node',))
            widths = np.full(n_nodes + 1, 15, dtype=np.int32)
            widths[0] = 10
            df_vgrid = pd.read_fwf(path_vgrid,
                                   header=None, skiprows=3, nrows=nvrt,
                                   widths=[10] + [15] * n_nodes,
                                   na_values=-9.,
                                   dtype=np.float32)
            self.ds['vgrid'] = xr.DataArray(df_vgrid.iloc[:, 1:].values + 1.0,
                                            dims=('nSCHISM_vgrid_layers',
                                                  'nSCHISM_hgrid_node',))

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


def get_topology_variable(dataset):
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
    mesh_name = grid.ds_var_names['Mesh2']
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
    ds_tri = grid.ds.copy()
    # Drop the original face_nodes variable
    varnames_to_drop = ["SCHISM_hgrid_face_nodes", "dryFlagElement",
                        "SCHISM_hgrid_face_x", "SCHISM_hgrid_face_y",
                        "Mesh2_face_dimension"]
    for varname in varnames_to_drop:
        if varname in ds_tri:
            ds_tri = ds_tri.drop_vars(varname)
    da_face_nodes = xr.DataArray(data=triangles,
                                 dims=(f"n{mesh_name}_face", "three"),
                                 name=f"{mesh_name}_face_nodes")
    ds_tri[da_face_nodes.name] = da_face_nodes
    da_elem_ind = xr.DataArray(data=triangle_original_ind,
                               dims=(f"n{mesh_name}_face"),
                               name=f"{mesh_name}_face_original")
    ds_tri[da_elem_ind.name] = da_elem_ind
    grid_tri = Grid(ds_tri, islation=False, mesh_type="ugrid")
    # grid_tri.Mesh2.attrs['start_index'] = 0
    return grid_tri


@njit
def face_average(val, face_nodes, face_geometry):
    """Calculate face average of a variable

    Calculate average of a variable at each face. No weighting is applied.

    Parameters
    ----------
    face_nodes: ndarray, required
        Face-node connectivity array
    face_geometry: ndarray, required
        Number of nodes per face

    Returns
    -------
    xarray.DataArray
        Face averaged variable
    """
    n_face, _ = face_nodes.shape

    # set initial area of each face to 0
    result = np.zeros(val.shape[:-1] + (n_face,))

    for face_idx, max_nodes in enumerate(face_geometry):
        avg = (val[..., face_nodes[face_idx, 0:max_nodes]].sum(axis=-1)
               / max_nodes)
        result[..., face_idx] = avg

    return result


def read_hgrid_gr3(path_hgrid):
    """ Read SCHISM hgrid.gr3 file and return suxarray grid """
    # read the header
    with open(path_hgrid, "r") as f:
        first_line = f.readline()
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
    ds['SCHISM_hgrid_node_x'] = xr.DataArray(
        data=df_nodes[1].values, dims="nSCHISM_hgrid_node")
    ds['SCHISM_hgrid_node_y'] = xr.DataArray(
        data=df_nodes[2].values, dims="nSCHISM_hgrid_node")
    # Replace NaN with -1
    df_faces = df_faces.fillna(0)
    ds['SCHISM_hgrid_face_nodes'] = xr.DataArray(
        data=df_faces[[2, 3, 4, 5]].astype(int).values - 1,
        dims=("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
        attrs={"start_index": 0,
               "cf_role": "face_node_connectivity",
               "_FillValue": -1}
        )
    ds['depth'] = df_nodes[3].values
    # Add dummy mesh_topology variable
    ds = Grid.add_topology_variable(ds)

    grid = Grid(ds, islation=False, mesh_type="ugrid")
    return grid
