""" Plotting routines for suxarray

We want to visualize the SCHISM mesh data easily for quick review using
Jupyter Notebook.
The goals of this module are not exact and production quality plotting.

Many of the ideas and implementation are borrowed from out libraries such as
holoviewes, xarray, xugrid, etc.

For now, we use holoviews only.
"""
import numpy as np
import xarray as xr
import holoviews as hv
from holoviews.operation.datashader import rasterize


# Load Bokeh extension for Notebook
if hv.extension and not getattr(hv.extension, "_loaded", False):
    hv.extension("bokeh", logo=False)


class PlotMethods:
    """Plotting methods class

    This class provides accessors to plotting routines.
    """

    def __init__(self, obj):
        self.grid = obj

    def trimesh(self, obj=None, **kwargs):
        m = TriMesh(self.grid)
        return m.plot_dataarray(obj, **kwargs)


class TriMesh:
    """Base class for the TriMesh plotter"""

    def __init__(self, grid):
        """Initialize the TriMesh plot

        Parameters
        ----------
        ds_out2d: suxarary.Grid
            Dataset with the mesh information
        """
        self.grid = grid
        self.process_mesh()

    def process_mesh(self):
        """Process the mesh data in the initialization

        This method is called in the initialization to create attributes
        for uxarray and other later use.
        """
        import suxarray as sx

        self.grid = sx.triangulate(self.grid)
        self.node_x = self.grid.Mesh2_node_x.values
        self.node_y = self.grid.Mesh2_node_y.values
        # self.node_z = self.grid.ds.depth.values
        self.node_z = np.zeros_like(self.node_x)
        self.node_coordinates = xr.DataArray(
            [self.grid.Mesh2_node_x, self.grid.Mesh2_node_y, self.node_z],
            dims=["three", "nSCHISM_hgrid_node"],
        ).transpose()

        self.points = hv.Points(self.node_coordinates.values, vdims=["z"])

    # def dynamic_mesh(self, dataarray, index, level=-1):
    #     da_t = dataarray.sel(time=index)
    #     if self.dim == 2:
    #         self.nodes.data[:, -1] = da_t.values
    #     elif self.dim == 3:
    #         self.nodes.data[:, -1] = da_t.sel(nSCHISM_vgrid_layers=level).values
    #     mesh = hv.TriMesh((self.ux_grid.ds.Mesh2_face_nodes.values, self.nodes))
    #     return mesh

    # def prepare_plot(self, dataarray: xr.DataArray):
    #     self.dataarray = dataarray
    #     self.dim = len(self.dataarray.dims)
    #     if self.dim == 3:
    #         self.dim_level = hv.Dimension(
    #             "level", values=dataarray.nSCHISM_vgrid_layers.values
    #         )

    def trimesh_variable(self, varname: str):
        da = self.grid.ds[varname]
        self.points.data[:, -1] = da.values
        p = hv.TriMesh((self.grid.Mesh2_face_nodes.values, self.points))
        return p

    def plot_dataarray(self, da: xr.DataArray, **kwargs):
        self.points.data[:, -1] = da.values
        p = rasterize(
            hv.TriMesh((self.grid.Mesh2_face_nodes.values, self.points), **kwargs)
        )
        return p

    # def plot(self,
    #          varname: str = None,
    #          dataarray: xr.DataArray = None,
    #          **kwargs):
    #     """Create a TriMesh plot"""
    #     if varname is not None:
    #         p = self.trimesh_variable(varname)
    #     elif dataarray is not None:
    #         p = self.trimesh_dataarray(dataarray)
    #     else:
    #         raise NotImplementedError("Either varname or dataarray must be provided")
    #     p = rasterize(self.trimesh(varname))
    #     # self.prepare_plot(dataarray)
    #     # if self.dim == 2:
    #     #     plot = rasterize(
    #     #         hv.DynamicMap(
    #     #             lambda t: self.dynamic_mesh(self.dataarray, t), kdims=self.dim_time
    #     #         )
    #     #     )
    #     # elif self.dim == 3:
    #     #     plot = rasterize(
    #     #         hv.DynamicMap(
    #     #             lambda t, l: self.dynamic_mesh(self.dataarray, t, l),
    #     #             kdims=[self.dim_time, self.dim_level],
    #     #         )
    #     #     )
    #     # else:
    #     #     raise ValueError("DataArray must be 2D or 3D")
    #     return p
