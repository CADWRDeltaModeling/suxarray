import numpy as np
import vtk
from vtkmodules.vtkCommonCore import (
    VTK_DOUBLE,
    VTK_FLOAT
)
from vtkmodules.vtkCommonDataModel import (
    VTK_QUAD,
    VTK_TRIANGLE
)
from vtk.util import numpy_support

def create_vtk_grid(grid):
    """ Create a VTK grid from the grid object

    Parameters
    ----------
    grid : suxarray.Grid
        Grid to create a VTK grid from
    var_name : str, required
        Variable name

    Returns
    -------
    vtk.vtkUnstructuredGrid
    """
    # Get he number of nodes
    n_points = grid.n_node
    # Create an unstructured grid object
    vtkgrid = vtk.vtkUnstructuredGrid()

    # Create an empty VTK points (or nodes)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(n_points)

    v_set_point = np.vectorize(
        lambda i, x, y: points.SetPoint(i, x, y, 0.))
    v_set_point(np.arange(n_points), grid.node_x, grid.node_y)
    vtkgrid.SetPoints(points)

    # Create faces (or cells)
    # Add VTK cells to the grid
    def insert_cell(face):
        if face[-1] < 0:
            vtkgrid.InsertNextCell(VTK_TRIANGLE, 3, face[:3])
        else:
            vtkgrid.InsertNextCell(VTK_QUAD, 4, face[:4])

    [insert_cell(face) for face in grid.face_node_connectivity.values]
    # xr.apply_ufunc(insert_cell, grid.Mesh2_face_nodes - 1,
    #                input_core_dims=[['nMaxSCHISM_hgrid_face_nodes',]],
    #                dask='parallelized',
    #                vectorize=True,
    #                output_dtypes=[None])
    return vtkgrid


def add_np_array_to_vtk(vtkgrid, np_array, name):
    """ Add an numpy array values to the VTK data
    """
    array = numpy_support.numpy_to_vtk(num_array=np_array,
                                       deep=True,
                                       array_type=VTK_FLOAT)
    array.SetName(name)
    vtkgrid.GetPointData().AddArray(array)


def write_vtk_grid(vtkgrid, fname):
    """ Write a vtk unstructured grid file

        Parameters
        ----------
        ugrid: vtk.vtkUnstructuredGrid
            VTK unstructured grid data to write
        fname: str
            file name to write
    """
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(vtkgrid)
    writer.Write()
