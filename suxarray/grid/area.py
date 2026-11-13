import numpy as np
from numba import njit


@njit(cache=True)
def _face_area_tri3(x, y):
    """Calculate the area of a triangle with three nodes

    Parameters
    ----------
    x : ndarray, required
        x coordinates of the triangle nodes
    y : ndarray, required
        y coordinates of the triangle nodes

    Returns
    -------
    float
        area of the triangle
    """
    return np.abs(0.5 * ((x[0] - x[2]) * (y[1] - y[2]) - (x[1] - x[2]) * (y[0] - y[2])))


@njit(cache=True)
def _face_area_quad4(x, y):
    """Calculate the planar area of a quadrilateral with four nodes using the shoelace formula.

    Parameters
    ----------
    x : ndarray, required
        x coordinates of the quad nodes (ordered CCW or CW)
    y : ndarray, required
        y coordinates of the quad nodes (ordered CCW or CW)

    Returns
    -------
    float
        area of the quadrilateral
    """
    return 0.5 * np.abs(
        (x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0])
        - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
    )


@njit(cache=True)
def _compute_planar_face_areas(node_x, node_y, connectivity):
    """Compute 2D planar face areas for all elements (triangles and quads).

    Parameters
    ----------
    node_x : ndarray
        x coordinates of all nodes
    node_y : ndarray
        y coordinates of all nodes
    connectivity : ndarray
        Face-node connectivity with negative fill values for triangles

    Returns
    -------
    ndarray
        Array of shape (n_face,) with planar areas in m^2 (or coordinate units^2)
    """
    n_face = connectivity.shape[0]
    areas = np.empty(n_face, dtype=np.float64)

    for i in range(n_face):
        element = connectivity[i, connectivity[i] >= 0]
        n = len(element)
        x = node_x[element]
        y = node_y[element]

        if n == 3:
            areas[i] = _face_area_tri3(x, y)
        elif n == 4:
            areas[i] = _face_area_quad4(x, y)
        else:
            # General polygon shoelace formula
            area = 0.0
            for j in range(n):
                j_next = (j + 1) % n
                area += x[j] * y[j_next] - y[j] * x[j_next]
            areas[i] = 0.5 * np.abs(area)

    return areas



@njit(cache=True)
def _integrate_triangle(x, y, z):
    """
    Integrate over a triangle using the three corner points.
    """
    area_2 = np.abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    result = np.empty((z.shape[0],), dtype=z.dtype)
    for k in range(z.shape[0]):
        result[k] = 0.5 * area_2 * z[k, :].mean()
    return result


@njit(cache=True)
def _gauss_quadrature_points_order2():
    return np.array([[-1 / np.sqrt(3), 1 / np.sqrt(3)]], dtype=np.float64), np.array(
        [1.0, 1.0]
    )


@njit(cache=True)
def _shape_functions_quad_bilinear(xi, eta, z_dtype):
    return np.array(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ],
        dtype=z_dtype,
    )


@njit(cache=True)
def _shape_functions_derivative_quad_bilinear(xi, eta, x_dtype):
    dxi = np.array(
        [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
        dtype=x_dtype,
    )
    deta = np.array(
        [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)],
        dtype=x_dtype,
    )
    return dxi, deta


@njit(cache=True)
def _integrate_quad(x, y, z):
    # points, weights = _gauss_quadrature_points_order2()
    points = np.array([[-0.577350269189626, 0.577350269189626]])
    weights = np.array([1.0, 1.0])
    result = np.zeros((z.shape[0],), dtype=z.dtype)

    for i in range(2):
        for j in range(2):
            xi, eta = points[0][i], points[0][j]
            shape = _shape_functions_quad_bilinear(xi, eta, z.dtype)
            dxi, deta = _shape_functions_derivative_quad_bilinear(xi, eta, x.dtype)

            dx_dxi = np.dot(dxi, x)
            dy_dxi = np.dot(dxi, y)
            dx_deta = np.dot(deta, x)
            dy_deta = np.dot(deta, y)

            jacobian_det = np.abs(dx_dxi * dy_deta - dx_deta * dy_dxi)
            for k in range(z.shape[0]):
                z_local = np.dot(shape, z[k, :])
                result[k] += weights[i] * weights[j] * z_local * jacobian_det

    return result


@njit(cache=True)
def _integrate_nodal_2d(node_x, node_y, values, connectivity):
    n_elements = connectivity.shape[0]
    result = np.zeros((values.shape[0], n_elements), dtype=values.dtype)

    for i in range(n_elements):
        # Select indices that are not negative
        element = connectivity[i, connectivity[i] >= 0]
        n = len(element)
        x = node_x[element]
        y = node_y[element]
        z = values[:, element]

        if n == 3:  # Triangle
            result[:, i] = _integrate_triangle(x, y, z)
        elif n == 4:  # Quadrilateral
            result[:, i] = _integrate_quad(x, y, z)

    return result


@njit(cache=True)
def _integrate_nodal(node_x, node_y, values, connectivity):
    if values.ndim == 1:
        v_2d = np.empty((1, len(values)), dtype=values.dtype)
        v_2d[0, :] = values
        res_2d = _integrate_nodal_2d(node_x, node_y, v_2d, connectivity)
        return res_2d[0]
    else:
        return _integrate_nodal_2d(node_x, node_y, values, connectivity)
