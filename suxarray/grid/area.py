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
def _integrate_triangle(x, y, z):
    """
    Integrate over a triangle using the three corner points.
    """
    return (
        np.abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        * np.array([z[i, :].mean() for i in range(z.shape[0])])
        * 0.5
    )


@njit(cache=True)
def _gauss_quadrature_points_order2():
    return np.array([[-1 / np.sqrt(3), 1 / np.sqrt(3)]], dtype=np.float64), np.array(
        [1.0, 1.0]
    )


@njit(cache=True)
def _shape_functions_quad_bilinear(xi, eta):
    return np.array(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ]
    )


@njit(cache=True)
def _shape_functions_derivative_quad_bilinear(xi, eta):
    dxi = np.array(
        [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
    )
    deta = np.array(
        [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)],
    )
    return dxi, deta


@njit(cache=True)
def _integrate_quad(x, y, z):
    # points, weights = _gauss_quadrature_points_order2()
    points = np.array([[-0.577350269189626, 0.577350269189626]])
    weights = np.array([1.0, 1.0])
    result = np.zeros((z.shape[0],))

    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            xi, eta = points[0][i], points[0][j]
            shape = _shape_functions_quad_bilinear(xi, eta)
            dxi, deta = _shape_functions_derivative_quad_bilinear(xi, eta)

            z_local = np.array([np.dot(shape, z[i, :]) for i in range(z.shape[0])])

            dx_dxi = np.dot(dxi, x)
            dy_dxi = np.dot(dxi, y)
            dx_deta = np.dot(deta, x)
            dy_deta = np.dot(deta, y)

            jacobian_det = np.abs(dx_dxi * dy_deta - dx_deta * dy_dxi)
            result += wi * wj * z_local * jacobian_det

    return result


@njit(cache=True)
def _integrate_nodal(node_x, node_y, values, connectivity):
    n_elements = connectivity.shape[0]
    result = np.zeros((values.shape[0], n_elements))

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
