import numpy as np
import numba as nb


@nb.njit(nopython=True, cache=True)
def _depth_average(v, zs, k, dry):
    """Calculate the depth average of a variable

    When a node is dry or the depth is zero, the depth-averaged value is set to NaN.

    Parameters
    ----------
    v : numpy.ndarray
        Variable to be averaged
    zs : numpy.ndarray
        z-coordinates of the grid
    k : numpy.ndarray
        Bottom index of the grid
    dry : numpy.ndarray
        Dry flag of the grid

    Returns
    -------
    numpy.ndarray
        Depth-averaged variable
    """
    result = np.zeros(v.shape[:-1], dtype=v.dtype)
    for t_i in range(v.shape[0]):
        for n_i in range(v.shape[1]):
            if dry[t_i, n_i] == 1.0:
                result[t_i, n_i] = np.nan
                continue
            k_i = k[n_i]
            thickness = zs[t_i, n_i, k_i + 1 :] - zs[t_i, n_i, k_i:-1]
            depth = zs[t_i, n_i, -1] - zs[t_i, n_i, k_i]
            if depth == 0.0:
                result[t_i, n_i] = np.nan
                continue
            result[t_i, n_i] = (
                0.5
                * np.sum(thickness * (v[t_i, n_i, k_i:-1] + v[t_i, n_i, k_i + 1 :]))
                / depth
            )
    return result
