import numpy as np
import numba as nb


@nb.njit(cache=True, nogil=True)
def _depth_average_3d(v, zs, k, dry, result):
    n_layer = v.shape[2]
    zero = result.dtype.type(0.0)
    half = result.dtype.type(0.5)
    for t_i in range(v.shape[0]):
        for n_i in range(v.shape[1]):
            dry_val = dry[t_i, n_i]
            if dry_val == 1.0 or np.isnan(dry_val):
                result[t_i, n_i] = np.nan
                continue
            k_val = k[n_i]
            if np.isnan(k_val):
                result[t_i, n_i] = np.nan
                continue
            k_i = int(k_val)
            if k_i < 0 or k_i >= n_layer - 1:
                result[t_i, n_i] = np.nan
                continue
            depth = zs[t_i, n_i, n_layer - 1] - zs[t_i, n_i, k_i]
            if not (depth > 0.0):
                result[t_i, n_i] = np.nan
                continue
            acc = zero
            for z_i in range(k_i, n_layer - 1):
                dz = zs[t_i, n_i, z_i + 1] - zs[t_i, n_i, z_i]
                acc += dz * (v[t_i, n_i, z_i] + v[t_i, n_i, z_i + 1])
            result[t_i, n_i] = half * acc / depth
    return result


@nb.njit(cache=True, nogil=True)
def _depth_average_2d(v, zs, k, dry, result):
    n_layer = v.shape[1]
    zero = result.dtype.type(0.0)
    half = result.dtype.type(0.5)
    for i in range(v.shape[0]):
        dry_val = dry[i]
        if dry_val == 1.0 or np.isnan(dry_val):
            result[i] = np.nan
            continue
        k_val = k[i]
        if np.isnan(k_val):
            result[i] = np.nan
            continue
        k_i = int(k_val)
        if k_i < 0 or k_i >= n_layer - 1:
            result[i] = np.nan
            continue
        depth = zs[i, n_layer - 1] - zs[i, k_i]
        if not (depth > 0.0):
            result[i] = np.nan
            continue
        acc = zero
        for z_i in range(k_i, n_layer - 1):
            dz = zs[i, z_i + 1] - zs[i, z_i]
            acc += dz * (v[i, z_i] + v[i, z_i + 1])
        result[i] = half * acc / depth
    return result


def _depth_average(v, zs, k, dry):
    """Calculate the depth average of a variable

    When a node is dry, the bottom index is out of range, or the depth is
    non-positive/NaN, the depth-averaged value is set to NaN.

    Parameters
    ----------
    v : numpy.ndarray
        Variable to be averaged, with layer axis as the last dimension.
    zs : numpy.ndarray
        z-coordinates of the grid, with layer axis as the last dimension.
    k : numpy.ndarray
        Zero-based bottom index of the grid.
    dry : numpy.ndarray
        Dry flag of the grid (1.0 indicates dry).

    Returns
    -------
    numpy.ndarray
        Depth-averaged variable
    """
    out_dtype = np.result_type(v.dtype, np.float32)
    if (
        v.ndim == 3
        and zs.ndim == 3
        and k.ndim == 1
        and dry.ndim == 2
        and zs.shape == v.shape
        and k.shape[0] == v.shape[1]
        and dry.shape == v.shape[:2]
    ):
        result = np.empty(v.shape[:2], dtype=out_dtype)
        return _depth_average_3d(v, zs, k, dry, result)

    n_layer = v.shape[-1]
    leading_shape = np.broadcast_shapes(
        v.shape[:-1], zs.shape[:-1], k.shape, dry.shape
    )
    result = np.empty(leading_shape, dtype=out_dtype)
    if result.size == 0:
        return result

    v_2d = np.broadcast_to(v, (*leading_shape, n_layer)).reshape(-1, n_layer)
    zs_2d = np.broadcast_to(zs, (*leading_shape, n_layer)).reshape(-1, n_layer)
    k_1d = np.broadcast_to(k, leading_shape).reshape(-1)
    dry_1d = np.broadcast_to(dry, leading_shape).reshape(-1)
    _depth_average_2d(v_2d, zs_2d, k_1d, dry_1d, result.reshape(-1))
    return result
