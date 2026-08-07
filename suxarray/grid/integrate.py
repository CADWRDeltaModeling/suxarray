import numpy as np
from .area import _integrate_nodal


def _coerce_single_timestep(arr, name):
    """Return a single-timestep 2D view for (node, layer) kernels.

    Accepts:
    - 2D arrays with shape (n_node, n_layer)
    - 3D arrays with shape (1, n_node, n_layer), which are squeezed

    Raises for multi-timestep 3D inputs.
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        raise ValueError(
            f"{name} must contain exactly one timestep for this kernel. "
            f"Got shape {arr.shape}. Select one timestep first (for example, "
            f"isel(time=i)) and parallelize externally with xarray/dask."
        )

    raise ValueError(
        f"{name} must be 2D (n_node, n_layer) or 3D with a single time "
        f"step (1, n_node, n_layer). Got ndim={arr.ndim}."
    )


def _earth_radius_at_latitude(latitude_deg):
    """Compute geocentric Earth radius (meters) at geodetic latitude.

    Uses the WGS84 ellipsoid semi-major/minor axes.
    """
    # WGS84 ellipsoid axes (meters)
    a = 6378137.0
    b = 6356752.314245

    lat = np.deg2rad(latitude_deg)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)

    a2 = a * a
    b2 = b * b

    numerator = (a2 * cos_lat) ** 2 + (b2 * sin_lat) ** 2
    denominator = (a * cos_lat) ** 2 + (b * sin_lat) ** 2
    return np.sqrt(numerator / denominator)


def _face_mean_from_nodal(values, connectivity):
    """Average nodal values onto faces.

    Parameters
    ----------
    values : ndarray, shape (n_node, n_interface)
        Nodal values to average.
    connectivity : ndarray, shape (n_face, n_max_face_nodes)
        Face-node connectivity with negative fill values.

    Returns
    -------
    ndarray, shape (n_interface, n_face)
        Face-averaged values.
    """
    n_node, n_interface = values.shape
    n_face = connectivity.shape[0]
    face_mean = np.zeros((n_interface, n_face), dtype=np.float64)

    for face_i in range(n_face):
        node_ids = connectivity[face_i]
        node_ids = node_ids[node_ids >= 0]
        face_mean[:, face_i] = values[node_ids, :].mean(axis=0)

    return face_mean


def _calc_layers_thickness(z, bottom_indices):
    """
    Calculate the thickness of each layer under a set of nodes.

    Parameters
    z : ndarray, shape (n_node, n_layer)
        z-coordinates of nodes
    bottom_indices : ndarray, shape (n_node,)
        0-based bottom layer index for each node

    Returns
        ndarray, shape (n_node, n_layer-1)
    """
    if z.ndim != 2:
        raise ValueError(
            "z must be 2D (n_node, n_layer) for thickness calculations."
        )

    thickness = z[:, 1:] - z[:, :-1]

    if np.any(bottom_indices > 0):
        layer_idx = np.arange(thickness.shape[-1])
        below_bottom = layer_idx[np.newaxis, :] < bottom_indices[:, np.newaxis]
        thickness = np.where(below_bottom, 0.0, thickness)

    return thickness


def _calculate_prism_volumes(node_x, node_y, z, connectivity, bottom_indices):
    """
    Calculate the volume of prism under a selection of nodes for each layer in
    the vertical direction.

    Parameters
    ----------
    node_x : ndarray, shape (n_node,)
        x coordinates of nodes
    node_y : ndarray, shape (n_node,)
        y coordinates of nodes
    z : ndarray, shape (n_node, n_layer)
        z-coordinates for a single timestep.
    connectivity : ndarray, shape (n_face, n_node_per_face)
        Node indices for each face
    bottom_indices : ndarray, shape (n_element,)
        0-based bottom layer index for each node

    Returns
    -------
    ndarray, shape (n_layer-1, n_face).
        Prism volumes per layer and face.
    """
    # Enforce float64 for numba
    node_x = node_x.astype(np.float64)
    node_y = node_y.astype(np.float64)
    z = z.astype(np.float64)

    z = _coerce_single_timestep(z, "z")

    # Calculate thickness of layers under our given nodes
    thickness = _calc_layers_thickness(z, bottom_indices)
    h = thickness.T

    # Integrate!
    integrated = _integrate_nodal(node_x, node_y, h, connectivity)

    return integrated


def projected_volume_ufunc_kernel(
    z,
    node_x,
    node_y,
    connectivity,
    bottom_indices,
):
    """Array-kernel companion for ``calc_volumes_from_projected``.

    This helper is designed for external orchestration with
    ``xarray.apply_ufunc(..., dask='parallelized')`` where ``z`` is passed as
    an ndarray block/chunk.

    Parameters
    ----------
    z : ndarray, shape (n_node, n_layer)
        Single-timestep z-coordinates.
    node_x : ndarray, shape (n_node,)
        x coordinates of nodes.
    node_y : ndarray, shape (n_node,)
        y coordinates of nodes.
    connectivity : ndarray, shape (n_face, n_node_per_face)
        Node indices for each face.
    bottom_indices : ndarray, shape (n_node,)
        0-based bottom layer index for each node.

    Returns
    -------
    ndarray, shape (n_layer-1, n_face)
        Prism volumes per layer and face for one timestep.
    """
    return _calculate_prism_volumes(
        node_x=node_x,
        node_y=node_y,
        z=z,
        connectivity=connectivity,
        bottom_indices=bottom_indices,
    )


def integrate_prism_variable(sxda, return_per_layer=False):
    """
    Integrate a variable under an element over the water column for nodal data.

    Parameters
    ----------
    sxda : SxDataArray
        SxDataArray with z-coordinates and bottom indices containing the
        variable to integrate. Variable must be a concentration or density-like
        quantity (e.g., chlorophyll, salt, energy) that can be integrated over
        the water column to produce a total quantity per face element.
        This function enforces single-timestep behavior.
    return_per_layer : bool, optional
        If True, return per-layer integrated values with shape
        (n_layer-1, n_face). If False (default), return total integrated
        values per face with shape (n_face).

    Returns
    -------
    ndarray
        If ``return_per_layer=False``: shape (n_face), integrated variable
        per face element summed over all layers.
        If ``return_per_layer=True``: shape (n_layer-1, n_face), integrated
        variable per layer and face.
    """

    grid = sxda.sxgrid

    # Extract grid properties
    node_x = grid.node_x.values.astype(np.float64)
    node_y = grid.node_y.values.astype(np.float64)
    connectivity = grid.face_node_connectivity.values
    z = _coerce_single_timestep(grid.sgrid_info.zCoordinates.values, "z")
    bottom_indices = grid.sgrid_info.bottom_index_node.values - 1
    thickness = _calc_layers_thickness(z, bottom_indices)

    # Pull variable to integrate from sxda
    var_arr = _coerce_single_timestep(sxda.values, "variable")

    # Average variable to layer intervals: shape (n_node, n_layer-1)
    var_layer = 0.5 * (var_arr[:, :-1] + var_arr[:, 1:])
    values = (var_layer * thickness).T

    # Integrate!
    integrated = _integrate_nodal(
        node_x, node_y, values.astype(np.float64), connectivity
    )

    if return_per_layer:
        return integrated

    return integrated.sum(axis=0)


def calc_volumes_from_projected(sxda):
    """
    Calculate the volume of prism under a selection of nodes for each layer in
    the vertical direction.

    Parameters
    ----------
    sxda : SxDataArray
        SxDataArray with z-coordinates and bottom indices.
        This function enforces single-timestep behavior.

    Returns
    -------
    ndarray, shape (n_layer-1, n_face)
        Volume of each prism layer under the selection of nodes.

    Notes
    -----
    For external dask/xarray parallelization over time, use
    ``calc_volumes_from_projected_kernel`` with ``xarray.apply_ufunc``.
    """
    if not hasattr(sxda, "sxgrid"):
        raise TypeError(
            "calc_volumes_from_projected expects an SxDataArray (with .sxgrid)."
            "If you are using xarray.apply_ufunc(..., dask='parallelized'), "
            "use calc_volumes_from_projected_kernel on z-coordinate array "
            "blocks and pass grid arrays (node_x, node_y, connectivity, "
            "bottom_indices) as static arguments."
        )

    grid = sxda.sxgrid
    z = _coerce_single_timestep(grid.sgrid_info.zCoordinates.values, "z")

    return _calculate_prism_volumes(
        node_x=grid.node_x.values,
        node_y=grid.node_y.values,
        z=z,
        connectivity=grid.face_node_connectivity.values,
        bottom_indices=grid.sgrid_info.bottom_index_node.values - 1,
    )


def _calculate_volumes_uxarray(sxda, radius_m=None):
    """Calculate per-face prism volumes using uxarray spherical face areas.

    This path uses uxarray's unit-sphere face areas from lat/lon geometry,
    then scales by a physical sphere radius in meters.

    Parameters
    ----------
    sxda : SxDataArray
        SxDataArray with z-coordinates and bottom indices.
        This function enforces single-timestep behavior.
    radius_m : float, optional
        Physical sphere radius in meters used to scale uxarray face areas.
        If not provided, computes a local geocentric Earth radius from the
        mean node latitude.

    Returns
    -------
    ndarray, shape (n_layer-1, n_face)
        Volume of each prism layer under each face element.
    """
    grid = sxda.sxgrid

    z = _coerce_single_timestep(
        grid.sgrid_info.zCoordinates.values.astype(np.float64), "z"
    )
    connectivity = grid.face_node_connectivity.values

    # Thickness at nodes: shape (n_node, n_interface)
    bottom_indices = grid.sgrid_info.bottom_index_node.values - 1
    thickness = _calc_layers_thickness(z, bottom_indices)

    # Face-average nodal thickness: shape (n_interface, n_face)
    # NOTE: This is an arithmetic node mean. For strict parity with the planar
    # path on quads, we may need to mirror its bilinear quadrature weighting.
    thickness_face = _face_mean_from_nodal(thickness, connectivity)

    # Compute unit-sphere face areas from lon/lat directly to avoid mutating
    # the source grid's planar cartesian coordinates.
    from uxarray.grid.area import get_all_face_area_from_coords

    node_lon = np.asarray(grid.node_lon.values, dtype=np.float64)
    node_lat = np.asarray(grid.node_lat.values, dtype=np.float64)
    lon_rad = np.deg2rad(node_lon)
    lat_rad = np.deg2rad(node_lat)
    node_x_unit = np.cos(lat_rad) * np.cos(lon_rad)
    node_y_unit = np.cos(lat_rad) * np.sin(lon_rad)
    node_z_unit = np.sin(lat_rad)

    n_nodes_per_face = np.sum(connectivity >= 0, axis=1).astype(np.int64)
    face_areas_unit, _ = get_all_face_area_from_coords(
        node_x_unit,
        node_y_unit,
        node_z_unit,
        connectivity,
        n_nodes_per_face,
        quadrature_rule="triangular",
        order=4,
        latitude_adjusted_area=False,
    )

    # Areas are on a unit sphere, so scale to m^2.
    if radius_m is None:
        mean_lat = np.asarray(grid.node_lat.values, dtype=np.float64).mean()
        radius_m = _earth_radius_at_latitude(mean_lat)
    else:
        radius_m = float(radius_m)
    face_areas_m2 = np.asarray(face_areas_unit, dtype=np.float64) * radius_m**2

    return thickness_face * face_areas_m2[np.newaxis, :]


# TODO: Test impact of gsw method on performance.

def _calculate_mass_in_ocean(sxda_salinity, sxda_temperature=None):
    """
    Calculate the total water mass in the domain using salinity and temperature
    to compute in-situ density via the TEOS-10 equations (gsw).

    This function is better applied in a ocean domain and not in shallow/estuary
    domains.

    Parameters
    ----------
    sxda_salinity : SxDataArray
        Practical salinity (PSU), shape (n_node, n_layer)
        or (1, n_node, n_layer).
    sxda_temperature : SxDataArray, optional
        In-situ temperature (°C), shape (n_node, n_layer)
        or (1, n_node, n_layer).
        Defaults to 15.0 °C if not provided.

    Returns
    -------
    ndarray, shape (n_layer-1, n_face)
        Water mass per prism layer per face element (kg).
    """
    import gsw

    grid = sxda_salinity.sxgrid

    # Pressure from z-coordinates: shape (n_node, n_layer)
    z = _coerce_single_timestep(grid.sgrid_info.zCoordinates.values, "z")
    lat = grid.node_lat.values[:, np.newaxis]
    lon = grid.node_lon.values[:, np.newaxis]
    pressure = gsw.p_from_z(z, lat)  # shape (n_node, n_layer)

    # Convert salinity to Absolute Salinity (g/kg)
    SP = _coerce_single_timestep(sxda_salinity.values, "salinity")
    SA = gsw.SA_from_SP(SP, pressure, lon, lat)  # shape (n_node, n_layer)

    # Convert temperature to Conservative Temperature
    if sxda_temperature is not None:
        t = _coerce_single_timestep(sxda_temperature.values, "temperature")
    else:
        t = 15.0
    CT = gsw.CT_from_t(SA, t, pressure)  # shape (n_node, n_layer)

    # density: shape (n_node, n_layer)
    rho = gsw.density.rho(SA, CT, pressure)

    # Average density to layer intervals: shape (n_node, n_layer-1)
    rho_layer = 0.5 * (rho[:, :-1] + rho[:, 1:])

    # Thickness per layer interval: shape (n_node, n_layer-1)
    bottom_indices = grid.sgrid_info.bottom_index_node.values - 1

    thickness = _calc_layers_thickness(z, bottom_indices)

    # rho * h at each node per layer: shape (n_node, n_layer-1)
    rho_h = rho_layer * thickness
    connectivity = grid.face_node_connectivity.values
    node_x = grid.node_x.values.astype(np.float64)
    node_y = grid.node_y.values.astype(np.float64)

    values = rho_h.T  # (n_layer-1, n_node)

    integrated = _integrate_nodal(
        node_x, node_y, values.astype(np.float64), connectivity
    )

    return integrated