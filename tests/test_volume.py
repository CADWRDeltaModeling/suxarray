import numpy as np
import pytest

from suxarray.grid.integrate import _calculate_prism_volumes


def test_calculate_prism_volume_synthetic_shaved_cells():
    """Validate prism volumes on a tiny synthetic triangular mesh.

    The mesh has 3 triangular faces of equal area (2 m^2 each), and node-wise
    shaved-cell bottom indices that mask lower interfaces at selected nodes.
    """
    node_x = np.array([0.0, 2.0, 0.0, -2.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 2.0, 0.0, -2.0], dtype=np.float64)

    # 3 triangles with -1 fill values in the 4th slot.
    connectivity = np.array(
        [
            [0, 1, 2, -1],
            [0, 2, 3, -1],
            [0, 3, 4, -1],
        ],
        dtype=np.int64,
    )
    # 0-based node bottom indices. k > 0 masks interfaces [:k] at that node.
    bottom_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)

    # z shape: (n_node=5, n_layer=4) -> 3 interfaces.
    z = np.array(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    volumes = _calculate_prism_volumes(
        node_x=node_x,
        node_y=node_y,
        z=z,
        connectivity=connectivity,
        bottom_indices=bottom_indices,
    )

    # Expected shape: (n_interface, n_face)
    assert volumes.shape == (3, 3)

    # Make the geometric factor explicit for readers:
    # expected volume = triangle_area * face-mean(thickness after masking)
    triangle_area_m2 = 2.0
    expected_face_mean_thickness = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected = triangle_area_m2 * expected_face_mean_thickness

    np.testing.assert_allclose(volumes, expected, rtol=0.0, atol=1e-12)

    # Spot-check a couple of entries with pytest.approx for readability.
    assert volumes[0, 0] == pytest.approx(2.0 / 3.0)
    assert volumes[2, 2] == pytest.approx(2.0)