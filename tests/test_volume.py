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


def test_calculate_prism_volume_synthetic_float32():
    """Validate that prism volume calculation preserves float32 precision on a mixed tri-quad mesh.

    Mesh consists of:
    - Face 0: Triangle with vertices (0,0), (2,0), (0,2). Area = 2 m^2.
    - Face 1: Quadrilateral with vertices (2,0), (4,0), (4,2), (2,2). Area = 4 m^2.
    """
    node_x = np.array([0.0, 2.0, 0.0, 4.0, 4.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 2.0, 0.0, 2.0, 2.0], dtype=np.float64)

    connectivity = np.array(
        [
            [0, 1, 2, -1],  # Triangle
            [1, 3, 4, 5],   # Quadrilateral
        ],
        dtype=np.int64,
    )
    # Shaved cells: node 0 and node 3 have bottom interface masked
    bottom_indices = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)

    # 4 layers -> 3 interfaces, all in float32
    z = np.tile(
        np.array([-3.0, -2.0, -1.0, 0.0], dtype=np.float32),
        (6, 1),
    )

    volumes = _calculate_prism_volumes(
        node_x=node_x,
        node_y=node_y,
        z=z,
        connectivity=connectivity,
        bottom_indices=bottom_indices,
    )

    assert volumes.dtype == np.float32
    assert volumes.shape == (3, 2)

    # Expected volumes per layer and face:
    # Layer 0:
    #   Tri [0,1,2]: thickness at nodes = [0, 1, 1], mean = 2/3 -> vol = 2.0 * (2/3) = 4/3
    #   Quad [1,3,4,5]: thickness at nodes = [1, 0, 1, 1], mean = 3/4 -> vol = 4.0 * 0.75 = 3.0
    # Layers 1 and 2:
    #   Tri: thickness = 1.0 -> vol = 2.0 * 1.0 = 2.0
    #   Quad: thickness = 1.0 -> vol = 4.0 * 1.0 = 4.0
    expected = np.array(
        [
            [4.0 / 3.0, 3.0],
            [2.0, 4.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(volumes, expected, rtol=1e-5, atol=1e-6)
