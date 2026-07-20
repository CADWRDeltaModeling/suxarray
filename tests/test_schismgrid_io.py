"""Tests for suxarray.io._schismgrid face/edge z-coordinate calculations.

Uses real SCHISM test data (out2d_1.nc / zCoordinates_1.nc).

Focuses on:
- _calculate_face_z: correct averaging for triangular (3-node) and
  quadrilateral (4-node) cells, with fill-value masking for tri cells.
- _calculate_edge_z: correct averaging over the two endpoint nodes.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from suxarray.io._schismgrid import (
    _calculate_edge_z,
    _calculate_face_z,
    _rename_coords,
)
from uxarray.io._ugrid import _read_ugrid

TESTDATA = Path(__file__).parent / "testdata"

# Expected face/edge counts for out2d_1.nc
N_NODES = 823
N_FACES = 791
N_EDGES = 1613
N_TRI = 87
N_QUAD = 704
N_LAYERS = 23
N_TIME = 48

# First tri face in the test mesh: index 64, nodes 63/80/79 (0-based)
TRI_IDX = 64
TRI_NODES = [63, 80, 79]

# First quad face in the test mesh: index 0, nodes 0/1/6/3 (0-based)
QUAD_IDX = 0
QUAD_NODES = [0, 1, 6, 3]

# First edge: index 0, nodes 0/1 (0-based)
EDGE_IDX = 0
EDGE_NODES = [0, 1]


@pytest.fixture(scope="module")
def calc_inputs():
    """Prepare processed ds_out2d and ds_zcoords ready for the calculation
    functions, mirroring the pre-processing done inside _read_schism_grid."""
    ds_out2d_raw = xr.open_dataset(str(TESTDATA / "out2d_1.nc"))
    ds_zcoords_raw = xr.open_dataset(str(TESTDATA / "zCoordinates_1.nc"))

    # Process out2d the same way _read_schism_grid does
    ds_out2d = ds_out2d_raw.drop_dims("time")
    ds_out2d, _ = _read_ugrid(ds_out2d)
    ds_out2d = _rename_coords(ds_out2d)

    # Rename zcoords dims the same way _read_schism_grid does
    ds_zcoords = ds_zcoords_raw.swap_dims(
        {"nSCHISM_hgrid_node": "n_node", "nSCHISM_vgrid_layers": "n_layer"}
    )

    return ds_out2d, ds_zcoords, ds_zcoords_raw


# Connectivity – sanity checks on the test mesh

def test_mesh_tri_and_quad_counts(calc_inputs):
    """Test mesh has the expected 87 tri and 704 quad cells."""
    ds_out2d, _, _ = calc_inputs
    fnc = ds_out2d.face_node_connectivity.values
    fill = ds_out2d.face_node_connectivity._FillValue
    assert int(np.sum(fnc[:, -1] == fill)) == N_TRI
    assert int(np.sum(fnc[:, -1] != fill)) == N_QUAD
    assert ds_out2d.face_node_connectivity.sizes["n_face"] == N_FACES


# _calculate_face_z

def test_calculate_face_z_output_shape(calc_inputs):
    """_calculate_face_z returns (n_face, time, n_layer)."""
    ds_out2d, ds_zcoords, _ = calc_inputs
    face_z = _calculate_face_z(ds_zcoords, ds_out2d)
    assert face_z.dims == ("n_face", "time", "n_layer")
    assert face_z.sizes == {"n_face": N_FACES,
                            "time": N_TIME,
                            "n_layer": N_LAYERS}


def test_calculate_face_z_tri_cell(calc_inputs):
    """face_z for a tri cell equals the mean of exactly its 3 node z-values."""
    ds_out2d, ds_zcoords, ds_zcoords_raw = calc_inputs
    face_z = _calculate_face_z(ds_zcoords, ds_out2d)
    z_raw = ds_zcoords_raw.zCoordinates.values  # (time, n_node, n_layer)
    expected = float(z_raw[0, TRI_NODES, -1].mean())
    assert float(face_z.values[TRI_IDX, 0, -1]) == pytest.approx(expected,
                                                                 rel=1e-5)


def test_calculate_face_z_tri_ignores_fill_node(calc_inputs):
    """Guard that the fill node is NOT included in the tri-cell average.

    If node index -1 (wrapping to the last node) were naively included,
    the result would match a 4-node mean instead of the correct 3-node mean.
    """
    ds_out2d, ds_zcoords, ds_zcoords_raw = calc_inputs
    face_z = _calculate_face_z(ds_zcoords, ds_out2d)
    z_raw = ds_zcoords_raw.zCoordinates.values
    # Wrong result if fill wraps to node[-1] = node[N_NODES - 1]
    wrong_mean = float(z_raw[0, TRI_NODES + [N_NODES - 1], -1].mean())
    assert float(face_z.values[TRI_IDX, 0, -1]) != pytest.approx(wrong_mean,
                                                                rel=1e-5)


def test_calculate_face_z_quad_cell(calc_inputs):
    """face_z for a quad cell equals the mean of all 4 node z-values."""
    ds_out2d, ds_zcoords, ds_zcoords_raw = calc_inputs
    face_z = _calculate_face_z(ds_zcoords, ds_out2d)
    z_raw = ds_zcoords_raw.zCoordinates.values
    expected = float(z_raw[0, QUAD_NODES, -1].mean())
    assert float(face_z.values[QUAD_IDX, 0, -1]) == pytest.approx(expected,
                                                                  rel=1e-5)


def test_calculate_face_z_tri_and_quad_differ(calc_inputs):
    """
    Tri and quad cells produce distinct z-values,
    confirming correct node counts.
    """
    ds_out2d, ds_zcoords, _ = calc_inputs
    face_z = _calculate_face_z(ds_zcoords, ds_out2d)
    assert float(face_z.values[TRI_IDX, 0, -1]) != pytest.approx(
        float(face_z.values[QUAD_IDX, 0, -1]), rel=1e-5
    )


# _calculate_edge_z

def test_calculate_edge_z_output_shape(calc_inputs):
    """_calculate_edge_z returns (n_edge, time, n_layer)."""
    ds_out2d, ds_zcoords, _ = calc_inputs
    edge_z = _calculate_edge_z(ds_zcoords, ds_out2d)
    assert edge_z.dims == ("n_edge", "time", "n_layer")
    assert edge_z.sizes == {"n_edge": N_EDGES,
                            "time": N_TIME,
                            "n_layer": N_LAYERS}


def test_calculate_edge_z_values(calc_inputs):
    """edge_z equals the two-node mean at every time step."""
    ds_out2d, ds_zcoords, ds_zcoords_raw = calc_inputs
    edge_z = _calculate_edge_z(ds_zcoords, ds_out2d)
    z_raw = ds_zcoords_raw.zCoordinates.values  # (time, n_node, n_layer)
    expected = z_raw[:, EDGE_NODES, -1].mean(axis=1)  # (time,)
    np.testing.assert_allclose(edge_z.values[EDGE_IDX, :, -1],
                               expected, rtol=1e-5)
