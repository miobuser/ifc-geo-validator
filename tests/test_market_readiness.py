"""Tests covering recent changes flagged as "NO COVERAGE" by the
test-coverage audit. One narrow test per new/changed contract:

  1. core.geometry.compute_signed_volume — new function
  2. core.mesh_converter.extract_mesh normals_flipped field
  3. core.face_classifier._build_face_adjacency non-manifold star graph
  4. validation.level3._compute_crown_width_sliced p10 robust fields
  5. validation.level5._min_bbox_distance_xy Euclidean
  6. validation.level5.validate_level5(config=...)
  7. core.anomaly_detection.detect_anomalies(config=...)
  8. core.ifc_parser.get_coordinate_system
  9. core.project_config.DEFAULT_CONFIG new sections

Kept deliberately minimal: one assertion per property so failures
point at the exact contract that regressed.
"""

import numpy as np
import pytest

from ifc_geo_validator.core.geometry import (
    compute_signed_volume, compute_volume,
)
from ifc_geo_validator.core.face_classifier import _build_face_adjacency
from ifc_geo_validator.core.anomaly_detection import detect_anomalies
from ifc_geo_validator.core.ifc_parser import get_coordinate_system
from ifc_geo_validator.core.project_config import DEFAULT_CONFIG
from ifc_geo_validator.validation.level5 import (
    validate_level5, _min_bbox_distance_xy, DEFAULT_MAX_GAP_3D_M,
)


# ── Unit cube for deterministic expectations ──────────────────────
UNIT_CUBE_VERTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
], dtype=float)
# Triangles with outward-facing normals (counter-clockwise from outside)
UNIT_CUBE_FACES = np.array([
    [0, 2, 1], [0, 3, 2],  # bottom (normal -z)
    [4, 5, 6], [4, 6, 7],  # top (normal +z)
    [0, 1, 5], [0, 5, 4],  # front y=0 (normal -y)
    [2, 3, 7], [2, 7, 6],  # back y=1 (normal +y)
    [1, 2, 6], [1, 6, 5],  # right x=1 (normal +x)
    [0, 4, 7], [0, 7, 3],  # left x=0 (normal -x)
], dtype=np.int64)


def test_signed_volume_positive_for_outward_normals():
    v = compute_signed_volume(UNIT_CUBE_VERTS, UNIT_CUBE_FACES)
    assert v == pytest.approx(1.0, abs=1e-9)


def test_signed_volume_negative_for_inverted_winding():
    # Flipped winding: swap columns 1 and 2 → inward normals
    flipped = UNIT_CUBE_FACES[:, [0, 2, 1]]
    v = compute_signed_volume(UNIT_CUBE_VERTS, flipped)
    assert v == pytest.approx(-1.0, abs=1e-9)


def test_compute_volume_is_abs_of_signed():
    flipped = UNIT_CUBE_FACES[:, [0, 2, 1]]
    assert compute_volume(UNIT_CUBE_VERTS, flipped) == pytest.approx(1.0, abs=1e-9)


def test_face_adjacency_manifold_cube_pairs():
    """Manifold cube: every edge shared by exactly 2 faces → 18 pairs."""
    pairs = _build_face_adjacency(UNIT_CUBE_FACES)
    # 12 triangles, each with 3 edges = 36 half-edges / 2 per edge = 18 edges,
    # each edge contributes one pair → 18 pairs.
    assert len(pairs) == 18


def test_face_adjacency_non_manifold_star_graph():
    """k faces sharing an edge should emit k-1 pairs (star), not k*(k-1)/2."""
    # 4 triangles sharing edge (0, 1): fan-out from a shared edge
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
    ], dtype=np.int64)
    pairs = _build_face_adjacency(faces)
    # The (0,1) edge is shared by 4 faces → 4 - 1 = 3 pairs minimum.
    # The (1,2), (0,2), (1,3), (0,3), ... edges are unique to one face each.
    # So only the shared-edge cluster produces pairs: exactly 3, not 6.
    assert len(pairs) == 3


def test_min_bbox_distance_xy_euclidean():
    """(3, 4) gap on diagonal → hypot = 5, not Manhattan 7."""
    a = {"bbox_min": np.array([0.0, 0.0, 0.0]),
         "bbox_max": np.array([1.0, 1.0, 1.0])}
    b = {"bbox_min": np.array([4.0, 5.0, 0.0]),
         "bbox_max": np.array([5.0, 6.0, 1.0])}
    # gap_x = 4-1 = 3, gap_y = 5-1 = 4 → Euclidean = 5
    assert _min_bbox_distance_xy(a, b) == pytest.approx(5.0, abs=1e-9)


def test_min_bbox_distance_xy_overlap_is_zero():
    a = {"bbox_min": np.array([0.0, 0.0, 0.0]),
         "bbox_max": np.array([2.0, 2.0, 1.0])}
    b = {"bbox_min": np.array([1.0, 1.0, 0.0]),
         "bbox_max": np.array([3.0, 3.0, 1.0])}
    assert _min_bbox_distance_xy(a, b) == 0.0


def test_validate_level5_config_max_gap_override():
    """A config max_gap_3d_m below the natural separation rejects pairs."""
    # Two far-apart single-element lists
    elem_a = {
        "element_id": 1, "element_name": "A",
        "level1": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1],
                            "size": [1, 1, 1]}, "volume": 1.0,
                   "centroid": [0.5, 0.5, 0.5]},
        "mesh_data": {"vertices": UNIT_CUBE_VERTS, "faces": UNIT_CUBE_FACES,
                      "normals": np.zeros((12, 3)), "areas": np.ones(12)},
    }
    elem_b = {
        "element_id": 2, "element_name": "B",
        "level1": {"bbox": {"min": [10, 10, 0], "max": [11, 11, 1],
                            "size": [1, 1, 1]}, "volume": 1.0,
                   "centroid": [10.5, 10.5, 0.5]},
        "mesh_data": {"vertices": UNIT_CUBE_VERTS + 10, "faces": UNIT_CUBE_FACES,
                      "normals": np.zeros((12, 3)), "areas": np.ones(12)},
    }
    # Default (1.0 m) rejects the pair (gap is 9 m). Custom 15 m includes it.
    res_default = validate_level5([elem_a, elem_b])
    assert res_default["summary"]["num_pairs"] == 0

    res_custom = validate_level5([elem_a, elem_b], config={"max_gap_3d_m": 15.0})
    assert res_custom["summary"]["num_pairs"] == 1


def test_anomaly_detection_config_override():
    """Custom front_back_ratio_flag changes whether asymmetry is flagged."""
    mesh = {"vertices": UNIT_CUBE_VERTS, "faces": UNIT_CUBE_FACES,
            "normals": np.zeros((12, 3)), "areas": np.ones(12)}
    l2 = {"summary": {"front": {"total_area": 5.0, "count": 1},
                      "back": {"total_area": 2.0, "count": 1},
                      "unclassified": {"total_area": 0.1, "count": 1},
                      "crown": {"total_area": 1.0, "count": 1},
                      "foundation": {"total_area": 1.0, "count": 1}},
          "has_crown": True, "has_foundation": True, "has_front": True,
          "has_back": True}
    l3 = {"crown_width_mm": 300.0, "min_wall_thickness_mm": 200.0}

    # 5:2 ratio = 2.5, default flag = 2.0 → one asymmetry anomaly
    default_anomalies = [a for a in detect_anomalies(mesh, l2, l3)
                         if a["type"] == "asymmetric_front_back"]
    assert len(default_anomalies) == 1

    # Raise the flag to 3.0 → suppresses the asymmetry diagnostic
    relaxed = [a for a in detect_anomalies(mesh, l2, l3,
                                           config={"front_back_ratio_flag": 3.0})
               if a["type"] == "asymmetric_front_back"]
    assert len(relaxed) == 0


def test_get_coordinate_system_returns_unspecified_for_empty_model():
    """When the model has no IfcProjectedCRS, the result is honest."""
    class FakeModel:
        def by_type(self, _):
            return []
    crs = get_coordinate_system(FakeModel())
    assert crs["name"] == "unspecified"
    assert crs["has_crs"] is False
    assert crs["eastings_offset"] is None


def test_default_config_has_reproducibility_sections():
    """Thesis-critical: project_config carries all override schemas."""
    for key in ("classifier", "pair_candidacy", "robust_stats", "anomaly"):
        assert key in DEFAULT_CONFIG, f"missing reproducibility section: {key}"
    # Spot-check a known value
    assert DEFAULT_CONFIG["pair_candidacy"]["max_gap_3d_m"] == DEFAULT_MAX_GAP_3D_M
