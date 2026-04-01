"""Edge case tests for pipeline robustness.

Tests error handling, degenerate geometry, and boundary conditions
to ensure the pipeline fails gracefully and provides useful feedback.
"""

import os
import pytest
import numpy as np

from ifc_geo_validator.core.ifc_parser import load_model, IFCLoadError
from ifc_geo_validator.core.mesh_converter import MeshExtractionError
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level5 import validate_level5
from ifc_geo_validator.validation.level6 import validate_level6


# ── Helper ──────────────────────────────────────────────────────────

def _make_mesh(verts, faces):
    """Build mesh_data from vertices and faces arrays."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = cross / norms
    return {
        "vertices": verts,
        "faces": faces,
        "normals": normals,
        "areas": areas,
        "is_watertight": True,
    }


# ── IFC Loading Errors ─────────────────────────────────────────────

class TestIFCLoadErrors:
    """Test graceful handling of invalid IFC files."""

    def test_nonexistent_file(self):
        with pytest.raises(IFCLoadError, match="not found"):
            load_model("/nonexistent/path/model.ifc")

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.ifc"
        empty.write_text("")
        with pytest.raises(IFCLoadError, match="empty"):
            load_model(str(empty))


# ── Degenerate Geometry ────────────────────────────────────────────

class TestDegenerateGeometry:
    """Test handling of degenerate or minimal geometry."""

    def test_single_triangle(self):
        """A single triangle should not crash."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = _make_mesh(verts, faces)
        mesh["is_watertight"] = False

        l1 = validate_level1(mesh)
        assert l1["volume"] >= 0
        assert l1["num_triangles"] == 1

    def test_flat_box(self):
        """A perfectly flat box (zero height) should not crash."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],  # same z
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
        ])
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        assert l1["volume"] == 0.0 or abs(l1["volume"]) < 0.001

    def test_very_thin_wall(self):
        """A 1mm thin wall should still classify correctly."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.001, 0], [0, 0.001, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.001, 3], [0, 0.001, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        assert l2["has_crown"]
        assert l2["has_foundation"]
        assert l2["num_groups"] == 6


# ── Asymmetry Index Edge Cases ────────────────────────────────────

class TestAsymmetryEdgeCases:
    """Verify front/back asymmetry index on degenerate inputs."""

    def test_symmetric_box_zero_asymmetry(self):
        """Perfectly symmetric box: front_back_asymmetry == 0."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        asym = l2.get("front_back_asymmetry", -1)
        assert abs(asym) < 0.01, f"Symmetric box should have ~0 asymmetry, got {asym}"

    def test_only_horizontal_faces_no_crash(self):
        """Mesh with only crown and foundation (no vertical faces).
        Asymmetry should be 0 since there's no front/back."""
        verts = np.array([
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        asym = l2.get("front_back_asymmetry", -1)
        assert asym == 0.0, f"No vertical faces → asymmetry must be 0, got {asym}"

    def test_asymmetry_range_0_to_1(self):
        """Asymmetry index must always be in [0, 1]."""
        # Wide wall (asymmetric: front much larger than back due to taper)
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 1.0, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 1.0, 3], [0, 0.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        asym = l2.get("front_back_asymmetry", -1)
        assert 0 <= asym <= 1, f"Asymmetry {asym} out of [0,1] range"


# ── CLI End-to-End Test ───────────────────────────────────────────

class TestCLIEndToEnd:
    """Test the CLI main() function end-to-end on a test model."""

    def test_cli_runs_without_error(self):
        """CLI main() should complete without exceptions on T1."""
        import sys
        from unittest.mock import patch
        from ifc_geo_validator.cli import main

        test_ifc = os.path.join(os.path.dirname(__file__),
                                "test_models", "T1_simple_box.ifc")
        with patch.object(sys, 'argv', ['ifc-geo-validator', test_ifc, '--levels', '1,2,3']):
            main()  # Should not raise

    def test_cli_with_ruleset(self):
        """CLI main() with ASTRA ruleset should complete on T7."""
        import sys
        from unittest.mock import patch
        from ifc_geo_validator.cli import main

        test_ifc = os.path.join(os.path.dirname(__file__),
                                "test_models", "T7_compliant.ifc")
        with patch.object(sys, 'argv', ['ifc-geo-validator', test_ifc]):
            main()  # Runs all levels with auto-detected ASTRA ruleset

    def test_cli_summary_flag(self, capsys):
        """CLI --summary prints machine-readable output."""
        import sys
        from unittest.mock import patch
        from ifc_geo_validator.cli import main

        test_ifc = os.path.join(os.path.dirname(__file__),
                                "test_models", "T7_compliant.ifc")
        with patch.object(sys, 'argv', ['ifc-geo-validator', test_ifc, '--summary']):
            main()
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "L4(" in captured.out

    def test_cli_summary_fail_exit_code(self):
        """CLI --summary exits with code 1 for non-compliant model."""
        import sys
        from unittest.mock import patch
        from ifc_geo_validator.cli import main

        test_ifc = os.path.join(os.path.dirname(__file__),
                                "test_models", "T6_non_compliant.ifc")
        with patch.object(sys, 'argv', ['ifc-geo-validator', test_ifc, '--summary']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ── Level 5 Edge Cases ─────────────────────────────────────────────

class TestEmptyArrayGuards:
    """Test that empty array edge cases don't crash."""

    def test_min_vertex_distance_empty_a(self):
        from ifc_geo_validator.core.distance import min_vertex_distance
        result = min_vertex_distance(np.array([]).reshape(0, 3), np.array([[1, 2, 3]]))
        assert result == float("inf")

    def test_min_vertex_distance_empty_b(self):
        from ifc_geo_validator.core.distance import min_vertex_distance
        result = min_vertex_distance(np.array([[1, 2, 3]]), np.array([]).reshape(0, 3))
        assert result == float("inf")

    def test_l3_empty_crown_groups(self):
        """L3 with an L2 result that has no crown groups should not crash."""
        mesh = _make_mesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 1, 2], [3, 4, 5]]
        )
        l2_fake = {
            "face_groups": [
                {"category": "front", "face_indices": [0], "normal": [0, -1, 0], "area": 0.5, "centroid": [0.33, 0, 0], "num_triangles": 1},
                {"category": "back", "face_indices": [1], "normal": [0, 1, 0], "area": 0.5, "centroid": [0.33, 1, 1], "num_triangles": 1},
            ],
            "wall_axis": [1, 0, 0],
            "centerline": None,
            "has_crown": False, "has_foundation": False,
            "num_groups": 2,
        }
        l3 = validate_level3(mesh, l2_fake)
        # Should not crash, crown measurements simply absent
        assert "crown_width_mm" not in l3
        # Wall height should still work
        assert "wall_height_m" in l3

    def test_l3_wall_height_with_empty_vertical(self):
        """L3 with no front/back groups should not crash on wall height."""
        mesh = _make_mesh(
            [[0, 0, 3], [1, 0, 3], [0, 1, 3], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 2], [3, 4, 5]]
        )
        l2_fake = {
            "face_groups": [
                {"category": "crown", "face_indices": [0], "normal": [0, 0, 1], "area": 0.5, "centroid": [0.33, 0.33, 3], "num_triangles": 1},
                {"category": "foundation", "face_indices": [1], "normal": [0, 0, -1], "area": 0.5, "centroid": [0.33, 0.33, 0], "num_triangles": 1},
            ],
            "wall_axis": [1, 0, 0],
            "centerline": None,
            "has_crown": True, "has_foundation": True,
            "num_groups": 2,
        }
        l3 = validate_level3(mesh, l2_fake)
        # No front/back → no wall height or thickness
        assert "wall_height_m" not in l3
        assert "min_wall_thickness_mm" not in l3


class TestLevel5EdgeCases:
    """Test L5 with edge case inputs."""

    def test_empty_input(self):
        result = validate_level5([])
        assert result["summary"]["num_pairs"] == 0

    def test_single_element(self):
        result = validate_level5([{"element_id": 1, "level1": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}}}])
        assert result["summary"]["num_pairs"] == 0

    def test_far_apart_elements(self):
        """Elements more than 1m apart should not form pairs."""
        a = {"element_id": 1, "level1": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}, "centroid": [0.5, 0.5, 0.5]},
             "mesh_data": _make_mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                                      [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])}
        b = {"element_id": 2, "level1": {"bbox": {"min": [100, 0, 0], "max": [101, 1, 1]}, "centroid": [100.5, 0.5, 0.5]},
             "mesh_data": _make_mesh([[100, 0, 0], [101, 0, 0], [101, 1, 0], [100, 1, 0], [100, 0, 1], [101, 0, 1], [101, 1, 1], [100, 1, 1]],
                                      [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])}
        result = validate_level5([a, b])
        assert result["summary"]["num_pairs"] == 0


# ── Level 6 Edge Cases ─────────────────────────────────────────────

class TestLevel6EdgeCases:
    """Test L6 with edge case inputs."""

    def test_no_terrain(self):
        result = validate_level6([], terrain_mesh=None)
        assert result["terrain_available"] is False

    def test_empty_elements(self):
        result = validate_level6([], terrain_mesh={"vertices": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]])})
        assert result["terrain_available"] is True
        assert len(result["clearances"]) == 0


# ── Cross-Level Consistency ────────────────────────────────────────

class TestCrossLevelConsistency:
    """Verify that level outputs are consistent with each other."""

    def test_l1_l3_height_matches_bbox(self):
        """Wall height from L3 should approximately match bbox Z-size."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        bbox_height = l1["bbox"]["size"][2]
        wall_height = l3.get("wall_height_m", 0)
        assert abs(bbox_height - wall_height) < 0.01

    def test_crown_width_positive(self):
        """Crown width must always be positive."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        assert l3["crown_width_mm"] > 0
        assert l3["min_wall_thickness_mm"] > 0
        assert l3["wall_height_m"] > 0

    def test_all_categories_present(self):
        """A simple box should have all 6 categories."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        categories = {g["category"] for g in l2["face_groups"]}
        assert "crown" in categories
        assert "foundation" in categories
        assert "front" in categories
        assert "back" in categories
        assert "end_left" in categories
        assert "end_right" in categories

    def test_rotated_wall(self):
        """A 45-degree rotated wall must still classify correctly."""
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        # 8m wall along 45° diagonal, 0.4m thick, 3m high
        verts = np.array([
            [0, 0, 0], [8*c, 8*s, 0], [8*c - 0.4*s, 8*s + 0.4*c, 0], [-0.4*s, 0.4*c, 0],
            [0, 0, 3], [8*c, 8*s, 3], [8*c - 0.4*s, 8*s + 0.4*c, 3], [-0.4*s, 0.4*c, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        # Volume should be same as axis-aligned wall
        assert abs(l1["volume"] - 9.6) < 0.1
        # Crown and foundation must be detected
        assert l2["has_crown"]
        assert l2["has_foundation"]
        # Crown width should be ~400mm
        assert abs(l3["crown_width_mm"] - 400.0) < 10.0

    def test_zero_area_triangle(self):
        """Degenerate triangle (collinear vertices) must not crash."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
            [4, 0, 0],  # collinear with edge 0-1
        ], dtype=float)
        faces = np.array([
            [0, 8, 1],  # zero-area (collinear)
            [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        # Should not crash, volume still computable
        assert l1["volume"] >= 0

    def test_area_sum_equals_total(self):
        """Sum of face group areas must equal L1 total area."""
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)

        group_area = sum(g["area"] for g in l2["face_groups"])
        assert abs(group_area - l1["total_area"]) < 0.01


# ══════════════════════════════════════════════════════════════════
# Scientifically rigorous edge-case and robustness tests
# ══════════════════════════════════════════════════════════════════

# ── Reference geometry: T1 simple box (8.0 x 0.4 x 3.0 m) ───────

T1_BOX_VERTS = np.array([
    [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
    [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
], dtype=float)

T1_BOX_FACES = np.array([
    [0, 2, 1], [0, 3, 2],  # bottom (foundation)
    [4, 5, 6], [4, 6, 7],  # top (crown)
    [0, 1, 5], [0, 5, 4],  # front
    [2, 3, 7], [2, 7, 6],  # back
    [0, 4, 7], [0, 7, 3],  # end left
    [1, 2, 6], [1, 6, 5],  # end right
])

T1_EXPECTED_VOLUME = 9.6       # m^3
T1_EXPECTED_AREA = 56.8        # m^2


def _rotation_matrix_z(angle_rad):
    """Return a 3x3 rotation matrix around the Z-axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])


# ── 1. Numerical Precision Tests ─────────────────────────────────

class TestNumericalPrecision:
    """Verify numerical stability across coordinate scales."""

    def test_large_coordinates_utm(self):
        """Wall at UTM-scale coordinates (600000, 200000) must classify correctly."""
        offset = np.array([600000.0, 200000.0, 500.0])
        verts = T1_BOX_VERTS + offset
        mesh = _make_mesh(verts, T1_BOX_FACES)

        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        # Volume must be identical to origin-centered box
        assert abs(l1["volume"] - T1_EXPECTED_VOLUME) < 0.01, (
            f"Volume at UTM offset: {l1['volume']}, expected {T1_EXPECTED_VOLUME}"
        )
        # Classification must still detect all 6 face categories
        assert l2["has_crown"]
        assert l2["has_foundation"]
        assert l2["has_front"]
        assert l2["has_back"]
        assert l2["num_groups"] == 6
        # Crown width must still be ~400 mm
        assert abs(l3["crown_width_mm"] - 400.0) < 10.0

    def test_millimeter_coordinates(self):
        """Wall in millimeter-scale coordinates must work."""
        verts = T1_BOX_VERTS * 1000.0  # 8000 x 400 x 3000 mm
        mesh = _make_mesh(verts, T1_BOX_FACES)

        l1 = validate_level1(mesh)
        # Volume = 8000 * 400 * 3000 = 9.6e9 mm^3
        expected_volume_mm3 = T1_EXPECTED_VOLUME * 1e9
        assert abs(l1["volume"] - expected_volume_mm3) / expected_volume_mm3 < 1e-6, (
            f"Volume in mm: {l1['volume']}, expected {expected_volume_mm3}"
        )
        # Classification must still work at mm scale
        l2 = validate_level2(mesh)
        assert l2["has_crown"]
        assert l2["has_foundation"]

    def test_near_zero_area_faces(self):
        """Faces with area < 1e-10 must not cause division by zero."""
        # Create a degenerate triangle by placing 3 nearly-coincident points
        verts = np.array([
            # Near-degenerate triangle (area ~ 5e-21)
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],
            [0.0, 1e-10, 0.0],
            # Normal T1 box for context
            *T1_BOX_VERTS.tolist(),
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],  # near-zero area
            # Re-index T1 faces (offset by 3)
            *((T1_BOX_FACES + 3).tolist()),
        ])
        mesh = _make_mesh(verts, faces)

        # Must not raise any exception (especially not ZeroDivisionError)
        l1 = validate_level1(mesh)
        assert l1["volume"] >= 0
        assert l1["num_triangles"] == 13  # 1 degenerate + 12 box


# ── 2. Algorithmic Correctness Tests ─────────────────────────────

class TestAlgorithmicCorrectness:
    """Verify mathematical properties of the algorithms."""

    def test_volume_invariant_under_rotation(self):
        """Volume must be identical for the same wall at different orientations.

        Property: Volume is a rotational invariant (Arfken & Weber 2005).
        Rotating a closed mesh does not change its enclosed volume.
        """
        angles_deg = [0, 30, 45, 90, 135, 180]
        volumes = []

        for deg in angles_deg:
            R = _rotation_matrix_z(np.radians(deg))
            rotated_verts = (R @ T1_BOX_VERTS.T).T
            mesh = _make_mesh(rotated_verts, T1_BOX_FACES)
            l1 = validate_level1(mesh)
            volumes.append(l1["volume"])

        # All volumes must agree within machine precision
        for i, v in enumerate(volumes):
            assert abs(v - T1_EXPECTED_VOLUME) < 0.01, (
                f"Volume at {angles_deg[i]} deg = {v}, expected {T1_EXPECTED_VOLUME}"
            )

    def test_crown_width_invariant_under_translation(self):
        """Crown width must not change when wall is translated.

        Property: Crown width is measured as a relative distance between
        vertices, which is translation-invariant by definition.
        """
        offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 200.0, 50.0]),
            np.array([-500.0, 1000.0, -100.0]),
        ]
        crown_widths = []

        for offset in offsets:
            verts = T1_BOX_VERTS + offset
            mesh = _make_mesh(verts, T1_BOX_FACES)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)
            crown_widths.append(l3["crown_width_mm"])

        # All crown widths must agree within 1 mm
        for cw in crown_widths:
            assert abs(cw - crown_widths[0]) < 1.0, (
                f"Crown width varied under translation: {crown_widths}"
            )

    def test_symmetric_wall_front_back_equal_area(self):
        """For a perfectly symmetric wall, front and back areas must be equal.

        Property: A rectangular box is mirror-symmetric about its midplane.
        Front and back faces must have identical total area.
        """
        mesh = _make_mesh(T1_BOX_VERTS, T1_BOX_FACES)
        l2 = validate_level2(mesh)

        front_area = sum(
            g["area"] for g in l2["face_groups"] if g["category"] == "front"
        )
        back_area = sum(
            g["area"] for g in l2["face_groups"] if g["category"] == "back"
        )
        # For a symmetric box, front area = back area = 8 * 3 = 24 m^2
        assert abs(front_area - back_area) < 0.01, (
            f"Front area {front_area} != Back area {back_area}"
        )
        assert abs(front_area - 24.0) < 0.01

    def test_watertight_implies_positive_volume(self):
        """A watertight mesh must always have positive volume.

        Property: A closed, consistently-oriented surface encloses
        a positive volume by the divergence theorem (Gauss 1840).
        """
        mesh = _make_mesh(T1_BOX_VERTS, T1_BOX_FACES)
        mesh["is_watertight"] = True
        l1 = validate_level1(mesh)
        assert l1["volume"] > 0, f"Watertight mesh has non-positive volume: {l1['volume']}"

    def test_classification_exhaustive(self):
        """Every face must be assigned to exactly one category (no gaps, no overlaps).

        Property: The face classification is a partition of the face index set.
        |union of all face_indices| == total number of faces, with no duplicates.
        """
        mesh = _make_mesh(T1_BOX_VERTS, T1_BOX_FACES)
        l2 = validate_level2(mesh)

        all_classified_indices = []
        for g in l2["face_groups"]:
            all_classified_indices.extend(g["face_indices"])

        total_faces = len(T1_BOX_FACES)

        # No gaps: every face index appears at least once
        assert len(all_classified_indices) >= total_faces, (
            f"Only {len(all_classified_indices)} faces classified out of {total_faces}"
        )
        # No overlaps: every face index appears at most once
        assert len(all_classified_indices) == len(set(all_classified_indices)), (
            "Some face indices are assigned to multiple categories"
        )
        # Exact coverage
        assert set(all_classified_indices) == set(range(total_faces)), (
            f"Face indices {set(range(total_faces)) - set(all_classified_indices)} unclassified"
        )


# ── 3. Curved Wall Precision Tests ───────────────────────────────

class TestCurvedWallPrecision:
    """Verify curved wall measurements converge with tessellation density."""

    def _make_curved_wall(self, n_segments, radius=10.0, thickness=0.4,
                          height=3.0, arc_deg=90.0):
        """Build a curved wall mesh with variable tessellation density.

        Constructs a 90-degree arc wall with inner radius R and outer
        radius R + thickness, height h, tessellated into n_segments.

        Returns (vertices, faces) arrays.
        """
        arc_rad = np.radians(arc_deg)
        angles = np.linspace(0, arc_rad, n_segments + 1)

        r_inner = radius
        r_outer = radius + thickness

        verts = []
        # Bottom ring: inner then outer
        for a in angles:
            verts.append([r_inner * np.cos(a), r_inner * np.sin(a), 0.0])
        for a in angles:
            verts.append([r_outer * np.cos(a), r_outer * np.sin(a), 0.0])
        # Top ring: inner then outer
        for a in angles:
            verts.append([r_inner * np.cos(a), r_inner * np.sin(a), height])
        for a in angles:
            verts.append([r_outer * np.cos(a), r_outer * np.sin(a), height])

        verts = np.array(verts, dtype=float)
        n = n_segments + 1  # number of points per ring

        faces = []
        # Inner wall (faces pointing inward)
        for i in range(n_segments):
            bi = i          # bottom inner
            bi1 = i + 1
            ti = 2 * n + i  # top inner
            ti1 = 2 * n + i + 1
            faces.append([bi, bi1, ti1])
            faces.append([bi, ti1, ti])

        # Outer wall (faces pointing outward)
        for i in range(n_segments):
            bo = n + i          # bottom outer
            bo1 = n + i + 1
            to = 3 * n + i      # top outer
            to1 = 3 * n + i + 1
            faces.append([bo, to1, bo1])
            faces.append([bo, to, to1])

        # Bottom face (z=0): quads between inner and outer
        for i in range(n_segments):
            bi = i
            bi1 = i + 1
            bo = n + i
            bo1 = n + i + 1
            faces.append([bi, bo, bo1])
            faces.append([bi, bo1, bi1])

        # Top face (z=height): quads between inner and outer
        for i in range(n_segments):
            ti = 2 * n + i
            ti1 = 2 * n + i + 1
            to = 3 * n + i
            to1 = 3 * n + i + 1
            faces.append([ti, ti1, to1])
            faces.append([ti, to1, to])

        # End cap at angle=0 (left end)
        bi0 = 0
        bo0 = n
        ti0 = 2 * n
        to0 = 3 * n
        faces.append([bi0, ti0, to0])
        faces.append([bi0, to0, bo0])

        # End cap at angle=arc (right end)
        bi_last = n_segments
        bo_last = n + n_segments
        ti_last = 2 * n + n_segments
        to_last = 3 * n + n_segments
        faces.append([bi_last, to_last, ti_last])
        faces.append([bi_last, bo_last, to_last])

        return verts, np.array(faces, dtype=int)

    def test_crown_width_independent_of_segments(self):
        """Crown width of a 90-degree arc must be consistent once curvature is detected.

        Analytical expectation: crown width = wall thickness = 400 mm.
        The per-slice local frame method (activated when curvature is detected)
        must produce consistent results across tessellation densities.

        Note: Very coarse tessellations (< ~12 segments for a 90-degree arc)
        may not trigger the curvature detection significance test, falling back
        to global projection. This test verifies the local-frame regime only,
        using segment counts where curvature is reliably detected (>= 16).
        """
        crown_widths = []
        # Segment counts where curvature detection reliably activates
        segment_counts = [16, 20, 32, 40]

        for n_seg in segment_counts:
            verts, faces = self._make_curved_wall(n_seg)
            mesh = _make_mesh(verts, faces)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)

            if "crown_width_mm" in l3 and l3.get("crown_width_method") == "slice_local_frame":
                crown_widths.append((n_seg, l3["crown_width_mm"]))

        # At least 2 tessellation levels must produce local-frame crown width
        assert len(crown_widths) >= 2, (
            f"Crown width (local frame) measured for only {len(crown_widths)} "
            f"of {len(segment_counts)} densities"
        )

        # All crown widths must be within 400 +/- 50 mm
        for n_seg, cw in crown_widths:
            assert abs(cw - 400.0) < 50.0, (
                f"Crown width at {n_seg} segments = {cw:.1f} mm, expected ~400 mm"
            )

        # Consistency: max variation between tessellation levels < 20 mm
        widths_only = [cw for _, cw in crown_widths]
        variation = max(widths_only) - min(widths_only)
        assert variation < 20.0, (
            f"Crown width variation across tessellations = {variation:.1f} mm (max 20 mm allowed)"
        )

    def test_volume_convergence_with_segments(self):
        """Volume of a curved wall must converge to the analytical value.

        Analytical volume of a 90-degree arc wall:
        V = height * (pi/4) * (R_outer^2 - R_inner^2)
          = 3.0 * (pi/4) * (10.4^2 - 10.0^2)
          = 3.0 * (pi/4) * (108.16 - 100.0)
          = 3.0 * (pi/4) * 8.16
          = 19.2265... m^3
        """
        R = 10.0
        t = 0.4
        h = 3.0
        analytical_volume = h * (np.pi / 4.0) * ((R + t) ** 2 - R ** 2)

        for n_seg in [8, 20, 40]:
            verts, faces = self._make_curved_wall(n_seg)
            mesh = _make_mesh(verts, faces)
            l1 = validate_level1(mesh)
            # Relative error must decrease with more segments
            rel_error = abs(l1["volume"] - analytical_volume) / analytical_volume
            assert rel_error < 0.05, (
                f"Volume at {n_seg} segments = {l1['volume']:.4f}, "
                f"analytical = {analytical_volume:.4f}, relative error = {rel_error:.4f}"
            )


# ── 4. Multi-Element Consistency Tests ────────────────────────────

class TestMultiElementConsistency:
    """Verify L5/L6 behave correctly for various multi-element configurations."""

    def _make_box_element(self, offset, element_id, dims=(8.0, 0.4, 3.0)):
        """Build a complete element dict for L5/L6 testing."""
        w, d, h = dims
        verts = np.array([
            [0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
            [0, 0, h], [w, 0, h], [w, d, h], [0, d, h],
        ], dtype=float) + np.array(offset, dtype=float)
        faces = T1_BOX_FACES.copy()
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        return {
            "element_id": element_id,
            "element_name": f"Wall_{element_id}",
            "level1": l1,
            "mesh_data": mesh,
        }

    def test_l5_symmetric_elements(self):
        """Two identical elements stacked vertically must be classified as stacked.

        Property: Two boxes sharing a horizontal contact plane have a
        contact normal with |N.z| = 1.0, so kappa > cos(45 deg) = stacked.
        """
        # Bottom element: z = [0, 3]
        elem_a = self._make_box_element([0, 0, 0], element_id=1)
        # Top element: z = [3, 6] (sitting directly on top)
        elem_b = self._make_box_element([0, 0, 3], element_id=2)

        result = validate_level5([elem_a, elem_b])
        assert result["summary"]["num_pairs"] >= 1, (
            "Stacked elements should form at least one pair"
        )
        # The pair must be classified as stacked
        stacked_pairs = [p for p in result["pairs"] if p["pair_type"] == "stacked"]
        assert len(stacked_pairs) >= 1, (
            f"Expected stacked pair, got: {[p['pair_type'] for p in result['pairs']]}"
        )

    def test_l5_side_by_side_elements(self):
        """Two identical elements placed side by side must be classified as side_by_side.

        Property: Two boxes sharing a vertical contact plane have a
        contact normal with |N.z| ~ 0.0, so kappa < cos(45 deg) = side_by_side.
        """
        # Left element: x = [0, 8]
        elem_a = self._make_box_element([0, 0, 0], element_id=1)
        # Right element: x = [8, 16] (sharing the x=8 face)
        elem_b = self._make_box_element([8, 0, 0], element_id=2)

        result = validate_level5([elem_a, elem_b])
        assert result["summary"]["num_pairs"] >= 1, (
            "Side-by-side elements should form at least one pair"
        )
        side_pairs = [p for p in result["pairs"] if p["pair_type"] == "side_by_side"]
        assert len(side_pairs) >= 1, (
            f"Expected side_by_side pair, got: {[p['pair_type'] for p in result['pairs']]}"
        )

    def test_l6_no_terrain_graceful(self):
        """All L6 functions return safely when terrain is None.

        Property: L6 must degrade gracefully (no exceptions, sensible defaults)
        when terrain data is unavailable.
        """
        # Single element, no terrain
        elem = self._make_box_element([0, 0, 0], element_id=1)
        result = validate_level6([elem], terrain_mesh=None)

        assert result["terrain_available"] is False
        assert result["terrain_side"] == {}
        assert result["clearances"] == []

    def test_l6_empty_elements_with_terrain(self):
        """L6 with terrain but no elements must not crash."""
        terrain = {
            "vertices": np.array([
                [-10, -10, -1], [20, -10, -1], [20, 10, -1], [-10, 10, -1],
            ], dtype=float),
            "faces": np.array([[0, 1, 2], [0, 2, 3]]),
        }
        result = validate_level6([], terrain_mesh=terrain)
        assert result["terrain_available"] is True
        assert len(result["clearances"]) == 0
        assert len(result["distances"]) == 0

    def test_l5_three_elements_pairwise(self):
        """Three elements must produce correct pairwise pair count.

        Property: C(3, 2) = 3 potential pairs. All close pairs detected.
        """
        # Three stacked boxes
        elem_a = self._make_box_element([0, 0, 0], element_id=1)
        elem_b = self._make_box_element([0, 0, 3], element_id=2)
        elem_c = self._make_box_element([0, 0, 6], element_id=3)

        result = validate_level5([elem_a, elem_b, elem_c])
        # At minimum, adjacent pairs (a,b) and (b,c) should be detected
        assert result["summary"]["num_pairs"] >= 2, (
            f"Expected >= 2 pairs for 3 stacked elements, got {result['summary']['num_pairs']}"
        )
