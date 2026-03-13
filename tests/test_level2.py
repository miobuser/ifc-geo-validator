"""Tests for Level 2: Face classification.

Test strategy:
  - Pure-numpy mesh tests (no IFC dependency) for algorithm verification
  - IFC file tests (skipped if models unavailable) for integration
"""

import os
import pytest
import numpy as np

from ifc_geo_validator.core.face_classifier import (
    classify_faces,
    _weld_vertices,
    _build_face_adjacency,
    _cluster_coplanar,
    _determine_wall_axis,
    FaceGroup,
    CROWN,
    FOUNDATION,
    FRONT,
    BACK,
    END_LEFT,
    END_RIGHT,
)
from ifc_geo_validator.validation.level2 import validate_level2


# ── Helper ──────────────────────────────────────────────────────────

def _make_mesh(verts, faces):
    """Build a mesh_data dict from vertices and faces arrays."""
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


# ── T1-style box: 8.0 x 0.4 x 3.0 ─────────────────────────────────

T1_VERTS = np.array([
    [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],  # bottom
    [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],  # top
], dtype=float)

T1_FACES = np.array([
    [0, 2, 1], [0, 3, 2],    # bottom  (z=0, normal -Z)
    [4, 5, 6], [4, 6, 7],    # top     (z=3, normal +Z)
    [0, 1, 5], [0, 5, 4],    # front   (y=0, normal -Y)
    [2, 3, 7], [2, 7, 6],    # back    (y=0.4, normal +Y)
    [0, 4, 7], [0, 7, 3],    # left    (x=0, normal -X)
    [1, 2, 6], [1, 6, 5],    # right   (x=8, normal +X)
])


class TestVertexWelding:
    """Test vertex deduplication."""

    def test_no_duplicates(self):
        """Vertices already unique → no change in face count."""
        welded_v, welded_f = _weld_vertices(T1_VERTS, T1_FACES)
        assert len(welded_v) == 8
        assert welded_f.shape == T1_FACES.shape

    def test_with_duplicates(self):
        """Duplicate vertices should be merged."""
        # Duplicate vertex 0 as vertex 8
        verts = np.vstack([T1_VERTS, T1_VERTS[0:1]])
        faces = T1_FACES.copy()
        welded_v, welded_f = _weld_vertices(verts, faces)
        assert len(welded_v) == 8  # duplicate removed


class TestFaceAdjacency:
    """Test adjacency detection from shared edges."""

    def test_box_adjacency(self):
        """A box has 12 triangles; each shares 1-3 edges with neighbors."""
        pairs = _build_face_adjacency(T1_FACES)
        # A box with 12 triangles and 18 edges → each internal edge
        # shared by exactly 2 faces.  12 internal adjacency pairs expected.
        assert len(pairs) >= 12

    def test_isolated_triangles(self):
        """Triangles with no shared edges → no adjacency."""
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        pairs = _build_face_adjacency(faces)
        assert len(pairs) == 0


class TestCoplanarClustering:
    """Test coplanar triangle grouping."""

    def test_box_six_groups(self):
        """A box should produce 6 coplanar groups (one per face)."""
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        pairs = _build_face_adjacency(T1_FACES)
        clusters = _cluster_coplanar(
            len(T1_FACES), pairs, mesh["normals"], np.radians(5.0)
        )
        assert len(clusters) == 6
        # Each cluster has exactly 2 triangles
        for c in clusters:
            assert len(c) == 2


class TestWallAxis:
    """Test longitudinal axis determination."""

    def test_elongated_box(self):
        """8m long, 0.4m wide → axis should be along X."""
        axis = _determine_wall_axis(T1_VERTS)
        # Axis should be approximately [±1, 0, 0]
        assert abs(abs(axis[0]) - 1.0) < 0.1
        assert abs(axis[1]) < 0.1
        assert abs(axis[2]) < 0.01

    def test_square_plan_fallback(self):
        """Square plan (1x1) → should still return a valid axis."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        axis = _determine_wall_axis(verts)
        # Should be a unit vector in XY plane
        assert abs(np.linalg.norm(axis[:2]) - 1.0) < 0.01


class TestClassifyFacesBox:
    """Full classification pipeline on T1-style box."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mesh = _make_mesh(T1_VERTS, T1_FACES)
        self.result = classify_faces(self.mesh)
        self.groups = self.result["face_groups"]

    def test_six_groups(self):
        assert self.result["num_groups"] == 6

    def test_has_crown(self):
        crowns = [g for g in self.groups if g.category == CROWN]
        assert len(crowns) == 1
        assert abs(crowns[0].area - 3.2) < 0.01  # 8.0 * 0.4

    def test_has_foundation(self):
        foundations = [g for g in self.groups if g.category == FOUNDATION]
        assert len(foundations) == 1
        assert abs(foundations[0].area - 3.2) < 0.01

    def test_has_front_and_back(self):
        fronts = [g for g in self.groups if g.category == FRONT]
        backs = [g for g in self.groups if g.category == BACK]
        assert len(fronts) == 1
        assert len(backs) == 1
        # Front and back each 8.0 * 3.0 = 24.0 m²
        assert abs(fronts[0].area - 24.0) < 0.01
        assert abs(backs[0].area - 24.0) < 0.01

    def test_has_end_faces(self):
        ends = [g for g in self.groups if g.category in (END_LEFT, END_RIGHT)]
        assert len(ends) == 2
        for e in ends:
            assert abs(e.area - 1.2) < 0.01  # 0.4 * 3.0

    def test_crown_normal_up(self):
        crown = [g for g in self.groups if g.category == CROWN][0]
        # Normal should point up (+Z)
        assert crown.normal[2] > 0.9

    def test_foundation_normal_down(self):
        found = [g for g in self.groups if g.category == FOUNDATION][0]
        assert found.normal[2] < -0.9

    def test_wall_axis_along_x(self):
        axis = self.result["wall_axis"]
        assert abs(abs(axis[0]) - 1.0) < 0.1

    def test_total_area_conserved(self):
        """Sum of group areas must equal total mesh area."""
        group_area = sum(g.area for g in self.groups)
        mesh_area = float(self.mesh["areas"].sum())
        assert abs(group_area - mesh_area) < 0.01


class TestValidateLevel2:
    """Integration test for the Level 2 validation wrapper."""

    def test_box_summary(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        result = validate_level2(mesh)

        assert result["has_crown"] is True
        assert result["has_foundation"] is True
        assert result["has_front"] is True
        assert result["has_back"] is True

        summary = result["summary"]
        assert CROWN in summary
        assert summary[CROWN]["count"] == 1
        assert abs(summary[CROWN]["total_area"] - 3.2) < 0.01

    def test_serialised_groups(self):
        """face_groups should be plain dicts, not FaceGroup instances."""
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        result = validate_level2(mesh)
        for g in result["face_groups"]:
            assert isinstance(g, dict)
            assert "category" in g


# ── Crown slope test (T3-style: 3% crown slope) ────────────────────

class TestCrownSlope:
    """A wall with 3% crown slope should still classify crown correctly."""

    def test_sloped_crown_classified(self):
        # 3% slope = rise/run = 0.03 → for 0.4m width, rise = 0.012m
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],        # bottom
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3.012], [0, 0.4, 3.012],  # top (sloped)
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        result = classify_faces(mesh)
        groups = result["face_groups"]

        crowns = [g for g in groups if g.category == CROWN]
        assert len(crowns) == 1
        # The crown normal should be nearly +Z (3% slope = 1.72°, well below 30° threshold)
        assert crowns[0].normal[2] > 0.99


# ── Unwelded mesh test (simulates IfcOpenShell output) ──────────────

class TestUnweldedMesh:
    """Test classification on mesh with duplicate vertices (as from IFC)."""

    def test_unwelded_produces_same_groups(self):
        # Create an unwelded version: each triangle gets its own vertices
        new_verts = []
        new_faces = []
        for i, tri in enumerate(T1_FACES):
            base = i * 3
            new_verts.append(T1_VERTS[tri[0]])
            new_verts.append(T1_VERTS[tri[1]])
            new_verts.append(T1_VERTS[tri[2]])
            new_faces.append([base, base + 1, base + 2])

        mesh = _make_mesh(np.array(new_verts), np.array(new_faces))
        result = classify_faces(mesh)

        # Should still produce 6 groups
        assert result["num_groups"] == 6

        categories = {g.category for g in result["face_groups"]}
        assert CROWN in categories
        assert FOUNDATION in categories


# ── IFC file tests (skipped if model not found) ─────────────────────

T1_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T1_simple_box.ifc")
T3_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T3_crown_slope.ifc")


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 model not found")
class TestT1FaceClassification:
    """Level 2 on T1 IFC file."""

    def test_six_groups(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T1_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        # Simple box → 6 groups
        assert result["num_groups"] == 6


T2_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T2_inclined_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T2_PATH), reason="T2 model not found")
class TestT2FaceClassification:
    """Level 2 on T2 IFC file (10:1 inclined front face)."""

    def test_inclined_front(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T2_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        assert result["has_front"]
        assert result["num_groups"] == 6

        # Crown should be smaller (0.35m wide) than foundation (0.65m wide)
        crown_area = result["summary"][CROWN]["total_area"]
        found_area = result["summary"][FOUNDATION]["total_area"]
        assert crown_area < found_area


@pytest.mark.skipif(not os.path.exists(T3_PATH), reason="T3 model not found")
class TestT3FaceClassification:
    """Level 2 on T3 IFC file (crown slope 3%)."""

    def test_crown_detected(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T3_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        crown_groups = [g for g in result["face_groups"] if g["category"] == CROWN]
        assert len(crown_groups) >= 1
