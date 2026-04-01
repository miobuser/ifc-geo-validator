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


T4_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T4_l_shaped.ifc")
T5_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T5_t_shaped.ifc")


@pytest.mark.skipif(not os.path.exists(T4_PATH), reason="T4 model not found")
class TestT4FaceClassification:
    """Level 2 on T4 IFC file (L-shaped retaining wall)."""

    def test_l_shaped_groups(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T4_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        # Multi-element: walls[0] = Mauersteg (simple box) → 6 groups
        assert result["num_groups"] == 6


@pytest.mark.skipif(not os.path.exists(T5_PATH), reason="T5 model not found")
class TestT5FaceClassification:
    """Level 2 on T5 IFC file (T-shaped wall with spur)."""

    def test_t_shaped_groups(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T5_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        assert result["has_front"]
        # Multi-element: walls[0] = Hauptwand (simple box) → 6 groups
        assert result["num_groups"] == 6


# ── T8: Curved wall (90° arc) ──────────────────────────────────────

T8_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T8_curved_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T8_PATH), reason="T8 model not found")
class TestT8FaceClassification:
    """Level 2 on T8 IFC file (90° arc, R=10m, 0.4m thick)."""

    def test_curved_wall_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T8_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        assert result["has_front"]
        assert result["has_back"]
        # After post-classification merge: 6 groups
        assert result["num_groups"] == 6

    def test_centerline_is_curved(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T8_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is True
        assert centerline.length > 5.0  # arc length > 5m


# ── T9: Stepped wall ────────────────────────────────────────────────

T9_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T9_stepped_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T9_PATH), reason="T9 model not found")
class TestT9FaceClassification:
    """Level 2 on T9 IFC file (stepped profile: 300mm crown, 600mm base)."""

    def test_stepped_wall_groups(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T9_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        assert result["has_front"]
        assert result["has_back"]
        # Multi-element: walls[0] = Oberer Steg (simple box) → 6 groups
        assert result["num_groups"] == 6

    def test_not_curved(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T9_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved == False


# ── T10: Complex curved wall ────────────────────────────────────────

T10_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T10_complex_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T10_PATH), reason="T10 model not found")
class TestT10FaceClassification:
    """Level 2 on T10 IFC file (60° arc, tapered, inclined, crown slope)."""

    def test_complex_curved_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T10_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level2(mesh_data)

        assert result["has_crown"]
        assert result["has_foundation"]
        assert result["has_front"]
        assert result["has_back"]
        assert result["num_groups"] == 6

        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is True


# ── T11-T14: Complex geometry models ────────────────────────────────

T11_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T11_s_curved.ifc")
T12_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T12_semicircle.ifc")
T13_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T13_polygonal.ifc")
T14_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T14_curved_l_profile.ifc")


@pytest.mark.skipif(not os.path.exists(T11_PATH), reason="T11 model not found")
class TestT11SCurved:
    """Level 2 on T11 (S-curve with inflection point)."""

    def test_s_curve_classified(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T11_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result.get("centerline").is_curved is True
        assert result["num_groups"] == 6


@pytest.mark.skipif(not os.path.exists(T12_PATH), reason="T12 model not found")
class TestT12Semicircle:
    """Level 2 on T12 (180° semicircle)."""

    def test_semicircle_classified(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T12_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result.get("centerline").is_curved is True
        assert result["num_groups"] == 6


@pytest.mark.skipif(not os.path.exists(T13_PATH), reason="T13 model not found")
class TestT13Polygonal:
    """Level 2 on T13 (3 straight segments at angles)."""

    def test_polygonal_not_curved(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T13_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result.get("centerline").is_curved == False
        assert result["num_groups"] == 6


@pytest.mark.skipif(not os.path.exists(T14_PATH), reason="T14 model not found")
class TestT14CurvedLProfile:
    """Level 2 on T14 (45° arc, L cross-section)."""

    def test_curved_l_classified(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T14_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result.get("centerline").is_curved is True
        # Multi-element: walls[0] = Mauersteg (simple curved box) → 6 groups
        assert result["num_groups"] == 6


# ── T6: Non-compliant wall (200mm thin) ─────────────────────────────

T6_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T6_non_compliant.ifc")


@pytest.mark.skipif(not os.path.exists(T6_PATH), reason="T6 model not found")
class TestT6FaceClassification:
    """Level 2 on T6 IFC file (non-compliant, 200mm thin wall)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T6_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6


# ── T7: Compliant wall ──────────────────────────────────────────────

T7_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T7_compliant.ifc")


@pytest.mark.skipif(not os.path.exists(T7_PATH), reason="T7 model not found")
class TestT7FaceClassification:
    """Level 2 on T7 IFC file (compliant, 300mm, inclined, crown slope)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T7_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6

    def test_not_curved(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T7_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))

        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is False


# ── T15: Variable height wall ───────────────────────────────────────

T15_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T15_variable_height.ifc")


@pytest.mark.skipif(not os.path.exists(T15_PATH), reason="T15 model not found")
class TestT15FaceClassification:
    """Level 2 on T15 IFC file (variable height wall)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T15_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6

    def test_not_curved(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T15_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))

        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is False


# ── T16: Height step (multi-element) ────────────────────────────────

T16_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T16_height_step.ifc")


@pytest.mark.skipif(not os.path.exists(T16_PATH), reason="T16 model not found")
class TestT16FaceClassification:
    """Level 2 on T16 IFC file (height step, multi-element)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T16_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        # Multi-element: walls[0] may be the upper section
        assert result["num_groups"] >= 6


# ── T17: Curved variable wall ───────────────────────────────────────

T17_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T17_curved_variable.ifc")


@pytest.mark.skipif(not os.path.exists(T17_PATH), reason="T17 model not found")
class TestT17FaceClassification:
    """Level 2 on T17 IFC file (curved with variable profile)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T17_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6

    def test_is_curved(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T17_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level2(extract_mesh(walls[0]))

        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is True


# ── T18: Buttressed wall (multi-element) ─────────────────────────────

T18_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T18_buttressed.ifc")


@pytest.mark.skipif(not os.path.exists(T18_PATH), reason="T18 model not found")
class TestT18FaceClassification:
    """Level 2 on T18 IFC file (buttressed, multi-element: walls[0]=Hauptmauer)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T18_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6


# ── T20: Triangulated representation ────────────────────────────────

T20_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T20_triangulated.ifc")


@pytest.mark.skipif(not os.path.exists(T20_PATH), reason="T20 model not found")
class TestT20FaceClassification:
    """Level 2 on T20 IFC file (triangulated, same geometry as T1)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T20_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_foundation"]
        assert result["has_front"]
        assert result["num_groups"] == 6


# ── T21: Extruded trapezoid ─────────────────────────────────────────

T21_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T21_extruded_trapezoid.ifc")


@pytest.mark.skipif(not os.path.exists(T21_PATH), reason="T21 model not found")
class TestT21FaceClassification:
    """Level 2 on T21 IFC file (extruded trapezoid cross-section)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T21_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6


# ── T22: Wall with terrain ──────────────────────────────────────────

T22_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T22_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T22_PATH), reason="T22 model not found")
class TestT22FaceClassification:
    """Level 2 on T22 IFC file (wall with terrain context)."""

    def test_classification(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T22_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"]
        assert result["has_front"]
        assert result["num_groups"] == 6


# ── T23: ASTRA compliant curved wall ─────────────────────────────────

T23_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T23_astra_compliant_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T23_PATH), reason="T23 model not found")
class TestT23ClassifyFaces:
    """Level 2 on T23 IFC file (ASTRA compliant curved wall, 3 elements)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T23_PATH)
        self.walls = get_elements(model, "IfcWall")
        self.stem_mesh = extract_mesh(self.walls[0])
        self.stem_result = validate_level2(self.stem_mesh)

    def test_num_elements(self):
        """T23 should contain 3 wall elements (stem, foundation, buttress)."""
        assert len(self.walls) == 3

    def test_has_crown(self):
        """Stem element should have a crown face."""
        assert self.stem_result["has_crown"] is True

    def test_has_foundation(self):
        """Stem element should have a foundation face."""
        assert self.stem_result["has_foundation"] is True

    def test_is_curved(self):
        """Stem element should be curved."""
        centerline = self.stem_result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is True

    def test_num_groups(self):
        """Stem element should have 6 face groups."""
        assert self.stem_result["num_groups"] == 6

    def test_all_categories_present(self):
        """All face categories should be present."""
        assert self.stem_result["has_front"] is True
        assert self.stem_result["has_back"] is True


# ── T24: Highway wall with terrain ───────────────────────────────────

T24_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T24_highway_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T24_PATH), reason="T24 model not found")
class TestT24ClassifyFaces:
    """Level 2 on T24 IFC file (highway wall with terrain, 2 elements)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T24_PATH)
        self.walls = get_elements(model, "IfcWall")
        self.stem_mesh = extract_mesh(self.walls[0])
        self.stem_result = validate_level2(self.stem_mesh)
        self.foundation_mesh = extract_mesh(self.walls[1])
        self.foundation_result = validate_level2(self.foundation_mesh)

    def test_num_elements(self):
        """T24 should contain 2 wall elements (stem + foundation)."""
        assert len(self.walls) == 2

    def test_stem_has_crown(self):
        """Stem element should have a crown face."""
        assert self.stem_result["has_crown"] is True

    def test_stem_groups(self):
        """Stem element should have 6 face groups."""
        assert self.stem_result["num_groups"] == 6

    def test_foundation_has_crown(self):
        """Foundation top is horizontal, so it should be detected as crown."""
        assert self.foundation_result["has_crown"] is True

    def test_not_curved(self):
        """Stem element should not be curved."""
        centerline = self.stem_result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is False

    def test_all_categories_present(self):
        """Stem should have all face categories."""
        assert self.stem_result["has_foundation"] is True
        assert self.stem_result["has_front"] is True
        assert self.stem_result["has_back"] is True


# ── T25: Multi-failure wall ──────────────────────────────────────────

T25_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T25_multi_failure.ifc")


@pytest.mark.skipif(not os.path.exists(T25_PATH), reason="T25 model not found")
class TestT25ClassifyFaces:
    """Level 2 on T25 IFC file (single non-compliant wall)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T25_PATH)
        self.walls = get_elements(model, "IfcWall")
        self.mesh = extract_mesh(self.walls[0])
        self.result = validate_level2(self.mesh)

    def test_single_element(self):
        """T25 should contain 1 wall element."""
        assert len(self.walls) == 1

    def test_has_all_categories(self):
        """Wall should have crown, foundation, front, back, and 2 ends."""
        assert self.result["has_crown"] is True
        assert self.result["has_foundation"] is True
        assert self.result["has_front"] is True
        assert self.result["has_back"] is True

    def test_num_groups(self):
        """Wall should have 6 face groups."""
        assert self.result["num_groups"] == 6


# ── T23-T25: Face classification tests (additional) ──────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "T23_astra_compliant_curved.ifc")),
    reason="T23 model not found",
)
class TestT23FaceClassification:
    """Face classification tests for T23 (ASTRA compliant curved, 3 elements)."""

    def _load(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        model = load_model(os.path.join(MODELS_DIR, "T23_astra_compliant_curved.ifc"))
        walls = get_elements(model, "IfcWall")
        return walls

    def test_three_elements(self):
        walls = self._load()
        assert len(walls) == 3

    def test_stem_has_crown(self):
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        walls = self._load()
        result = validate_level2(extract_mesh(walls[0]))
        assert result["has_crown"] is True

    def test_stem_is_curved(self):
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        walls = self._load()
        result = validate_level2(extract_mesh(walls[0]))
        centerline = result.get("centerline")
        assert centerline is not None
        assert centerline.is_curved is True


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "T24_highway_with_terrain.ifc")),
    reason="T24 model not found",
)
class TestT24FaceClassification:
    """Face classification tests for T24 (highway wall with terrain, 2 elements)."""

    def _load(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        model = load_model(os.path.join(MODELS_DIR, "T24_highway_with_terrain.ifc"))
        walls = get_elements(model, "IfcWall")
        return walls

    def test_two_elements(self):
        walls = self._load()
        assert len(walls) == 2

    def test_stem_six_groups(self):
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        walls = self._load()
        result = validate_level2(extract_mesh(walls[0]))
        assert result["num_groups"] == 6


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "T25_multi_failure.ifc")),
    reason="T25 model not found",
)
class TestT25FaceClassification:
    """Face classification tests for T25 (single non-compliant wall)."""

    def _load(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        model = load_model(os.path.join(MODELS_DIR, "T25_multi_failure.ifc"))
        walls = get_elements(model, "IfcWall")
        return walls

    def test_single_element(self):
        walls = self._load()
        assert len(walls) == 1

    def test_six_groups(self):
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        walls = self._load()
        result = validate_level2(extract_mesh(walls[0]))
        assert result["num_groups"] == 6


# ── T26: Extruded curved front profile ────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "T26_extruded_curved.ifc")),
    reason="T26 model not found",
)
class TestT26FaceClassification:
    """Face classification tests for T26 (extruded profile with curved front)."""

    def _load(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(os.path.join(MODELS_DIR, "T26_extruded_curved.ifc"))
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        return validate_level2(mesh)

    def test_has_crown(self):
        result = self._load()
        assert result["has_crown"] is True

    def test_six_groups(self):
        result = self._load()
        assert result["num_groups"] == 6


# ── Asymmetry index tests ─────────────────────────────────────────

class TestAsymmetryIndex:
    """Test front/back asymmetry index is computed and meaningful."""

    def test_symmetric_wall_low_asymmetry(self):
        """T1 simple box: symmetric front/back -> asymmetry near 0."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(os.path.join(MODELS_DIR, "T1_simple_box.ifc"))
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        result = validate_level2(mesh)
        asym = result.get("front_back_asymmetry", -1)
        assert 0 <= asym <= 0.1, f"Symmetric box should have low asymmetry, got {asym}"

    def test_inclined_wall_has_asymmetry(self):
        """T2 inclined wall (10:1): slight asymmetry due to inclination.
        At 10:1 ratio (~5.7 deg), front/back areas differ only slightly.
        Asymmetry should be > 0 but small (wall is nearly symmetric)."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(os.path.join(MODELS_DIR, "T2_inclined_wall.ifc"))
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        result = validate_level2(mesh)
        asym = result.get("front_back_asymmetry", -1)
        assert asym > 0, f"Inclined wall should have non-zero asymmetry, got {asym}"
        assert asym < 0.1, f"10:1 wall is nearly symmetric, got {asym}"

    def test_asymmetry_in_range(self):
        """Asymmetry index must be in [0, 1]."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        for name in ["T1_simple_box.ifc", "T2_inclined_wall.ifc", "T7_compliant.ifc"]:
            path = os.path.join(MODELS_DIR, name)
            if not os.path.exists(path):
                continue
            model = load_model(path)
            walls = get_elements(model, "IfcWall")
            mesh = extract_mesh(walls[0])
            result = validate_level2(mesh)
            asym = result.get("front_back_asymmetry", -1)
            assert 0 <= asym <= 1, f"{name}: asymmetry {asym} out of range [0,1]"
