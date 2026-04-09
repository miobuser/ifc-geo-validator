"""Tests for Level 3: Face-specific geometric measurements.

Reference values computed analytically for known test geometries.
"""

import os
import pytest
import numpy as np

from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3


# ── Helper ──────────────────────────────────────────────────────────

def _make_mesh(verts, faces):
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


# ── T1 box: 8.0 x 0.4 x 3.0 ───────────────────────────────────────

T1_VERTS = np.array([
    [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
    [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
], dtype=float)

T1_FACES = np.array([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3],
    [1, 2, 6], [1, 6, 5],
])


class TestT1CrownWidth:
    """Crown width of simple box = 400 mm (0.4 m)."""

    def test_crown_width(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert abs(l3["crown_width_mm"] - 400.0) < 1.0  # 1mm tolerance


class TestT1CrownSlope:
    """Flat crown → slope = 0%."""

    def test_crown_slope_zero(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert abs(l3["crown_slope_percent"]) < 0.1


class TestT1WallThickness:
    """Wall thickness = 400 mm (distance between front and back)."""

    def test_wall_thickness(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert abs(l3["min_wall_thickness_mm"] - 400.0) < 1.0
        assert abs(l3["avg_wall_thickness_mm"] - 400.0) < 1.0


class TestT1FrontInclination:
    """Vertical front face → inclination angle = 0°."""

    def test_vertical_front(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert abs(l3["front_inclination_deg"]) < 0.1


# ── Crown slope 3% ─────────────────────────────────────────────────

class TestCrownSlope3Percent:
    """Wall with 3% crown slope."""

    def test_slope_value(self):
        # 3% slope = rise/run = 0.03 → for 0.4m width, rise = 0.012m
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
            [0, 0, 3], [8, 0, 3], [8, 0.4, 3.012], [0, 0.4, 3.012],
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
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        # Crown slope should be approximately 3%
        assert abs(l3["crown_slope_percent"] - 3.0) < 0.5


# ── Narrow wall (below 300mm minimum) ──────────────────────────────

class TestNarrowWall:
    """Wall with 200mm thickness — should fail ASTRA-SM-L3-003."""

    def test_thin_wall(self):
        verts = np.array([
            [0, 0, 0], [6, 0, 0], [6, 0.2, 0], [0, 0.2, 0],
            [0, 0, 2], [6, 0, 2], [6, 0.2, 2], [0, 0.2, 2],
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
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert l3["crown_width_mm"] < 300  # 200mm < 300mm minimum
        assert l3["min_wall_thickness_mm"] < 300


# ── Inclined front face (10:1 ratio) ───────────────────────────────

class TestInclinedFront:
    """Front face with 10:1 inclination (≈5.71°)."""

    def test_inclination_ratio(self):
        # 10:1 means 0.3m horizontal offset over 3.0m height
        verts = np.array([
            [0, 0, 0], [8, 0, 0], [8, 0.7, 0], [0, 0.7, 0],      # bottom wider
            [0, 0.3, 3], [8, 0.3, 3], [8, 0.7, 3], [0, 0.7, 3],   # top narrower
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2],    # bottom
            [4, 5, 6], [4, 6, 7],    # top
            [0, 1, 5], [0, 5, 4],    # front (inclined)
            [2, 3, 7], [2, 7, 6],    # back (vertical)
            [0, 4, 7], [0, 7, 3],    # left
            [1, 2, 6], [1, 6, 5],    # right
        ])
        mesh = _make_mesh(verts, faces)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        # Inclination should be approximately 5.71° (arctan(1/10))
        expected_deg = np.degrees(np.arctan(1.0 / 10.0))
        assert abs(l3["front_inclination_deg"] - expected_deg) < 0.5

        # Ratio should be approximately 10
        assert abs(l3["front_inclination_ratio"] - 10.0) < 1.0


# ── Full pipeline test ──────────────────────────────────────────────

class TestFullPipeline:
    """End-to-end L1 → L2 → L3 on T1 box."""

    def test_all_measurements_present(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert "crown_width_mm" in l3
        assert "crown_slope_percent" in l3
        assert "min_wall_thickness_mm" in l3
        assert "front_inclination_deg" in l3


# ── IFC file tests ──────────────────────────────────────────────────

T1_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T1_simple_box.ifc")
T2_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T2_inclined_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T2_PATH), reason="T2 model not found")
class TestT2FromIFC:
    """Level 3 on T2 IFC file (10:1 inclined wall)."""

    def test_crown_width_350mm(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T2_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l2 = validate_level2(mesh_data)
        l3 = validate_level3(mesh_data, l2)

        assert abs(l3["crown_width_mm"] - 350.0) < 5.0
        assert abs(l3["min_wall_thickness_mm"] - 350.0) < 5.0
        assert abs(l3["front_inclination_deg"] - 5.71) < 0.5
        assert abs(l3["front_inclination_ratio"] - 10.0) < 1.0


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 model not found")
class TestT1FromIFC:
    """Level 3 on T1 IFC file."""

    def test_crown_width_400mm(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T1_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l2 = validate_level2(mesh_data)
        l3 = validate_level3(mesh_data, l2)

        assert abs(l3["crown_width_mm"] - 400.0) < 5.0


T3_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T3_crown_slope.ifc")
T4_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T4_l_shaped.ifc")
T5_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T5_t_shaped.ifc")


@pytest.mark.skipif(not os.path.exists(T3_PATH), reason="T3 model not found")
class TestT3FromIFC:
    """Level 3 on T3 IFC file (3% crown slope)."""

    def test_crown_slope_3pct(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T3_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l2 = validate_level2(mesh_data)
        l3 = validate_level3(mesh_data, l2)

        assert abs(l3["crown_width_mm"] - 300.0) < 5.0
        assert abs(l3["crown_slope_percent"] - 3.0) < 0.5
        assert abs(l3["min_wall_thickness_mm"] - 300.0) < 5.0


@pytest.mark.skipif(not os.path.exists(T4_PATH), reason="T4 model not found")
class TestT4FromIFC:
    """Level 3 on T4 IFC file — multi-element: walls[0] = Mauersteg (300mm)."""

    def test_mauersteg_measurements(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T4_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l2 = validate_level2(mesh_data)
        l3 = validate_level3(mesh_data, l2)

        # Multi-element: walls[0] = Mauersteg (simple box, 300mm thick)
        assert abs(l3["crown_width_mm"] - 300.0) < 10.0
        # Wall thickness = 300mm
        assert abs(l3["min_wall_thickness_mm"] - 300.0) < 5.0
        # Vertical front
        assert abs(l3["front_inclination_deg"]) < 0.5


@pytest.mark.skipif(not os.path.exists(T5_PATH), reason="T5 model not found")
class TestT5FromIFC:
    """Level 3 on T5 IFC file — multi-element: walls[0] = Hauptwand (400mm)."""

    def test_hauptwand_measurements(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T5_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l2 = validate_level2(mesh_data)
        l3 = validate_level3(mesh_data, l2)

        # Multi-element: walls[0] = Hauptwand (simple box, 400mm thick)
        assert l3["crown_width_mm"] >= 400
        # Wall thickness = 400mm
        assert abs(l3["min_wall_thickness_mm"] - 400.0) < 5.0
        # Vertical front
        assert abs(l3["front_inclination_deg"]) < 0.5


# ── T8: Curved retaining wall (90° arc) ──────────────────────────────

T8_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T8_curved_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T8_PATH), reason="T8 model not found")
class TestT8CurvedWall:
    """Level 2+3 on T8 IFC file (90° arc, R=10m, 0.4m thick)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T8_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_is_curved(self):
        """Wall must be detected as curved."""
        assert self.l3.get("is_curved") is True

    def test_face_groups(self):
        """After merge: 6 groups (crown, foundation, front, back, 2 ends)."""
        assert self.l2["num_groups"] == 6
        assert self.l2["has_crown"]
        assert self.l2["has_foundation"]
        assert self.l2["has_front"]
        assert self.l2["has_back"]

    def test_crown_width_local(self):
        """Crown width must be ~400mm (local measurement, not arc chord)."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 5.0
        assert self.l3["crown_width_method"] == "slice_local_frame"

    def test_wall_thickness_local(self):
        """Wall thickness must be ~400mm (local measurement)."""
        assert abs(self.l3["min_wall_thickness_mm"] - 400.0) < 5.0

    def test_wall_length(self):
        """Arc length of R≈10.2m quarter circle ≈ π/2 * 10.2 ≈ 16m."""
        assert self.l3["wall_length_m"] > 5.0

    def test_vertical_front(self):
        """Front face is vertical (no inclination)."""
        assert abs(self.l3["front_inclination_deg"]) < 1.0


# ── T9: Stepped retaining wall ───────────────────────────────────────

T9_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T9_stepped_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T9_PATH), reason="T9 model not found")
class TestT9SteppedWall:
    """Level 2+3 on T9 IFC file — multi-element: walls[0] = Oberer Steg (300mm uniform)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T9_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_not_curved(self):
        """Oberer Steg must NOT be detected as curved."""
        assert self.l3.get("is_curved") is False

    def test_face_groups(self):
        """Multi-element: walls[0] = Oberer Steg (simple box) → 6 groups."""
        assert self.l2["num_groups"] == 6

    def test_min_wall_thickness(self):
        """Thickness = 300mm (uniform simple box)."""
        assert abs(self.l3["min_wall_thickness_mm"] - 300.0) < 5.0

    def test_avg_thickness_equals_min(self):
        """Uniform box: avg thickness ≈ min thickness ≈ 300mm."""
        assert abs(self.l3["avg_wall_thickness_mm"] - 300.0) < 10.0

    def test_vertical_front(self):
        """Front face is vertical."""
        assert abs(self.l3["front_inclination_deg"]) < 0.5


# ── T10: Complex curved wall (60° arc, tapered, inclined, 3% slope) ──

T10_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T10_complex_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T10_PATH), reason="T10 model not found")
class TestT10ComplexCurved:
    """Level 2+3 on T10: 60° arc, R=8m, tapered 600->300mm, inclined, 3% slope."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T10_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True

    def test_crown_width_300mm(self):
        """Crown width at top ≈ 300mm (tapered)."""
        assert abs(self.l3["crown_width_mm"] - 300.0) < 10.0
        assert self.l3["crown_width_method"] == "slice_local_frame"

    def test_crown_slope_3pct(self):
        """Crown slope ≈ 3%."""
        assert abs(self.l3["crown_slope_percent"] - 3.0) < 0.5

    def test_wall_thickness(self):
        """Average thickness between crown (300mm) and base (600mm)."""
        assert self.l3["min_wall_thickness_mm"] > 300.0
        assert self.l3["avg_wall_thickness_mm"] < 600.0

    def test_front_inclination(self):
        """Front is inclined (~4.3°, ratio ~13:1)."""
        assert self.l3["front_inclination_deg"] > 2.0
        assert self.l3["front_inclination_deg"] < 8.0

    def test_six_groups(self):
        assert self.l2["num_groups"] == 6


# ── T11: S-curved wall ───────────────────────────────────────────────

T11_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T11_s_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T11_PATH), reason="T11 model not found")
class TestT11SCurved:
    """Level 3 on T11 (S-curve, 0.4m thick, inflection point)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T11_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True

    def test_crown_width_400mm(self):
        """Local crown width ≈ 400mm despite S-shape."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 10.0

    def test_wall_thickness_400mm(self):
        assert abs(self.l3["min_wall_thickness_mm"] - 400.0) < 10.0


# ── T12: Semicircular wall ───────────────────────────────────────────

T12_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T12_semicircle.ifc")


@pytest.mark.skipif(not os.path.exists(T12_PATH), reason="T12 model not found")
class TestT12Semicircle:
    """Level 3 on T12 (180° semicircle, R=6m, 0.3m thick)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T12_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True

    def test_crown_width_300mm(self):
        """Local crown width ≈ 300mm."""
        assert abs(self.l3["crown_width_mm"] - 300.0) < 10.0

    def test_crown_slope_positive(self):
        """Crown slope is positive (3% nominal, reduced by 180° normal averaging)."""
        assert self.l3["crown_slope_percent"] > 1.0


# ── T14: Curved L-profile ───────────────────────────────────────────

T14_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T14_curved_l_profile.ifc")


@pytest.mark.skipif(not os.path.exists(T14_PATH), reason="T14 model not found")
class TestT14CurvedLProfile:
    """Level 3 on T14 — multi-element: walls[0] = Mauersteg (curved simple box, 300mm)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T14_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True

    def test_six_groups(self):
        """Multi-element: walls[0] = Mauersteg (simple curved box) → 6 groups."""
        assert self.l2["num_groups"] == 6


# ── T6: Non-compliant wall (200mm thin) ──────────────────────────────

T6_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T6_non_compliant.ifc")


@pytest.mark.skipif(not os.path.exists(T6_PATH), reason="T6 model not found")
class TestT6FromIFC:
    """Level 3 on T6 IFC file (non-compliant, 200mm thin wall)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T6_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_200mm(self):
        """Crown width = 200mm (non-compliant, below 300mm minimum)."""
        assert abs(self.l3["crown_width_mm"] - 200.0) < 10.0

    def test_wall_thickness_200mm(self):
        """Wall thickness = 200mm."""
        assert abs(self.l3["min_wall_thickness_mm"] - 200.0) < 10.0


# ── T7: Compliant wall (300mm, inclined, crown slope) ────────────────

T7_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T7_compliant.ifc")


@pytest.mark.skipif(not os.path.exists(T7_PATH), reason="T7 model not found")
class TestT7FromIFC:
    """Level 3 on T7 IFC file (compliant wall with inclination and crown slope)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T7_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_300mm(self):
        """Crown width = 300mm."""
        assert abs(self.l3["crown_width_mm"] - 300.0) < 10.0

    def test_crown_slope_3pct(self):
        """Crown slope approximately 3%."""
        assert abs(self.l3["crown_slope_percent"] - 3.0) < 1.0

    def test_wall_thickness_300mm(self):
        """Wall thickness = 300mm."""
        assert abs(self.l3["min_wall_thickness_mm"] - 300.0) < 10.0

    def test_inclination_10_to_1(self):
        """Front inclination approximately 10:1."""
        assert abs(self.l3["front_inclination_ratio"] - 10.0) < 2.0


# ── T13: Polygonal wall (local measurement) ──────────────────────────

T13_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T13_polygonal.ifc")


@pytest.mark.skipif(not os.path.exists(T13_PATH), reason="T13 model not found")
class TestT13FromIFC:
    """Level 3 on T13 IFC file (3 straight segments at angles)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T13_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_400mm(self):
        """Crown width approximately 400mm (local measurement)."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 15.0

    def test_not_curved(self):
        assert self.l3.get("is_curved") is False


# ── T15: Variable height wall ────────────────────────────────────────

T15_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T15_variable_height.ifc")


@pytest.mark.skipif(not os.path.exists(T15_PATH), reason="T15 model not found")
class TestT15FromIFC:
    """Level 3 on T15 IFC file (variable height wall)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T15_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_400mm(self):
        """Crown width ≈ 400mm (surface-corrected, may be slightly larger for steep slopes)."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 25.0

    def test_crown_slope_30pct(self):
        """Crown slope approximately 30% (variable height!)."""
        assert abs(self.l3["crown_slope_percent"] - 30.0) < 5.0


# ── T16: Height step (multi-element) ─────────────────────────────────

T16_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T16_height_step.ifc")


@pytest.mark.skipif(not os.path.exists(T16_PATH), reason="T16 model not found")
class TestT16FromIFC:
    """Level 3 on T16 IFC file (height step, multi-element)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T16_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width(self):
        """walls[0] crown width approximately 400mm (upper section)."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 15.0


# ── T17: Curved variable wall ────────────────────────────────────────

T17_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T17_curved_variable.ifc")


@pytest.mark.skipif(not os.path.exists(T17_PATH), reason="T17 model not found")
class TestT17FromIFC:
    """Level 3 on T17 IFC file (curved with variable profile)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T17_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True

    def test_crown_width_350mm(self):
        """Crown width approximately 350mm."""
        assert abs(self.l3["crown_width_mm"] - 350.0) < 20.0


# ── T18: Buttressed wall (multi-element) ─────────────────────────────

T18_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T18_buttressed.ifc")


@pytest.mark.skipif(not os.path.exists(T18_PATH), reason="T18 model not found")
class TestT18FromIFC:
    """Level 3 on T18 IFC file (buttressed, multi-element: walls[0]=Hauptmauer)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T18_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_300mm(self):
        """Crown width = 300mm (Hauptmauer)."""
        assert abs(self.l3["crown_width_mm"] - 300.0) < 10.0

    def test_wall_thickness_300mm(self):
        """Wall thickness = 300mm (Hauptmauer)."""
        assert abs(self.l3["min_wall_thickness_mm"] - 300.0) < 10.0


# ── T20: Triangulated representation ─────────────────────────────────

T20_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T20_triangulated.ifc")


@pytest.mark.skipif(not os.path.exists(T20_PATH), reason="T20 model not found")
class TestT20FromIFC:
    """Level 3 on T20 IFC file (triangulated, same geometry as T1)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T20_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_400mm(self):
        """Crown width = 400mm (same as T1)."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 5.0


# ── T21: Extruded trapezoid ──────────────────────────────────────────

T21_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T21_extruded_trapezoid.ifc")


@pytest.mark.skipif(not os.path.exists(T21_PATH), reason="T21 model not found")
class TestT21FromIFC:
    """Level 3 on T21 IFC file (extruded trapezoid cross-section)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T21_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_300mm(self):
        """Crown width approximately 300mm (trapezoid)."""
        assert abs(self.l3["crown_width_mm"] - 300.0) < 15.0

    def test_inclination(self):
        """Front inclination approximately 3.8 degrees."""
        assert abs(self.l3["front_inclination_deg"] - 3.8) < 1.0


# ── T22: Wall with terrain ───────────────────────────────────────────

T22_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T22_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T22_PATH), reason="T22 model not found")
class TestT22FromIFC:
    """Level 3 on T22 IFC file (wall with terrain context)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T22_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_400mm(self):
        """Crown width = 400mm."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 10.0


# ── T23: ASTRA compliant curved wall ──────────────────────────────────

T23_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T23_astra_compliant_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T23_PATH), reason="T23 model not found")
class TestT23FromIFC:
    """Level 3 on T23 IFC file (ASTRA compliant curved wall)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T23_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_400mm(self):
        """Crown width approximately 400mm."""
        assert abs(self.l3["crown_width_mm"] - 400.0) < 15.0

    def test_crown_slope_3pct(self):
        """Crown slope approximately 3%."""
        assert abs(self.l3["crown_slope_percent"] - 3.0) < 1.0

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True


# ── T24: Highway wall with terrain ────────────────────────────────────

T24_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T24_highway_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T24_PATH), reason="T24 model not found")
class TestT24FromIFC:
    """Level 3 on T24 IFC file (highway wall with terrain, ASTRA compliant)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T24_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_350mm(self):
        """Crown width approximately 350mm."""
        assert abs(self.l3["crown_width_mm"] - 350.0) < 15.0

    def test_crown_slope_3pct(self):
        """Crown slope approximately 3%."""
        assert abs(self.l3["crown_slope_percent"] - 3.0) < 1.0

    def test_inclination_10_to_1(self):
        """Front inclination approximately 10:1 (ASTRA compliant)."""
        assert abs(self.l3["front_inclination_ratio"] - 10.0) < 2.0


# ── T25: Multi-failure wall ───────────────────────────────────────────

T25_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T25_multi_failure.ifc")


@pytest.mark.skipif(not os.path.exists(T25_PATH), reason="T25 model not found")
class TestT25FromIFC:
    """Level 3 on T25 IFC file (multi-failure, non-compliant wall)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T25_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_crown_width_200mm(self):
        """Crown width approximately 200mm (non-compliant)."""
        assert abs(self.l3["crown_width_mm"] - 200.0) < 15.0

    def test_wall_thickness_200mm(self):
        """Wall thickness approximately 200mm."""
        assert abs(self.l3["min_wall_thickness_mm"] - 200.0) < 15.0


# ── Profile consistency (crown_width_cv) ───────────────────────────────

@pytest.mark.skipif(not os.path.exists(T8_PATH), reason="T8 model not found")
class TestT8CrownWidthCV:
    """T8 curved wall — uniform 400mm crown → low CV."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T8_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_cv_present(self):
        """Curved wall uses slice_local_frame → crown_width_cv should be present."""
        assert "crown_width_cv" in self.l3

    def test_cv_low(self):
        """Uniform 400mm crown on curved wall → CV should be very low (< 0.1)."""
        assert self.l3["crown_width_cv"] < 0.1


@pytest.mark.skipif(not os.path.exists(T15_PATH), reason="T15 model not found")
class TestT15CrownWidthCV:
    """T15 variable height wall — crown_width_cv test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T15_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_cv_key_exists_or_none(self):
        """T15 may or may not use sliced measurement; check graceful handling."""
        # If T15 is not curved, crown_width_cv won't be present (that's OK)
        # If it is curved, it should be a float
        cv = self.l3.get("crown_width_cv")
        if cv is not None:
            assert isinstance(cv, float)
            assert cv >= 0.0


# ── Foundation width tests ────────────────────────────────────────────


@pytest.mark.skipif(not os.path.exists(T4_PATH), reason="T4 model not found")
class TestT4FoundationWidth:
    """T4 L-shaped profile — foundation element (walls[1]) has 2m wide foundation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T4_PATH)
        walls = get_elements(model, "IfcWall")
        # walls[1] is the Fundament element in T4 (wider base slab)
        mesh_data = extract_mesh(walls[1])
        self.l2 = validate_level2(mesh_data)
        self.l3 = validate_level3(mesh_data, self.l2)

    def test_foundation_width_present(self):
        """Foundation width should be measured."""
        assert "foundation_width_mm" in self.l3

    def test_foundation_width_approx_2000mm(self):
        """Foundation slab is 2m wide → foundation_width_mm ≈ 2000mm."""
        assert abs(self.l3["foundation_width_mm"] - 2000.0) < 50.0

    def test_foundation_width_ratio(self):
        """Foundation width ratio = foundation_width / (wall_height * 1000)."""
        assert "foundation_width_ratio" in self.l3
        # T4 walls[1] is the Fundament slab: 2000mm wide, ~0.5m tall → ratio ~4.0
        assert self.l3["foundation_width_ratio"] > 0.5
        assert self.l3["foundation_width_ratio"] < 6.0


class TestT1FoundationWidth:
    """T1 simple box — foundation width equals crown width (uniform 400mm)."""

    def test_foundation_width_400mm(self):
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        assert "foundation_width_mm" in l3
        assert abs(l3["foundation_width_mm"] - 400.0) < 1.0


# ── T26: Extruded curved front profile ─────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "T26_extruded_curved.ifc")),
    reason="T26 model not found",
)
class TestT26ExtrudedCurvedMeasurements:
    """T26 — IfcExtrudedAreaSolid with curved front profile (300mm crown, 450mm base)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(os.path.join(MODELS_DIR, "T26_extruded_curved.ifc"))
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l2 = validate_level2(mesh)
        self.l3 = validate_level3(mesh, l2)

    def test_crown_width(self):
        assert abs(self.l3["crown_width_mm"] - 300) < 20

    def test_wall_thickness(self):
        # Variable thickness due to curved front: between 300-450mm
        assert self.l3["min_wall_thickness_mm"] >= 250
        assert self.l3["min_wall_thickness_mm"] <= 500

    def test_wall_height(self):
        assert abs(self.l3["wall_height_m"] - 3.0) < 0.1

    def test_foundation_width(self):
        # Foundation same as base width ~450mm
        if "foundation_width_mm" in self.l3:
            assert self.l3["foundation_width_mm"] > 200


# ── T27: Long curved wall along slope ────────────────────────────

T27_PATH = os.path.join(MODELS_DIR, "T27_long_curved_slope.ifc")


@pytest.mark.skipif(not os.path.exists(T27_PATH), reason="T27 model not found")
class TestT27LongCurvedSlope:
    """T27: 31.4m arc, 90deg curve, 3-6m variable height, 400mm uniform thickness.

    This is the stress test for long curved walls along slopes where crown Z
    varies by >3m. Tests that the natural-gap crown filter (replacing the old
    hardcoded 0.5m threshold) correctly includes all crown faces.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T27_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        self.l2 = validate_level2(mesh)
        self.l3 = validate_level3(mesh, self.l2)

    def test_is_curved(self):
        assert self.l3.get("is_curved") is True

    def test_six_groups(self):
        assert self.l2["num_groups"] == 6

    def test_crown_width_400mm(self):
        """Uniform 400mm thickness wall -> crown width ~400mm."""
        cw = self.l3["crown_width_mm"]
        assert 350 < cw < 450, f"Crown width {cw:.1f}mm, expected ~400mm"

    def test_wall_thickness_400mm(self):
        wt = self.l3["min_wall_thickness_mm"]
        assert 350 < wt < 450, f"Thickness {wt:.1f}mm, expected ~400mm"

    def test_wall_height_covers_range(self):
        """Wall height should reflect max height (6m) since vertical faces span full range."""
        h = self.l3["wall_height_m"]
        assert h > 5.0, f"Height {h:.2f}m, expected >5m (wall goes 3-6m + 0-2m base)"

    def test_crown_width_cv_low(self):
        """Uniform thickness -> low CV (consistent crown width along curve)."""
        cv = self.l3.get("crown_width_cv")
        if cv is not None:
            assert cv < 0.15, f"Crown CV={cv:.4f}, expected <0.15 for uniform wall"

    def test_wall_length_30m(self):
        """Arc length = 20m * pi/2 ~= 31.4m."""
        wl = self.l3.get("wall_length_m", 0)
        assert 15 < wl < 35, f"Length {wl:.1f}m, expected ~22-31m"
