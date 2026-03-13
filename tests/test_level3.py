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
