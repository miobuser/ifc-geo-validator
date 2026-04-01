"""Tests for Level 1: Geometric properties (Volume, Area, BBox, Centroid).

Reference values computed analytically for known test geometries.
"""

import os
import pytest
import numpy as np

from ifc_geo_validator.core.geometry import (
    compute_volume,
    compute_total_area,
    compute_bbox,
    compute_centroid,
)
from ifc_geo_validator.core.mesh_converter import _check_watertight
from ifc_geo_validator.validation.level1 import validate_level1


# ── Unit cube test data ──────────────────────────────────────────────

UNIT_CUBE_VERTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
], dtype=float)

UNIT_CUBE_FACES = np.array([
    [0, 2, 1], [0, 3, 2],  # bottom (z=0, outward = -z)
    [4, 5, 6], [4, 6, 7],  # top (z=1, outward = +z)
    [0, 1, 5], [0, 5, 4],  # front (y=0)
    [2, 3, 7], [2, 7, 6],  # back (y=1)
    [0, 4, 7], [0, 7, 3],  # left (x=0)
    [1, 2, 6], [1, 6, 5],  # right (x=1)
])


def _compute_areas(verts, faces):
    """Helper: compute per-face areas."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


class TestSimpleBox:
    """Test against T1: Simple box retaining wall (8.0 x 0.4 x 3.0 m)."""

    EXPECTED_VOLUME = 9.6       # m³
    EXPECTED_AREA = 56.8        # m²
    EXPECTED_BBOX = [8.0, 0.4, 3.0]  # m
    TOLERANCE = 0.001           # 1mm tolerance

    # Construct a box mesh at origin with dimensions 8.0 x 0.4 x 3.0
    VERTS = np.array([
        [0, 0, 0], [8, 0, 0], [8, 0.4, 0], [0, 0.4, 0],
        [0, 0, 3], [8, 0, 3], [8, 0.4, 3], [0, 0.4, 3],
    ], dtype=float)

    FACES = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ])

    AREAS = _compute_areas(VERTS, FACES)

    def test_volume(self):
        volume = compute_volume(self.VERTS, self.FACES)
        assert abs(volume - self.EXPECTED_VOLUME) < self.TOLERANCE

    def test_area(self):
        area = compute_total_area(self.AREAS)
        assert abs(area - self.EXPECTED_AREA) < self.TOLERANCE

    def test_bbox(self):
        bbox = compute_bbox(self.VERTS)
        for actual, expected in zip(bbox["size"], self.EXPECTED_BBOX):
            assert abs(actual - expected) < self.TOLERANCE

    def test_centroid(self):
        centroid = compute_centroid(self.VERTS, self.FACES, self.AREAS)
        expected = np.array([4.0, 0.2, 1.5])
        np.testing.assert_allclose(centroid, expected, atol=self.TOLERANCE)


T2_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T2_inclined_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T2_PATH), reason="T2 model not found")
class TestInclinedWall:
    """Test against T2: Inclined retaining wall (trapezoid, 10:1 Anzug)."""

    EXPECTED_VOLUME = 12.0
    EXPECTED_AREA = 59.12
    EXPECTED_BBOX = [8.0, 0.65, 3.0]
    TOLERANCE = 0.01

    def test_full_pipeline(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T2_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - self.EXPECTED_VOLUME) < self.TOLERANCE
        assert abs(result["total_area"] - self.EXPECTED_AREA) < self.TOLERANCE
        bbox_size = result["bbox"]["size"]
        for actual, expected in zip(sorted(bbox_size), sorted(self.EXPECTED_BBOX)):
            assert abs(actual - expected) < self.TOLERANCE


# ── Standalone math tests (no IFC dependency) ───────────────────────

class TestVolumeComputation:
    """Test volume computation on known numpy arrays."""

    def test_unit_cube_volume(self):
        """A unit cube should have volume 1.0."""
        volume = compute_volume(UNIT_CUBE_VERTS, UNIT_CUBE_FACES)
        assert abs(volume - 1.0) < 1e-10

    def test_triangle_area(self):
        """A right triangle with legs 3 and 4 should have area 6.0."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([3.0, 0.0, 0.0])
        v2 = np.array([0.0, 4.0, 0.0])
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        assert abs(area - 6.0) < 1e-10


class TestBoundingBox:
    """Test bounding box computation."""

    def test_unit_cube_bbox(self):
        bbox = compute_bbox(UNIT_CUBE_VERTS)
        assert bbox["min"] == [0.0, 0.0, 0.0]
        assert bbox["max"] == [1.0, 1.0, 1.0]
        assert bbox["size"] == [1.0, 1.0, 1.0]


class TestCentroid:
    """Test area-weighted centroid."""

    def test_unit_cube_centroid(self):
        areas = _compute_areas(UNIT_CUBE_VERTS, UNIT_CUBE_FACES)
        centroid = compute_centroid(UNIT_CUBE_VERTS, UNIT_CUBE_FACES, areas)
        np.testing.assert_allclose(centroid, [0.5, 0.5, 0.5], atol=1e-10)


class TestWatertight:
    """Test watertight check."""

    def test_unit_cube_is_watertight(self):
        assert _check_watertight(UNIT_CUBE_FACES) is True

    def test_open_mesh_is_not_watertight(self):
        # Remove last two triangles (one face open)
        open_faces = UNIT_CUBE_FACES[:10]
        assert _check_watertight(open_faces) is False


class TestValidateLevel1:
    """Integration test for the full Level 1 pipeline."""

    def test_unit_cube_pipeline(self):
        areas = _compute_areas(UNIT_CUBE_VERTS, UNIT_CUBE_FACES)
        v0 = UNIT_CUBE_VERTS[UNIT_CUBE_FACES[:, 0]]
        v1 = UNIT_CUBE_VERTS[UNIT_CUBE_FACES[:, 1]]
        v2 = UNIT_CUBE_VERTS[UNIT_CUBE_FACES[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals = cross / norms

        mesh_data = {
            "vertices": UNIT_CUBE_VERTS,
            "faces": UNIT_CUBE_FACES,
            "normals": normals,
            "areas": areas,
            "is_watertight": True,
        }

        result = validate_level1(mesh_data)

        assert abs(result["volume"] - 1.0) < 1e-10
        assert abs(result["total_area"] - 6.0) < 1e-10
        assert result["bbox"]["size"] == [1.0, 1.0, 1.0]
        np.testing.assert_allclose(result["centroid"], [0.5, 0.5, 0.5], atol=1e-10)
        assert result["is_watertight"] is True
        assert result["num_triangles"] == 12
        assert result["num_vertices"] == 8


# ── IFC file test (runs only if test model exists) ───────────────────

T1_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T1_simple_box.ifc")


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 test model not found")
class TestT1FromIFC:
    """Test Level 1 against actual T1 IFC file."""

    EXPECTED_VOLUME = 9.6
    EXPECTED_AREA = 56.8
    EXPECTED_BBOX = [8.0, 0.4, 3.0]
    TOLERANCE = 0.01

    def test_full_pipeline(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T1_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) > 0

        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - self.EXPECTED_VOLUME) < self.TOLERANCE
        assert abs(result["total_area"] - self.EXPECTED_AREA) < self.TOLERANCE
        bbox_size = result["bbox"]["size"]
        for actual, expected in zip(sorted(bbox_size), sorted(self.EXPECTED_BBOX)):
            assert abs(actual - expected) < self.TOLERANCE


T4_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T4_l_shaped.ifc")
T5_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T5_t_shaped.ifc")


@pytest.mark.skipif(not os.path.exists(T4_PATH), reason="T4 model not found")
class TestT4FromIFC:
    """Test Level 1 against T4 IFC file (L-shaped)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T4_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - 4.50) < 0.1
        assert result["is_watertight"] is True
        bbox_size = sorted(result["bbox"]["size"])
        assert abs(bbox_size[0] - 0.3) < 0.05
        assert abs(bbox_size[1] - 2.5) < 0.1
        assert abs(bbox_size[2] - 6.0) < 0.1


@pytest.mark.skipif(not os.path.exists(T5_PATH), reason="T5 model not found")
class TestT5FromIFC:
    """Test Level 1 against T5 IFC file (T-shaped with spur)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T5_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - 9.60) < 0.1
        assert result["is_watertight"] is True
        bbox_size = sorted(result["bbox"]["size"])
        assert abs(bbox_size[0] - 0.4) < 0.05
        assert abs(bbox_size[1] - 3.0) < 0.1
        assert abs(bbox_size[2] - 8.0) < 0.1


# ── T8-T10: Curved and stepped wall models ──────────────────────────

T8_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T8_curved_wall.ifc")
T9_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T9_stepped_wall.ifc")
T10_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T10_complex_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T8_PATH), reason="T8 model not found")
class TestT8FromIFC:
    """Test Level 1 against T8 IFC file (90° curved wall, R=10m)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T8_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - 19.245) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T9_PATH), reason="T9 model not found")
class TestT9FromIFC:
    """Test Level 1 against T9 IFC file (stepped wall, 300/600mm)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T9_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - 3.6) < 0.1
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T10_PATH), reason="T10 model not found")
class TestT10FromIFC:
    """Test Level 1 against T10 IFC file (complex curved, 60° arc, tapered)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T10_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        result = validate_level1(mesh_data)

        assert abs(result["volume"] - 15.53) < 0.5
        assert result["is_watertight"] is True


T11_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T11_s_curved.ifc")
T12_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T12_semicircle.ifc")
T13_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T13_polygonal.ifc")
T14_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T14_curved_l_profile.ifc")


@pytest.mark.skipif(not os.path.exists(T11_PATH), reason="T11 model not found")
class TestT11FromIFC:
    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T11_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 21.83) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T12_PATH), reason="T12 model not found")
class TestT12FromIFC:
    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T12_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 17.37) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T13_PATH), reason="T13 model not found")
class TestT13FromIFC:
    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T13_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 12.99) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T14_PATH), reason="T14 model not found")
class TestT14FromIFC:
    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T14_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 7.15) < 0.5
        assert result["is_watertight"] is True


# ── T3, T6, T7: Crown slope, non-compliant, compliant ────────────────

T3_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T3_crown_slope.ifc")
T6_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T6_non_compliant.ifc")
T7_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T7_compliant.ifc")


@pytest.mark.skipif(not os.path.exists(T3_PATH), reason="T3 model not found")
class TestT3FromIFC:
    """Test Level 1 against T3 IFC file (crown slope 3%)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T3_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 7.21) < 0.1
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T6_PATH), reason="T6 model not found")
class TestT6FromIFC:
    """Test Level 1 against T6 IFC file (non-compliant thin wall)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T6_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 2.0) < 0.1
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T7_PATH), reason="T7 model not found")
class TestT7FromIFC:
    """Test Level 1 against T7 IFC file (norm-compliant 10:1 Anzug, 3% slope)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T7_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 10.81) < 0.1
        assert result["is_watertight"] is True


# ── T15-T18: Variable height, height step, curved variable, buttressed ──

T15_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T15_variable_height.ifc")
T16_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T16_height_step.ifc")
T17_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T17_curved_variable.ifc")
T18_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T18_buttressed.ifc")


@pytest.mark.skipif(not os.path.exists(T15_PATH), reason="T15 model not found")
class TestT15FromIFC:
    """Test Level 1 against T15 IFC file (variable height wall)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T15_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        # 10m × 0.4m × avg 3.5m = 14.0 m³
        assert abs(result["volume"] - 14.0) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T16_PATH), reason="T16 model not found")
class TestT16FromIFC:
    """Test Level 1 against T16 IFC file (height step 4.0m / 2.5m)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T16_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 13.0) < 0.5


@pytest.mark.skipif(not os.path.exists(T17_PATH), reason="T17 model not found")
class TestT17FromIFC:
    """Test Level 1 against T17 IFC file (curved wall with variable height)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T17_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 13.64) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T18_PATH), reason="T18 model not found")
class TestT18FromIFC:
    """Test Level 1 against T18 IFC file (buttressed wall, multi-element)."""

    def test_hauptmauer_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T18_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) >= 1
        # walls[0] is Hauptmauer
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 10.8) < 0.5
        assert result["is_watertight"] is True

    def test_multi_element(self):
        """T18 should contain multiple wall elements (Hauptmauer + Strebepfeiler)."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        model = load_model(T18_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) == 4


# ── T20-T22: TriangulatedFaceSet, extruded trapezoid, with terrain ───

T20_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T20_triangulated.ifc")
T21_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T21_extruded_trapezoid.ifc")
T22_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T22_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T20_PATH), reason="T20 model not found")
class TestT20FromIFC:
    """Test Level 1 against T20 IFC file (IfcTriangulatedFaceSet, same as T1)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T20_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 9.6) < 0.1
        assert result["is_watertight"] is True

    @pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 model not found")
    def test_same_as_t1(self):
        """IfcTriangulatedFaceSet must produce identical results to IfcFacetedBrep."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model_t1 = load_model(T1_PATH)
        walls_t1 = get_elements(model_t1, "IfcWall")
        result_t1 = validate_level1(extract_mesh(walls_t1[0]))

        model_t20 = load_model(T20_PATH)
        walls_t20 = get_elements(model_t20, "IfcWall")
        result_t20 = validate_level1(extract_mesh(walls_t20[0]))

        assert abs(result_t20["volume"] - result_t1["volume"]) < 0.01
        assert abs(result_t20["total_area"] - result_t1["total_area"]) < 0.01
        assert result_t20["is_watertight"] == result_t1["is_watertight"]


@pytest.mark.skipif(not os.path.exists(T21_PATH), reason="T21 model not found")
class TestT21FromIFC:
    """Test Level 1 against T21 IFC file (extruded trapezoid profile, 10:1)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T21_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 10.8) < 0.5
        assert result["is_watertight"] is True


@pytest.mark.skipif(not os.path.exists(T22_PATH), reason="T22 model not found")
class TestT22FromIFC:
    """Test Level 1 against T22 IFC file (simple box with terrain)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T22_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 9.6) < 0.1
        assert result["is_watertight"] is True

    def test_terrain_available(self):
        """T22 should have extractable terrain geometry."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_terrain_mesh
        model = load_model(T22_PATH)
        terrain = get_terrain_mesh(model)
        assert terrain is not None
        assert terrain["vertices"].shape[0] > 0
        assert terrain["faces"].shape[0] > 0


# ── T23-T25: ASTRA compliant curved, highway with terrain, multi-failure ──

T23_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T23_astra_compliant_curved.ifc")
T24_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T24_highway_with_terrain.ifc")
T25_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T25_multi_failure.ifc")


@pytest.mark.skipif(not os.path.exists(T23_PATH), reason="T23 model not found")
class TestT23FromIFC:
    """Test Level 1 against T23 IFC file (ASTRA compliant curved wall)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T23_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 24.0) < 0.5
        assert result["is_watertight"] is True

    def test_multi_element(self):
        """T23 should contain 3 wall elements."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        model = load_model(T23_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) == 3


@pytest.mark.skipif(not os.path.exists(T24_PATH), reason="T24 model not found")
class TestT24FromIFC:
    """Test Level 1 against T24 IFC file (highway wall with terrain)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T24_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 22.1) < 0.5
        assert result["is_watertight"] is True

    def test_multi_element(self):
        """T24 should contain 2 wall elements."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        model = load_model(T24_PATH)
        walls = get_elements(model, "IfcWall")
        assert len(walls) == 2

    def test_terrain_available(self):
        """T24 should have extractable terrain geometry."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_terrain_mesh
        model = load_model(T24_PATH)
        terrain = get_terrain_mesh(model)
        assert terrain is not None
        assert terrain["vertices"].shape[0] > 0
        assert terrain["faces"].shape[0] > 0


@pytest.mark.skipif(not os.path.exists(T25_PATH), reason="T25 model not found")
class TestT25FromIFC:
    """Test Level 1 against T25 IFC file (multi-failure, non-compliant wall)."""

    def test_volume_and_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T25_PATH)
        walls = get_elements(model, "IfcWall")
        result = validate_level1(extract_mesh(walls[0]))
        assert abs(result["volume"] - 2.4) < 0.5
        assert result["is_watertight"] is True


# ── T26: Extruded curved front profile ────────────────────────────────

T26_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T26_extruded_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T26_PATH), reason="T26 model not found")
class TestT26ExtrudedCurved:
    """Test Level 1 against T26 IFC file (extruded profile with curved front)."""

    def test_volume_positive(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T26_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l1 = validate_level1(mesh_data)
        assert l1["volume"] > 0

    def test_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T26_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l1 = validate_level1(mesh_data)
        assert l1["is_watertight"] is True


# ── T27: Long curved wall along slope ─────────────────────────────

T27_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T27_long_curved_slope.ifc")


@pytest.mark.skipif(not os.path.exists(T27_PATH), reason="T27 model not found")
class TestT27LongCurvedSlope:
    """T27: 31m curved wall along slope — stress test for variable Z crown."""

    def test_volume_positive(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T27_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l1 = validate_level1(mesh_data)
        assert l1["volume"] > 30, f"Volume {l1['volume']:.1f}, expected >30m3 for 31m wall"

    def test_watertight(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        model = load_model(T27_PATH)
        walls = get_elements(model, "IfcWall")
        mesh_data = extract_mesh(walls[0])
        l1 = validate_level1(mesh_data)
        assert l1["is_watertight"] is True
