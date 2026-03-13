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

        assert abs(result["volume"] - 10.5) < 0.01
        assert result["is_watertight"] is True
        bbox_size = sorted(result["bbox"]["size"])
        assert abs(bbox_size[0] - 2.0) < 0.01
        assert abs(bbox_size[1] - 3.0) < 0.01
        assert abs(bbox_size[2] - 6.0) < 0.01


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

        assert abs(result["volume"] - 10.725) < 0.01
        assert result["is_watertight"] is True
        bbox_size = sorted(result["bbox"]["size"])
        assert abs(bbox_size[0] - 1.9) < 0.01
        assert abs(bbox_size[1] - 3.0) < 0.01
        assert abs(bbox_size[2] - 8.0) < 0.01
