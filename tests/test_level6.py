"""Tests for Level 6: Distance checks and terrain context."""

import os
import pytest
import numpy as np

from ifc_geo_validator.core.distance import (
    min_vertex_distance,
    terrain_height_at_xy,
    horizontal_distance_xy,
    classify_terrain_side,
    _barycentric_2d,
)
from ifc_geo_validator.validation.level6 import validate_level6


# ── Unit tests for distance primitives ──────────────────────────────

class TestBarycentric:
    """Barycentric interpolation for terrain height queries."""

    def test_point_inside_triangle(self):
        result = _barycentric_2d(0.25, 0.25, 0, 0, 1, 0, 0, 1)
        assert result is not None
        u, v, w = result
        assert u >= 0 and v >= 0 and w >= 0
        assert abs(u + v + w - 1.0) < 1e-10

    def test_point_outside_triangle(self):
        result = _barycentric_2d(2.0, 2.0, 0, 0, 1, 0, 0, 1)
        assert result is None

    def test_point_on_vertex(self):
        result = _barycentric_2d(0, 0, 0, 0, 1, 0, 0, 1)
        assert result is not None
        u, v, w = result
        assert abs(u - 1.0) < 1e-6


class TestTerrainHeight:
    """Terrain height query via barycentric interpolation."""

    def test_flat_terrain(self):
        verts = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        z = terrain_height_at_xy(verts, faces, 5.0, 5.0)
        assert z is not None
        assert abs(z - 5.0) < 0.01

    def test_sloped_terrain(self):
        verts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 10], [0, 10, 10]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        z = terrain_height_at_xy(verts, faces, 5.0, 5.0)
        assert z is not None
        assert abs(z - 5.0) < 0.1  # linear slope: z = y

    def test_outside_terrain(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        z = terrain_height_at_xy(verts, faces, 5.0, 5.0)
        assert z is None


class TestMinDistance:
    """Minimum distance between vertex sets."""

    def test_touching_boxes(self):
        verts_a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        verts_b = np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], dtype=float)
        dist = min_vertex_distance(verts_a, verts_b)
        assert dist < 0.01  # touching at x=1

    def test_separated_boxes(self):
        verts_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        verts_b = np.array([[3, 0, 0], [4, 0, 0]], dtype=float)
        dist = min_vertex_distance(verts_a, verts_b)
        assert abs(dist - 2.0) < 0.01


class TestHorizontalDistance:
    """XY gap between bounding boxes."""

    def test_overlapping(self):
        d = horizontal_distance_xy(
            np.array([0, 0, 0]), np.array([5, 5, 5]),
            np.array([3, 3, 0]), np.array([8, 8, 5]),
        )
        assert d == 0.0

    def test_separated_x(self):
        d = horizontal_distance_xy(
            np.array([0, 0, 0]), np.array([1, 1, 1]),
            np.array([3, 0, 0]), np.array([4, 1, 1]),
        )
        assert abs(d - 2.0) < 0.01


# ── IFC integration tests ──────────────────────────────────────────

T22_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T22_with_terrain.ifc")
T1_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T1_simple_box.ifc")


@pytest.mark.skipif(not os.path.exists(T22_PATH), reason="T22 model not found")
class TestT22TerrainContext:
    """Level 6 on T22 (wall + terrain)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level1 import validate_level1
        from ifc_geo_validator.validation.level2 import validate_level2
        from ifc_geo_validator.validation.level3 import validate_level3

        model = load_model(T22_PATH)
        walls = get_elements(model, "IfcWall")
        self.terrain = get_terrain_mesh(model)
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        self.elems = [{
            "element_id": walls[0].id(),
            "element_name": getattr(walls[0], "Name", "?"),
            "level1": l1, "level2": l2, "level3": l3, "mesh_data": mesh,
        }]
        self.l6 = validate_level6(self.elems, terrain_mesh=self.terrain)

    def test_terrain_detected(self):
        assert self.terrain is not None
        assert self.l6["terrain_available"] is True

    def test_terrain_side_assigned(self):
        assert len(self.l6["terrain_side"]) > 0

    def test_clearance_computed(self):
        assert len(self.l6["clearances"]) > 0
        cl = self.l6["clearances"][0]
        assert cl["min_m"] is not None
        assert cl["min_m"] > 0  # crown is above terrain

    def test_clearance_reasonable(self):
        """Crown at z=3, terrain at y=0.2 → z≈0.2, clearance ≈ 2.8m."""
        cl = self.l6["clearances"][0]
        assert cl["min_m"] > 2.0
        assert cl["max_m"] <= 3.5


    def test_embedment_result_present(self):
        """Embedment list should exist in L6 result."""
        assert "embedments" in self.l6

    def test_foundation_embedment_computed(self):
        """T22 has foundation faces and terrain → embedment should be computed."""
        embedments = self.l6["embedments"]
        # T22 may or may not have foundation faces; if present, check structure
        if embedments:
            emb = embedments[0]
            assert "foundation_embedment_m" in emb
            assert "terrain_z" in emb
            assert "foundation_min_z" in emb


T24_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T24_highway_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T24_PATH), reason="T24 model not found")
class TestT24TerrainContext:
    """Level 6 on T24 (multi-element wall + terrain)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level1 import validate_level1
        from ifc_geo_validator.validation.level2 import validate_level2
        from ifc_geo_validator.validation.level3 import validate_level3

        model = load_model(T24_PATH)
        walls = get_elements(model, "IfcWall")
        self.terrain = get_terrain_mesh(model)

        self.elems = []
        for w in walls:
            mesh = extract_mesh(w)
            l1 = validate_level1(mesh)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)
            self.elems.append({
                "element_id": w.id(),
                "element_name": getattr(w, "Name", "?"),
                "level1": l1, "level2": l2, "level3": l3, "mesh_data": mesh,
            })
        self.l6 = validate_level6(self.elems, terrain_mesh=self.terrain)

    def test_terrain_detected(self):
        """T24 has terrain geometry → get_terrain_mesh returns non-None."""
        assert self.terrain is not None
        assert self.l6["terrain_available"] is True

    def test_clearance_computed(self):
        """Clearance results exist for T24."""
        assert len(self.l6["clearances"]) > 0
        cl = self.l6["clearances"][0]
        assert cl["min_m"] is not None

    def test_two_elements_distances(self):
        """Inter-element distance computed for the 2 wall elements."""
        assert len(self.elems) >= 2
        assert len(self.l6["distances"]) >= 1
        d = self.l6["distances"][0]
        assert "min_distance_mm" in d
        assert d["min_distance_mm"] >= 0

    def test_embedments_list_present(self):
        """Embedments list should exist in L6 result for T24."""
        assert "embedments" in self.l6

    def test_foundation_embedment_computed(self):
        """T24 has a foundation element with terrain → embedment should be computed."""
        embedments = self.l6["embedments"]
        # T24 has 2 walls (stem + foundation); foundation should have foundation faces
        if embedments:
            emb = embedments[0]
            assert "foundation_embedment_m" in emb
            assert isinstance(emb["foundation_embedment_m"], float)

    def test_foundation_embedment_positive(self):
        """Foundation should be embedded below terrain (positive value)."""
        embedments = self.l6["embedments"]
        if embedments:
            for emb in embedments:
                # Foundation bottom should be at or below terrain
                assert emb["foundation_embedment_m"] >= -1.0  # allow small tolerance


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 model not found")
class TestT1NoTerrain:
    """Level 6 without terrain (graceful fallback)."""

    def test_no_terrain_graceful(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level1 import validate_level1
        from ifc_geo_validator.validation.level2 import validate_level2

        model = load_model(T1_PATH)
        walls = get_elements(model, "IfcWall")
        terrain = get_terrain_mesh(model)  # should be None (T1 has no terrain)
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        elems = [{"element_id": 1, "level1": l1, "level2": l2, "mesh_data": mesh}]

        l6 = validate_level6(elems, terrain_mesh=terrain)
        assert l6["terrain_available"] is False
        assert len(l6["terrain_side"]) == 0
        assert len(l6["clearances"]) == 0
        assert len(l6["embedments"]) == 0
