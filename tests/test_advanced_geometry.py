"""Tests for advanced geometric analysis."""

import numpy as np
import pytest
import math

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.core.advanced_geometry import (
    compute_taper_profile,
    compute_planarity,
    check_overlap,
    compute_profile_variation,
    check_plumbness,
)


class TestTaperProfile:
    """Wall taper (Anzug) analysis."""

    def test_t7_10_to_1_taper(self):
        """T7 has 10:1 inclination → taper ratio ≈ 10."""
        model = load_model("tests/test_models/T7_compliant.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        taper = compute_taper_profile(mesh, l2["face_groups"],
                                       np.array(l2["wall_axis"]))
        assert taper["is_tapered"]
        assert taper["taper_ratio"] == pytest.approx(10.0, rel=0.15)
        assert taper["min_thickness_mm"] < taper["max_thickness_mm"]

    def test_t1_no_taper(self):
        """T1 simple box has no taper (constant thickness)."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        taper = compute_taper_profile(mesh, l2["face_groups"],
                                       np.array(l2["wall_axis"]))
        assert not taper["is_tapered"]
        assert abs(taper["min_thickness_mm"] - taper["max_thickness_mm"]) < 1

    def test_t28_taper(self):
        """T28 showcase has 10:1 taper."""
        model = load_model("tests/test_models/T28_showcase.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        taper = compute_taper_profile(mesh, l2["face_groups"],
                                       np.array(l2["wall_axis"]))
        assert taper["is_tapered"]
        assert taper["min_thickness_mm"] == pytest.approx(350, rel=0.1)
        assert taper["max_thickness_mm"] == pytest.approx(700, rel=0.1)


class TestPlanarity:
    """Surface planarity analysis."""

    def test_t1_front_is_planar(self):
        """T1 box front face is perfectly planar."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        plan = compute_planarity(mesh, l2["face_groups"], "front")
        assert plan["is_planar"]
        assert plan["rms_deviation_mm"] < 0.01

    def test_t8_curved_not_planar(self):
        """T8 curved wall front face is not planar (it's curved)."""
        model = load_model("tests/test_models/T8_curved_wall.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        plan = compute_planarity(mesh, l2["face_groups"], "front")
        assert not plan["is_planar"]
        assert plan["rms_deviation_mm"] > 10  # significant deviation


class TestOverlap:
    """Overlap / collision detection."""

    def test_separated_no_overlap(self):
        """Two separated boxes should not overlap."""
        def _box_mesh(origin, size):
            x0, y0, z0 = origin
            x1, y1, z1 = x0+size[0], y0+size[1], z0+size[2]
            v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                          [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
            return {"vertices": v, "faces": np.zeros((0, 3), dtype=int)}

        a = _box_mesh([0, 0, 0], [1, 1, 1])
        b = _box_mesh([5, 5, 5], [1, 1, 1])
        result = check_overlap(a, b)
        assert result["clear"]
        assert not result["aabb_overlap"]

    def test_overlapping_boxes(self):
        """Two overlapping boxes should detect overlap."""
        def _box_mesh(origin, size):
            x0, y0, z0 = origin
            x1, y1, z1 = x0+size[0], y0+size[1], z0+size[2]
            v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                          [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
            return {"vertices": v, "faces": np.zeros((0, 3), dtype=int)}

        a = _box_mesh([0, 0, 0], [2, 2, 2])
        b = _box_mesh([1, 1, 1], [2, 2, 2])
        result = check_overlap(a, b)
        assert result["aabb_overlap"]
        assert result["overlap_volume_m3"] > 0


class TestProfileVariation:
    """Cross-section profile variation analysis."""

    def test_t1_constant_profile(self):
        """T1 box has constant profile along its length."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        prof = compute_profile_variation(mesh, l2.get("centerline"), l2["face_groups"])
        if prof:
            assert prof["width_cv"] < 0.01
            assert not prof["is_variable"]

    def test_t15_variable_height(self):
        """T15 has variable height → height CV > 0."""
        model = load_model("tests/test_models/T15_variable_height.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        prof = compute_profile_variation(mesh, l2.get("centerline"), l2["face_groups"])
        if prof:
            assert prof["height_cv"] > 0.01


class TestPlumbness:
    """Plumbness (verticality) check."""

    def test_t1_is_plumb(self):
        """T1 box walls are perfectly plumb (vertical)."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        plumb = check_plumbness(l2["face_groups"])
        assert plumb["is_plumb"]
        assert plumb["max_deviation_deg"] < 0.1

    def test_t7_inclined_not_plumb(self):
        """T7 has 10:1 inclination → front not perfectly plumb."""
        model = load_model("tests/test_models/T7_compliant.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        plumb = check_plumbness(l2["face_groups"])
        front_dev = plumb.get("front_plumbness_deg", 0)
        # 10:1 → arctan(1/10) ≈ 5.7°
        assert front_dev == pytest.approx(5.7, abs=0.5)
        assert not plumb["is_plumb"]
