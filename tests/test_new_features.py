"""Tests for cross-section, smart rule filtering, and HTML report.

Covers features added in the production-hardening phase.
"""

import os
import numpy as np
import pytest

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.viz.cross_section import extract_cross_section
from ifc_geo_validator.report.html_report import generate_html_report


RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "astra_fhb_stuetzmauer.yaml",
)

T28_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T28_showcase.ifc")
T1_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T1_simple_box.ifc")


# ── Helpers ───────────────────────────────────────────────────────

def _make_mesh(verts, faces):
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return {"vertices": verts, "faces": faces, "normals": cross / norms,
            "areas": areas, "is_watertight": True}


def _box(o, s):
    x0, y0, z0 = o
    x1, y1, z1 = o[0]+s[0], o[1]+s[1], o[2]+s[2]
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5]])
    return v, f


# ── Cross-Section Tests ──────────────────────────────────────────

class TestCrossSection:
    """Test cross-section extraction at various positions."""

    def test_t1_mid_section(self):
        """T1 simple box: cross-section at 50% should have correct dims."""
        model = load_model(T1_PATH)
        elem = get_elements(model, "IfcWall")[0]
        mesh = extract_mesh(elem)
        l2 = validate_level2(mesh)
        section = extract_cross_section(mesh, l2.get("centerline"), 0.5)
        assert section is not None
        assert abs(section["width_mm"] - 400) < 50  # ~400mm wide
        assert abs(section["height_m"] - 3.0) < 0.1  # ~3m tall

    def test_t28_sections_consistent(self):
        """T28 curved wall: sections at different positions should be similar."""
        model = load_model(T28_PATH)
        elem = get_elements(model, "IfcWall")[0]  # Stem
        mesh = extract_mesh(elem)
        l2 = validate_level2(mesh)
        cl = l2.get("centerline")

        widths = []
        for frac in [0.2, 0.4, 0.6, 0.8]:
            s = extract_cross_section(mesh, cl, frac)
            if s:
                widths.append(s["width_mm"])

        assert len(widths) >= 3
        # Width should be relatively consistent (~700mm ± 10%)
        mean_w = np.mean(widths)
        for w in widths:
            assert abs(w - mean_w) < mean_w * 0.15, (
                f"Width {w:.0f}mm deviates >15% from mean {mean_w:.0f}mm"
            )

    def test_no_centerline_returns_none(self):
        """Without centerline, should return None."""
        mesh = _make_mesh(*_box([0, 0, 0], [8, 0.4, 3]))
        section = extract_cross_section(mesh, None, 0.5)
        assert section is None

    def test_section_has_outline(self):
        """Section should contain a closed polygon outline."""
        model = load_model(T1_PATH)
        elem = get_elements(model, "IfcWall")[0]
        mesh = extract_mesh(elem)
        l2 = validate_level2(mesh)
        section = extract_cross_section(mesh, l2.get("centerline"), 0.5)
        if section is not None:
            outline = section["outline_2d"]
            assert len(outline) >= 4  # at least a quad
            # Check it's closed (first ≈ last point)
            assert np.linalg.norm(outline[0] - outline[-1]) < 0.01


# ── Smart Rule Filtering Tests ────────────────────────────────────

class TestSmartRuleFiltering:
    """Test that non-wall elements get wall rules skipped."""

    def test_slab_skips_l3_rules(self):
        """A flat slab should have L3 wall rules skipped."""
        mesh = _make_mesh(*_box([0, 0, 0], [8, 6, 0.3]))
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)

        l4 = validate_level4(l1, l3, ruleset, level2_result=l2)
        skipped = [c for c in l4["checks"] if c["status"] == "SKIP"]
        l3_skipped = [c for c in skipped if "L3" in c["rule_id"]]

        assert len(l3_skipped) > 0, "Slab should have L3 rules skipped"
        # Check skip message mentions element role or geometry
        for c in l3_skipped:
            msg = c["message"].lower()
            assert "platte" in msg or "not wall-like" in msg or "wall-specific" in msg

    def test_wall_does_not_skip(self):
        """A standard wall should NOT have L3 rules skipped."""
        mesh = _make_mesh(*_box([0, 0, 0], [8, 0.4, 3]))
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)

        l4 = validate_level4(l1, l3, ruleset, level2_result=l2)
        l3_checks = [c for c in l4["checks"] if "L3" in c["rule_id"]]
        l3_skipped = [c for c in l3_checks if c["status"] == "SKIP"
                      and ("not wall-like" in c.get("message", "")
                           or "wall-specific" in c.get("message", ""))]

        assert len(l3_skipped) == 0, "Wall should not have L3 rules skipped for geometry"

    def test_without_l2_no_filtering(self):
        """Without L2 result, no smart filtering should occur (backward compat)."""
        mesh = _make_mesh(*_box([0, 0, 0], [8, 6, 0.3]))
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)

        # Without level2_result parameter
        l4 = validate_level4(l1, l3, ruleset)
        l3_wall_skipped = [c for c in l4["checks"]
                           if "L3" in c["rule_id"] and "not wall-like" in c.get("message", "")]
        assert len(l3_wall_skipped) == 0, "Without L2 result, no smart filtering"


# ── HTML Report Tests ─────────────────────────────────────────────

class TestHTMLReport:
    """Test HTML report generation."""

    def test_generates_valid_html(self):
        """Report should be valid HTML with key sections."""
        results = [{
            "element_id": 1,
            "element_name": "Test Wall",
            "level1": {"volume": 10.0, "total_area": 50.0, "is_watertight": True,
                       "num_triangles": 12, "num_vertices": 8, "bbox": {"size": [8, 0.4, 3]}},
            "level2": {"confidence": 0.85, "diagnostics": ["Test diagnostic"]},
            "level3": {"crown_width_mm": 400, "crown_slope_percent": 3.0,
                       "min_wall_thickness_mm": 400},
            "level4": {"checks": [
                {"rule_id": "TEST-001", "name": "Test Rule", "status": "PASS",
                 "severity": "ERROR", "message": "Check passed"},
            ], "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0}},
        }]

        html = generate_html_report(results, ifc_filename="test.ifc",
                                     ruleset_name="Test Ruleset")
        assert "<!DOCTYPE html>" in html
        assert "Test Wall" in html
        assert "test.ifc" in html
        assert "Test Ruleset" in html
        assert "PASS" in html
        assert "400" in html  # crown width

    def test_handles_errors(self):
        """Report should handle error results gracefully."""
        results = [
            {"element_id": 1, "element_name": "Good", "level1": {"volume": 5},
             "level2": {"confidence": 0.8}, "level3": {}},
            {"element_id": 2, "element_name": "Bad", "error": "Mesh failed"},
        ]
        html = generate_html_report(results)
        assert "Good" in html
        assert "Bad" in html
        assert "Mesh failed" in html

    def test_empty_results(self):
        """Report should handle empty results."""
        html = generate_html_report([])
        assert "<!DOCTYPE html>" in html
        assert "0 bestanden" in html

    def test_with_slope_data(self):
        """Report should include slope analysis when present."""
        results = [{
            "element_id": 1, "element_name": "Wall",
            "level1": {"volume": 10}, "level2": {"confidence": 0.9},
            "level3": {"crown_width_mm": 300},
            "slope_analysis": {
                "area_weighted_cross_pct": 3.0,
                "area_weighted_long_pct": 0.5,
            },
        }]
        html = generate_html_report(results)
        assert "Quergefälle" in html or "3.00" in html


# ── Actionable FAIL Messages Tests ────────────────────────────────

class TestActionableMessages:
    """Test that FAIL messages contain actionable information."""

    def test_fail_shows_actual_and_expected(self):
        """FAIL message should show actual value and requirement."""
        mesh = _make_mesh(*_box([0, 0, 0], [5, 0.2, 2]))  # 200mm width < 300mm
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)

        l4 = validate_level4(l1, l3, ruleset, level2_result=l2)
        fails = [c for c in l4["checks"] if c["status"] == "FAIL"]

        assert len(fails) > 0
        # Check that at least one FAIL has actual value in message
        has_actual = any("Ist:" in c.get("message", "") for c in fails)
        assert has_actual, "FAIL messages should contain 'Ist:' with actual value"

    def test_fail_shows_reference(self):
        """FAIL message should include reference or quote."""
        mesh = _make_mesh(*_box([0, 0, 0], [5, 0.2, 2]))
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)

        l4 = validate_level4(l1, l3, ruleset, level2_result=l2)
        fails = [c for c in l4["checks"] if c["status"] == "FAIL"]

        has_ref = any("Quelle:" in c.get("message", "") or "Referenz:" in c.get("message", "")
                       for c in fails)
        assert has_ref, "FAIL messages should contain source reference"
