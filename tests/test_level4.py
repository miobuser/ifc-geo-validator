"""Tests for Level 4: Requirement comparison against YAML ruleset."""

import os
import pytest
import numpy as np

from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset


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


# ── T1 box mesh ────────────────────────────────────────────────────

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

RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "astra_fhb_stuetzmauer.yaml"
)


class TestLoadRuleset:
    """Test YAML ruleset loading."""

    @pytest.mark.skipif(not os.path.exists(RULESET_PATH), reason="Ruleset not found")
    def test_load(self):
        rs = load_ruleset(RULESET_PATH)
        assert rs["metadata"]["name"] == "ASTRA FHB T/G — Stützmauern"
        assert len(rs["level_1"]) == 2
        assert len(rs["level_3"]) == 4


class TestT1Validation:
    """T1 box (8.0 x 0.4 x 3.0) against ASTRA ruleset.

    Expected results:
      - L1-001 Volume > 0.1:      PASS (9.6 m³)
      - L1-002 Watertight:         PASS (True)
      - L3-001 Crown ≥ 300mm:     PASS (400mm)
      - L3-002 Crown slope 2.5-3.5%: FAIL (0%)
      - L3-003 Wall thickness ≥ 300mm: PASS (400mm)
      - L3-004 Inclination 10:1:  FAIL (vertical → inf)
      - L4-001 Crown composite:   FAIL (L3-002 failed)
      - L4-002 Wall composite:    FAIL (L3-004 failed)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(RULESET_PATH):
            pytest.skip("Ruleset not found")

        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        self.result = validate_level4(l1, l3, ruleset)
        self.checks = {c["rule_id"]: c for c in self.result["checks"]}

    def test_volume_pass(self):
        assert self.checks["ASTRA-SM-L1-001"]["status"] == "PASS"

    def test_crown_width_pass(self):
        assert self.checks["ASTRA-SM-L3-001"]["status"] == "PASS"

    def test_crown_slope_fail(self):
        """Flat crown (0%) → fails 2.5-3.5% requirement."""
        assert self.checks["ASTRA-SM-L3-002"]["status"] == "FAIL"

    def test_wall_thickness_pass(self):
        assert self.checks["ASTRA-SM-L3-003"]["status"] == "PASS"

    def test_composite_crown_fail(self):
        """Crown composite fails because slope check failed."""
        assert self.checks["ASTRA-SM-L4-001"]["status"] == "FAIL"

    def test_summary_counts(self):
        s = self.result["summary"]
        assert s["total"] == 8  # 2 L1 + 4 L3 + 2 L4
        assert s["passed"] >= 3
        assert s["failed"] >= 2


class TestNarrowWallFails:
    """Wall with 200mm thickness should fail thickness check."""

    @pytest.mark.skipif(not os.path.exists(RULESET_PATH), reason="Ruleset not found")
    def test_thickness_fail(self):
        verts = np.array([
            [0, 0, 0], [6, 0, 0], [6, 0.2, 0], [0, 0.2, 0],
            [0, 0, 2], [6, 0, 2], [6, 0.2, 2], [0, 0.2, 2],
        ], dtype=float)
        faces = T1_FACES.copy()

        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        assert checks["ASTRA-SM-L3-001"]["status"] == "FAIL"  # 200mm < 300mm
        assert checks["ASTRA-SM-L3-003"]["status"] == "FAIL"  # 200mm < 300mm


class TestCompliantWall:
    """Wall that meets all ASTRA requirements."""

    @pytest.mark.skipif(not os.path.exists(RULESET_PATH), reason="Ruleset not found")
    def test_all_pass(self):
        # 8.0 x 0.4m wall, 3.0m height, 3% crown slope, 10:1 inclination
        # Crown slope: 3% of 0.4m = 0.012m rise
        # 10:1 inclination: 0.3m offset over 3.0m height
        verts = np.array([
            [0, 0, 0],    [8, 0, 0],    [8, 0.7, 0],    [0, 0.7, 0],      # bottom
            [0, 0.3, 3],  [8, 0.3, 3],  [8, 0.7, 3.012], [0, 0.7, 3.012], # top
        ], dtype=float)
        faces = T1_FACES.copy()

        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        assert checks["ASTRA-SM-L1-001"]["status"] == "PASS"
        assert checks["ASTRA-SM-L3-001"]["status"] == "PASS"  # crown ≥ 300mm
        assert checks["ASTRA-SM-L3-003"]["status"] == "PASS"  # thickness ≥ 300mm
