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
        assert len(rs["level_3"]) == 7


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

    def test_composite_crown_pass(self):
        """Crown composite (L4-001) passes — only depends on L3-001 (width), not slope."""
        assert self.checks["ASTRA-SM-L4-001"]["status"] == "PASS"

    def test_composite_slope_fail(self):
        """Slope composite (L4-003) fails because T1 has no crown slope."""
        assert self.checks["ASTRA-SM-L4-003"]["status"] == "FAIL"

    def test_summary_counts(self):
        s = self.result["summary"]
        assert s["total"] == 18  # 2 L1 + 7 L3 + 2 L5 + 2 L6 + 1 L7 + 4 L4
        assert s["passed"] >= 4
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


# ── IFC integration tests ─────────────────────────────────────────────

T2_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T2_inclined_wall.ifc")
T3_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T3_crown_slope.ifc")


@pytest.mark.skipif(
    not os.path.exists(T2_PATH) or not os.path.exists(RULESET_PATH),
    reason="T2 model or ruleset not found",
)
class TestT2FullPipeline:
    """T2 (10:1 inclined, no slope) against ASTRA ruleset."""

    def test_inclination_pass(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T2_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        assert checks["ASTRA-SM-L3-004"]["status"] == "PASS"   # 10:1 inclination
        assert checks["ASTRA-SM-L3-001"]["status"] == "PASS"   # crown ≥ 300mm (350mm)
        assert checks["ASTRA-SM-L3-002"]["status"] == "FAIL"   # no crown slope


@pytest.mark.skipif(
    not os.path.exists(T3_PATH) or not os.path.exists(RULESET_PATH),
    reason="T3 model or ruleset not found",
)
class TestT3FullPipeline:
    """T3 (3% crown slope, vertical) — crown slope should PASS."""

    def test_crown_slope_pass(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T3_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        assert checks["ASTRA-SM-L3-001"]["status"] == "PASS"   # crown ≥ 300mm
        assert checks["ASTRA-SM-L3-002"]["status"] == "PASS"   # 3% slope
        assert checks["ASTRA-SM-L4-001"]["status"] == "PASS"   # crown composite


T6_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T6_non_compliant.ifc")
T7_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T7_compliant.ifc")


@pytest.mark.skipif(
    not os.path.exists(T6_PATH) or not os.path.exists(RULESET_PATH),
    reason="T6 model or ruleset not found",
)
class TestT6NonCompliant:
    """T6 (thin, no slope, vertical) — all L3 rules should FAIL."""

    def test_all_l3_fail(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T6_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        assert checks["ASTRA-SM-L3-001"]["status"] == "FAIL"   # 200mm < 300mm
        assert checks["ASTRA-SM-L3-002"]["status"] == "FAIL"   # 0% slope
        assert checks["ASTRA-SM-L3-003"]["status"] == "FAIL"   # 200mm < 300mm
        assert checks["ASTRA-SM-L3-004"]["status"] == "FAIL"   # vertical
        assert result["summary"]["passed"] == 2   # only L1 rules pass


@pytest.mark.skipif(
    not os.path.exists(T7_PATH) or not os.path.exists(RULESET_PATH),
    reason="T7 model or ruleset not found",
)
class TestT7FullyCompliant:
    """T7 (10:1 incl, 3% slope, 300mm crown) — ALL rules should PASS."""

    def test_all_pass(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T7_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)

        assert result["summary"]["passed"] == 11  # L1+L3(6)+L4 all pass (T7 is fully compliant)
        assert result["summary"]["failed"] == 0
        # L5+L6+L7 rules are SKIPPED (no context provided)


# ── Alternative Ruleset (SIA 262) ─────────────────────────────────

SIA_RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "sia_262_stuetzmauer.yaml"
)


@pytest.mark.skipif(not os.path.exists(SIA_RULESET_PATH), reason="SIA ruleset not found")
class TestSIARuleset:
    """Test that alternative rulesets (SIA 262) work correctly."""

    def test_load_sia_ruleset(self):
        rs = load_ruleset(SIA_RULESET_PATH)
        assert rs["metadata"]["name"] == "SIA 262 — Stützmauern (Betonbau)"

    def test_t1_passes_sia_rules(self):
        """T1 (400mm thick) passes SIA 262 minimum (200mm)."""
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        rs = load_ruleset(SIA_RULESET_PATH)
        result = validate_level4(l1, l3, rs)
        # SIA has lower thresholds — T1 should pass everything
        assert result["summary"]["failed"] == 0
        assert result["summary"]["passed"] >= 5

    def test_sia_l5_with_context(self):
        """SIA L5-001 (foundation gap) should evaluate when context provided."""
        mesh = _make_mesh(T1_VERTS, T1_FACES)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        rs = load_ruleset(SIA_RULESET_PATH)
        # Provide L5 context
        l5_ctx = {"wall_foundation_gap_mm": 0.5}
        result = validate_level4(l1, l3, rs, level5_context=l5_ctx)
        # SIA-SM-L5-001 should PASS (0.5mm <= 10mm)
        l5_check = next((c for c in result["checks"] if "L5-001" in c["rule_id"]), None)
        assert l5_check is not None
        assert l5_check["status"] == "PASS"

    def test_different_thresholds(self):
        """SIA allows 200mm minimum vs ASTRA 300mm — verify a 250mm wall
        passes SIA but fails ASTRA."""
        # 250mm thin wall
        verts = np.array([
            [0, 0, 0], [6, 0, 0], [6, 0.25, 0], [0, 0.25, 0],
            [0, 0, 2], [6, 0, 2], [6, 0.25, 2], [0, 0.25, 2],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = _make_mesh(verts, faces)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        # SIA: 200mm min → 250mm passes
        sia_rs = load_ruleset(SIA_RULESET_PATH)
        sia_result = validate_level4(l1, l3, sia_rs)
        sia_thick = next(c for c in sia_result["checks"] if "L3-001" in c["rule_id"])
        assert sia_thick["status"] == "PASS"

        # ASTRA: 300mm min → 250mm fails
        astra_rs = load_ruleset(RULESET_PATH)
        astra_result = validate_level4(l1, l3, astra_rs)
        astra_thick = next(c for c in astra_result["checks"] if "L3-003" in c["rule_id"])
        assert astra_thick["status"] == "FAIL"


# ── T24: Highway wall — fully ASTRA compliant wall stem ──────────────

T24_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T24_highway_with_terrain.ifc")


@pytest.mark.skipif(
    not os.path.exists(T24_PATH) or not os.path.exists(RULESET_PATH),
    reason="T24 model or ruleset not found",
)
class TestT24FullyCompliant:
    """T24 (inclined, crown slope, 350mm) — all mandatory (ERROR) rules should PASS."""

    def test_all_mandatory_pass(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T24_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        # All L1 rules pass
        assert checks["ASTRA-SM-L1-001"]["status"] == "PASS"
        assert checks["ASTRA-SM-L1-002"]["status"] == "PASS"
        # All L3 rules pass (ASTRA compliant wall stem)
        assert checks["ASTRA-SM-L3-001"]["status"] == "PASS"   # crown >= 300mm (350mm)
        assert checks["ASTRA-SM-L3-002"]["status"] == "PASS"   # crown slope ~3%
        assert checks["ASTRA-SM-L3-003"]["status"] == "PASS"   # thickness >= 300mm
        assert checks["ASTRA-SM-L3-004"]["status"] == "PASS"   # inclination ~10:1
        # No ERROR-level failures
        assert result["summary"]["errors"] == 0


# ── T25: Multi-failure — multiple ERROR rules should FAIL ────────────

T25_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T25_multi_failure.ifc")


@pytest.mark.skipif(
    not os.path.exists(T25_PATH) or not os.path.exists(RULESET_PATH),
    reason="T25 model or ruleset not found",
)
class TestT25MultiFailure:
    """T25 (thin, no slope, vertical) — multiple ERROR rules should FAIL."""

    def test_multiple_failures(self):
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh

        model = load_model(T25_PATH)
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        ruleset = load_ruleset(RULESET_PATH)
        result = validate_level4(l1, l3, ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}

        # L1 rules still pass (volume > 0, watertight)
        assert checks["ASTRA-SM-L1-001"]["status"] == "PASS"
        # L3 rules fail (200mm thin, no slope, vertical)
        assert checks["ASTRA-SM-L3-001"]["status"] == "FAIL"   # 200mm < 300mm crown
        assert checks["ASTRA-SM-L3-003"]["status"] == "FAIL"   # 200mm < 300mm thickness
        # Multiple ERROR-level failures
        assert result["summary"]["errors"] >= 2


# ── L5 context in L4 ──────────────────────────────────────────────

class TestL5ContextInL4:
    """Test L5 inter-element context passed to validate_level4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(RULESET_PATH):
            pytest.skip("Ruleset not found")
        self.mesh = _make_mesh(T1_VERTS, T1_FACES)
        self.l1 = validate_level1(self.mesh)
        self.l2 = validate_level2(self.mesh)
        self.l3 = validate_level3(self.mesh, self.l2)
        self.ruleset = load_ruleset(RULESET_PATH)

    def test_l5_foundation_overhang_pass(self):
        """L5 context with good values → L5 rules PASS."""
        l5_ctx = {
            "foundation_extends_beyond_wall": True,
            "wall_foundation_gap_mm": 0.5,
        }
        result = validate_level4(self.l1, self.l3, self.ruleset, level5_context=l5_ctx)
        checks = {c["rule_id"]: c for c in result["checks"]}
        assert checks["ASTRA-SM-L5-001"]["status"] == "PASS"
        assert checks["ASTRA-SM-L5-002"]["status"] == "PASS"

    def test_l5_gap_fail(self):
        """L5 context with bad values → L5 rules FAIL."""
        l5_ctx = {
            "foundation_extends_beyond_wall": False,
            "wall_foundation_gap_mm": 50.0,
        }
        result = validate_level4(self.l1, self.l3, self.ruleset, level5_context=l5_ctx)
        checks = {c["rule_id"]: c for c in result["checks"]}
        assert checks["ASTRA-SM-L5-001"]["status"] == "FAIL"
        assert checks["ASTRA-SM-L5-002"]["status"] == "FAIL"

    def test_l5_no_context_skips(self):
        """No L5 context provided → L5 rules SKIP."""
        result = validate_level4(self.l1, self.l3, self.ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}
        assert checks["ASTRA-SM-L5-001"]["status"] == "SKIP"
        assert checks["ASTRA-SM-L5-002"]["status"] == "SKIP"


# ── L6 context in L4 ──────────────────────────────────────────────

class TestL6ContextInL4:
    """Test L6 terrain/distance context passed to validate_level4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(RULESET_PATH):
            pytest.skip("Ruleset not found")
        self.mesh = _make_mesh(T1_VERTS, T1_FACES)
        self.l1 = validate_level1(self.mesh)
        self.l2 = validate_level2(self.mesh)
        self.l3 = validate_level3(self.mesh, self.l2)
        self.ruleset = load_ruleset(RULESET_PATH)

    def test_l6_earth_side_pass(self):
        """L6 context with terrain data → L6 rules PASS."""
        l6_ctx = {
            "earth_side_determined": True,
            "crown_slope_towards_earth_side": True,
        }
        result = validate_level4(self.l1, self.l3, self.ruleset, level6_context=l6_ctx)
        checks = {c["rule_id"]: c for c in result["checks"]}
        assert checks["ASTRA-SM-L6-001"]["status"] == "PASS"
        assert checks["ASTRA-SM-L6-002"]["status"] == "PASS"

    def test_l6_no_terrain_skips(self):
        """No L6 context provided → L6 rules SKIP."""
        result = validate_level4(self.l1, self.l3, self.ruleset)
        checks = {c["rule_id"]: c for c in result["checks"]}
        assert checks["ASTRA-SM-L6-001"]["status"] == "SKIP"
        assert checks["ASTRA-SM-L6-002"]["status"] == "SKIP"
