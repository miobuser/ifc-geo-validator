"""Tests for IFC property injection (Pset_GeoValidation)."""

import os
import tempfile

import ifcopenshell
import pytest

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.report.ifc_property_writer import (
    inject_properties, inject_all, PSET_NAME,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")
RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "astra_fhb_stuetzmauer.yaml",
)


def _run_pipeline(model_name):
    """Run full L1-L4 pipeline on a test model, return (model, element, result)."""
    path = os.path.join(MODELS_DIR, model_name)
    model = load_model(path)
    walls = get_elements(model, "IfcWall")
    assert walls, f"No walls in {model_name}"
    elem = walls[0]
    mesh = extract_mesh(elem)
    l1 = validate_level1(mesh)
    l2 = validate_level2(mesh)
    l3 = validate_level3(mesh, l2)
    ruleset = load_ruleset(RULESET_PATH)
    l4 = validate_level4(l1, l3, ruleset)
    result = {
        "element_id": elem.id(),
        "element_name": getattr(elem, "Name", ""),
        "level1": l1,
        "level2": l2,
        "level3": l3,
        "level4": l4,
    }
    return model, elem, result


def _read_pset(element):
    """Read Pset_GeoValidation from an element, return dict of properties."""
    for rel in getattr(element, "IsDefinedBy", []):
        if not hasattr(rel, "RelatingPropertyDefinition"):
            continue
        pdef = rel.RelatingPropertyDefinition
        if hasattr(pdef, "Name") and pdef.Name == PSET_NAME:
            return {
                p.Name: p.NominalValue.wrappedValue
                for p in pdef.HasProperties
            }
    return None


class TestInjectProperties:
    """Test property injection on T1 simple box."""

    def test_creates_pset(self):
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert props is not None

    def test_l1_properties(self):
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert abs(props["Volume_m3"] - 9.6) < 0.01
        assert props["IsWatertight"] is True
        assert props["NumTriangles"] > 0

    def test_l3_properties(self):
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert abs(props["CrownWidth_mm"] - 400.0) < 1.0
        assert "MinWallThickness_mm" in props

    def test_l4_properties(self):
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert props["RulesTotal"] > 0
        assert "Rule_ASTRA-SM-L3-001" in props

    def test_skips_error_result(self):
        model, elem, _ = _run_pipeline("T1_simple_box.ifc")
        inject_properties(model, elem, {"error": "test"})
        props = _read_pset(elem)
        assert props is None


class TestL5L6Context:
    """Test that L5/L6 context properties are written."""

    def test_l5_context_properties(self):
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        result["level5_context"] = {
            "foundation_extends_beyond_wall": True,
            "wall_foundation_gap_mm": 0.5,
        }
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert props["FoundationOverhang"] is True
        assert abs(props["FoundationGap_mm"] - 0.5) < 0.1

    def test_l6_context_properties(self):
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        result["level6_context"] = {
            "earth_side_determined": True,
        }
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert props["EarthSideDetermined"] is True

    def test_no_context_no_crash(self):
        """Missing L5/L6 context should not crash."""
        model, elem, result = _run_pipeline("T1_simple_box.ifc")
        # No level5_context or level6_context keys
        inject_properties(model, elem, result)
        props = _read_pset(elem)
        assert "FoundationOverhang" not in props
        assert "EarthSideDetermined" not in props


class TestInjectAll:
    """Test batch injection with file output."""

    def test_round_trip_t7(self):
        model, elem, result = _run_pipeline("T7_compliant.ifc")
        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
            tmp_path = f.name

        try:
            inject_all(model, [elem], [result], tmp_path)
            assert os.path.exists(tmp_path)

            # Re-read and verify
            model2 = ifcopenshell.open(tmp_path)
            walls2 = model2.by_type("IfcWall")
            assert len(walls2) >= 1
            props = _read_pset(walls2[0])
            assert props is not None
            assert props["RulesPassed"] == 11   # L1+L3(6)+L4 pass; L3-007 cross_slope SKIP (no slope data)
            assert props["RulesTotal"] == 18   # 2 L1 + 7 L3 + 4 L4 + 2 L5 + 2 L6 + 1 L7
        finally:
            os.unlink(tmp_path)

    def test_round_trip_t6(self):
        model, elem, result = _run_pipeline("T6_non_compliant.ifc")
        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
            tmp_path = f.name

        try:
            inject_all(model, [elem], [result], tmp_path)
            model2 = ifcopenshell.open(tmp_path)
            walls2 = model2.by_type("IfcWall")
            props = _read_pset(walls2[0])
            assert props["RulesFailed"] > 0
            assert props["Rule_ASTRA-SM-L3-001"] == "FAIL"
        finally:
            os.unlink(tmp_path)
