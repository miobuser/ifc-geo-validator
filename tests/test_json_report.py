"""Tests for JSON report generation and CLI --output integration.

Verifies that the JSON report:
  - Is valid JSON
  - Contains expected structure (elements, levels, rules)
  - Round-trips correctly (write + read back)
  - Has correct values for known test models
"""

import json
import os
import sys
import tempfile

import pytest

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.report.json_report import generate_report, write_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")
RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "astra_fhb_stuetzmauer.yaml",
)


def _run_pipeline(model_name):
    """Run L1-L4 pipeline, return results list."""
    path = os.path.join(MODELS_DIR, model_name)
    model = load_model(path)
    walls = get_elements(model, "IfcWall")
    ruleset = load_ruleset(RULESET_PATH)
    results = []
    for wall in walls:
        mesh = extract_mesh(wall)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        l4 = validate_level4(l1, l3, ruleset)
        results.append({
            "element_id": wall.id(),
            "element_name": getattr(wall, "Name", ""),
            "level1": l1, "level2": l2, "level3": l3, "level4": l4,
        })
    return results, ruleset


class TestGenerateReport:
    """Test the generate_report() function."""

    def test_report_structure(self):
        results, ruleset = _run_pipeline("T1_simple_box.ifc")
        report = generate_report("T1_simple_box.ifc", results, ruleset)
        assert "report" in report
        assert "elements" in report
        assert "summary" in report
        assert report["report"]["generator"] == "ifc-geo-validator"

    def test_report_has_elements(self):
        results, ruleset = _run_pipeline("T1_simple_box.ifc")
        report = generate_report("T1_simple_box.ifc", results, ruleset)
        assert len(report["elements"]) == 1

    def test_report_element_has_geometry(self):
        results, ruleset = _run_pipeline("T1_simple_box.ifc")
        report = generate_report("T1_simple_box.ifc", results, ruleset)
        elem = report["elements"][0]
        assert "geometry" in elem
        assert abs(elem["geometry"]["volume_m3"] - 9.6) < 0.01

    def test_report_element_has_measurements(self):
        results, ruleset = _run_pipeline("T1_simple_box.ifc")
        report = generate_report("T1_simple_box.ifc", results, ruleset)
        elem = report["elements"][0]
        assert "measurements" in elem
        assert abs(elem["measurements"]["crown_width_mm"] - 400) < 1

    def test_report_element_has_rule_checks(self):
        results, ruleset = _run_pipeline("T1_simple_box.ifc")
        report = generate_report("T1_simple_box.ifc", results, ruleset)
        elem = report["elements"][0]
        assert "rule_checks" in elem
        assert elem["rule_checks"]["summary"]["total"] == 18


class TestWriteReport:
    """Test JSON file write and round-trip."""

    def test_write_valid_json(self):
        results, ruleset = _run_pipeline("T1_simple_box.ifc")
        report = generate_report("T1_simple_box.ifc", results, ruleset)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            write_report(report, tmp)
            with open(tmp, encoding="utf-8") as f:
                data = json.load(f)  # Must not raise
            assert data["report"]["version"] == "1.0.0"
        finally:
            os.unlink(tmp)

    def test_round_trip_preserves_values(self):
        results, ruleset = _run_pipeline("T7_compliant.ifc")
        report = generate_report("T7_compliant.ifc", results, ruleset)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            write_report(report, tmp)
            with open(tmp, encoding="utf-8") as f:
                data = json.load(f)
            elem = data["elements"][0]
            assert abs(elem["geometry"]["volume_m3"] - 10.811) < 0.01
            assert elem["rule_checks"]["summary"]["failed"] == 0
        finally:
            os.unlink(tmp)

    def test_multi_element_report(self):
        results, ruleset = _run_pipeline("T4_l_shaped.ifc")
        report = generate_report("T4_l_shaped.ifc", results, ruleset)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            write_report(report, tmp)
            with open(tmp, encoding="utf-8") as f:
                data = json.load(f)
            assert len(data["elements"]) == 2
            assert data["summary"]["validated"] == 2
        finally:
            os.unlink(tmp)


class TestCLIOutputFlag:
    """Test CLI --output flag produces valid JSON."""

    def test_cli_output_creates_file(self):
        import subprocess
        model = os.path.join(MODELS_DIR, "T1_simple_box.ifc")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ifc_geo_validator.cli", model,
                 "--output", tmp, "--levels", "1,2,3,4"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(tmp)
            with open(tmp, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) >= 1
            assert "level1" in data[0]
            assert abs(data[0]["level1"]["volume"] - 9.6) < 0.1
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
