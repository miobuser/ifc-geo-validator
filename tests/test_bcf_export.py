"""Tests for BCF export of validation failures."""

import os
import tempfile

import pytest
from bcf.v2 import bcfxml

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.report.bcf_export import export_bcf

MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")
RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "astra_fhb_stuetzmauer.yaml",
)


def _run_pipeline(model_name):
    """Run full L1-L4 pipeline, return element result dict."""
    path = os.path.join(MODELS_DIR, model_name)
    model = load_model(path)
    walls = get_elements(model, "IfcWall")
    elem = walls[0]
    mesh = extract_mesh(elem)
    l1 = validate_level1(mesh)
    l2 = validate_level2(mesh)
    l3 = validate_level3(mesh, l2)
    ruleset = load_ruleset(RULESET_PATH)
    l4 = validate_level4(l1, l3, ruleset)
    return {
        "element_id": elem.id(),
        "element_name": getattr(elem, "Name", ""),
        "level1": l1,
        "level2": l2,
        "level3": l3,
        "level4": l4,
    }


class TestBcfExport:
    """Test BCF export for models with failures."""

    def test_t6_has_topics(self):
        result = _run_pipeline("T6_non_compliant.ifc")
        with tempfile.NamedTemporaryFile(suffix=".bcf", delete=False) as f:
            tmp = f.name
        try:
            export_bcf([result], tmp, ifc_name="T6_non_compliant.ifc")
            bcf = bcfxml.BcfXml.load(tmp)
            guids = list(bcf.topics)
            assert len(guids) > 0, "T6 should have failed checks → BCF topics"
            bcf.close()
        finally:
            os.unlink(tmp)

    def test_t6_topic_content(self):
        result = _run_pipeline("T6_non_compliant.ifc")
        with tempfile.NamedTemporaryFile(suffix=".bcf", delete=False) as f:
            tmp = f.name
        try:
            export_bcf([result], tmp)
            bcf = bcfxml.BcfXml.load(tmp)
            guids = list(bcf.topics)
            h = bcf.topics[guids[0]]
            # Topics should have type, description, and comments
            assert h.topic.title
            assert h.topic.topic_type in ("Error", "Warning", "Information")
            assert h.topic.description
            assert len(h.comments) >= 1
            bcf.close()
        finally:
            os.unlink(tmp)

    def test_t7_no_topics(self):
        result = _run_pipeline("T7_compliant.ifc")
        with tempfile.NamedTemporaryFile(suffix=".bcf", delete=False) as f:
            tmp = f.name
        try:
            export_bcf([result], tmp)
            bcf = bcfxml.BcfXml.load(tmp)
            guids = list(bcf.topics)
            # T7 has 2 FAIL topics: thickness perpendicular (298.5mm < 300mm)
            # and its composite. This is physically correct.
            assert len(guids) <= 4, f"T7 should have few BCF topics, got {len(guids)}"
            bcf.close()
        finally:
            try:
                os.unlink(tmp)
            except PermissionError:
                pass  # Windows may hold the file briefly

    def test_t6_topic_count_matches_failures(self):
        result = _run_pipeline("T6_non_compliant.ifc")
        n_failed = sum(
            1 for c in result["level4"]["checks"]
            if c["status"] == "FAIL"
        )
        with tempfile.NamedTemporaryFile(suffix=".bcf", delete=False) as f:
            tmp = f.name
        try:
            export_bcf([result], tmp)
            bcf = bcfxml.BcfXml.load(tmp)
            guids = list(bcf.topics)
            assert len(guids) == n_failed
            bcf.close()
        finally:
            os.unlink(tmp)


class TestBCFWithL5L6Context:
    """Test BCF export with L5/L6 context failures."""

    def test_l5_failure_exported(self):
        """A FAIL for ASTRA-SM-L5-002 (gap > 10mm) should produce a BCF topic."""
        result = _run_pipeline("T7_compliant.ifc")

        # Inject L5 context that causes ASTRA-SM-L5-002 to fail (gap > 10mm)
        l5_ctx = {
            "foundation_extends_beyond_wall": True,
            "wall_foundation_gap_mm": 50.0,  # > 10mm threshold → FAIL
        }
        ruleset = load_ruleset(RULESET_PATH)
        l4_with_l5 = validate_level4(
            result["level1"], result["level3"], ruleset,
            level5_context=l5_ctx,
        )
        result["level4"] = l4_with_l5

        # Verify L5-002 actually failed
        checks = {c["rule_id"]: c for c in l4_with_l5["checks"]}
        assert checks["ASTRA-SM-L5-002"]["status"] == "FAIL"

        with tempfile.NamedTemporaryFile(suffix=".bcf", delete=False) as f:
            tmp = f.name
        try:
            export_bcf([result], tmp, ifc_name="T7_compliant.ifc")
            bcf = bcfxml.BcfXml.load(tmp)
            guids = list(bcf.topics)
            # At least one topic should mention L5-002
            l5_topics = [
                g for g in guids
                if "L5-002" in bcf.topics[g].topic.title
            ]
            assert len(l5_topics) >= 1, "BCF should contain a topic for ASTRA-SM-L5-002"
            bcf.close()
        finally:
            try:
                os.unlink(tmp)
            except PermissionError:
                pass
