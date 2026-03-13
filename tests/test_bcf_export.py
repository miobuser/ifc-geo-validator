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
            assert len(guids) == 0, "T7 passes all rules → no BCF topics"
            bcf.close()
        finally:
            os.unlink(tmp)

    def test_t6_topic_count_matches_failures(self):
        result = _run_pipeline("T6_non_compliant.ifc")
        n_failed = sum(
            1 for c in result["level4"]["checks"]
            if c["status"] != "PASS"
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
