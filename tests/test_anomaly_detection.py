"""Tests for geometric anomaly detection."""

import numpy as np
import pytest

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.core.anomaly_detection import detect_anomalies


class TestMissingFaces:
    """Test detection of missing face categories."""

    def test_complete_wall_no_anomaly(self):
        """T1 box has all expected faces → no missing face anomaly."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        anomalies = detect_anomalies(mesh, l2, l3)
        missing = [a for a in anomalies if a["type"] == "missing_face"]
        assert len(missing) == 0


class TestClassificationQuality:
    """Test detection of classification quality issues."""

    def test_normal_wall_no_quality_issue(self):
        """Standard wall should have no classification anomalies."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        anomalies = detect_anomalies(mesh, l2, l3)
        quality = [a for a in anomalies if a["type"] == "high_unclassified"]
        assert len(quality) == 0


class TestAspectRatio:
    """Test aspect ratio anomaly detection."""

    def test_normal_wall_no_anomaly(self):
        """Standard wall proportions → no aspect ratio anomaly."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        anomalies = detect_anomalies(mesh, l2, l3)
        aspect = [a for a in anomalies if a["type"] == "extreme_aspect_ratio"]
        assert len(aspect) == 0


class TestNormalConsistency:
    """Test normal direction consistency check."""

    def test_valid_mesh_consistent(self):
        """A valid closed mesh has consistent normals."""
        model = load_model("tests/test_models/T7_compliant.ifc")
        mesh = extract_mesh(get_elements(model, "IfcWall")[0])
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        anomalies = detect_anomalies(mesh, l2, l3)
        normal_issues = [a for a in anomalies if a["type"] == "inconsistent_normals"]
        assert len(normal_issues) == 0


class TestAnomalyOnRealModel:
    """Test anomaly detection on real models."""

    @pytest.mark.skipif(
        not __import__("os").path.exists(
            __import__("os").path.join(
                __import__("os").path.expanduser("~"),
                "OneDrive - Berner Fachhochschule",
                "Semester 7", "Minor", "IFC 4.0.2.1", "Infra-Bridge.ifc",
            )
        ),
        reason="Bridge model not available",
    )
    def test_bridge_walls_no_critical_anomalies(self):
        """Real bridge walls should have no critical anomalies."""
        import os
        path = os.path.join(
            os.path.expanduser("~"),
            "OneDrive - Berner Fachhochschule",
            "Semester 7", "Minor", "IFC 4.0.2.1", "Infra-Bridge.ifc",
        )
        model = load_model(path)
        for w in get_elements(model, "IfcWall")[:2]:
            mesh = extract_mesh(w)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)
            anomalies = detect_anomalies(mesh, l2, l3)
            errors = [a for a in anomalies if a["severity"] == "error"]
            assert len(errors) == 0, f"Unexpected errors: {errors}"
