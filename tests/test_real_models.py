"""Integration tests with real IFC models.

These tests validate the full pipeline on real-world IFC files.
Tests are skipped if the model files are not available (they are
not part of the repository).

Purpose: verify that the pipeline handles real BIM models without
crashing, produces plausible results, and handles edge cases like
multi-body elements and non-manifold geometry gracefully.
"""

import os
import pytest

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh, MeshExtractionError
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.validation.level5 import validate_level5

# Path to the IFC4 bridge reference model (buildingSMART sample)
BRIDGE_PATH = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Berner Fachhochschule",
    "Semester 7", "Minor", "IFC 4.0.2.1", "Infra-Bridge.ifc",
)

RULESET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "ifc_geo_validator",
    "rules", "rulesets", "astra_fhb_stuetzmauer.yaml",
)

bridge_available = pytest.mark.skipif(
    not os.path.exists(BRIDGE_PATH),
    reason="Bridge model not available (not part of repository)",
)


@bridge_available
class TestInfraBridge:
    """End-to-end tests on the IFC4 Infra-Bridge reference model."""

    @pytest.fixture(scope="class")
    def model(self):
        return load_model(BRIDGE_PATH)

    def test_model_loads(self, model):
        """Model loads without errors."""
        assert model is not None

    def test_walls_found(self, model):
        """Model contains IfcWall elements."""
        walls = get_elements(model, "IfcWall")
        assert len(walls) == 4

    def test_multiple_entity_types(self, model):
        """Model contains multiple structural entity types."""
        types_found = {}
        for etype in ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
                       "IfcMember", "IfcFooting", "IfcBuildingElementProxy"]:
            elems = get_elements(model, etype)
            if elems:
                types_found[etype] = len(elems)
        assert len(types_found) >= 5, f"Expected >= 5 entity types, got {types_found}"

    def test_wall_pipeline_end_to_end(self, model):
        """Full L1-L4 pipeline runs on all walls without errors."""
        walls = get_elements(model, "IfcWall")
        ruleset = load_ruleset(RULESET_PATH)

        for wall in walls:
            mesh = extract_mesh(wall)
            l1 = validate_level1(mesh)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)
            l4 = validate_level4(l1, l3, ruleset)

            # Plausibility checks on results
            assert l1["volume"] > 1.0, "Wall volume should be > 1 m³"
            assert l1["num_triangles"] > 10
            assert l2["has_crown"]
            assert l2["has_front"] or l2["has_back"]
            assert "crown_width_mm" in l3
            assert l3["crown_width_mm"] > 100  # at least 10cm
            assert l4["summary"]["total"] == 17  # all 17 ASTRA rules evaluated

    def test_multi_body_element_survives(self, model):
        """Elements with multiple bodies (e.g. name signs) don't crash."""
        proxies = get_elements(model, "IfcBuildingElementProxy")
        processed = 0
        for elem in proxies:
            try:
                mesh = extract_mesh(elem)
                l1 = validate_level1(mesh)
                l2 = validate_level2(mesh)
                assert l2.get("n_bodies", 1) >= 1
                processed += 1
            except MeshExtractionError:
                pass  # Some proxies may lack geometry
        assert processed > 0

    def test_l5_inter_element_pairs(self, model):
        """L5 detects geometric relationships between walls and footings."""
        walls = get_elements(model, "IfcWall")
        footings = get_elements(model, "IfcFooting")
        all_elems = []

        for e in walls + footings:
            try:
                mesh = extract_mesh(e)
                l1 = validate_level1(mesh)
                all_elems.append({
                    "element_id": e.id(),
                    "element_name": getattr(e, "Name", "") or f"#{e.id()}",
                    "level1": l1,
                    "mesh_data": mesh,
                })
            except MeshExtractionError:
                pass

        if len(all_elems) >= 2:
            l5 = validate_level5(all_elems)
            assert l5["summary"]["num_elements"] >= 2
            # Pairs may or may not be detected depending on geometry proximity

    def test_mesh_quality_reported(self, model):
        """Mesh quality metrics are populated for real model elements."""
        walls = get_elements(model, "IfcWall")
        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)

        q = l1.get("mesh_quality")
        assert q is not None
        assert "n_degenerate" in q
        assert "non_manifold_edges" in q
        assert "edge_length_median" in q
        assert q["edge_length_median"] > 0
