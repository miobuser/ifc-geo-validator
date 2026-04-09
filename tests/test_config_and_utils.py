"""Tests for auto-config, project config, and utility modules."""

import os
import tempfile
import pytest

from ifc_geo_validator.core.auto_config import auto_configure
from ifc_geo_validator.core.project_config import (
    create_default_config, load_config, find_config, CONFIG_FILENAME,
)
from ifc_geo_validator.core.ifc_parser import load_model


class TestAutoConfig:
    """Test automatic model configuration."""

    def test_t28_detected_as_stuetzbauwerk(self):
        """T28 with 'Stütz' in name → detected as Stützbauwerk."""
        model = load_model("tests/test_models/T28_showcase.ifc")
        config = auto_configure(model)
        assert "Stütz" in config["description"]
        assert config["has_terrain"] is True
        assert "IfcWall" in config["entity_types"]

    def test_t1_has_walls(self):
        """T1 simple box → IfcWall detected."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        config = auto_configure(model)
        assert "IfcWall" in config["entity_types"]
        assert config["element_count"] >= 1

    def test_bridge_multi_type(self):
        """Bridge model → multiple entity types detected."""
        path = os.path.join(
            os.path.expanduser("~"),
            "OneDrive - Berner Fachhochschule",
            "Semester 7", "Minor", "IFC 4.0.2.1", "Infra-Bridge.ifc",
        )
        if not os.path.exists(path):
            pytest.skip("Bridge model not available")
        model = load_model(path)
        config = auto_configure(model)
        assert len(config["entity_types"]) >= 3
        assert config["element_count"] >= 10

    def test_returns_valid_ruleset(self):
        """Auto-config always returns a valid ruleset filename."""
        model = load_model("tests/test_models/T1_simple_box.ifc")
        config = auto_configure(model)
        assert config["ruleset"].endswith(".yaml")


class TestProjectConfig:
    """Test project configuration file."""

    def test_create_default(self):
        """--init creates a valid .igv.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_default_config(tmpdir)
            assert os.path.exists(path)
            config = load_config(path)
            assert "filter_type" in config
            assert "project" in config

    def test_load_config(self):
        """Config file is loadable and has expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_default_config(tmpdir)
            config = load_config(os.path.join(tmpdir, CONFIG_FILENAME))
            assert config["auto"] is True
            assert isinstance(config["filter_type"], list)
            assert isinstance(config["levels"], list)

    def test_find_config_in_parent(self):
        """Config file found in parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_default_config(tmpdir)
            subdir = os.path.join(tmpdir, "sub")
            os.makedirs(subdir)
            found = find_config(subdir)
            assert found is not None
            assert CONFIG_FILENAME in found

    def test_find_config_missing(self):
        """No config file → returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            found = find_config(tmpdir)
            assert found is None

    def test_custom_config_values(self):
        """Custom values in config are preserved."""
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, CONFIG_FILENAME)
            with open(path, "w") as f:
                yaml.dump({
                    "project": "Test Projekt",
                    "author": "Test Author",
                    "filter_type": ["IfcWall", "IfcSlab"],
                    "distances": True,
                }, f)
            config = load_config(path)
            assert config["project"] == "Test Projekt"
            assert config["author"] == "Test Author"
            assert config["distances"] is True
            assert "IfcSlab" in config["filter_type"]
