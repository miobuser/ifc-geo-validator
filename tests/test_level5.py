"""Tests for Level 5: Inter-element geometric context.

Tests pair detection, classification, gap measurement, and foundation
overhang checks for multi-element retaining wall models.
"""

import os
import pytest

from ifc_geo_validator.validation.level5 import validate_level5


# ── Helper ──────────────────────────────────────────────────────────

def _load_and_prepare(model_path):
    """Load model, extract meshes, run L1 for each element."""
    from ifc_geo_validator.core.ifc_parser import load_model, get_elements
    from ifc_geo_validator.core.mesh_converter import extract_mesh
    from ifc_geo_validator.validation.level1 import validate_level1

    model = load_model(model_path)
    walls = get_elements(model, "IfcWall")

    elements_data = []
    for w in walls:
        mesh = extract_mesh(w)
        l1 = validate_level1(mesh)
        elements_data.append({
            "element_id": w.id(),
            "element_name": getattr(w, "Name", "Unnamed"),
            "level1": l1,
            "mesh_data": mesh,
        })
    return elements_data


# ── Single element (no pairs) ──────────────────────────────────────

T1_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T1_simple_box.ifc")


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 model not found")
class TestSingleElement:
    """Single-element model should produce no pairs."""

    def test_no_pairs(self):
        data = _load_and_prepare(T1_PATH)
        l5 = validate_level5(data)
        assert l5["summary"]["num_pairs"] == 0
        assert l5["summary"]["num_elements"] == 1


# ── T4: Wall stem + foundation (stacked) ───────────────────────────

T4_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T4_l_shaped.ifc")


@pytest.mark.skipif(not os.path.exists(T4_PATH), reason="T4 model not found")
class TestT4StemFoundation:
    """T4: Mauersteg sits on Fundament (stacked pair)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T4_PATH)
        self.l5 = validate_level5(data)

    def test_one_pair(self):
        assert self.l5["summary"]["num_pairs"] == 1

    def test_stacked(self):
        pair = self.l5["pairs"][0]
        assert pair["pair_type"] == "stacked"

    def test_no_gap(self):
        """Stem bottom touches foundation top → gap ≈ 0mm."""
        pair = self.l5["pairs"][0]
        assert abs(pair["vertical_gap_mm"]) < 1.0

    def test_foundation_extends(self):
        """Foundation (2.0m wide) extends beyond stem (0.3m wide)."""
        pair = self.l5["pairs"][0]
        assert pair["foundation_extends_beyond_wall"] is True


# ── T5: Wall + spur (side by side) ────────────────────────────────

T5_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T5_t_shaped.ifc")


@pytest.mark.skipif(not os.path.exists(T5_PATH), reason="T5 model not found")
class TestT5WallAndSpur:
    """T5: Hauptwand + Sporn as side-by-side pair."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T5_PATH)
        self.l5 = validate_level5(data)

    def test_one_pair(self):
        assert self.l5["summary"]["num_pairs"] == 1

    def test_side_by_side(self):
        pair = self.l5["pairs"][0]
        assert pair["pair_type"] == "side_by_side"

    def test_touching(self):
        """Wall and spur touch (gap = 0mm)."""
        pair = self.l5["pairs"][0]
        assert pair["horizontal_gap_mm"] < 1.0


# ── T18: Wall + 3 buttresses ──────────────────────────────────────

T18_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T18_buttressed.ifc")


@pytest.mark.skipif(not os.path.exists(T18_PATH), reason="T18 model not found")
class TestT18Buttresses:
    """T18: Hauptmauer + 3 Strebepfeiler."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T18_PATH)
        self.l5 = validate_level5(data)

    def test_three_pairs(self):
        """Each buttress pairs with the main wall."""
        assert self.l5["summary"]["num_pairs"] == 3

    def test_all_side_by_side(self):
        for pair in self.l5["pairs"]:
            assert pair["pair_type"] == "side_by_side"

    def test_all_touching(self):
        for pair in self.l5["pairs"]:
            assert pair["horizontal_gap_mm"] < 1.0


# ── T9: Upper wall + lower course (stacked) ──────────────────────

T9_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T9_stepped_wall.ifc")


@pytest.mark.skipif(not os.path.exists(T9_PATH), reason="T9 model not found")
class TestT9SteppedStacked:
    """T9: Oberer Steg on Unterer Sockel."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T9_PATH)
        self.l5 = validate_level5(data)

    def test_stacked(self):
        assert self.l5["pairs"][0]["pair_type"] == "stacked"

    def test_no_gap(self):
        assert abs(self.l5["pairs"][0]["vertical_gap_mm"]) < 1.0

    def test_foundation_extends(self):
        """Lower course (0.6m) extends beyond upper (0.3m)."""
        assert self.l5["pairs"][0]["foundation_extends_beyond_wall"] is True


# ── T23: Curved multi-element (stem + foundation + buttress) ─────

T23_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T23_astra_compliant_curved.ifc")


@pytest.mark.skipif(not os.path.exists(T23_PATH), reason="T23 model not found")
class TestT23CurvedMultiElement:
    """T23: Curved wall with 3 elements (stem, foundation, buttress)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T23_PATH)
        self.data = data
        self.l5 = validate_level5(data)

    def test_three_elements(self):
        """T23 has 3 wall elements."""
        assert self.l5["summary"]["num_elements"] == 3

    def test_pairs_detected(self):
        """At least 2 pairs detected among 3 elements."""
        assert self.l5["summary"]["num_pairs"] >= 2

    def test_has_stacked(self):
        """At least one stacked pair (stem on foundation)."""
        stacked = [p for p in self.l5["pairs"] if p["pair_type"] == "stacked"]
        assert len(stacked) >= 1


# ── T24: Highway wall stem + foundation (stacked) ────────────────

T24_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T24_highway_with_terrain.ifc")


@pytest.mark.skipif(not os.path.exists(T24_PATH), reason="T24 model not found")
class TestT24HighwayStacked:
    """T24: Stem + foundation (stacked pair)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T24_PATH)
        self.data = data
        self.l5 = validate_level5(data)

    def test_two_elements(self):
        """T24 has 2 wall elements."""
        assert self.l5["summary"]["num_elements"] == 2

    def test_stacked(self):
        """Stem sits on foundation → stacked pair."""
        assert self.l5["summary"]["num_pairs"] >= 1
        pair = self.l5["pairs"][0]
        assert pair["pair_type"] == "stacked"

    def test_foundation_extends(self):
        """Foundation extends beyond wall stem."""
        pair = self.l5["pairs"][0]
        assert pair["foundation_extends_beyond_wall"] is True


# ── T14: Curved L-profile (Mauersteg + Fundament) ────────────────

T14_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T14_curved_l_profile.ifc")


@pytest.mark.skipif(not os.path.exists(T14_PATH), reason="T14 model not found")
class TestT14CurvedLProfile:
    """T14: Curved Mauersteg + Fundament."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _load_and_prepare(T14_PATH)
        self.data = data
        self.l5 = validate_level5(data)

    def test_two_elements(self):
        """T14 has 2 wall elements."""
        assert self.l5["summary"]["num_elements"] == 2

    def test_stacked_curved(self):
        """Stacked pair detected even with curved geometry."""
        assert self.l5["summary"]["num_pairs"] >= 1
        stacked = [p for p in self.l5["pairs"] if p["pair_type"] == "stacked"]
        assert len(stacked) >= 1

    def test_no_gap(self):
        """Gap between stem and foundation <= 1mm."""
        stacked = [p for p in self.l5["pairs"] if p["pair_type"] == "stacked"]
        assert len(stacked) >= 1
        assert abs(stacked[0]["vertical_gap_mm"]) <= 1.0


# ── T27: Single element (long curved slope) ──────────────────────

T27_PATH = os.path.join(os.path.dirname(__file__), "test_models", "T27_long_curved_slope.ifc")


@pytest.mark.skipif(not os.path.exists(T27_PATH), reason="T27 model not found")
class TestT27SingleElement:
    """T27 is a single element — L5 should return empty pairs gracefully."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = _load_and_prepare(T27_PATH)
        self.l5 = validate_level5(self.data)

    def test_single_element(self):
        assert self.l5["summary"]["num_elements"] == 1

    def test_no_pairs(self):
        assert self.l5["summary"]["num_pairs"] == 0
        assert self.l5["pairs"] == []
