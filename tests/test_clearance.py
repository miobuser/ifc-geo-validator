"""Tests for clearance profile checking (Lichtraumprofil)."""

import numpy as np
import pytest

from ifc_geo_validator.validation.clearance import (
    check_clearance, _points_in_polygon, astra_road_clearance,
)
from ifc_geo_validator.core.face_classifier import WallCenterline


# ── Point-in-Polygon Tests ────────────────────────────────────────

class TestPointInPolygon:
    """Verify ray casting algorithm for point-in-polygon test."""

    def test_inside_square(self):
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        pts = np.array([[1, 1]])
        assert _points_in_polygon(pts, poly)[0] == True

    def test_outside_square(self):
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        pts = np.array([[3, 1]])
        assert _points_in_polygon(pts, poly)[0] == False

    def test_multiple_points(self):
        poly = np.array([[0, 0], [4, 0], [4, 3], [0, 3]])
        pts = np.array([[2, 1.5], [-1, 0], [4.1, 3], [2, 0.01]])
        result = _points_in_polygon(pts, poly)
        assert result[0] == True   # inside
        assert result[1] == False  # outside left
        assert result[2] == False  # outside right
        assert result[3] == True   # just inside

    def test_triangle(self):
        poly = np.array([[0, 0], [1, 0], [0.5, 1]])
        inside = np.array([[0.5, 0.3]])
        outside = np.array([[0.5, 1.1]])
        assert _points_in_polygon(inside, poly)[0] == True
        assert _points_in_polygon(outside, poly)[0] == False


# ── Clearance Profile Tests ───────────────────────────────────────

class TestClearanceCheck:
    """Test clearance envelope checking."""

    def _straight_centerline(self, length=10.0):
        pts = np.array([[0, 0], [length, 0]])
        return WallCenterline.from_polyline(pts)

    def test_no_violation(self):
        """Element outside clearance envelope → clear."""
        # Wall at Y=5 (outside the ±4m envelope)
        verts = np.array([
            [0, 5, 0], [10, 5, 0], [10, 5.4, 0], [0, 5.4, 0],
            [0, 5, 3], [10, 5, 3], [10, 5.4, 3], [0, 5.4, 3],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = {"vertices": verts, "faces": faces}
        cl = self._straight_centerline()
        profile = astra_road_clearance(8.0, 4.5)

        result = check_clearance(mesh, cl, profile, n_slices=5)
        assert result["clear"] == True
        assert result["n_violations"] == 0

    def test_violation_detected(self):
        """Element inside clearance envelope → violation."""
        # Wall at Y=0.5 (inside the ±4m envelope)
        verts = np.array([
            [0, 0.5, 0.5], [10, 0.5, 0.5], [10, 0.9, 0.5], [0, 0.9, 0.5],
            [0, 0.5, 3.5], [10, 0.5, 3.5], [10, 0.9, 3.5], [0, 0.9, 3.5],
        ], dtype=float)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ])
        mesh = {"vertices": verts, "faces": faces}
        cl = self._straight_centerline()
        profile = astra_road_clearance(8.0, 4.5)

        result = check_clearance(mesh, cl, profile, n_slices=5)
        assert result["clear"] == False
        assert result["n_violations"] > 0
        assert result["max_penetration_mm"] > 0

    def test_no_centerline_returns_clear(self):
        """Without centerline, assume clear (can't check)."""
        mesh = {"vertices": np.zeros((4, 3)), "faces": np.array([[0, 1, 2]])}
        result = check_clearance(mesh, None, np.zeros((4, 2)))
        assert result["clear"] == True

    def test_astra_profile_dimensions(self):
        """ASTRA road clearance has correct dimensions."""
        profile = astra_road_clearance(8.0, 4.5)
        assert len(profile) == 4
        assert profile[:, 0].min() == -4.0  # half width
        assert profile[:, 0].max() == 4.0
        assert profile[:, 1].max() == 4.5   # height
