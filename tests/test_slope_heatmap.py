"""Tests for slope heatmap computation and alignment centerline.

Verifies mathematical correctness of per-triangle slope decomposition
into cross-slope (Quergefälle) and longitudinal slope (Längsgefälle),
with both global axis and local centerline frames.
"""

import numpy as np
import pytest

from ifc_geo_validator.core.face_classifier import WallCenterline
from ifc_geo_validator.viz.slope_heatmap import (
    compute_triangle_slopes,
    compute_surface_slopes,
)


# ── Helpers ───────────────────────────────────────────────────────

def _make_mesh(verts, faces):
    """Build mesh_data from vertices and faces arrays."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
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


def _flat_quad(z=0, slope_y_pct=0.0):
    """Create a flat quad (2 triangles) with optional cross-slope in Y direction.

    slope_y_pct: the Y-direction slope in percent.
    Wall axis is along X, so Y-slope = cross-slope.
    """
    # A 10m x 2m quad, with Z rising in Y by slope_y_pct/100 * 2m
    dz = slope_y_pct / 100.0 * 2.0
    verts = np.array([
        [0, 0, z], [10, 0, z], [10, 2, z + dz], [0, 2, z + dz],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return verts, faces


# ── WallCenterline.from_polyline ──────────────────────────────────

class TestFromPolyline:
    """Test WallCenterline.from_polyline factory."""

    def test_straight_line(self):
        pts = np.array([[0, 0], [5, 0], [10, 0]])
        cl = WallCenterline.from_polyline(pts)
        assert cl.is_curved is False
        assert abs(cl.length - 10.0) < 0.01
        np.testing.assert_allclose(cl.tangents[0], [1, 0, 0], atol=0.01)

    def test_90_degree_arc(self):
        angles = np.linspace(0, np.pi / 2, 50)
        pts = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
        cl = WallCenterline.from_polyline(pts, source="test_arc")
        assert cl.is_curved is True
        assert abs(cl.length - 10 * np.pi / 2) < 0.5  # ~15.7m
        assert cl.to_dict()["source"] == "test_arc"

    def test_s_curve(self):
        t = np.linspace(0, 2 * np.pi, 100)
        pts = np.column_stack([t * 5, 3 * np.sin(t)])
        cl = WallCenterline.from_polyline(pts)
        assert cl.is_curved is True
        assert cl.length > 30

    def test_tangent_perpendicular(self):
        """Tangent and normal are perpendicular at every point."""
        angles = np.linspace(0, np.pi, 40)
        pts = np.column_stack([5 * np.cos(angles), 5 * np.sin(angles)])
        cl = WallCenterline.from_polyline(pts)
        for i in range(len(cl.tangents)):
            dot = np.dot(cl.tangents[i], cl.normals[i])
            assert abs(dot) < 1e-10, f"Not perpendicular at point {i}: dot={dot}"

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            WallCenterline.from_polyline(np.array([[0, 0]]))

    def test_get_local_frame(self):
        """Local frame changes along a curve."""
        angles = np.linspace(0, np.pi / 2, 50)
        pts = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
        cl = WallCenterline.from_polyline(pts)

        # At start (angle=0): tangent ≈ [0, 1], normal ≈ [-1, 0]
        t0, n0, _ = cl.get_local_frame(pts[0])
        assert abs(t0[1]) > 0.9  # mostly Y-direction

        # At end (angle=90°): tangent ≈ [-1, 0], normal ≈ [0, -1]
        te, ne, _ = cl.get_local_frame(pts[-1])
        assert abs(te[0]) > 0.9  # mostly X-direction


# ── compute_triangle_slopes ──────────────────────────────────────

class TestTriangleSlopes:
    """Test per-triangle slope computation."""

    def test_flat_surface_zero_slope(self):
        """Perfectly flat surface has 0% slope everywhere."""
        mesh = _make_mesh(*_flat_quad(slope_y_pct=0))
        slopes = compute_triangle_slopes(mesh, axis=np.array([1, 0, 0]))
        np.testing.assert_allclose(slopes["total_slope_pct"], 0.0, atol=0.01)
        np.testing.assert_allclose(slopes["cross_slope_pct"], 0.0, atol=0.01)
        np.testing.assert_allclose(slopes["long_slope_pct"], 0.0, atol=0.01)

    def test_3pct_cross_slope(self):
        """3% cross-slope (Y-direction) with wall axis along X."""
        mesh = _make_mesh(*_flat_quad(slope_y_pct=3.0))
        slopes = compute_triangle_slopes(mesh, axis=np.array([1, 0, 0]))
        np.testing.assert_allclose(slopes["cross_slope_pct"], 3.0, atol=0.05)
        np.testing.assert_allclose(slopes["long_slope_pct"], 0.0, atol=0.05)

    def test_5pct_cross_slope(self):
        """5% cross-slope (ASTRA max)."""
        mesh = _make_mesh(*_flat_quad(slope_y_pct=5.0))
        slopes = compute_triangle_slopes(mesh, axis=np.array([1, 0, 0]))
        np.testing.assert_allclose(slopes["cross_slope_pct"], 5.0, atol=0.1)

    def test_longitudinal_slope(self):
        """Slope along the wall axis = longitudinal slope."""
        # Z rises by 0.3m over 10m in X direction = 3% Längsgefälle
        verts = np.array([
            [0, 0, 0], [10, 0, 0.3], [10, 2, 0.3], [0, 2, 0],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = _make_mesh(verts, faces)
        slopes = compute_triangle_slopes(mesh, axis=np.array([1, 0, 0]))
        np.testing.assert_allclose(slopes["long_slope_pct"], 3.0, atol=0.1)
        np.testing.assert_allclose(slopes["cross_slope_pct"], 0.0, atol=0.1)

    def test_combined_slope(self):
        """Both cross and longitudinal slope present."""
        # 3% cross (Y) + 2% longitudinal (X)
        verts = np.array([
            [0, 0, 0], [10, 0, 0.2], [10, 2, 0.2 + 0.06], [0, 2, 0.06],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = _make_mesh(verts, faces)
        slopes = compute_triangle_slopes(mesh, axis=np.array([1, 0, 0]))
        np.testing.assert_allclose(slopes["cross_slope_pct"], 3.0, atol=0.15)
        np.testing.assert_allclose(slopes["long_slope_pct"], 2.0, atol=0.15)

    def test_vertical_face_clamped(self):
        """Vertical face slope is clamped to 9000%."""
        verts = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 3], [0, 0, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = _make_mesh(verts, faces)
        slopes = compute_triangle_slopes(mesh)
        assert all(s >= 9000 for s in slopes["total_slope_pct"])


# ── Local frame for curves ────────────────────────────────────────

class TestLocalFrameSlopes:
    """Test that local frame gives correct cross-slope on curves."""

    def test_curved_uniform_slope(self):
        """A curved surface with uniform 3% tilt should report ~3% everywhere."""
        # Build a 90° arc road surface with 3% cross-slope
        n_seg = 20
        angles = np.linspace(0, np.pi / 2, n_seg + 1)
        r_inner, r_outer = 9.8, 10.2  # 400mm wide road

        all_verts = []
        all_faces = []
        for i in range(n_seg):
            a0, a1 = angles[i], angles[i + 1]
            # Inner edge at Z=0, outer edge at Z=0.012 (3% of 0.4m = 0.012m)
            dz = 0.03 * (r_outer - r_inner)  # 3% cross slope
            v_base = len(all_verts)
            all_verts.extend([
                [r_inner * np.cos(a0), r_inner * np.sin(a0), 0],
                [r_outer * np.cos(a0), r_outer * np.sin(a0), dz],
                [r_inner * np.cos(a1), r_inner * np.sin(a1), 0],
                [r_outer * np.cos(a1), r_outer * np.sin(a1), dz],
            ])
            all_faces.extend([
                [v_base, v_base + 1, v_base + 3],
                [v_base, v_base + 3, v_base + 2],
            ])

        mesh = _make_mesh(np.array(all_verts), np.array(all_faces))

        # Build centerline (arc midline)
        r_mid = (r_inner + r_outer) / 2
        cl_pts = np.column_stack([r_mid * np.cos(angles), r_mid * np.sin(angles)])
        cl = WallCenterline.from_polyline(cl_pts)

        # With local frame: should get ~3% cross-slope everywhere
        slopes_local = compute_triangle_slopes(mesh, centerline=cl)
        cross_local = slopes_local["cross_slope_pct"]

        # With global frame: will get varying values due to rotating axis
        slopes_global = compute_triangle_slopes(mesh)
        cross_global = slopes_global["cross_slope_pct"]

        # Local frame should give uniform ~3% (within tessellation noise)
        assert cross_local.std() < cross_local.mean() * 0.15, (
            f"Local frame cross-slope should be uniform: mean={cross_local.mean():.2f}%, "
            f"std={cross_local.std():.2f}%"
        )
        # Global frame should show more variation
        assert cross_global.std() > cross_local.std() * 1.5 or n_seg < 5, (
            "Global frame should have more variation than local on curves"
        )
        # Local mean should be close to 3%
        assert abs(cross_local.mean() - 3.0) < 0.5, (
            f"Expected ~3% cross-slope, got {cross_local.mean():.2f}%"
        )


# ── compute_surface_slopes ────────────────────────────────────────

class TestSurfaceSlopes:
    """Test category-filtered slope computation."""

    def test_category_filter(self):
        """Only selected category faces are included in statistics."""
        from ifc_geo_validator.validation.level2 import validate_level2

        # Simple wall box (8x0.4x3m)
        def box(o, s):
            x0, y0, z0 = o
            x1, y1, z1 = o[0]+s[0], o[1]+s[1], o[2]+s[2]
            v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                          [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
            f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                          [2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5]])
            return v, f

        mesh = _make_mesh(*box([0, 0, 0], [8, 0.4, 3]))
        l2 = validate_level2(mesh)

        # Crown only
        slopes_crown = compute_surface_slopes(
            mesh, l2["face_groups"], categories=["crown"]
        )
        assert slopes_crown is not None
        assert int(slopes_crown["face_mask"].sum()) == 2  # 2 crown triangles

        # Front only
        slopes_front = compute_surface_slopes(
            mesh, l2["face_groups"], categories=["front"]
        )
        assert slopes_front is not None
        assert int(slopes_front["face_mask"].sum()) > 0

        # Non-existent category
        slopes_none = compute_surface_slopes(
            mesh, l2["face_groups"], categories=["nonexistent"]
        )
        assert slopes_none is None

    def test_statistics_populated(self):
        """Slope statistics (min, max, avg) are computed correctly."""
        mesh = _make_mesh(*_flat_quad(slope_y_pct=3.0))
        groups = [{"category": "crown", "face_indices": [0, 1]}]
        slopes = compute_surface_slopes(mesh, groups, categories=["crown"],
                                        axis=np.array([1, 0, 0]))
        assert slopes is not None
        assert abs(slopes["area_weighted_cross_pct"] - 3.0) < 0.1
        assert abs(slopes["min_cross_pct"] - 3.0) < 0.1
        assert abs(slopes["max_cross_pct"] - 3.0) < 0.1


# ── Integration with real test models ─────────────────────────────

class TestSlopeOnTestModels:
    """Test slope computation on the actual test models."""

    def test_t3_crown_slope_3pct(self):
        """T3 has exactly 3% crown slope."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level2 import validate_level2

        model = load_model("tests/test_models/T3_crown_slope.ifc")
        elem = get_elements(model, "IfcWall")[0]
        mesh = extract_mesh(elem)
        l2 = validate_level2(mesh)
        slopes = compute_surface_slopes(
            mesh, l2["face_groups"], categories=["crown"],
            axis=np.array(l2["wall_axis"]),
            centerline=l2.get("centerline"),
        )
        assert slopes is not None
        assert abs(slopes["area_weighted_cross_pct"] - 3.0) < 0.1

    def test_t1_flat_crown(self):
        """T1 simple box has 0% crown slope."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level2 import validate_level2

        model = load_model("tests/test_models/T1_simple_box.ifc")
        elem = get_elements(model, "IfcWall")[0]
        mesh = extract_mesh(elem)
        l2 = validate_level2(mesh)
        slopes = compute_surface_slopes(
            mesh, l2["face_groups"], categories=["crown"],
            axis=np.array(l2["wall_axis"]),
        )
        assert slopes is not None
        assert slopes["area_weighted_cross_pct"] < 0.1
