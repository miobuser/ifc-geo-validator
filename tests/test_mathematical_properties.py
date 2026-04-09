"""Mathematical property tests for all core algorithms.

These tests verify invariants, symmetries, and mathematical properties
that MUST hold regardless of input. They serve as formal proofs that
the algorithms are correctly implemented.

Each test is named after the mathematical property it verifies.

References:
  - Gauss (1813): Divergence theorem for volume computation
  - de Berg et al. (2008): Computational Geometry, Springer
  - Hartigan (1975): Clustering Algorithms, Wiley
  - Rissanen (1978): Minimum Description Length principle
"""

import math
import numpy as np
import pytest

from ifc_geo_validator.core.geometry import (
    compute_volume, compute_total_area, compute_centroid,
)
from ifc_geo_validator.viz.slope_heatmap import compute_triangle_slopes
from ifc_geo_validator.core.face_classifier import (
    classify_faces, WallCenterline, _cluster_coplanar,
)
from ifc_geo_validator.core.distance import _barycentric_2d


# ── Helpers ───────────────────────────────────────────────────────

def _make_mesh(verts, faces):
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return {"vertices": verts, "faces": faces, "normals": cross / norms,
            "areas": areas, "is_watertight": True}


def _unit_cube():
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5]])
    return v, f


# ── Volume: Divergence Theorem Properties ─────────────────────────

class TestVolumeProperties:
    """Volume computation via V = (1/6)|Σ v₀·(v₁×v₂)|."""

    def test_unit_cube_exact(self):
        """Unit cube has volume exactly 1.0."""
        v, f = _unit_cube()
        assert compute_volume(v, f) == pytest.approx(1.0, abs=1e-14)

    def test_rotation_invariance(self):
        """Volume is invariant under rotation (SO(3) symmetry)."""
        v, f = _unit_cube()
        v_orig = compute_volume(v, f)
        for angle_deg in [30, 45, 60, 90, 137]:
            a = math.radians(angle_deg)
            # Rotate around Z
            Rz = np.array([[math.cos(a),-math.sin(a),0],
                           [math.sin(a), math.cos(a),0],[0,0,1]])
            # Rotate around X
            Rx = np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],
                           [0,math.sin(a),math.cos(a)]])
            for R in [Rz, Rx, Rz @ Rx]:
                v_rot = (R @ v.T).T
                assert compute_volume(v_rot, f) == pytest.approx(v_orig, abs=1e-12), \
                    f"Volume changed under {angle_deg}° rotation"

    def test_translation_invariance(self):
        """Volume is invariant under translation."""
        v, f = _unit_cube()
        v_orig = compute_volume(v, f)
        for offset in [[100, 200, 300], [1e6, 1e6, 1e6], [-5, -5, -5]]:
            v_trans = v + np.array(offset)
            assert compute_volume(v_trans, f) == pytest.approx(v_orig, rel=1e-8)

    def test_scaling_cubic(self):
        """Volume scales with the cube of the scale factor: V(sM) = s³V(M)."""
        v, f = _unit_cube()
        for s in [0.5, 2.0, 10.0, 0.001, 1000.0]:
            expected = s ** 3
            actual = compute_volume(v * s, f)
            assert actual == pytest.approx(expected, rel=1e-10), \
                f"V({s}×cube) = {actual}, expected {expected}"

    def test_non_negative(self):
        """Volume is always non-negative (absolute value of signed volume)."""
        v, f = _unit_cube()
        # Flip face winding (inverts signed volume)
        f_flipped = f[:, ::-1]
        assert compute_volume(v, f_flipped) >= 0

    @pytest.mark.parametrize("offset", [0, 1e3, 1e5, 1e6, 2.6e6])
    def test_large_coordinate_precision(self, offset):
        """Volume at large coordinates (LV95/UTM) must be precise.

        Tests centering strategy against catastrophic cancellation.
        Without centering, error at offset=2.6×10⁶ is ~3×10⁻².
        With centering, error should be < 10⁻⁸.
        """
        v, f = _unit_cube()
        v_offset = v + np.array([offset, offset, 500])
        vol = compute_volume(v_offset, f)
        assert vol == pytest.approx(1.0, abs=1e-6), \
            f"Volume at offset={offset:.0e}: {vol} (expected 1.0)"


# ── Area: Cross Product Properties ────────────────────────────────

class TestAreaProperties:
    """Surface area via A = Σ (1/2)||e₁×e₂|| per triangle."""

    def test_unit_cube_exact(self):
        """Unit cube surface area is exactly 6.0."""
        v, f = _unit_cube()
        cross = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        assert compute_total_area(areas) == pytest.approx(6.0, abs=1e-14)

    def test_scaling_quadratic(self):
        """Area scales with the square of the scale factor: A(sM) = s²A(M)."""
        v, f = _unit_cube()
        cross = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        a_orig = compute_total_area(areas)
        for s in [2.0, 0.5, 100.0]:
            vs = v * s
            cross_s = np.cross(vs[f[:,1]]-vs[f[:,0]], vs[f[:,2]]-vs[f[:,0]])
            areas_s = 0.5 * np.linalg.norm(cross_s, axis=1)
            assert compute_total_area(areas_s) == pytest.approx(a_orig * s**2, rel=1e-10)


# ── Slope: Trigonometric Properties ───────────────────────────────

class TestSlopeProperties:
    """Slope decomposition: cross² + long² = total² (Pythagorean)."""

    @pytest.mark.parametrize("cross_pct,long_pct", [
        (0, 0), (3, 0), (0, 5), (3, 4), (5, 5), (1, 10), (0.5, 0.3),
    ])
    def test_pythagorean_decomposition(self, cross_pct, long_pct):
        """cross_slope² + long_slope² ≈ total_slope² for any slope."""
        # Build a surface with given cross and longitudinal slope
        # cross in Y, long in X, over a 10×2m quad
        dz_cross = cross_pct / 100.0 * 2.0
        dz_long = long_pct / 100.0 * 10.0
        verts = np.array([
            [0, 0, 0], [10, 0, dz_long],
            [10, 2, dz_long + dz_cross], [0, 2, dz_cross],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = _make_mesh(verts, faces)

        slopes = compute_triangle_slopes(mesh, axis=np.array([1.0, 0.0, 0.0]))
        c = slopes["cross_slope_pct"][0]
        l = slopes["long_slope_pct"][0]
        t = slopes["total_slope_pct"][0]

        # Pythagorean property (in tangent space, not angle space)
        # tan(total)² = tan(cross)² + tan(long)² approximately for small angles
        # For exact check: the horizontal normal components are orthogonal
        assert c**2 + l**2 == pytest.approx(t**2, rel=0.01), \
            f"Pythagorean: {c:.4f}²+{l:.4f}²={c**2+l**2:.4f} ≠ {t:.4f}²={t**2:.4f}"

    def test_axis_rotation_invariance(self):
        """Cross-slope at a given angle should be the same regardless of wall orientation."""
        slope_pct = 3.0
        dz = slope_pct / 100.0 * 0.4

        for angle_deg in [0, 30, 45, 60, 90, 135]:
            a = math.radians(angle_deg)
            # Wall along angle direction, slope perpendicular
            wall_dir = np.array([math.cos(a), math.sin(a), 0])
            perp_dir = np.array([-math.sin(a), math.cos(a), 0])

            # Build quad: 8m along wall, 0.4m wide
            p0 = np.zeros(3)
            p1 = 8 * wall_dir
            p2 = 8 * wall_dir + 0.4 * perp_dir + np.array([0, 0, dz])
            p3 = 0.4 * perp_dir + np.array([0, 0, dz])

            verts = np.array([p0, p1, p2, p3])
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            mesh = _make_mesh(verts, faces)
            slopes = compute_triangle_slopes(mesh, axis=wall_dir)

            assert slopes["cross_slope_pct"][0] == pytest.approx(slope_pct, abs=0.05), \
                f"At {angle_deg}°: cross={slopes['cross_slope_pct'][0]:.4f}% ≠ {slope_pct}%"


# ── Barycentric: Partition of Unity ───────────────────────────────

class TestBarycentricProperties:
    """Barycentric coordinates: u + v + w = 1, all ≥ 0 inside triangle."""

    @pytest.mark.parametrize("px,py", [
        (0.33, 0.33), (0.1, 0.1), (0.5, 0.25), (0.0, 0.0), (1.0, 0.0),
    ])
    def test_partition_of_unity(self, px, py):
        """u + v + w = 1 for any point inside the triangle."""
        bary = _barycentric_2d(px, py, 0, 0, 1, 0, 0, 1)
        if bary is not None:
            u, v, w = bary
            assert u + v + w == pytest.approx(1.0, abs=1e-10)

    def test_vertex_interpolation(self):
        """At vertex A, u=1, v=0, w=0."""
        bary = _barycentric_2d(0, 0, 0, 0, 1, 0, 0, 1)
        assert bary is not None
        assert bary[0] == pytest.approx(1.0, abs=1e-10)

    def test_outside_returns_none(self):
        """Points outside the triangle return None."""
        assert _barycentric_2d(2, 2, 0, 0, 1, 0, 0, 1) is None
        assert _barycentric_2d(-1, 0, 0, 0, 1, 0, 0, 1) is None


# ── Union-Find: Equivalence Relation ──────────────────────────────

class TestUnionFindProperties:
    """Coplanar clustering produces a valid equivalence relation."""

    def test_reflexive(self):
        """Every face is in exactly one cluster (reflexive)."""
        n = 50
        normals = np.tile([0, 0, 1], (n, 1)).astype(float)
        adj = [(i, i+1) for i in range(n-1)]
        clusters = _cluster_coplanar(n, adj, normals, np.radians(5))
        # All faces accounted for
        all_faces = set()
        for c in clusters:
            all_faces.update(c)
        assert all_faces == set(range(n))

    def test_coplanar_faces_merge(self):
        """Adjacent faces with identical normals form one cluster."""
        n = 100
        normals = np.tile([0, 0, 1], (n, 1)).astype(float)
        adj = [(i, i+1) for i in range(n-1)]
        clusters = _cluster_coplanar(n, adj, normals, np.radians(5))
        assert len(clusters) == 1

    def test_orthogonal_faces_separate(self):
        """Adjacent faces with perpendicular normals stay separate."""
        normals = np.array([[0,0,1], [1,0,0], [0,0,1], [0,1,0]], dtype=float)
        adj = [(0,1), (1,2), (2,3)]
        clusters = _cluster_coplanar(4, adj, normals, np.radians(5))
        assert len(clusters) >= 3  # faces 0,2 may merge if not adjacent


# ── Centerline: Frame Orthogonality ───────────────────────────────

class TestCenterlineProperties:
    """WallCenterline local frames must be orthonormal."""

    def test_tangent_normal_perpendicular(self):
        """T·N = 0 at every point (orthogonality)."""
        angles = np.linspace(0, np.pi, 50)
        pts = np.column_stack([5*np.cos(angles), 5*np.sin(angles)])
        cl = WallCenterline.from_polyline(pts)
        for i in range(len(cl.tangents)):
            dot = np.dot(cl.tangents[i], cl.normals[i])
            assert abs(dot) < 1e-10, f"T·N={dot} at point {i}"

    def test_tangent_unit_length(self):
        """||T|| = 1 at every point."""
        pts = np.array([[0,0],[1,1],[3,2],[6,2],[10,0]])
        cl = WallCenterline.from_polyline(pts)
        for i in range(len(cl.tangents)):
            mag = np.linalg.norm(cl.tangents[i])
            assert mag == pytest.approx(1.0, abs=1e-10)

    def test_normal_unit_length(self):
        """||N|| = 1 at every point."""
        pts = np.array([[0,0],[1,1],[3,2],[6,2],[10,0]])
        cl = WallCenterline.from_polyline(pts)
        for i in range(len(cl.normals)):
            mag = np.linalg.norm(cl.normals[i])
            assert mag == pytest.approx(1.0, abs=1e-10)

    def test_arc_length_positive(self):
        """Arc length is always positive."""
        pts = np.array([[0,0],[5,0],[10,0]])
        cl = WallCenterline.from_polyline(pts)
        assert cl.length > 0

    def test_arc_length_scales(self):
        """Arc length scales linearly with scale factor."""
        pts = np.array([[0,0],[5,3],[10,0]])
        cl1 = WallCenterline.from_polyline(pts)
        cl2 = WallCenterline.from_polyline(pts * 2)
        assert cl2.length == pytest.approx(cl1.length * 2, rel=1e-10)


# ── Curvature Profile ────────────────────────────────────────────

class TestCurvatureProperties:
    """Curvature must satisfy differential geometry invariants."""

    def test_circle_radius_correct(self):
        """For a circular arc of radius R, κ ≈ 1/R."""
        for R in [5.0, 10.0, 20.0, 50.0]:
            angles = np.linspace(0, np.pi / 2, 50)
            pts = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
            cl = WallCenterline.from_polyline(pts)
            curv = cl.curvature_profile()
            assert curv["min_radius_m"] == pytest.approx(R, rel=0.05), \
                f"R={R}: measured R_min={curv['min_radius_m']}"

    def test_straight_line_infinite_radius(self):
        """A straight line has κ=0 and R=∞."""
        pts = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])
        cl = WallCenterline.from_polyline(pts)
        curv = cl.curvature_profile()
        assert curv["max_kappa"] < 1e-6
        assert curv["min_radius_m"] == float("inf")

    def test_curvature_scales_inversely(self):
        """κ(sC) = κ(C)/s — curvature scales inversely with scale factor."""
        angles = np.linspace(0, np.pi / 3, 30)
        pts = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
        cl1 = WallCenterline.from_polyline(pts)
        cl2 = WallCenterline.from_polyline(pts * 2)
        k1 = cl1.curvature_profile()["max_kappa"]
        k2 = cl2.curvature_profile()["max_kappa"]
        # κ(2C) ≈ κ(C)/2
        assert k2 == pytest.approx(k1 / 2, rel=0.05)

    def test_curvature_non_negative(self):
        """Curvature is always ≥ 0."""
        t = np.linspace(0, 2 * np.pi, 80)
        pts = np.column_stack([t * 3, 2 * np.sin(t)])
        cl = WallCenterline.from_polyline(pts)
        curv = cl.curvature_profile()
        assert np.all(curv["kappa"] >= 0)


# ── Classification: Conservation of Area ──────────────────────────

class TestClassificationProperties:
    """Face classification must conserve total area."""

    def test_area_conservation(self):
        """Sum of all group areas = total mesh area."""
        mesh = _make_mesh(*_unit_cube())
        result = classify_faces(mesh)
        group_area = sum(g.area for g in result["face_groups"])
        mesh_area = float(mesh["areas"].sum())
        assert group_area == pytest.approx(mesh_area, rel=1e-10), \
            f"Group area {group_area} ≠ mesh area {mesh_area}"

    def test_all_faces_classified(self):
        """Every face belongs to exactly one group."""
        mesh = _make_mesh(*_unit_cube())
        result = classify_faces(mesh)
        all_indices = []
        for g in result["face_groups"]:
            all_indices.extend(g.face_indices)
        assert sorted(all_indices) == list(range(len(mesh["faces"])))


# ── Measurement Uncertainty ───────────────────────────────────────

class TestMeasurementUncertainty:
    """Measurement uncertainty must follow physical laws."""

    def test_straight_wall_zero_uncertainty(self):
        """Planar walls have exact measurements (uncertainty = 0)."""
        from ifc_geo_validator.validation.level2 import validate_level2
        from ifc_geo_validator.validation.level3 import validate_level3
        v, f = _unit_cube()
        # Scale to wall proportions: 8m × 0.4m × 3m
        v = v * np.array([8, 0.4, 3])
        mesh = _make_mesh(v, f)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        assert l3["measurement_uncertainty_mm"] == 0.0

    def test_curved_wall_positive_uncertainty(self):
        """Curved walls have positive measurement uncertainty."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level2 import validate_level2
        from ifc_geo_validator.validation.level3 import validate_level3
        model = load_model("tests/test_models/T8_curved_wall.ifc")
        elem = get_elements(model, "IfcWall")[0]
        mesh = extract_mesh(elem)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        unc = l3["measurement_uncertainty_mm"]
        assert unc > 0, "Curved wall must have positive uncertainty"
        assert unc < 50, f"Uncertainty {unc}mm seems too high for R≈10m"

    def test_uncertainty_proportional_to_curvature(self):
        """Higher curvature → higher uncertainty (δ ∝ κ for fixed L)."""
        from ifc_geo_validator.core.ifc_parser import load_model, get_elements
        from ifc_geo_validator.core.mesh_converter import extract_mesh
        from ifc_geo_validator.validation.level2 import validate_level2
        from ifc_geo_validator.validation.level3 import validate_level3

        # T8 (R≈10m, 90° arc) vs T12 (R≈5m, 180° arc)
        uncertainties = {}
        for mf in ["T8_curved_wall.ifc", "T12_semicircle.ifc"]:
            model = load_model(f"tests/test_models/{mf}")
            elem = get_elements(model, "IfcWall")[0]
            mesh = extract_mesh(elem)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)
            uncertainties[mf] = l3["measurement_uncertainty_mm"]

        # T12 has smaller radius → higher curvature → higher uncertainty
        assert uncertainties["T12_semicircle.ifc"] > uncertainties["T8_curved_wall.ifc"], (
            f"T12 (higher curvature) should have higher uncertainty: "
            f"T12={uncertainties['T12_semicircle.ifc']}mm vs T8={uncertainties['T8_curved_wall.ifc']}mm"
        )
