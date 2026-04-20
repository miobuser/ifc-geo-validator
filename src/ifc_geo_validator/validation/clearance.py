"""Clearance profile checking (Lichtraumprofil).

Verifies that structural elements do not intrude into a defined
clearance envelope. The envelope is specified as a 2D polygon in the
cross-section plane (perpendicular to the alignment/wall axis).

Algorithm:
  1. Slice the mesh at regular intervals along the centerline
  2. At each slice, project vertices onto the local (normal, Z) plane
  3. Check if any projected vertex lies inside the clearance polygon
  4. Report violations with position, penetration depth, and vertex count

The clearance polygon is defined in local coordinates:
  - X-axis = perpendicular to alignment (cross-section width)
  - Y-axis = vertical (height above reference)

Reference:
  - ASTRA FHB T/G 24 001-10201: Tunnelquerschnitt, Normalprofil
  - SIA 197/2: Strassenquerschnitt im Tunnel

Example clearance profiles:
  - Road tunnel: rectangular with rounded corners
  - Rail tunnel: horseshoe shape
  - Retaining wall: min. distance from road edge
"""

import numpy as np


def check_clearance(
    mesh_data: dict,
    centerline,
    clearance_polygon: np.ndarray,
    n_slices: int = 20,
    reference_z: float = 0.0,
) -> dict:
    """Check if a mesh violates a clearance profile.

    Args:
        mesh_data: dict with vertices, faces.
        centerline: WallCenterline with local frames.
        clearance_polygon: (K, 2) array of polygon vertices in local
                          (perpendicular, Z-reference_z) coordinates.
                          The polygon defines the CLEAR area — points
                          inside are violations.
        n_slices: number of cross-sections to check along the centerline.
        reference_z: Z-coordinate of the reference plane (e.g. road surface).

    Returns:
        dict with:
            n_violations: int — total number of violating vertices
            n_slices_checked: int
            violations: list of per-slice violation dicts
            max_penetration_mm: float — deepest penetration into envelope
            clear: bool — True if no violations found
    """
    if centerline is None or not hasattr(centerline, "points_2d"):
        return {"n_violations": 0, "n_slices_checked": 0,
                "violations": [], "max_penetration_mm": 0.0, "clear": True}

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    n_pts = len(centerline.points_2d)

    # Slice positions along centerline
    fractions = np.linspace(0.05, 0.95, n_slices)
    slice_indices = np.clip((fractions * (n_pts - 1)).astype(int), 0, n_pts - 1)

    # Compute slice tolerance from centerline spacing
    if n_pts > 1:
        diffs = np.diff(centerline.points_2d, axis=0)
        spacing = float(np.median(np.linalg.norm(diffs, axis=1)))
        tol = spacing * 0.6
    else:
        tol = 0.5

    violations = []
    max_pen = 0.0
    total_violations = 0

    for idx in slice_indices:
        pt = centerline.points_2d[idx]
        tangent = centerline.tangents[idx]
        normal = centerline.normals[idx]

        tang_2d = tangent[:2]
        tang_mag = np.linalg.norm(tang_2d)
        if tang_mag < 1e-10:
            continue
        tang_2d = tang_2d / tang_mag

        # Select vertices near this slice
        verts_xy = vertices[:, :2]
        t_proj = (verts_xy - pt) @ tang_2d
        mask = np.abs(t_proj) < tol
        if mask.sum() == 0:
            continue

        selected = vertices[mask]

        # Also sample edge midpoints to catch cases where a triangle
        # edge crosses the clearance boundary but neither vertex is inside.
        # For each pair of selected vertices, add the midpoint.
        if len(selected) >= 2:
            n_sel = len(selected)
            # Sample midpoints of edges between close vertices
            mid_pts = []
            for k in range(min(n_sel, 50)):
                for l in range(k + 1, min(n_sel, 50)):
                    mid = (selected[k] + selected[l]) / 2.0
                    mid_pts.append(mid)
            if mid_pts:
                selected = np.vstack([selected, np.array(mid_pts)])

        # Project onto local (normal, Z) plane
        norm_2d = normal[:2]
        norm_mag = np.linalg.norm(norm_2d)
        if norm_mag < 1e-10:
            continue
        norm_2d = norm_2d / norm_mag

        perp_proj = (selected[:, :2] - pt) @ norm_2d
        z_proj = selected[:, 2] - reference_z

        # Check each vertex against clearance polygon
        local_pts = np.column_stack([perp_proj, z_proj])
        inside = _points_in_polygon(local_pts, clearance_polygon)

        n_inside = int(inside.sum())
        if n_inside > 0:
            # Compute penetration depth (distance from polygon boundary)
            pen_depths = _penetration_depths(local_pts[inside], clearance_polygon)
            max_slice_pen = float(pen_depths.max()) if len(pen_depths) > 0 else 0.0
            max_pen = max(max_pen, max_slice_pen)
            total_violations += n_inside

            # Position along centerline
            if idx > 0:
                cum = float(np.linalg.norm(
                    np.diff(centerline.points_2d[:idx + 1], axis=0), axis=1
                ).sum())
            else:
                cum = 0.0

            violations.append({
                "position_m": round(cum, 2),
                "n_vertices": n_inside,
                "max_penetration_mm": round(max_slice_pen * 1000, 1),
            })

    return {
        "n_violations": total_violations,
        "n_slices_checked": len(slice_indices),
        "violations": violations,
        "max_penetration_mm": round(max_pen * 1000, 1),
        "clear": total_violations == 0,
    }


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Test if 2D points are inside a polygon (ray casting algorithm).

    Uses the even-odd rule: cast a ray from each point in the +X direction
    and count how many polygon edges it crosses. Inside = odd count.

    Complexity: O(N × K) where N = number of points, K = polygon vertices.

    Reference: Shimrat, M. (1962). Algorithm 112: Position of point
    relative to polygon. Communications of the ACM, 5(8), 434.
    """
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)

    for i in range(n):
        j = (i + 1) % n
        yi, yj = polygon[i, 1], polygon[j, 1]
        xi, xj = polygon[i, 0], polygon[j, 0]

        # For each point, check if the ray crosses this edge
        cond1 = (yi > points[:, 1]) != (yj > points[:, 1])
        if not np.any(cond1):
            continue

        # X-coordinate of intersection
        slope = (xj - xi) / (yj - yi) if abs(yj - yi) > 1e-30 else 0
        x_intersect = xi + slope * (points[:, 1] - yi)
        crosses = cond1 & (points[:, 0] < x_intersect)
        inside ^= crosses  # XOR toggles inside/outside

    return inside


def _penetration_depths(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Compute minimum distance from points to polygon boundary.

    For points inside the polygon, this is the penetration depth.
    Uses point-to-segment distance for each polygon edge.
    """
    n_poly = len(polygon)
    min_dists = np.full(len(points), float("inf"))

    for i in range(n_poly):
        j = (i + 1) % n_poly
        a = polygon[i]
        b = polygon[j]

        # Point-to-segment distance (vectorized over points)
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-30:
            dists = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip(((points - a) @ ab) / ab_sq, 0, 1)
            closest = a + t[:, np.newaxis] * ab
            dists = np.linalg.norm(points - closest, axis=1)

        min_dists = np.minimum(min_dists, dists)

    return min_dists


# ── Predefined clearance profiles ─────────────────────────────────

def astra_road_clearance(width_m: float = 8.0, height_m: float = 4.5) -> np.ndarray:
    """ASTRA road tunnel clearance profile (rectangular Lichtraumprofil).

    The rectangular polygon is a conservative envelope for a two-lane
    cross-section on a *Nationalstrasse 1. Klasse* — real profiles are
    arched but the rectangular outer bound is what the validator tests
    against (any intrusion into the rectangle also intrudes into the
    real arch, so this is a sufficient condition for non-compliance).

    Args:
        width_m: total road width. Default 8.0 m = 2 × 3.5 m Fahrstreifen
                 + 2 × 0.5 m Randstreifen, per ASTRA FHB T/G 24 001-10201
                 §8.2 "Regelquerschnitte" Tabelle 2 (NS 1. Klasse).
        height_m: required clearance height above driving surface.
                 Default 4.5 m per ASTRA FHB T/G 24 001-10201 §7.3
                 "Lichtraumprofil" Absatz 1 (Regelhöhe für
                 Nationalstrassen).

    Returns:
        (4, 2) polygon in local coordinates (perpendicular, height).

    Reference: ASTRA FHB T/G 24 001-10201, §7 "Lichtraumprofil" und
               §8 "Regelquerschnitte".
    """
    hw = width_m / 2
    return np.array([
        [-hw, 0], [hw, 0], [hw, height_m], [-hw, height_m],
    ])


def astra_pedestrian_clearance(width_m: float = 1.5, height_m: float = 2.5) -> np.ndarray:
    """ASTRA pedestrian clearance profile (Fluchtweg/Notausstieg).

    Defaults are the minimum usable passage for a stretcher-carrying
    rescue team per ASTRA FHB T/G 24 001-10701 §3.4 "Fluchtweg-
    Abmessungen": 1.5 m breadth, 2.5 m height.

    Reference: ASTRA FHB T/G 24 001-10701, §3.4 "Fluchtwege und
               Notausstiege".
    """
    hw = width_m / 2
    return np.array([
        [-hw, 0], [hw, 0], [hw, height_m], [-hw, height_m],
    ])
