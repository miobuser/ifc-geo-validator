"""Level 3 Validation: Face-specific geometric measurements.

Computes measurements from the classified face groups (Level 2 output):
  - Crown width (mm)
  - Crown slope (%)
  - Minimum wall thickness (mm)
  - Foundation width (mm)
  - Foundation width ratio (dimensionless)
  - Front face inclination ratio (e.g. 10:1)

For curved and polygonal walls (WallCenterline.use_local_measurement), crown
width and wall thickness are measured per-slice along the centerline using
local coordinate frames, giving correct results regardless of wall geometry.

Crown groups are filtered to only the topmost horizontal faces, preventing
step surfaces in L-shaped or stepped profiles from inflating the crown width.

Crown slope is computed per-face then area-weighted, avoiding cancellation
of horizontal normal components on curved walls.
"""

import numpy as np

from ifc_geo_validator.core.face_classifier import (
    WallCenterline,
    CROWN,
    FOUNDATION,
    FRONT,
    BACK,
)


def validate_level3(mesh_data: dict, level2_result: dict) -> dict:
    """Run Level 3 face-specific measurements.

    Args:
        mesh_data: dict from mesh_converter.extract_mesh().
        level2_result: dict from validate_level2() with face_groups.

    Returns:
        dict with computed face-specific measurements.
    """
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    groups = level2_result["face_groups"]
    wall_axis = np.array(level2_result["wall_axis"])
    centerline = level2_result.get("centerline")

    # Scale-relative zero threshold: derived from bbox diagonal.
    # A measurement is considered zero if it is < 1e-8 × model size.
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    eps = bbox_diag * 1e-8 if bbox_diag > 0 else 1e-12

    results = {}

    # ── Measurement uncertainty ─────────────────────────────────────
    # Every measurement on a tessellated mesh has an inherent uncertainty
    # proportional to the tessellation resolution (median edge length).
    # For a vertex-based extent measurement (e.g. crown width), the
    # true boundary lies within ±(median_edge_length/2) of a vertex.
    #
    # This is the Nyquist-Shannon analog for spatial sampling:
    # features smaller than the sampling interval cannot be resolved.
    #
    # Reference: Botsch et al. (2010). Polygon Mesh Processing, §1.3.
    edge_lengths = np.linalg.norm(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1
    )
    median_edge = float(np.median(edge_lengths)) if len(edge_lengths) > 0 else 0.0
    results["measurement_uncertainty_mm"] = round(median_edge * 1000.0 / 2.0, 1)

    # ── Centerline metadata ──────────────────────────────────────────
    if centerline is not None and isinstance(centerline, WallCenterline):
        results["is_curved"] = centerline.is_curved
        results["wall_length_m"] = round(centerline.length, 4)

    # ── Crown measurements ──────────────────────────────────────────
    crown_groups = [g for g in groups if g["category"] == CROWN]
    if crown_groups:
        # Filter to topmost crown groups only (excludes step surfaces)
        crown_groups = _filter_topmost_crown(crown_groups)

    if crown_groups:  # Re-check after filtering (may be empty)
        cw = _compute_crown_width(vertices, faces, crown_groups, wall_axis, centerline, eps)
        results["crown_width_mm"] = cw["width_mm"]
        results["crown_width_method"] = cw["method"]
        if "width_min_mm" in cw:
            results["crown_width_min_mm"] = cw["width_min_mm"]
            results["crown_width_max_mm"] = cw["width_max_mm"]
        if "width_cv" in cw:
            results["crown_width_cv"] = cw["width_cv"]

        cs = _compute_crown_slope(crown_groups, mesh_data)
        results["crown_slope_percent"] = cs["slope_percent"]
        results["crown_slope_direction"] = cs["direction"]

    # ── Wall thickness ──────────────────────────────────────────────
    front_groups = [g for g in groups if g["category"] == FRONT]
    back_groups = [g for g in groups if g["category"] == BACK]
    if front_groups and back_groups:
        wt = _compute_wall_thickness(
            vertices, faces, front_groups, back_groups, wall_axis, centerline, eps
        )
        results["min_wall_thickness_mm"] = wt["min_thickness_mm"]
        results["avg_wall_thickness_mm"] = wt["avg_thickness_mm"]

    # ── Foundation width ─────────────────────────────────────────────
    foundation_groups = [g for g in groups if g["category"] == FOUNDATION]
    if foundation_groups:
        fw = _compute_foundation_width(vertices, faces, foundation_groups, wall_axis, centerline, eps)
        results["foundation_width_mm"] = fw["width_mm"]
        results["foundation_width_method"] = fw["method"]
        if "width_cv" in fw:
            results["foundation_width_cv"] = fw["width_cv"]

    # ── Front inclination ───────────────────────────────────────────
    if front_groups:
        inc = _compute_front_inclination(front_groups)
        results["front_inclination_deg"] = inc["angle_deg"]
        results["front_inclination_ratio"] = inc["ratio"]

    # ── Wall height ──────────────────────────────────────────────
    all_vertical = front_groups + back_groups
    if all_vertical:
        vert_verts = _collect_vertices(vertices, faces, all_vertical)
        if len(vert_verts) > 0:
            z_vals = vert_verts[:, 2]
            results["wall_height_m"] = round(float(z_vals.max() - z_vals.min()), 4)

    # ── Foundation width ratio ────────────────────────────────────
    if "foundation_width_mm" in results and "wall_height_m" in results:
        wall_height_m = results["wall_height_m"]
        if wall_height_m > 0:
            results["foundation_width_ratio"] = round(
                results["foundation_width_mm"] / (wall_height_m * 1000.0), 6
            )

    return results


# ── Crown height filter ────────────────────────────────────────────

def _filter_topmost_crown(crown_groups):
    """Filter crown groups to only include the topmost horizontal faces.

    Uses natural-gap clustering on the z-coordinates of crown group
    centroids: the largest gap in the sorted z-values is the natural
    boundary between "actual crown" and "lower step surfaces." Only
    splits if the gap exceeds the intra-cluster spread on both sides
    (single-linkage cluster separation criterion, Hartigan 1975).

    This is scale-invariant — no absolute threshold needed.
    """
    if len(crown_groups) <= 1:
        return crown_groups

    z_vals = np.array([g["centroid"][2] for g in crown_groups])
    order = np.argsort(z_vals)
    z_sorted = z_vals[order]

    if len(z_sorted) < 2:
        return crown_groups

    # Find the largest gap between consecutive z-values
    gaps = np.diff(z_sorted)
    if len(gaps) == 0 or gaps.max() < 1e-6:
        return crown_groups  # all at same height

    max_gap_idx = int(np.argmax(gaps))
    gap_size = float(gaps[max_gap_idx])

    # Intra-cluster spread on each side of the gap
    below = z_sorted[:max_gap_idx + 1]
    above = z_sorted[max_gap_idx + 1:]
    range_below = float(below.max() - below.min()) if len(below) > 1 else 0.0
    range_above = float(above.max() - above.min()) if len(above) > 1 else 0.0

    # Split only if gap exceeds both intra-cluster ranges (meaningful separation)
    if gap_size > max(range_below, range_above, 1e-3):
        threshold_z = (z_sorted[max_gap_idx] + z_sorted[max_gap_idx + 1]) / 2.0
        top_groups = [g for g in crown_groups if g["centroid"][2] > threshold_z]
        return top_groups if top_groups else crown_groups

    return crown_groups


# ── Crown width ─────────────────────────────────────────────────────

def _compute_crown_width(vertices, faces, crown_groups, wall_axis, centerline=None, eps=1e-12):
    """Compute crown width perpendicular to wall axis.

    For simple straight walls: uses global perpendicular projection.
    For curved/polygonal walls: uses per-slice local frame measurement.
    """
    use_local = (centerline is not None
                 and isinstance(centerline, WallCenterline)
                 and centerline.use_local_measurement)

    if use_local:
        return _compute_crown_width_sliced(vertices, faces, crown_groups, centerline, eps)

    # Global approach for simple straight walls
    z_axis = np.array([0.0, 0.0, 1.0])
    perp_axis = np.cross(wall_axis, z_axis)
    perp_norm = np.linalg.norm(perp_axis)
    if perp_norm > 1e-10:
        perp_axis /= perp_norm
    else:
        perp_axis = np.array([1.0, 0.0, 0.0])

    crown_verts = _collect_vertices(vertices, faces, crown_groups)
    if len(crown_verts) == 0:
        return {"width_mm": 0.0, "method": "empty_crown"}
    projections = crown_verts @ perp_axis
    width_m = float(projections.max() - projections.min())

    return {
        "width_mm": width_m * 1000.0,
        "method": "vertex_extent_perpendicular",
    }


def _slice_tolerance(centerline):
    """Compute slice tolerance from centerline point spacing.

    Uses median spacing (robust to outliers from irregular tessellation)
    instead of mean. The multiplier 0.6 ensures slices don't overlap
    excessively on curves while still capturing enough vertices per slice.
    """
    n_pts = len(centerline.points_2d)
    if n_pts > 1:
        diffs = np.diff(centerline.points_2d, axis=0)
        spacings = np.linalg.norm(diffs, axis=1)
        return float(np.median(spacings)) * 0.6
    return 0.5


def _compute_crown_width_sliced(vertices, faces, crown_groups, centerline, eps=1e-12):
    """Compute crown width using per-slice local frames along the centerline."""
    crown_verts = _collect_vertices(vertices, faces, crown_groups)
    crown_xy = crown_verts[:, :2]

    local_widths = []
    n_pts = len(centerline.points_2d)
    slice_tol = _slice_tolerance(centerline)

    for i in range(n_pts):
        pt = centerline.points_2d[i]
        tangent = centerline.tangents[i]
        normal = centerline.normals[i]

        tang_2d = tangent[:2]
        tang_mag = np.linalg.norm(tang_2d)
        if tang_mag < eps:
            continue
        tang_2d = tang_2d / tang_mag

        t_proj = (crown_xy - pt) @ tang_2d
        mask = np.abs(t_proj) < slice_tol
        if mask.sum() < 2:
            continue

        local_verts = crown_verts[mask]
        n_proj = local_verts @ normal
        local_width = float(n_proj.max() - n_proj.min())
        if local_width > eps:
            local_widths.append(local_width)

    if not local_widths:
        return _compute_crown_width(vertices, faces, crown_groups,
                                     centerline.wall_axis, centerline=None, eps=eps)

    widths_mm = [w * 1000.0 for w in local_widths]
    widths_arr = np.array(widths_mm)
    mean_w = float(widths_arr.mean())
    std_w = float(widths_arr.std())
    cv = std_w / mean_w if mean_w > eps else 0.0

    return {
        "width_mm": min(widths_mm),
        "width_min_mm": min(widths_mm),
        "width_max_mm": max(widths_mm),
        "width_cv": round(cv, 6),
        "method": "slice_local_frame",
    }


# ── Foundation width ────────────────────────────────────────────────

def _compute_foundation_width(vertices, faces, foundation_groups, wall_axis, centerline=None, eps=1e-12):
    """Compute foundation width perpendicular to wall axis.

    Same approach as crown width but using foundation face groups.
    For simple straight walls: uses global perpendicular projection.
    For curved/polygonal walls: uses per-slice local frame measurement.
    """
    use_local = (centerline is not None
                 and isinstance(centerline, WallCenterline)
                 and centerline.use_local_measurement)

    if use_local:
        return _compute_foundation_width_sliced(vertices, faces, foundation_groups, centerline, eps)

    # Global approach for simple straight walls
    z_axis = np.array([0.0, 0.0, 1.0])
    perp_axis = np.cross(wall_axis, z_axis)
    perp_norm = np.linalg.norm(perp_axis)
    if perp_norm > 1e-10:
        perp_axis /= perp_norm
    else:
        perp_axis = np.array([1.0, 0.0, 0.0])

    foundation_verts = _collect_vertices(vertices, faces, foundation_groups)
    if len(foundation_verts) == 0:
        return {"width_mm": 0.0, "method": "empty_foundation"}
    projections = foundation_verts @ perp_axis
    width_m = float(projections.max() - projections.min())

    return {
        "width_mm": width_m * 1000.0,
        "method": "vertex_extent_perpendicular",
    }


def _compute_foundation_width_sliced(vertices, faces, foundation_groups, centerline, eps=1e-12):
    """Compute foundation width using per-slice local frames along the centerline."""
    foundation_verts = _collect_vertices(vertices, faces, foundation_groups)
    foundation_xy = foundation_verts[:, :2]

    local_widths = []
    n_pts = len(centerline.points_2d)
    slice_tol = _slice_tolerance(centerline)

    for i in range(n_pts):
        pt = centerline.points_2d[i]
        tangent = centerline.tangents[i]
        normal = centerline.normals[i]

        tang_2d = tangent[:2]
        tang_mag = np.linalg.norm(tang_2d)
        if tang_mag < eps:
            continue
        tang_2d = tang_2d / tang_mag

        t_proj = (foundation_xy - pt) @ tang_2d
        mask = np.abs(t_proj) < slice_tol
        if mask.sum() < 2:
            continue

        local_verts = foundation_verts[mask]
        n_proj = local_verts @ normal
        local_width = float(n_proj.max() - n_proj.min())
        if local_width > eps:
            local_widths.append(local_width)

    if not local_widths:
        return _compute_foundation_width(vertices, faces, foundation_groups,
                                          centerline.wall_axis, centerline=None, eps=eps)

    widths_mm = [w * 1000.0 for w in local_widths]
    widths_arr = np.array(widths_mm)
    mean_w = float(widths_arr.mean())
    std_w = float(widths_arr.std())
    cv = std_w / mean_w if mean_w > eps else 0.0

    return {
        "width_mm": min(widths_mm),
        "width_min_mm": min(widths_mm),
        "width_max_mm": max(widths_mm),
        "width_cv": round(cv, 6),
        "method": "slice_local_frame",
    }


# ── Crown slope ─────────────────────────────────────────────────────

def _compute_crown_slope(crown_groups, mesh_data=None):
    """Compute crown slope from per-triangle normal deviation from vertical.

    Computes slope per individual triangle and area-weights the results.
    This avoids cancellation of horizontal normal components that occurs
    when averaging normals over a curved crown (e.g. 180° semicircle).
    Falls back to per-group computation when mesh_data is not available.
    """
    total_area = sum(g["area"] for g in crown_groups)
    if total_area <= 0:
        return {"slope_percent": 0.0, "direction": [0, 0, 0]}

    # Use per-triangle normals if mesh data available (more accurate for curves)
    if mesh_data is not None:
        normals = mesh_data["normals"]
        areas = mesh_data["areas"]
        weighted_slope = 0.0
        weighted_area = 0.0
        for g in crown_groups:
            for fi in g["face_indices"]:
                n = normals[fi]
                a = float(areas[fi])
                cos_z = np.clip(np.dot(n, [0.0, 0.0, 1.0]), -1.0, 1.0)
                angle_rad = np.arccos(cos_z)
                # Clamp tan to avoid infinity for near-vertical faces
                slope_val = min(float(np.tan(angle_rad) * 100.0), 1e6)
                weighted_slope += slope_val * a
                weighted_area += a
        if weighted_area > 0:
            slope_percent = weighted_slope / weighted_area
        else:
            slope_percent = 0.0
    else:
        # Fallback: per-group computation
        weighted_slope = 0.0
        for g in crown_groups:
            n = np.array(g["normal"])
            cos_z = np.clip(np.dot(n, [0.0, 0.0, 1.0]), -1.0, 1.0)
            angle_rad = np.arccos(cos_z)
            # Clamp tan to avoid infinity for near-vertical faces
            weighted_slope += min(float(np.tan(angle_rad) * 100.0), 1e6) * g["area"]
        slope_percent = weighted_slope / total_area

    # Direction: average horizontal component (for display)
    avg_normal = np.zeros(3)
    for g in crown_groups:
        avg_normal += np.array(g["normal"]) * g["area"]
    avg_normal /= total_area
    direction = np.array([avg_normal[0], avg_normal[1], 0.0])
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-10:
        direction /= d_norm

    return {
        "slope_percent": slope_percent,
        "direction": direction.tolist(),
    }


# ── Wall thickness ──────────────────────────────────────────────────

def _compute_wall_thickness(vertices, faces, front_groups, back_groups,
                            wall_axis, centerline=None, eps=1e-12):
    """Compute wall thickness as distance between front and back faces.

    For simple straight walls: uses global perpendicular projection.
    For curved/polygonal walls: uses per-slice local frame measurement.
    """
    use_local = (centerline is not None
                 and isinstance(centerline, WallCenterline)
                 and centerline.use_local_measurement)

    if use_local:
        return _compute_wall_thickness_sliced(
            vertices, faces, front_groups, back_groups, centerline, eps
        )

    # Global approach for simple straight walls
    z_axis = np.array([0.0, 0.0, 1.0])
    perp_axis = np.cross(wall_axis, z_axis)
    perp_norm = np.linalg.norm(perp_axis)
    if perp_norm > 1e-10:
        perp_axis /= perp_norm
    else:
        perp_axis = np.array([1.0, 0.0, 0.0])

    front_verts = _collect_vertices(vertices, faces, front_groups)
    back_verts = _collect_vertices(vertices, faces, back_groups)

    if len(front_verts) == 0 or len(back_verts) == 0:
        return {"min_thickness_mm": 0.0, "avg_thickness_mm": 0.0}

    front_proj = front_verts @ perp_axis
    back_proj = back_verts @ perp_axis

    # Robust thickness estimation using median projections.
    #
    # The median is the L1-optimal estimator of location — it minimizes
    # the sum of absolute deviations and is robust against up to 50%
    # outlier contamination (breakdown point = 0.5).
    #
    # For a wall with N front vertices and M back vertices projected
    # onto the perpendicular axis:
    #   avg_thickness = |median(front_proj) - median(back_proj)|
    #   min_thickness = estimated from interquartile ranges
    #
    # This replaces the previous extent-based method which used
    # min/max projections (sensitive to single outlier vertices from
    # tessellation artifacts or boolean operation residuals).
    #
    # Reference: Huber, P.J. (1981). Robust Statistics. Wiley.
    front_median = float(np.median(front_proj))
    back_median = float(np.median(back_proj))
    avg_thickness = abs(front_median - back_median)

    # Min thickness: use the closest percentiles (10th/90th) to estimate
    # the narrowest point, robust against outliers at the edges.
    if len(front_proj) >= 4 and len(back_proj) >= 4:
        # Use 10th and 90th percentiles for robust min/max
        if front_median < back_median:
            front_high = float(np.percentile(front_proj, 90))
            back_low = float(np.percentile(back_proj, 10))
        else:
            front_high = float(np.percentile(front_proj, 10))
            back_low = float(np.percentile(back_proj, 90))
        min_thickness = abs(back_low - front_high)
        min_thickness = max(min_thickness, 0.0)
    else:
        min_thickness = avg_thickness

    if min_thickness == 0.0:
        min_thickness = avg_thickness

    return {
        "min_thickness_mm": min_thickness * 1000.0,
        "avg_thickness_mm": avg_thickness * 1000.0,
    }


def _compute_wall_thickness_sliced(vertices, faces, front_groups, back_groups,
                                   centerline, eps=1e-12):
    """Compute wall thickness using per-slice local frames along the centerline."""
    front_verts = _collect_vertices(vertices, faces, front_groups)
    back_verts = _collect_vertices(vertices, faces, back_groups)
    front_xy = front_verts[:, :2]
    back_xy = back_verts[:, :2]

    local_thicknesses = []
    n_pts = len(centerline.points_2d)

    slice_tol = _slice_tolerance(centerline)

    for i in range(n_pts):
        pt = centerline.points_2d[i]
        tangent = centerline.tangents[i]
        normal = centerline.normals[i]

        tang_2d = tangent[:2]
        tang_mag = np.linalg.norm(tang_2d)
        if tang_mag < eps:
            continue
        tang_2d = tang_2d / tang_mag

        front_t = (front_xy - pt) @ tang_2d
        front_mask = np.abs(front_t) < slice_tol
        back_t = (back_xy - pt) @ tang_2d
        back_mask = np.abs(back_t) < slice_tol

        if front_mask.sum() < 1 or back_mask.sum() < 1:
            continue

        front_n = front_verts[front_mask] @ normal
        back_n = back_verts[back_mask] @ normal

        # Use median instead of midpoint for robustness against outlier
        # vertices from tessellation artifacts or boolean operations.
        front_median = float(np.median(front_n))
        back_median = float(np.median(back_n))
        thickness = abs(front_median - back_median)

        if thickness > eps:
            local_thicknesses.append(thickness)

    if not local_thicknesses:
        return _compute_wall_thickness(
            vertices, faces, front_groups, back_groups,
            centerline.wall_axis, centerline=None, eps=eps
        )

    thicknesses_mm = [t * 1000.0 for t in local_thicknesses]
    return {
        "min_thickness_mm": min(thicknesses_mm),
        "avg_thickness_mm": sum(thicknesses_mm) / len(thicknesses_mm),
    }


# ── Front inclination ──────────────────────────────────────────────

def _compute_front_inclination(front_groups):
    """Compute inclination of the front face from vertical.

    Inclination ratio n:1 means horizontal offset / vertical rise = 1/n.
    A perfectly vertical wall has infinite ratio (reported as 0:1 → angle=0°).
    A 10:1 wall leans 1 unit horizontally per 10 units vertically.
    """
    total_area = sum(g["area"] for g in front_groups)
    if total_area <= 0:
        return {"angle_deg": 0.0, "ratio": float("inf")}

    avg_normal = np.zeros(3)
    for g in front_groups:
        n = np.array(g["normal"])
        avg_normal += n * g["area"]
    avg_normal /= total_area
    mag = np.linalg.norm(avg_normal)
    if mag > 0:
        avg_normal /= mag

    z_component = abs(avg_normal[2])
    horiz_component = np.sqrt(avg_normal[0] ** 2 + avg_normal[1] ** 2)

    if horiz_component < 1e-10:
        return {"angle_deg": 90.0, "ratio": 0.0}

    angle_from_vertical_rad = np.arctan2(z_component, horiz_component)
    angle_deg = float(np.degrees(angle_from_vertical_rad))

    if angle_deg < 0.01:
        ratio = float("inf")  # perfectly vertical
    else:
        ratio = 1.0 / np.tan(angle_from_vertical_rad)

    return {
        "angle_deg": angle_deg,
        "ratio": ratio,
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _collect_vertices(vertices, faces, groups):
    """Collect all unique vertices belonging to a list of face groups."""
    all_indices = []
    for g in groups:
        for fi in g["face_indices"]:
            all_indices.extend(faces[fi].tolist())
    unique_idx = list(set(all_indices))
    return vertices[unique_idx]
