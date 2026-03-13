"""Level 3 Validation: Face-specific geometric measurements.

Computes measurements from the classified face groups (Level 2 output):
  - Crown width (mm)
  - Crown slope (%)
  - Minimum wall thickness (mm)
  - Front face inclination ratio (e.g. 10:1)
"""

import numpy as np

from ifc_geo_validator.core.face_classifier import (
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

    results = {}

    # ── Crown measurements ──────────────────────────────────────────
    crown_groups = [g for g in groups if g["category"] == CROWN]
    if crown_groups:
        cw = _compute_crown_width(vertices, faces, crown_groups, wall_axis)
        results["crown_width_mm"] = cw["width_mm"]
        results["crown_width_method"] = cw["method"]

        cs = _compute_crown_slope(crown_groups)
        results["crown_slope_percent"] = cs["slope_percent"]
        results["crown_slope_direction"] = cs["direction"]

    # ── Wall thickness ──────────────────────────────────────────────
    front_groups = [g for g in groups if g["category"] == FRONT]
    back_groups = [g for g in groups if g["category"] == BACK]
    if front_groups and back_groups:
        wt = _compute_wall_thickness(
            vertices, faces, front_groups, back_groups, wall_axis
        )
        results["min_wall_thickness_mm"] = wt["min_thickness_mm"]
        results["avg_wall_thickness_mm"] = wt["avg_thickness_mm"]

    # ── Front inclination ───────────────────────────────────────────
    if front_groups:
        inc = _compute_front_inclination(front_groups)
        results["front_inclination_deg"] = inc["angle_deg"]
        results["front_inclination_ratio"] = inc["ratio"]

    return results


# ── Crown width ─────────────────────────────────────────────────────

def _compute_crown_width(vertices, faces, crown_groups, wall_axis):
    """Compute crown width perpendicular to wall axis.

    Width = extent of crown face vertices in the direction perpendicular
    to the wall axis, projected onto the horizontal plane.
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    perp_axis = np.cross(wall_axis, z_axis)
    perp_norm = np.linalg.norm(perp_axis)
    if perp_norm > 1e-10:
        perp_axis /= perp_norm
    else:
        # Wall axis is vertical (unusual) — fall back to X
        perp_axis = np.array([1.0, 0.0, 0.0])

    # Collect all vertices belonging to crown faces
    crown_verts = _collect_vertices(vertices, faces, crown_groups)

    # Project onto perpendicular axis
    projections = crown_verts @ perp_axis
    width_m = float(projections.max() - projections.min())

    return {
        "width_mm": width_m * 1000.0,
        "method": "vertex_extent_perpendicular",
    }


# ── Crown slope ─────────────────────────────────────────────────────

def _compute_crown_slope(crown_groups):
    """Compute crown slope from face normal deviation from vertical.

    Slope % = tan(angle_from_vertical) * 100
    Uses the area-weighted average normal of all crown groups.
    """
    # Area-weighted average of all crown normals
    total_area = sum(g["area"] for g in crown_groups)
    if total_area <= 0:
        return {"slope_percent": 0.0, "direction": [0, 0, 0]}

    avg_normal = np.zeros(3)
    for g in crown_groups:
        n = np.array(g["normal"])
        avg_normal += n * g["area"]
    avg_normal /= total_area
    mag = np.linalg.norm(avg_normal)
    if mag > 0:
        avg_normal /= mag

    # Angle from vertical (+Z)
    cos_z = np.clip(np.dot(avg_normal, [0.0, 0.0, 1.0]), -1.0, 1.0)
    angle_rad = np.arccos(cos_z)
    slope_percent = float(np.tan(angle_rad) * 100.0)

    # Direction of slope: horizontal component of the normal
    direction = np.array([avg_normal[0], avg_normal[1], 0.0])
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-10:
        direction /= d_norm

    return {
        "slope_percent": slope_percent,
        "direction": direction.tolist(),
    }


# ── Wall thickness ──────────────────────────────────────────────────

def _compute_wall_thickness(vertices, faces, front_groups, back_groups, wall_axis):
    """Compute wall thickness as distance between front and back faces.

    Projects the centroids of front and back face groups onto the axis
    perpendicular to the wall, giving the distance between the two sides.
    Also samples vertex positions for min/max thickness.
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    perp_axis = np.cross(wall_axis, z_axis)
    perp_norm = np.linalg.norm(perp_axis)
    if perp_norm > 1e-10:
        perp_axis /= perp_norm
    else:
        perp_axis = np.array([1.0, 0.0, 0.0])

    # Collect front and back vertex projections
    front_verts = _collect_vertices(vertices, faces, front_groups)
    back_verts = _collect_vertices(vertices, faces, back_groups)

    front_proj = front_verts @ perp_axis
    back_proj = back_verts @ perp_axis

    # Thickness = distance between the two faces along perpendicular axis
    front_extent = [float(front_proj.min()), float(front_proj.max())]
    back_extent = [float(back_proj.min()), float(back_proj.max())]

    # The faces should be on opposite sides of the wall
    front_mean = np.mean(front_extent)
    back_mean = np.mean(back_extent)

    avg_thickness = abs(front_mean - back_mean)

    # Min thickness: closest approach between front and back
    # Using the overlap region along the perpendicular axis
    if front_mean < back_mean:
        min_thickness = abs(back_extent[0] - front_extent[1])
    else:
        min_thickness = abs(front_extent[0] - back_extent[1])

    # Ensure min is not negative (overlap case)
    min_thickness = max(min_thickness, 0.0)
    # For simple walls, min ≈ avg
    if min_thickness == 0.0:
        min_thickness = avg_thickness

    return {
        "min_thickness_mm": min_thickness * 1000.0,
        "avg_thickness_mm": avg_thickness * 1000.0,
    }


# ── Front inclination ──────────────────────────────────────────────

def _compute_front_inclination(front_groups):
    """Compute inclination of the front face from vertical.

    Inclination ratio n:1 means horizontal offset / vertical rise = 1/n.
    A perfectly vertical wall has infinite ratio (reported as 0:1 → angle=0°).
    A 10:1 wall leans 1 unit horizontally per 10 units vertically.
    """
    # Area-weighted average normal of front groups
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

    # The front face normal should be approximately horizontal
    # Inclination = angle of the face from vertical
    # Normal of a vertical face is horizontal → angle_from_horizontal = 0
    # If the face tilts, the normal tilts away from horizontal
    z_component = abs(avg_normal[2])
    horiz_component = np.sqrt(avg_normal[0] ** 2 + avg_normal[1] ** 2)

    if horiz_component < 1e-10:
        return {"angle_deg": 90.0, "ratio": 0.0}

    angle_from_vertical_rad = np.arctan2(z_component, horiz_component)
    angle_deg = float(np.degrees(angle_from_vertical_rad))

    # Ratio: vertical / horizontal = 1 / tan(angle)
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
