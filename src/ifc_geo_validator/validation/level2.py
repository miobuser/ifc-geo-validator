"""Level 2 Validation: Face classification (Crown, Foundation, Front, Back, Ends).

Orchestrates the face classifier and produces a structured result dict
suitable for downstream Level 3 (face-specific measurements) and reporting.

Includes a geometry pre-check and classification confidence score to detect
elements that are not wall-like (slabs, columns, irregular shapes) and
report the reliability of the classification.
"""

import numpy as np

from ifc_geo_validator.core.face_classifier import (
    classify_faces,
    WallCenterline,
    CROWN,
    FOUNDATION,
    FRONT,
    BACK,
    END_LEFT,
    END_RIGHT,
    UNCLASSIFIED,
)


def validate_level2(mesh_data: dict, thresholds: dict = None) -> dict:
    """Run Level 2 face classification on extracted mesh data.

    Args:
        mesh_data: dict from mesh_converter.extract_mesh().
        thresholds: optional dict with horizontal_deg, coplanar_deg, lateral_deg.

    Returns:
        dict with:
            face_groups:  list of serialised FaceGroup dicts
            wall_axis:    [x, y, z]
            num_groups:   int
            summary:      dict mapping category → {count, total_area}
            has_crown:    bool
            has_foundation: bool
            has_front:    bool
            has_back:     bool
            geometry_check: dict — pre-classification geometry assessment
            confidence:     float — classification confidence 0.0–1.0
            diagnostics:    list[str] — actionable messages for issues
    """
    vertices = mesh_data["vertices"]
    diagnostics = []

    # ── Geometry pre-check ──────────────────────────────────────
    geo_check = _check_wall_geometry(vertices, mesh_data["areas"])

    if not geo_check["is_wall_like"]:
        diagnostics.append(
            f"Geometry does not appear wall-like: {geo_check['reason']}. "
            f"Classification results may be unreliable."
        )

    # ── Face classification ──────────────────────────────────────
    result = classify_faces(mesh_data, thresholds)

    groups = result["face_groups"]

    # Build summary by category
    summary: dict[str, dict] = {}
    for g in groups:
        cat = g.category
        if cat not in summary:
            summary[cat] = {"count": 0, "total_area": 0.0}
        summary[cat]["count"] += 1
        summary[cat]["total_area"] += g.area

    # Serialise FaceGroup dataclasses to dicts
    group_dicts = []
    for g in groups:
        group_dicts.append({
            "category": g.category,
            "face_indices": g.face_indices,
            "normal": g.normal,
            "area": g.area,
            "centroid": g.centroid,
            "num_triangles": g.num_triangles,
        })

    has_crown = CROWN in summary
    has_foundation = FOUNDATION in summary
    has_front = FRONT in summary
    has_back = BACK in summary
    asymmetry = result.get("front_back_asymmetry", 0.0)

    # If classification found all expected wall categories despite bbox
    # being non-wall-like (e.g. curved walls with square bbox), check the
    # vertical/horizontal face area ratio to confirm it's actually wall-like.
    # A wall has vertical area (front+back) >> horizontal area (crown+foundation).
    # A slab has horizontal area >> vertical area.
    if not geo_check["is_wall_like"] and has_crown and has_foundation and (has_front or has_back):
        vert_area = summary.get(FRONT, {}).get("total_area", 0) + summary.get(BACK, {}).get("total_area", 0)
        horiz_area = summary.get(CROWN, {}).get("total_area", 0) + summary.get(FOUNDATION, {}).get("total_area", 0)
        # Wall: vertical faces dominate. Slab: horizontal faces dominate.
        # Threshold: vertical > 30% of horizontal → wall-like classification is credible.
        if horiz_area <= 0 or vert_area / horiz_area > 0.3:
            geo_check = {**geo_check, "is_wall_like": True,
                         "reason": "classified as wall (overrides bbox shape)"}
            diagnostics = []  # Clear the non-wall-like warning

    # ── Element role detection ─────────────────────────────────────
    element_role = _detect_element_role(geo_check, summary, has_crown, has_foundation)

    # ── Classification confidence ─────────────────────────────────
    confidence, conf_diagnostics = _compute_confidence(
        summary, has_crown, has_foundation, has_front, has_back,
        asymmetry, geo_check,
    )
    diagnostics.extend(conf_diagnostics)

    # Extract centerline metadata
    centerline = result.get("centerline")
    centerline_dict = centerline.to_dict() if centerline else None

    return {
        "face_groups": group_dicts,
        "wall_axis": result["wall_axis"],
        "centerline": centerline,
        "centerline_info": centerline_dict,
        "num_groups": result["num_groups"],
        "thresholds_used": result["thresholds_used"],
        "summary": summary,
        "has_crown": has_crown,
        "has_foundation": has_foundation,
        "has_front": has_front,
        "has_back": has_back,
        "front_back_asymmetry": asymmetry,
        "n_bodies": result.get("n_bodies", 1),
        "geometry_check": geo_check,
        "confidence": confidence,
        "element_role": element_role,
        "diagnostics": diagnostics,
    }


# ── Geometry pre-check ────────────────────────────────────────────

def _check_wall_geometry(vertices: np.ndarray, areas: np.ndarray) -> dict:
    """Assess whether the element geometry is wall-like.

    A wall-like element has:
      1. Elongation: one horizontal dimension >> the other (plan aspect ratio > 2)
      2. Verticality: height (Z-extent) > min horizontal dimension
      3. Not a slab: height > 0.1 × max horizontal dimension

    Returns dict with is_wall_like (bool), reason (str), and metrics.

    These are soft checks — classification runs regardless, but confidence
    is reduced for non-wall-like elements.
    """
    if len(vertices) < 4:
        return {"is_wall_like": False, "reason": "too few vertices", "plan_aspect": 0, "vert_ratio": 0}

    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    dx, dy, dz = float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])

    # Sort horizontal dimensions
    h_dims = sorted([dx, dy], reverse=True)
    h_max, h_min = h_dims[0], h_dims[1]

    # Prevent division by zero
    eps = max(h_max * 1e-10, 1e-12)
    plan_aspect = h_max / max(h_min, eps)
    vert_ratio = dz / max(h_min, eps)

    reasons = []

    # Check 1: Plan elongation — a wall is longer than it is wide
    # Threshold: plan_aspect > 2.0 (wall is at least 2× longer than thick)
    # Relaxed: also accept square plan if tall (column-like still works)
    if plan_aspect < 1.5 and dz < h_max * 0.5:
        reasons.append(f"plan nearly square ({plan_aspect:.1f}:1) and flat")

    # Check 2: Not a slab — height should be significant
    if dz < h_max * 0.05 and dz < h_min * 0.5:
        reasons.append(f"very flat (Z={dz:.3f}m vs XY={h_max:.1f}×{h_min:.1f}m)")

    is_wall = len(reasons) == 0
    reason = "; ".join(reasons) if reasons else "wall-like geometry"

    return {
        "is_wall_like": is_wall,
        "reason": reason,
        "plan_aspect": round(plan_aspect, 2),
        "vert_ratio": round(vert_ratio, 2),
        "bbox_dims_m": [round(dx, 3), round(dy, 3), round(dz, 3)],
    }


# ── Element role detection ─────────────────────────────────────────

def _detect_element_role(geo_check: dict, summary: dict,
                         has_crown: bool, has_foundation: bool) -> str:
    """Detect the structural role of an element from its geometry.

    Uses bounding box aspect ratios and face area distribution:
      - wall_stem:   tall and thin (height > width, plan elongated)
      - foundation:  short and wide (height < width/3)
      - parapet:     short and thin (height < 1.5m, thin)
      - column:      tall, square plan (plan_aspect < 2)
      - slab:        very flat (height < 5% of max horizontal)
      - unknown:     doesn't match any pattern

    Returns one of: "wall_stem", "foundation", "parapet", "column", "slab", "unknown".
    """
    dims = geo_check.get("bbox_dims_m", [0, 0, 0])
    if len(dims) < 3:
        return "unknown"

    dx, dy, dz = dims
    h_dims = sorted([dx, dy], reverse=True)
    h_max, h_min = h_dims[0], h_dims[1]
    plan_aspect = geo_check.get("plan_aspect", 1.0)
    vert_ratio = geo_check.get("vert_ratio", 1.0)

    # Vertical face area vs horizontal face area
    vert_area = summary.get(FRONT, {}).get("total_area", 0) + summary.get(BACK, {}).get("total_area", 0)
    horiz_area = summary.get(CROWN, {}).get("total_area", 0) + summary.get(FOUNDATION, {}).get("total_area", 0)

    eps = max(h_max * 1e-6, 1e-6)

    # Slab: very flat
    if dz < h_max * 0.05 and dz < h_min * 0.5:
        return "slab"

    # Foundation: short and wide (height < width/3, horiz > vert)
    if dz < h_min / 2 and horiz_area > vert_area * 2:
        return "foundation"

    # Column: tall, nearly square plan
    if plan_aspect < 2.0 and dz > h_max * 0.8:
        return "column"

    # Parapet: short and thin (height < 1.5m, narrow)
    if dz < 1.5 and h_min < 0.5 and plan_aspect > 3:
        return "parapet"

    # Wall stem: elongated plan, significant height, vertical faces dominate
    if plan_aspect > 2.0 and vert_area > horiz_area * 0.3:
        return "wall_stem"

    # Default: if wall-like geometry detected, assume stem
    if geo_check.get("is_wall_like", False):
        return "wall_stem"

    return "unknown"


# ── Classification confidence ─────────────────────────────────────

def _compute_confidence(
    summary, has_crown, has_foundation, has_front, has_back,
    asymmetry, geo_check,
) -> tuple[float, list[str]]:
    """Compute a classification confidence score between 0.0 and 1.0.

    Factors (each contributes to the score):
      1. Has crown AND foundation (essential for a wall)        → 0.25
      2. Has front AND back (essential for a wall)              → 0.25
      3. Low unclassified area fraction                         → 0.20
      4. Geometry is wall-like                                  → 0.15
      5. Front/back asymmetry > 0 (distinguishable sides)      → 0.15

    Returns (confidence, diagnostics).
    """
    score = 0.0
    diag = []

    # Factor 1: Crown + Foundation
    if has_crown and has_foundation:
        score += 0.25
    elif has_crown or has_foundation:
        score += 0.10
        missing = "foundation" if has_crown else "crown"
        diag.append(f"No {missing} face detected — element may not be a complete wall.")
    else:
        diag.append("Neither crown nor foundation detected — element is likely not a wall.")

    # Factor 2: Front + Back
    if has_front and has_back:
        score += 0.25
    elif has_front or has_back:
        score += 0.10
        diag.append("Only one vertical face side detected (front or back, not both).")
    else:
        diag.append("No front/back faces detected — vertical face classification failed.")

    # Factor 3: Classified area coverage
    total_area = sum(s["total_area"] for s in summary.values())
    unclass_area = summary.get(UNCLASSIFIED, {}).get("total_area", 0.0)
    if total_area > 0:
        classified_ratio = 1.0 - (unclass_area / total_area)
        score += 0.20 * classified_ratio
        if classified_ratio < 0.8:
            diag.append(
                f"{unclass_area / total_area * 100:.0f}% of surface area is unclassified."
            )
    else:
        score += 0.20

    # Factor 4: Geometry check
    if geo_check.get("is_wall_like", False):
        score += 0.15
    # (diagnostic already added in main function)

    # Factor 5: Front/back asymmetry
    if asymmetry > 0.05:
        score += 0.15
    elif asymmetry > 0.01:
        score += 0.08
        diag.append(
            f"Front/back asymmetry very low ({asymmetry:.3f}) — "
            f"side assignment has low confidence. Use terrain (L6) for definitive result."
        )
    else:
        diag.append(
            f"Front/back faces are symmetric (asymmetry={asymmetry:.3f}) — "
            f"assignment is arbitrary without terrain context."
        )

    return round(min(score, 1.0), 3), diag
