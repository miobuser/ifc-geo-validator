"""Wall-to-Alignment geometric checks.

If the IFC model contains an IfcAlignment (IFC 4.3 infrastructure
extension), we can measure each retaining wall against the alignment:

  - Perpendicular XY distance from wall centroid to the nearest
    alignment polyline vertex (stays in horizontal plane; vertical
    alignment is not yet considered)
  - Wall curvature radius vs. the local alignment radius (a wall
    curving tighter than the road it supports is geometrically
    unsafe — the driving surface the wall holds back cannot turn
    as tightly as the wall itself)

Both outputs are exposed as L6 context variables so existing L4 YAML
rules can consume them without code changes:

    min_alignment_distance_m
    alignment_radius_ratio   (wall radius / local alignment radius;
                              >= 1.0 means wall ≥ alignment curve)

References:
  - ASTRA FHB T/G §"Stützmauer vs. Achse": Wand muss die Strasse
    jederzeit stützen, d.h. Kurvenradius der Wand darf den lokalen
    Fahrbahnradius nicht unterschreiten.
  - Farin 2002, §7 (discrete curvature of a polyline).
"""

from __future__ import annotations

import numpy as np


def compute_alignment_context(
    element: dict,
    alignments: list[dict],
) -> dict:
    """Compute alignment-aware context for a single element.

    Args:
        element: an element result dict with a ``level1.centroid`` and
            optionally ``level3.min_radius_m``.
        alignments: list from ``ifc_parser.get_alignments`` — each
            item has ``points_xy`` (N×2) and ``points_3d``.

    Returns:
        dict with:
            has_alignment: bool — True if at least one alignment exists
            min_alignment_distance_m: float — perpendicular XY distance
                from the wall centroid to the closest alignment polyline
                point, across all alignments. None if no alignment.
            nearest_alignment_name: str | None — name of the winning
                alignment (useful when several are present).
            alignment_radius_ratio: float | None — wall radius divided
                by local alignment radius at the nearest point; None if
                either radius is unavailable.
    """
    if not alignments:
        return {
            "has_alignment": False,
            "min_alignment_distance_m": None,
            "nearest_alignment_name": None,
            "alignment_radius_ratio": None,
        }

    centroid = np.asarray((element.get("level1") or {}).get("centroid"),
                          dtype=float)
    if centroid.size != 3:
        return {"has_alignment": True, "min_alignment_distance_m": None,
                "nearest_alignment_name": None, "alignment_radius_ratio": None}

    c_xy = centroid[:2]

    best_dist = float("inf")
    best_align = None
    best_nearest_idx = -1
    best_poly = None

    for align in alignments:
        pts = np.asarray(align.get("points_xy"))
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        # Perpendicular distance from centroid to each polyline point.
        # For a proper point-to-segment distance we'd need to project
        # onto every segment; here the nearest vertex is sufficient
        # given IfcAlignment tessellations are typically ≤ 1 m apart.
        dists = np.linalg.norm(pts - c_xy, axis=1)
        k = int(np.argmin(dists))
        if dists[k] < best_dist:
            best_dist = float(dists[k])
            best_align = align
            best_nearest_idx = k
            best_poly = pts

    if best_poly is None:
        return {"has_alignment": True, "min_alignment_distance_m": None,
                "nearest_alignment_name": None, "alignment_radius_ratio": None}

    # Local alignment radius via three consecutive points (discrete
    # curvature, inverse of the circumradius of the triangle).
    local_radius_m = _three_point_radius(best_poly, best_nearest_idx)

    wall_radius_m = (element.get("level3") or {}).get("min_radius_m")
    if wall_radius_m is not None and local_radius_m is not None and local_radius_m > 0:
        ratio = float(wall_radius_m) / float(local_radius_m)
    else:
        ratio = None

    return {
        "has_alignment": True,
        "min_alignment_distance_m": round(best_dist, 3),
        "nearest_alignment_name": best_align.get("name"),
        "alignment_radius_ratio": round(ratio, 3) if ratio is not None else None,
    }


def _three_point_radius(points: np.ndarray, idx: int) -> float | None:
    """Circumradius of the triangle (p[idx-1], p[idx], p[idx+1]) in 2D.

    For three collinear points the triangle is degenerate and the
    radius is infinite — we return None so callers know the curve is
    locally straight. For endpoints we fall back to the neighbour
    triple.
    """
    n = len(points)
    if n < 3:
        return None
    i = max(1, min(idx, n - 2))
    a = points[i - 1]
    b = points[i]
    c = points[i + 1]
    # Side lengths
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    # Area via cross product (positive magnitude)
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    if area2 < 1e-10:
        return None
    # Circumradius = (ab · bc · ca) / (2 · area)
    return float((ab * bc * ca) / (2.0 * area2))
