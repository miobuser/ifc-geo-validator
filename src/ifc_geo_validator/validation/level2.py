"""Level 2 Validation: Face classification (Crown, Foundation, Front, Back, Ends).

Orchestrates the face classifier and produces a structured result dict
suitable for downstream Level 3 (face-specific measurements) and reporting.
"""

from ifc_geo_validator.core.face_classifier import (
    classify_faces,
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
    """
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

    return {
        "face_groups": group_dicts,
        "wall_axis": result["wall_axis"],
        "num_groups": result["num_groups"],
        "thresholds_used": result["thresholds_used"],
        "summary": summary,
        "has_crown": CROWN in summary,
        "has_foundation": FOUNDATION in summary,
        "has_front": FRONT in summary,
        "has_back": BACK in summary,
    }
