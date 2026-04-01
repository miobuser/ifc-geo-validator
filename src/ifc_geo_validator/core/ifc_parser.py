"""IFC file loading and element filtering."""

import os
import ifcopenshell


class IFCLoadError(Exception):
    """Raised when an IFC file cannot be loaded."""
    pass


def load_model(path: str) -> ifcopenshell.file:
    """Load an IFC file and return the model object.

    Raises:
        IFCLoadError: If the file doesn't exist, is empty, or is not valid IFC.
    """
    if not os.path.exists(path):
        raise IFCLoadError(f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise IFCLoadError(f"File is empty: {path}")
    try:
        return ifcopenshell.open(path)
    except Exception as e:
        raise IFCLoadError(f"Failed to parse IFC file '{path}': {e}") from e


def get_elements(model, entity_type="IfcWall", predefined_type=None) -> list:
    """Filter IFC elements by entity type and optional PredefinedType.

    For IfcWall, PredefinedType may be set directly on the element or
    inherited from the associated IfcWallType via the type relationship.
    """
    elements = model.by_type(entity_type)

    if predefined_type is None:
        return elements

    filtered = []
    for elem in elements:
        ptype = _get_predefined_type(elem)
        if ptype and ptype.upper() == predefined_type.upper():
            filtered.append(elem)
    return filtered


def _get_predefined_type(element) -> str | None:
    """Get PredefinedType, falling back to the element's type definition."""
    # Direct attribute
    ptype = getattr(element, "PredefinedType", None)
    if ptype and ptype != "USERDEFINED" and ptype != "NOTDEFINED":
        return ptype

    # Fallback: check the associated type (e.g., IfcWallType)
    if hasattr(element, "IsTypedBy"):
        for rel in element.IsTypedBy:
            type_obj = rel.RelatingType
            tp = getattr(type_obj, "PredefinedType", None)
            if tp:
                return tp

    return ptype  # May still be USERDEFINED/NOTDEFINED/None


def get_alignments(model) -> list[dict]:
    """Extract horizontal alignments from the model.

    Searches for IfcAlignment entities (IFC 4.3) and tessellates their
    geometric representation to polylines.  Each alignment is returned as
    a dict with name, points_xy (N×2), and points_3d (N×3).

    Falls back to IfcAlignmentCurve (IFC 4.0 experimental) if available.

    Returns an empty list if no alignments are found.
    """
    import numpy as np

    results = []

    # Try IFC 4.3 IfcAlignment
    for entity_type in ("IfcAlignment", "IfcAlignmentCurve"):
        try:
            alignments = model.by_type(entity_type)
        except Exception:
            continue

        for align in alignments:
            name = getattr(align, "Name", None) or f"#{align.id()}"

            # Try to extract geometry via IfcOpenShell tessellation
            if align.Representation is not None:
                try:
                    from ifc_geo_validator.core.mesh_converter import SETTINGS
                    import ifcopenshell.geom

                    shape = ifcopenshell.geom.create_shape(SETTINGS, align)
                    verts = np.array(shape.geometry.verts).reshape(-1, 3)

                    if len(verts) >= 2:
                        results.append({
                            "name": name,
                            "entity_id": align.id(),
                            "entity_type": entity_type,
                            "points_3d": verts,
                            "points_xy": verts[:, :2],
                        })
                        continue
                except Exception:
                    pass

            # Fallback: try to extract from nested IfcAlignmentHorizontal
            if hasattr(align, "IsNestedBy"):
                for rel in align.IsNestedBy:
                    for child in rel.RelatedObjects:
                        if child.is_a("IfcAlignmentHorizontal"):
                            pts = _extract_horizontal_alignment(child)
                            if pts is not None and len(pts) >= 2:
                                results.append({
                                    "name": name,
                                    "entity_id": align.id(),
                                    "entity_type": entity_type,
                                    "points_3d": np.column_stack([pts, np.zeros(len(pts))]),
                                    "points_xy": pts,
                                })

    return results


def _extract_horizontal_alignment(horiz_align) -> "np.ndarray | None":
    """Extract XY polyline from IfcAlignmentHorizontal segments.

    Evaluates LINE and CIRCULARARC segments parametrically.
    CLOTHOID segments are approximated as circular arcs.
    """
    import numpy as np

    segments = []
    if hasattr(horiz_align, "IsNestedBy"):
        for rel in horiz_align.IsNestedBy:
            for child in rel.RelatedObjects:
                if child.is_a("IfcAlignmentSegment"):
                    dp = child.DesignParameters
                    if dp is not None:
                        segments.append(dp)

    if not segments:
        return None

    points = []
    for seg in segments:
        seg_type = seg.is_a()
        start = getattr(seg, "StartPoint", None)
        if start is None:
            continue

        coords = start.Coordinates if hasattr(start, "Coordinates") else None
        if coords is None or len(coords) < 2:
            continue

        sx, sy = float(coords[0]), float(coords[1])
        length = float(getattr(seg, "SegmentLength", 0) or 0)
        direction = float(getattr(seg, "StartDirection", 0) or 0)

        if "Line" in seg_type or length < 1e-6:
            # Straight segment: start + end
            ex = sx + length * np.cos(direction)
            ey = sy + length * np.sin(direction)
            points.append([sx, sy])
            points.append([ex, ey])

        elif "Circular" in seg_type:
            # Circular arc: tessellate with ~1m spacing
            radius = float(getattr(seg, "StartRadiusOfCurvature", 0) or 0)
            if abs(radius) < 1e-6:
                points.append([sx, sy])
                continue
            n_pts = max(10, int(length / 1.0))
            for k in range(n_pts + 1):
                t = k / n_pts * length
                angle = direction + t / radius
                px = sx + radius * (np.sin(angle) - np.sin(direction))
                py = sy - radius * (np.cos(angle) - np.cos(direction))
                points.append([px, py])
        else:
            # Unknown segment type: just add start point
            points.append([sx, sy])

    if len(points) < 2:
        return None

    return np.array(points)


def get_terrain_mesh(model) -> dict | None:
    """Extract terrain mesh from IfcSite geometry, if available.

    Searches all IfcSite elements for one with a Representation attribute.
    If found, tessellates the geometry to a triangulated mesh using the
    same pipeline as wall element extraction.

    Returns:
        dict with vertices, faces, normals, areas (same as extract_mesh()),
        or None if no IfcSite has geometry.
    """
    from ifc_geo_validator.core.mesh_converter import extract_mesh

    sites = model.by_type("IfcSite")
    for site in sites:
        if site.Representation is not None:
            try:
                return extract_mesh(site)
            except Exception:
                continue
    return None
