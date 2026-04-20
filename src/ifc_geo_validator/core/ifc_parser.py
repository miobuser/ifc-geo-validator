"""IFC file loading, element filtering, and geographic reference extraction.

All geometric measurements returned by this validator are in the IFC's native
coordinate system. For Swiss infrastructure this is typically LV95 (EPSG:2056)
with height reference LN02 or LHN95 — a fact that is load-bearing for any
downstream GIS integration. Use :func:`get_coordinate_system` to extract the
declared frame so every report carries its provenance.

References:
  - Krijnen, T. & Beetz, J. (2020). IfcOpenShell documentation.
  - buildingSMART International (2024). IFC 4.3 ADD2 §8.8 IfcRepresentationResource
    (IfcMapConversion, IfcProjectedCRS).
  - swisstopo (2023). *Das Schweizerische Bezugssystem LV95.*
"""

import os
import ifcopenshell


class IFCLoadError(Exception):
    """Raised when an IFC file cannot be loaded."""
    pass


def get_coordinate_system(model: ifcopenshell.file) -> dict:
    """Extract the IFC geographic reference declaration.

    Reads IfcProjectedCRS (IFC 4.0+) and IfcMapConversion (linear offset
    from local engineering coordinates to the projected CRS). For files
    without an explicit CRS declaration, reports "unspecified" so the
    downstream report does not silently claim a frame that is not in
    the IFC.

    Returns a dict with:
        name:         str — e.g. "EPSG:2056" (Swiss LV95) or "unspecified"
        description:  str — human-readable CRS label
        geodetic_datum: str — e.g. "CH1903+"
        vertical_datum: str — e.g. "LN02" or "LHN95"
        map_unit:     str — e.g. "METRE"
        eastings_offset:  float | None — additive X offset (metres)
        northings_offset: float | None — additive Y offset (metres)
        orthogonal_height_offset: float | None — additive Z offset (metres)
        has_crs:      bool — True if a CRS was declared in the IFC
    """
    result = {
        "name": "unspecified",
        "description": "No IfcProjectedCRS or IfcMapConversion declared",
        "geodetic_datum": None,
        "vertical_datum": None,
        "map_unit": None,
        "eastings_offset": None,
        "northings_offset": None,
        "orthogonal_height_offset": None,
        "has_crs": False,
    }
    try:
        crs_list = model.by_type("IfcProjectedCRS")
    except Exception:
        crs_list = []
    if crs_list:
        crs = crs_list[0]
        result["name"] = getattr(crs, "Name", None) or "unspecified"
        result["description"] = getattr(crs, "Description", None) or result["name"]
        result["geodetic_datum"] = getattr(crs, "GeodeticDatum", None)
        result["vertical_datum"] = getattr(crs, "VerticalDatum", None)
        map_unit = getattr(crs, "MapUnit", None)
        if map_unit is not None:
            result["map_unit"] = getattr(map_unit, "Name", None) or str(map_unit)
        result["has_crs"] = True

    try:
        mc_list = model.by_type("IfcMapConversion")
    except Exception:
        mc_list = []
    if mc_list:
        mc = mc_list[0]
        result["eastings_offset"] = _as_float(getattr(mc, "Eastings", None))
        result["northings_offset"] = _as_float(getattr(mc, "Northings", None))
        result["orthogonal_height_offset"] = _as_float(
            getattr(mc, "OrthogonalHeight", None))
    return result


def _as_float(v):
    """Coerce to float, return None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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


def get_elements(model: ifcopenshell.file,
                 entity_type: str = "IfcWall",
                 predefined_type: str | None = None) -> list:
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


def get_alignments(model: ifcopenshell.file) -> list[dict]:
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


def get_terrain_mesh(model: ifcopenshell.file) -> dict | None:
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
