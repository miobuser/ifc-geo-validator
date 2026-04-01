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
