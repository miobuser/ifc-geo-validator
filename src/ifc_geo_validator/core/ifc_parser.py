"""IFC file loading and element filtering."""

import ifcopenshell


def load_model(path: str) -> ifcopenshell.file:
    """Load an IFC file and return the model object."""
    return ifcopenshell.open(path)


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
