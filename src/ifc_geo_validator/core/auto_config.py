"""Automatic model configuration.

Analyzes an IFC model and recommends the best validation settings:
  - Which entity types to validate
  - Which ruleset to use
  - What analysis modes are relevant

This enables zero-configuration usage: just pass the IFC file and
the tool figures out what to check.

Usage:
    from ifc_geo_validator.core.auto_config import auto_configure
    config = auto_configure(model)
    # config["entity_types"] = ["IfcWall", "IfcFooting"]
    # config["ruleset"] = "astra_fhb_komplett.yaml"
    # config["has_terrain"] = True
"""

import ifcopenshell


def auto_configure(model: ifcopenshell.file) -> dict:
    """Analyze an IFC model and return recommended configuration.

    Examines: schema version, entity types present, alignment availability,
    terrain geometry, spatial structure, and element names.

    Returns:
        dict with recommended settings:
            entity_types:   list[str] — IFC types to validate
            ruleset:        str — recommended ruleset filename
            has_terrain:    bool — IfcSite has geometry
            has_alignment:  bool — IfcAlignment present
            schema:         str — IFC schema version
            description:    str — human-readable model description
            element_count:  int — total elements to validate
    """
    schema = model.schema

    # Discover entity types with geometry.
    #
    # Intentional design: model.by_type(etype) raises when `etype` is not
    # defined in the loaded IFC schema (e.g. asking for IFC4.3-only types
    # on an IFC2x3 file). Probing is the correct strategy for auto-
    # detection — we don't want the pipeline to fail because one probe
    # missed. Every except clause in this function is a bounded, scoped
    # probe; the alternative (a large try/except around the whole scan)
    # would mask genuine bugs. Keep the blanket Exception catch because
    # IfcOpenShell raises several concrete subclasses (RuntimeError,
    # AttributeError, and custom errors) and probing each is just as
    # noisy.
    structural_types = [
        "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcMember",
        "IfcFooting", "IfcPile", "IfcPlate", "IfcRailing",
        "IfcBuildingElementProxy",
    ]

    found_types = {}
    for etype in structural_types:
        try:
            elems = model.by_type(etype)
            with_geom = [e for e in elems if e.Representation is not None]
            if with_geom:
                found_types[etype] = len(with_geom)
        except Exception:
            # Type not available in this IFC schema version — skip.
            pass

    # Check for terrain
    has_terrain = False
    try:
        sites = model.by_type("IfcSite")
        has_terrain = any(s.Representation is not None for s in sites)
    except Exception:
        pass

    # Check for alignment
    has_alignment = False
    for atype in ("IfcAlignment", "IfcAlignmentCurve"):
        try:
            if model.by_type(atype):
                has_alignment = True
                break
        except Exception:
            pass

    # Determine best entity types to validate
    entity_types = []
    # Prioritize structural wall types
    for etype in ["IfcWall", "IfcSlab", "IfcFooting", "IfcColumn"]:
        if etype in found_types:
            entity_types.append(etype)
    # Add proxy only if no specific types found
    if not entity_types and "IfcBuildingElementProxy" in found_types:
        entity_types.append("IfcBuildingElementProxy")
    # Always include railings for distance checks
    if "IfcRailing" in found_types and entity_types:
        entity_types.append("IfcRailing")

    if not entity_types:
        entity_types = list(found_types.keys())[:3]

    # Determine best ruleset
    has_walls = "IfcWall" in found_types
    has_tunnel_elements = "IfcBuildingElementProxy" in found_types and not has_walls
    total_elements = sum(found_types.get(t, 0) for t in entity_types)

    if has_walls:
        ruleset = "astra_fhb_komplett.yaml"
        desc = "Infrastrukturbauwerk mit Wänden"
    elif has_tunnel_elements:
        ruleset = "astra_fhb_tunnel.yaml"
        desc = "Tunnelbauwerk"
    else:
        ruleset = "astra_fhb_komplett.yaml"
        desc = "Allgemeines Bauwerk"

    # Check element names for hints
    all_names = []
    for etype in entity_types:
        try:
            for e in model.by_type(etype):
                name = getattr(e, "Name", "") or ""
                if name:
                    all_names.append(name.lower())
        except Exception:
            pass

    if any("tunnel" in n for n in all_names):
        ruleset = "astra_fhb_tunnel.yaml"
        desc = "Tunnelbauwerk (Name-basiert)"
    elif any("stütz" in n or "stuetz" in n or "retaining" in n for n in all_names):
        desc = "Stützbauwerk"

    return {
        "entity_types": entity_types,
        "ruleset": ruleset,
        "has_terrain": has_terrain,
        "has_alignment": has_alignment,
        "schema": schema,
        "description": desc,
        "element_count": total_elements,
        "found_types": found_types,
    }
