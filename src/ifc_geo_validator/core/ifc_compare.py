"""Compare two IFC models for geometric deviations.

Matches elements by name or GlobalId, then compares their geometric
properties (volume, dimensions, crown width, thickness, etc.) and
reports deviations exceeding a tolerance.

Use case: As-designed vs As-built comparison.

Usage:
    from ifc_geo_validator.core.ifc_compare import compare_models
    deviations = compare_models("design.ifc", "asbuilt.ifc")
"""

import numpy as np

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3


def compare_models(
    path_a: str,
    path_b: str,
    entity_type: str = "IfcWall",
    match_by: str = "name",
    tolerance_mm: float = 10.0,
) -> dict:
    """Compare geometry of matched elements between two IFC models.

    Args:
        path_a: path to reference model (e.g. as-designed).
        path_b: path to comparison model (e.g. as-built).
        entity_type: IFC entity type to compare.
        match_by: "name" or "globalid" — how to match elements.
        tolerance_mm: deviations below this are considered equal.

    Returns:
        dict with:
            matched: list of matched element comparisons
            unmatched_a: elements only in model A
            unmatched_b: elements only in model B
            summary: aggregate deviation statistics
    """
    model_a = load_model(path_a)
    model_b = load_model(path_b)

    elems_a = get_elements(model_a, entity_type)
    elems_b = get_elements(model_b, entity_type)

    # Build lookup dicts
    def _key(elem):
        if match_by == "globalid":
            return getattr(elem, "GlobalId", None)
        return getattr(elem, "Name", None) or f"#{elem.id()}"

    dict_a = {_key(e): e for e in elems_a}
    dict_b = {_key(e): e for e in elems_b}

    matched = []
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())
    common = keys_a & keys_b

    for key in sorted(common):
        ea, eb = dict_a[key], dict_b[key]
        try:
            props_a = _extract_properties(ea)
            props_b = _extract_properties(eb)
            devs = _compute_deviations(props_a, props_b, tolerance_mm)
            matched.append({
                "name": key,
                "id_a": ea.id(),
                "id_b": eb.id(),
                "properties_a": props_a,
                "properties_b": props_b,
                "deviations": devs,
                "has_deviation": any(d["exceeds_tolerance"] for d in devs),
            })
        except Exception as ex:
            matched.append({
                "name": key,
                "error": str(ex),
            })

    # Stats
    n_with_dev = sum(1 for m in matched if m.get("has_deviation"))

    return {
        "matched": matched,
        "unmatched_a": sorted(keys_a - keys_b),
        "unmatched_b": sorted(keys_b - keys_a),
        "summary": {
            "total_matched": len(matched),
            "with_deviations": n_with_dev,
            "unmatched_a": len(keys_a - keys_b),
            "unmatched_b": len(keys_b - keys_a),
            "tolerance_mm": tolerance_mm,
        },
    }


def _extract_properties(element) -> dict:
    """Extract all measurable properties from an IFC element."""
    mesh = extract_mesh(element)
    l1 = validate_level1(mesh)
    l2 = validate_level2(mesh)
    l3 = validate_level3(mesh, l2)

    return {
        "volume_m3": round(l1["volume"], 4),
        "total_area_m2": round(l1["total_area"], 4),
        "crown_width_mm": round(l3.get("crown_width_mm", 0), 1),
        "crown_slope_pct": round(l3.get("crown_slope_percent", 0), 2),
        "min_thickness_mm": round(l3.get("min_wall_thickness_mm", 0), 1),
        "wall_height_m": round(l3.get("wall_height_m", 0), 3),
        "is_curved": l3.get("is_curved", False),
    }


def _compute_deviations(props_a: dict, props_b: dict,
                         tolerance_mm: float) -> list[dict]:
    """Compare properties and return deviations."""
    devs = []
    comparisons = [
        ("volume_m3", "m³", 0.001),
        ("total_area_m2", "m²", 0.01),
        ("crown_width_mm", "mm", tolerance_mm),
        ("crown_slope_pct", "%", 0.1),
        ("min_thickness_mm", "mm", tolerance_mm),
        ("wall_height_m", "m", tolerance_mm / 1000),
    ]

    for key, unit, tol in comparisons:
        va = props_a.get(key, 0)
        vb = props_b.get(key, 0)
        if va is None or vb is None:
            continue
        diff = abs(va - vb)
        devs.append({
            "property": key,
            "value_a": va,
            "value_b": vb,
            "difference": round(diff, 4),
            "unit": unit,
            "tolerance": tol,
            "exceeds_tolerance": diff > tol,
        })

    return devs
