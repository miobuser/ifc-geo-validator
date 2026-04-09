"""Inject validation results into IFC model as IfcPropertySet.

Creates a custom property set 'Pset_GeoValidation' on each validated
element, containing key metrics from L1, L3, and L4.  The enriched
model is written to a new file — the original is never modified.

Usage (standalone):
    from ifc_geo_validator.report.ifc_property_writer import inject_properties
    inject_properties(model, element, elem_result)
    model.write("enriched.ifc")
"""

import math

import ifcopenshell
import ifcopenshell.api

PSET_NAME = "Pset_GeoValidation"


def inject_properties(
    model: ifcopenshell.file,
    element,
    elem_result: dict,
) -> None:
    """Add validation result properties to an IFC element.

    Args:
        model: The opened IFC model (will be modified in place).
        element: The IFC element (e.g., IfcWall) to annotate.
        elem_result: Pipeline result dict with level1/level2/level3/level4 keys.
    """
    props = _collect_properties(elem_result)
    if not props:
        return

    pset = ifcopenshell.api.run(
        "pset.add_pset", model, product=element, name=PSET_NAME,
    )
    ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=props)


def inject_all(
    model: ifcopenshell.file,
    elements: list,
    results: list[dict],
    output_path: str,
) -> str:
    """Inject properties for all elements and write enriched IFC.

    Args:
        model: The opened IFC model.
        elements: List of IFC elements in same order as results.
        results: List of per-element result dicts.
        output_path: Path for the enriched IFC output file.

    Returns:
        The output path.
    """
    for element, result in zip(elements, results):
        if "error" not in result:
            inject_properties(model, element, result)

    model.write(output_path)
    return output_path


def _collect_properties(elem_result: dict) -> dict:
    """Extract flat property dict from nested validation results."""
    props = {}

    # Level 1: geometry metrics
    l1 = elem_result.get("level1")
    if l1:
        props["Volume_m3"] = round(l1["volume"], 4)
        props["TotalArea_m2"] = round(l1["total_area"], 4)
        props["IsWatertight"] = l1["is_watertight"]
        props["NumTriangles"] = l1["num_triangles"]
        bbox = l1.get("bbox", {})
        size = bbox.get("size", [0, 0, 0])
        props["BboxLength_m"] = round(max(size), 4)
        props["BboxHeight_m"] = round(size[2] if len(size) > 2 else 0, 4)

        # Mesh quality diagnostics
        n_degen = l1.get("n_degenerate_filtered", 0)
        if n_degen > 0:
            props["DegenerateTrianglesFiltered"] = n_degen
        q = l1.get("mesh_quality", {})
        nm = q.get("non_manifold_edges", 0)
        if nm > 0:
            props["NonManifoldEdges"] = nm

    # Level 2: classification summary
    l2 = elem_result.get("level2")
    if l2:
        props["NumFaceGroups"] = l2["num_groups"]
        n_bodies = l2.get("n_bodies", 1)
        if n_bodies > 1:
            props["NumBodies"] = n_bodies
        props["HasCrown"] = l2.get("has_crown", False)
        props["HasFoundation"] = l2.get("has_foundation", False)
        props["HasFront"] = l2.get("has_front", False)
        props["HasBack"] = l2.get("has_back", False)
        asym = l2.get("front_back_asymmetry")
        if asym is not None:
            props["FrontBackAsymmetry"] = round(asym, 4)

    # Level 3: face measurements
    l3 = elem_result.get("level3")
    if l3:
        if "crown_width_mm" in l3:
            props["CrownWidth_mm"] = round(l3["crown_width_mm"], 1)
        if "crown_slope_percent" in l3:
            props["CrownSlope_pct"] = round(l3["crown_slope_percent"], 2)
        if "min_wall_thickness_mm" in l3:
            props["MinWallThickness_mm"] = round(l3["min_wall_thickness_mm"], 1)
        if "front_inclination_deg" in l3:
            props["FrontInclination_deg"] = round(l3["front_inclination_deg"], 2)
        ratio = l3.get("front_inclination_ratio")
        if ratio is not None:
            if isinstance(ratio, float) and math.isinf(ratio):
                props["FrontInclination_ratio"] = "vertical"
            else:
                props["FrontInclination_ratio"] = f"{ratio:.1f}:1"
        if "wall_height_m" in l3:
            props["WallHeight_m"] = round(l3["wall_height_m"], 4)
        if "is_curved" in l3:
            props["IsCurved"] = l3["is_curved"]
        if "wall_length_m" in l3:
            props["WallLength_m"] = round(l3["wall_length_m"], 4)
        if "crown_width_cv" in l3:
            props["CrownWidthCV"] = round(l3["crown_width_cv"], 6)

    # Min distance to nearest element
    if l3:
        min_d = l3.get("min_distance_to_nearest_mm")
        if min_d is not None:
            props["MinDistanceToNearest_mm"] = round(min_d, 1)

    # Curvature (if available)
    if l3:
        min_r = l3.get("min_radius_m")
        if min_r is not None and min_r != float("inf"):
            props["MinRadius_m"] = round(min_r, 2)

    # Clearance (if computed)
    clearance = elem_result.get("clearance")
    if clearance:
        props["ClearanceClear"] = clearance.get("clear", True)
        pen = clearance.get("max_penetration_mm", 0)
        if pen > 0:
            props["ClearancePenetration_mm"] = round(pen, 1)

    # Slope analysis (if computed)
    slope_data = elem_result.get("slope_analysis")
    if slope_data:
        props["CrossSlope_avg_pct"] = round(slope_data["area_weighted_cross_pct"], 2)
        props["CrossSlope_max_pct"] = round(slope_data["max_cross_pct"], 2)
        props["LongSlope_avg_pct"] = round(slope_data["area_weighted_long_pct"], 2)
        props["LongSlope_max_pct"] = round(slope_data["max_long_pct"], 2)

    # Level 5: inter-element context
    l5 = elem_result.get("level5_context")
    if l5:
        if "foundation_extends_beyond_wall" in l5:
            props["FoundationOverhang"] = l5["foundation_extends_beyond_wall"]
        if "wall_foundation_gap_mm" in l5:
            props["FoundationGap_mm"] = round(l5["wall_foundation_gap_mm"], 1)

    # Level 6: terrain context
    l6 = elem_result.get("level6_context")
    if l6:
        if "earth_side_determined" in l6:
            props["EarthSideDetermined"] = l6["earth_side_determined"]
        if "foundation_embedment_m" in l6:
            props["FoundationEmbedment_m"] = round(l6["foundation_embedment_m"], 4)

    # Level 4: rule check summary
    l4 = elem_result.get("level4")
    if l4:
        s = l4.get("summary", {})
        props["RulesPassed"] = s.get("passed", 0)
        props["RulesFailed"] = s.get("failed", 0)
        props["RulesTotal"] = s.get("total", 0)
        # Individual rule results as "PASS"/"FAIL"/"SKIP"
        for chk in l4.get("checks", []):
            props[f"Rule_{chk['rule_id']}"] = chk["status"]

    return props
