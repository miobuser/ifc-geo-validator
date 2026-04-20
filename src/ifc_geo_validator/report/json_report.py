"""Structured JSON report generation for validation results.

Produces a self-contained JSON document with metadata, element results,
and summary statistics suitable for post-processing and thesis documentation.
"""

import json
import math
from datetime import datetime
from pathlib import Path


def generate_report(
    ifc_path: str,
    elements_results: list[dict],
    ruleset: dict = None,
    coordinate_system: dict = None,
) -> dict:
    """Generate a structured validation report.

    Args:
        ifc_path: Path to the validated IFC file.
        elements_results: list of per-element result dicts from the CLI pipeline.
        ruleset: optional parsed YAML ruleset dict.
        coordinate_system: dict from ifc_parser.get_coordinate_system().
            When provided, is recorded in the report so every measurement
            carries the frame it was taken in (critical for Swiss LV95
            infrastructure models).

    Returns:
        dict: Complete report structure.
    """
    try:
        from ifc_geo_validator import get_version
        version = get_version()
    except Exception:
        version = "unknown"

    report = {
        "report": {
            "generator": "ifc-geo-validator",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "ifc_file": str(Path(ifc_path).name),
            "ifc_path": str(ifc_path),
        },
        "coordinate_system": coordinate_system or {
            "name": "unspecified",
            "has_crs": False,
        },
        "ruleset": None,
        "elements": [],
        "summary": {},
    }

    if ruleset:
        report["ruleset"] = {
            "name": ruleset["metadata"]["name"],
            "version": ruleset["metadata"].get("version", "unknown"),
            "source": ruleset["metadata"].get("source", ""),
        }

    # Process element results
    for elem in elements_results:
        report["elements"].append(_process_element(elem))

    # Compute summary
    report["summary"] = _compute_summary(elements_results)

    return report


def write_report(report: dict, output_path: str) -> None:
    """Write report to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=json_default)


def _process_element(elem: dict) -> dict:
    """Clean up a single element result for the report."""
    result = {
        "element_id": elem.get("element_id"),
        "element_name": elem.get("element_name"),
    }

    if "error" in elem:
        result["error"] = elem["error"]
        return result

    # Level 1
    if "level1" in elem:
        l1 = elem["level1"]
        result["geometry"] = {
            "volume_m3": _round(l1.get("volume")),
            "total_area_m2": _round(l1.get("total_area")),
            "bounding_box": l1.get("bbox"),
            "centroid": l1.get("centroid"),
            "is_watertight": l1.get("is_watertight"),
            "num_triangles": l1.get("num_triangles"),
            "num_vertices": l1.get("num_vertices"),
        }

    # Level 2
    if "level2" in elem:
        l2 = elem["level2"]
        result["face_classification"] = {
            "wall_axis": l2.get("wall_axis"),
            "num_groups": l2.get("num_groups"),
            "summary": l2.get("summary"),
            "groups": [
                {
                    "category": g["category"],
                    "area_m2": _round(g["area"]),
                    "normal": g["normal"],
                    "num_triangles": g["num_triangles"],
                }
                for g in l2.get("face_groups", [])
            ],
        }

    # Level 3
    if "level3" in elem:
        l3 = elem["level3"]
        result["measurements"] = {
            "crown_width_mm": _round(l3.get("crown_width_mm")),
            "crown_slope_percent": _round(l3.get("crown_slope_percent")),
            "min_wall_thickness_mm": _round(l3.get("min_wall_thickness_mm")),
            "avg_wall_thickness_mm": _round(l3.get("avg_wall_thickness_mm")),
            "front_inclination_deg": _round(l3.get("front_inclination_deg")),
            "front_inclination_ratio": _safe_ratio(l3.get("front_inclination_ratio")),
            "wall_height_m": _round(l3.get("wall_height_m")),
            "is_curved": l3.get("is_curved"),
            "wall_length_m": _round(l3.get("wall_length_m")),
        }

    # Level 4
    if "level4" in elem:
        l4 = elem["level4"]
        result["rule_checks"] = {
            "summary": l4.get("summary"),
            "checks": l4.get("checks"),
        }

    # Level 5 (inter-element context, if available)
    if "level5" in elem:
        result["inter_element"] = elem["level5"]

    # Level 6 (terrain/distance context, if available)
    if "level6" in elem:
        result["terrain_context"] = elem["level6"]

    return result


def _compute_summary(elements_results: list[dict]) -> dict:
    """Compute aggregate statistics across all elements."""
    total = len(elements_results)
    errors = sum(1 for e in elements_results if "error" in e)
    validated = total - errors

    # Aggregate L4 results
    all_checks = []
    for elem in elements_results:
        if "level4" in elem:
            all_checks.extend(elem["level4"].get("checks", []))

    l4_summary = {}
    if all_checks:
        l4_summary = {
            "total_checks": len(all_checks),
            "passed": sum(1 for c in all_checks if c["status"] == "PASS"),
            "failed": sum(1 for c in all_checks if c["status"] == "FAIL"),
            "skipped": sum(1 for c in all_checks if c["status"] == "SKIP"),
        }

    return {
        "total_elements": total,
        "validated": validated,
        "errors": errors,
        "rule_checks": l4_summary,
    }


def _round(value, decimals=3):
    if value is None:
        return None
    return round(float(value), decimals)


def _safe_ratio(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isinf(value):
        return "vertical"
    return _round(value)


def json_default(obj):
    """JSON serialisation fallback for numpy scalars and special floats.

    Exported so that cli.py, app.py, and any other caller share the
    same conversion rules and do not drift. Raises TypeError for
    genuinely non-serialisable types so json.dumps surfaces the
    failure instead of silently dropping data.
    """
    if hasattr(obj, "item"):
        return obj.item()  # numpy scalar
    if isinstance(obj, float) and math.isinf(obj):
        return "Infinity"
    raise TypeError(f"Not JSON serializable: {type(obj)}")
