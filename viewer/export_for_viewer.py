"""Export validation results as JSON for the browser-based 3D viewer.

Usage:
    python viewer/export_for_viewer.py tests/test_models/T8_curved_wall.ifc
    python viewer/export_for_viewer.py tests/test_models/T4_l_shaped.ifc

Opens viewer/viewer.html automatically with the exported data.
"""
import sys
import os
import json
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.validation.level5 import validate_level5
from ifc_geo_validator.validation.level6 import validate_level6
from ifc_geo_validator.core.face_classifier import (
    CROWN, FOUNDATION, FRONT, BACK, END_LEFT, END_RIGHT, UNCLASSIFIED,
)
import math
import numpy as np

COLORS = {
    CROWN:        [0.13, 0.59, 0.95],  # blue
    FOUNDATION:   [0.47, 0.33, 0.28],  # brown
    FRONT:        [0.96, 0.26, 0.21],  # red
    BACK:         [1.00, 0.60, 0.00],  # orange
    END_LEFT:     [0.30, 0.69, 0.31],  # green
    END_RIGHT:    [0.55, 0.76, 0.29],  # light green
    UNCLASSIFIED: [0.62, 0.62, 0.62],  # grey
}


def export_model(ifc_path: str, output_path: str = None):
    """Export IFC model with classification data for the 3D viewer."""
    model = load_model(ifc_path)
    walls = get_elements(model, "IfcWall")
    terrain = get_terrain_mesh(model)

    if not walls:
        print(f"No IfcWall elements found in {ifc_path}")
        return

    # Phase 1: Per-element L1-L3
    elem_data = []
    for wall in walls:
        wname = getattr(wall, "Name", None) or f"#{wall.id()}"
        mesh = extract_mesh(wall)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        elem_data.append({
            "wall": wall, "name": wname, "mesh": mesh,
            "l1": l1, "l2": l2, "l3": l3,
            "element_id": wall.id(), "element_name": wname,
            "level1": l1, "level2": l2, "level3": l3, "mesh_data": mesh,
        })

    # Phase 2: L5 inter-element
    l5_global = validate_level5(elem_data) if len(elem_data) > 1 else None
    l5_pairs = l5_global.get("pairs", []) if l5_global else []

    # Phase 3: L6 terrain
    l6_global = validate_level6(elem_data, terrain_mesh=terrain) if (terrain or len(elem_data) > 1) else None
    l6_data = {}
    if l6_global:
        l6_data = {
            "terrain_available": l6_global.get("terrain_available", False),
            "clearances": l6_global.get("clearances", []),
            "embedments": l6_global.get("embedments", []),
            "distances": l6_global.get("distances", []),
        }

    elements = []
    for ed in elem_data:
        wall = ed["wall"]
        name = ed["name"]
        mesh = ed["mesh"]
        l1, l2, l3 = ed["l1"], ed["l2"], ed["l3"]

        verts = mesh["vertices"].tolist()
        faces = mesh["faces"].tolist()

        # Build per-vertex colors (average of adjacent face colors)
        n_verts = len(mesh["vertices"])
        vert_colors = [[0.5, 0.5, 0.5]] * n_verts
        vert_counts = [0] * n_verts

        face_data = []
        for g in l2["face_groups"]:
            cat = g["category"]
            color = COLORS.get(cat, [0.5, 0.5, 0.5])
            for fi in g["face_indices"]:
                tri = mesh["faces"][fi]
                face_data.append({
                    "indices": tri.tolist(),
                    "category": cat,
                    "color": color,
                })
                for vi in tri:
                    if vert_counts[vi] == 0:
                        vert_colors[vi] = list(color)
                    else:
                        # Blend
                        for c in range(3):
                            vert_colors[vi][c] = (vert_colors[vi][c] * vert_counts[vi] + color[c]) / (vert_counts[vi] + 1)
                    vert_counts[vi] += 1

        # Centerline
        cl = l2.get("centerline")
        centerline_data = None
        if cl and hasattr(cl, "points_2d") and len(cl.points_2d) > 2:
            centerline_data = {
                "points": cl.points_2d.tolist(),
                "is_curved": bool(cl.is_curved),
                "length": round(cl.length, 2),
            }

        # Compute per-triangle slope data for heatmap overlay
        from ifc_geo_validator.viz.slope_heatmap import compute_triangle_slopes
        slope_result = compute_triangle_slopes(mesh, centerline=cl)
        slope_data = {
            "cross": [round(float(v), 2) for v in slope_result["cross_slope_pct"]],
            "long": [round(float(v), 2) for v in slope_result["long_slope_pct"]],
            "total": [round(float(v), 2) for v in np.clip(slope_result["total_slope_pct"], 0, 100)],
        }

        elements.append({
            "name": name,
            "id": wall.id(),
            "vertices": verts,
            "faces": faces,
            "face_data": face_data,
            "vertex_colors": vert_colors,
            "slope_per_face": slope_data,
            "groups": [
                {
                    "category": g["category"],
                    "area": round(g["area"], 3),
                    "normal": [round(n, 4) for n in g["normal"]],
                    "centroid": [round(c, 3) for c in g["centroid"]],
                    "triangles": g["num_triangles"],
                    "face_indices": g["face_indices"],
                }
                for g in l2["face_groups"]
            ],
            "metrics": {
                "volume_m3": round(l1["volume"], 3),
                "total_area_m2": round(l1["total_area"], 3),
                "watertight": l1["is_watertight"],
                "crown_width_mm": round(l3.get("crown_width_mm", 0), 1),
                "wall_thickness_mm": round(l3.get("min_wall_thickness_mm", 0), 1),
                "crown_slope_pct": round(l3.get("crown_slope_percent", 0), 2),
                "wall_height_m": round(l3.get("wall_height_m", 0), 2),
                "is_curved": l3.get("is_curved", False),
                "crown_width_cv": round(l3.get("crown_width_cv", 0), 6) if l3.get("crown_width_cv") is not None else None,
                "foundation_width_mm": round(l3.get("foundation_width_mm", 0), 1) if l3.get("foundation_width_mm") else None,
                "inclination": _format_inclination(l3),
            },
            "centerline": centerline_data,
            "rules": _get_rules(l1, l3, wall.id(), l5_global, l6_global),
        })

    # Terrain
    terrain_data = None
    if terrain:
        terrain_data = {
            "vertices": terrain["vertices"].tolist(),
            "faces": terrain["faces"].tolist(),
        }

    data = {
        "source": os.path.basename(ifc_path),
        "elements": elements,
        "terrain": terrain_data,
        "l5_pairs": l5_pairs,
        "l6": l6_data,
        "categories": {
            cat: {"color": COLORS[cat], "label": {
                CROWN: "Krone", FOUNDATION: "Fundament", FRONT: "Ansichtsfläche",
                BACK: "Rückseite", END_LEFT: "Stirnfläche L", END_RIGHT: "Stirnfläche R",
                UNCLASSIFIED: "Unklassifiziert",
            }[cat]} for cat in COLORS
        },
    }

    if output_path is None:
        output_path = str(Path(__file__).parent / "model_data.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=None)  # compact for performance

    print(f"Exported {len(elements)} elements to {output_path}")
    size_kb = os.path.getsize(output_path) / 1024
    print(f"File size: {size_kb:.0f} KB")
    return output_path


RULESET_PATH = Path(__file__).resolve().parents[1] / "src" / "ifc_geo_validator" / "rules" / "rulesets" / "astra_fhb_stuetzmauer.yaml"


def _format_inclination(l3):
    inc_r = l3.get("front_inclination_ratio")
    if inc_r and math.isinf(inc_r):
        return "vertikal"
    elif inc_r:
        return f"{inc_r:.1f}:1"
    return "—"


def _get_rules(l1, l3, eid=None, l5_global=None, l6_global=None):
    """Run L4 rule evaluation with full L5/L6 context for the viewer."""
    if not RULESET_PATH.exists():
        return []
    try:
        ruleset = load_ruleset(str(RULESET_PATH))

        # Build L5 context
        l5_ctx = {}
        if l5_global and eid:
            for p in l5_global.get("pairs", []):
                if p.get("element_a_id") == eid or p.get("element_b_id") == eid:
                    if p["pair_type"] == "stacked":
                        l5_ctx["foundation_extends_beyond_wall"] = p.get("foundation_extends_beyond_wall", False)
                        l5_ctx["wall_foundation_gap_mm"] = p.get("vertical_gap_mm", 0)

        # Build L6 context
        l6_ctx = {}
        if l6_global and eid:
            l6_ctx["earth_side_determined"] = eid in l6_global.get("terrain_side", {})

        l4 = validate_level4(l1, l3, ruleset, level5_context=l5_ctx, level6_context=l6_ctx)
        return [
            {
                "id": c["rule_id"],
                "name": c["name"],
                "status": c["status"],
                "severity": c["severity"],
                "message": c.get("message", ""),
                "check": c.get("check_expr", ""),
            }
            for c in l4.get("checks", [])
        ]
    except Exception:
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viewer/export_for_viewer.py <ifc_file>")
        print("Example: python viewer/export_for_viewer.py tests/test_models/T8_curved_wall.ifc")
        sys.exit(1)

    ifc_path = sys.argv[1]
    out = export_model(ifc_path)
    if out:
        viewer_html = str(Path(__file__).parent / "viewer.html")
        print(f"\nOpen in browser: {viewer_html}")
        webbrowser.open(f"file:///{viewer_html}")
