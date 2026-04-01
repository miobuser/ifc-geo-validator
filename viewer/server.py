"""Local validation server — Upload IFC, run pipeline, view results in 3D.

Usage:
    python viewer/server.py

Opens http://localhost:8080 with the 3D viewer.
Drag & drop an IFC file onto the page to validate it.
"""
import sys
import os
import json
import tempfile
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs

# Add project to path
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
import numpy as np
import math

PORT = 8080
VIEWER_DIR = Path(__file__).parent

COLORS = {
    CROWN:        [0.13, 0.59, 0.95],
    FOUNDATION:   [0.47, 0.33, 0.28],
    FRONT:        [0.96, 0.26, 0.21],
    BACK:         [1.00, 0.60, 0.00],
    END_LEFT:     [0.30, 0.69, 0.31],
    END_RIGHT:    [0.55, 0.76, 0.29],
    UNCLASSIFIED: [0.62, 0.62, 0.62],
}

RULESET_PATH = Path(__file__).resolve().parents[1] / "src" / "ifc_geo_validator" / "rules" / "rulesets" / "astra_fhb_stuetzmauer.yaml"


def run_pipeline(ifc_path: str) -> dict:
    """Run the full L1-L6 pipeline and return viewer-compatible JSON."""
    model = load_model(ifc_path)
    walls = get_elements(model, "IfcWall")
    terrain = get_terrain_mesh(model)
    ruleset = load_ruleset(str(RULESET_PATH)) if RULESET_PATH.exists() else None

    if not walls:
        return {"error": "No IfcWall elements found", "elements": []}

    all_elems_data = []
    elements = []

    for wall in walls:
        name = getattr(wall, "Name", None) or f"#{wall.id()}"
        mesh = extract_mesh(wall)
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)

        elem_data = {
            "element_id": wall.id(),
            "element_name": name,
            "level1": l1, "level2": l2, "level3": l3,
            "mesh_data": mesh,
        }
        all_elems_data.append(elem_data)

        verts = mesh["vertices"].tolist()
        faces_list = mesh["faces"].tolist()

        # Per-vertex colors
        n_verts = len(mesh["vertices"])
        vert_colors = [[0.5, 0.5, 0.5]] * n_verts
        vert_counts = [0] * n_verts

        face_data = []
        for g in l2["face_groups"]:
            cat = g["category"]
            color = COLORS.get(cat, [0.5, 0.5, 0.5])
            for fi in g["face_indices"]:
                tri = mesh["faces"][fi]
                face_data.append({"indices": tri.tolist(), "category": cat, "color": color})
                for vi in tri:
                    if vert_counts[vi] == 0:
                        vert_colors[vi] = list(color)
                    else:
                        for c in range(3):
                            vert_colors[vi][c] = (vert_colors[vi][c] * vert_counts[vi] + color[c]) / (vert_counts[vi] + 1)
                    vert_counts[vi] += 1

        cl = l2.get("centerline")
        centerline_data = None
        if cl and hasattr(cl, "points_2d") and len(cl.points_2d) > 2:
            centerline_data = {
                "points": cl.points_2d.tolist(),
                "is_curved": bool(cl.is_curved),
                "length": round(cl.length, 2),
            }

        inc_r = l3.get("front_inclination_ratio")
        inc_str = "vertikal" if inc_r and math.isinf(inc_r) else f"{inc_r:.1f}:1" if inc_r else "N/A"

        elements.append({
            "name": name,
            "id": wall.id(),
            "vertices": verts,
            "faces": faces_list,
            "face_data": face_data,
            "vertex_colors": vert_colors,
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
                "inclination": inc_str,
            },
            "centerline": centerline_data,
            "rules": [],  # populated below after L5/L6
        })

    # L5 — compute once for all elements
    l5 = None
    l5_pairs = []
    if len(all_elems_data) > 1:
        l5 = validate_level5(all_elems_data)
        l5_pairs = l5.get("pairs", [])

    # L6
    l6 = None
    l6_data = {}
    if terrain or len(all_elems_data) > 1:
        l6 = validate_level6(all_elems_data, terrain_mesh=terrain)
        l6_data = {
            "terrain_available": l6.get("terrain_available", False),
            "clearances": l6.get("clearances", []),
            "embedments": l6.get("embedments", []),
            "distances": l6.get("distances", []),
        }

    # L4 rules with full L5/L6 context
    if ruleset:
        for i, elem_view in enumerate(elements):
            eid = elem_view["id"]
            ed = all_elems_data[i]
            l1, l3 = ed["level1"], ed["level3"]

            l5_ctx = {}
            if l5:
                for p in l5.get("pairs", []):
                    if p.get("element_a_id") == eid or p.get("element_b_id") == eid:
                        if p["pair_type"] == "stacked":
                            l5_ctx["foundation_extends_beyond_wall"] = p.get("foundation_extends_beyond_wall", False)
                            l5_ctx["wall_foundation_gap_mm"] = p.get("vertical_gap_mm", 0)

            l6_ctx = {}
            if l6_data.get("terrain_available") and l6:
                l6_ctx["earth_side_determined"] = eid in l6.get("terrain_side", {})
                for emb in l6.get("embedments", []):
                    if emb.get("element_id") == eid:
                        l6_ctx["foundation_embedment_m"] = emb["foundation_embedment_m"]
                        break

            l4 = validate_level4(l1, l3, ruleset, level5_context=l5_ctx, level6_context=l6_ctx)
            elem_view["rules"] = [
                {"id": c["rule_id"], "name": c["name"], "status": c["status"],
                 "severity": c["severity"], "message": c.get("message", "")}
                for c in l4.get("checks", [])
            ]

    # Terrain mesh
    terrain_data = None
    if terrain:
        terrain_data = {
            "vertices": terrain["vertices"].tolist(),
            "faces": terrain["faces"].tolist(),
        }

    return {
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


class ValidatorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/validate":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
                f.write(body)
                tmp_path = f.name

            try:
                print(f"Validating: {tmp_path} ({content_length / 1024:.0f} KB)")
                result = run_pipeline(tmp_path)
                response = json.dumps(result)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))
                print(f"  → {len(result.get('elements', []))} elements exported")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                print(f"  → Error: {e}")
            finally:
                os.unlink(tmp_path)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress verbose GET logs
        if "POST" in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    print(f"IFC Geo Validator — Server starting on http://localhost:{PORT}")
    print(f"Viewer: http://localhost:{PORT}/viewer.html")
    print(f"Upload IFC files via drag & drop in the browser.")
    print()

    server = HTTPServer(("localhost", PORT), ValidatorHandler)
    webbrowser.open(f"http://localhost:{PORT}/viewer.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
