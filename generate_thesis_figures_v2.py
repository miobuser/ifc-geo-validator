"""Generate all thesis figures (Chapter 4+5) from current code state.

Usage:
    python generate_thesis_figures_v2.py

Generates figures in viz_output/:
    - validation_matrix.csv + .md
    - sensitivity plots (via sensitivity_analysis.py)
    - measurement summary table
"""

import sys
import os
import csv
import math

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
os.makedirs("viz_output", exist_ok=True)

from pathlib import Path
from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.viz.slope_heatmap import compute_surface_slopes
from ifc_geo_validator.core.advanced_geometry import compute_taper_profile, check_plumbness
import numpy as np


RULESET = load_ruleset("src/ifc_geo_validator/rules/rulesets/astra_fhb_stuetzmauer.yaml")
TEST_DIR = Path("tests/test_models")


def main():
    print("=" * 60)
    print("Thesis Figure Generator v2")
    print("=" * 60)
    print()

    # 1. Complete measurement table for ALL models
    print("1. Generating measurement table...")
    models = sorted(TEST_DIR.glob("*.ifc"))
    rows = []

    for mf in models:
        model = load_model(str(mf))
        for e in get_elements(model, "IfcWall"):
            name = getattr(e, "Name", "")[:25]
            mesh = extract_mesh(e)
            l1 = validate_level1(mesh)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)

            # Slope
            sl = compute_surface_slopes(
                mesh, l2["face_groups"], ["crown"],
                np.array(l2["wall_axis"]), l2.get("centerline"),
            )

            # Taper
            taper = compute_taper_profile(mesh, l2["face_groups"], np.array(l2["wall_axis"]))

            # Plumbness
            plumb = check_plumbness(l2["face_groups"])

            # Curvature
            cl = l2.get("centerline")
            r_min = float("inf")
            if cl and hasattr(cl, "curvature_profile"):
                r_min = cl.curvature_profile()["min_radius_m"]

            # L4
            l4 = validate_level4(l1, l3, RULESET, level2_result=l2)
            s = l4["summary"]

            row = {
                "Model": mf.stem,
                "Element": name,
                "Role": l2.get("element_role", "?"),
                "Conf": f"{l2.get('confidence', 0):.0%}",
                "V(m³)": round(l1["volume"], 2),
                "Tris": l1["num_triangles"],
                "CW(mm)": round(l3.get("crown_width_mm", 0), 0) if l3.get("crown_width_mm") else "-",
                "Slope(%)": round(l3.get("crown_slope_percent", 0), 1) if l3.get("crown_slope_percent") is not None else "-",
                "Th(mm)": round(l3.get("min_wall_thickness_mm", 0), 0) if l3.get("min_wall_thickness_mm") else "-",
                "H(m)": round(l3.get("wall_height_m", 0), 2) if l3.get("wall_height_m") else "-",
                "Curved": "Ja" if l3.get("is_curved") else "",
                "R(m)": round(r_min, 1) if not math.isinf(r_min) else "-",
                "Taper": f"{taper['taper_ratio']:.0f}:1" if taper.get("is_tapered") else "-",
                "Cross(%)": round(sl["area_weighted_cross_pct"], 1) if sl else "-",
                "Unc(mm)": round(l3.get("measurement_uncertainty_mm", 0), 1),
                "Score": f"{s['passed']}/{s['total']}",
            }
            rows.append(row)

    # Write CSV
    csv_path = "viz_output/thesis_measurement_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"   → {csv_path} ({len(rows)} elements)")

    # Write Markdown
    md_path = "viz_output/thesis_measurement_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        cols = list(rows[0].keys())
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"   → {md_path}")

    # 2. Summary statistics
    print("\n2. Summary statistics:")
    n_elem = len(rows)
    n_curved = sum(1 for r in rows if r["Curved"] == "Ja")
    n_tapered = sum(1 for r in rows if r["Taper"] != "-")
    scores = [r["Score"] for r in rows]
    passed_total = sum(int(s.split("/")[0]) for s in scores)
    rules_total = sum(int(s.split("/")[1]) for s in scores)
    print(f"   Elements: {n_elem}")
    print(f"   Curved: {n_curved} ({n_curved/n_elem*100:.0f}%)")
    print(f"   Tapered: {n_tapered}")
    print(f"   Rules evaluated: {rules_total}")
    print(f"   Rules passed: {passed_total} ({passed_total/rules_total*100:.0f}%)")

    print("\n" + "=" * 60)
    print("Done. Figures saved to viz_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
