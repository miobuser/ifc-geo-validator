"""Generate all thesis figures from test models.

Usage:
    py -3 generate_thesis_figures.py

Output:
    viz_output/T*_classified.png — Face classification plots
    viz_output/validation_matrix.md — Summary table (Markdown)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import math
from pathlib import Path

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.viz.face_plot import plot_model_from_ifc

MODELS_DIR = Path(__file__).parent / "tests" / "test_models"
RULESET = Path(__file__).parent / "src" / "ifc_geo_validator" / "rules" / "rulesets" / "astra_fhb_stuetzmauer.yaml"
OUTPUT_DIR = Path(__file__).parent / "viz_output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    ruleset = load_ruleset(str(RULESET))

    models = sorted(MODELS_DIR.glob("T*.ifc"))
    print(f"Generating figures for {len(models)} models...")
    print()

    # ── 1. Face classification screenshots ────────────────────────
    for model_path in models:
        name = model_path.name
        out_png = OUTPUT_DIR / name.replace(".ifc", "_classified.png")
        print(f"  Rendering {name}...")
        try:
            plot_model_from_ifc(str(model_path), show=False, screenshot=str(out_png))
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── 2. Validation matrix ─────────────────────────────────────
    print("\n  Generating validation matrix...")
    rows = []
    for model_path in models:
        name = model_path.name
        model = load_model(str(model_path))
        walls = get_elements(model, "IfcWall")
        if not walls:
            continue

        mesh = extract_mesh(walls[0])
        l1 = validate_level1(mesh)
        l2 = validate_level2(mesh)
        l3 = validate_level3(mesh, l2)
        l4 = validate_level4(l1, l3, ruleset)

        checks = {c["rule_id"]: c["status"] for c in l4["checks"]}
        inc_r = l3.get("front_inclination_ratio")
        inc_s = "vert." if inc_r and math.isinf(inc_r) else f"{inc_r:.0f}:1" if inc_r else "—"

        is_c = l3.get("is_curved", False)

        rows.append({
            "model": name.replace(".ifc", ""),
            "vol": f"{l1['volume']:.3f}",
            "wt": "Ja" if l1["is_watertight"] else "Nein",
            "groups": l2["num_groups"],
            "curved": "Ja" if is_c else "Nein",
            "crown_mm": f"{l3.get('crown_width_mm', 0):.0f}",
            "slope_pct": f"{l3.get('crown_slope_percent', 0):.2f}",
            "thick_mm": f"{l3.get('min_wall_thickness_mm', 0):.0f}",
            "incl": inc_s,
            "L3-001": checks.get("ASTRA-SM-L3-001", "—"),
            "L3-002": checks.get("ASTRA-SM-L3-002", "—"),
            "L3-003": checks.get("ASTRA-SM-L3-003", "—"),
            "L3-004": checks.get("ASTRA-SM-L3-004", "—"),
            "score": f"{l4['summary']['passed']}/{l4['summary']['total']}",
        })

    # Write Markdown table
    md_path = OUTPUT_DIR / "validation_matrix.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Validierungsmatrix — Testmodelle T1–T25\n\n")
        f.write("| Modell | V (m³) | WT | Gr. | Kurve | Krone (mm) | Neig. (%) | Dicke (mm) | Anzug | L3-001 | L3-002 | L3-003 | L3-004 | Score |\n")
        f.write("|--------|--------|----|-----|-------|------------|-----------|------------|-------|--------|--------|--------|--------|-------|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['vol']} | {r['wt']} | {r['groups']} | {r['curved']} | {r['crown_mm']} | {r['slope_pct']} | {r['thick_mm']} | {r['incl']} | {r['L3-001']} | {r['L3-002']} | {r['L3-003']} | {r['L3-004']} | {r['score']} |\n")

        f.write("\n**Regeln:**\n")
        f.write("- L3-001: Kronenbreite ≥ 300 mm\n")
        f.write("- L3-002: Kronenneigung 2.5–3.5 %\n")
        f.write("- L3-003: Mindestbauteilstärke ≥ 300 mm\n")
        f.write("- L3-004: Empfohlene Neigung 10:1\n")

    print(f"  Saved: {md_path}")

    # Also print to console
    print("\n" + "=" * 100)
    print(f"{'Modell':<25} {'V (m3)':>8} {'WT':>4} {'Gr':>3} {'Kurv':>5} {'Krone':>7} {'Neig':>7} {'Dicke':>7} {'Anzug':>7} {'L3-1':>5} {'L3-2':>5} {'L3-3':>5} {'L3-4':>5} {'Score':>6}")
    print("-" * 110)
    for r in rows:
        print(f"{r['model']:<25} {r['vol']:>8} {r['wt']:>4} {r['groups']:>3} {r['curved']:>5} {r['crown_mm']:>6}mm {r['slope_pct']:>6}% {r['thick_mm']:>6}mm {r['incl']:>7} {r['L3-001']:>5} {r['L3-002']:>5} {r['L3-003']:>5} {r['L3-004']:>5} {r['score']:>6}")

    print(f"\n{len(models)} models processed, {len(rows)} validated.")


if __name__ == "__main__":
    main()
