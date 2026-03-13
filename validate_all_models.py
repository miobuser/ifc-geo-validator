"""Run full validation pipeline on all test models and generate report.

Usage:
    py -3 validate_all_models.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import math
from pathlib import Path

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.report.json_report import generate_report, write_report

TEST_DIR = Path(__file__).parent / "tests" / "test_models"
RULESET = Path(__file__).parent / "src" / "ifc_geo_validator" / "rules" / "rulesets" / "astra_fhb_stuetzmauer.yaml"

# Expected reference values for verification
EXPECTED = {
    "T1_simple_box.ifc": {"volume": 9.6, "crown_width_mm": 400.0},
    "T2_inclined_wall.ifc": {"volume": 12.0, "crown_width_mm": 350.0, "inclination": 10.0},
    "T3_crown_slope.ifc": {"volume": 7.211, "crown_width_mm": 300.0},
    "T4_l_shaped.ifc": {"volume": 10.5},
    "T5_t_shaped.ifc": {"volume": 10.725},
    "T6_non_compliant.ifc": {"volume": 2.0, "crown_width_mm": 200.0},
    "T7_compliant.ifc": {"volume": 10.811, "crown_width_mm": 300.0, "inclination": 10.0},
}


def main():
    ruleset = load_ruleset(str(RULESET))
    models = sorted(TEST_DIR.glob("T*.ifc"))

    print(f"IFC Geometry Validator — Batch Validation")
    print(f"{'='*70}")
    print(f"Models: {len(models)}")
    print(f"Ruleset: {ruleset['metadata']['name']} v{ruleset['metadata']['version']}")
    print()

    all_results = []

    for model_path in models:
        name = model_path.name
        print(f"{'─'*70}")
        print(f"Model: {name}")

        model = load_model(str(model_path))
        walls = get_elements(model, "IfcWall")

        if not walls:
            print(f"  No IfcWall elements found")
            continue

        for wall in walls:
            wall_name = getattr(wall, "Name", None) or "Unnamed"

            try:
                mesh = extract_mesh(wall)
                l1 = validate_level1(mesh)
                l2 = validate_level2(mesh)
                l3 = validate_level3(mesh, l2)
                l4 = validate_level4(l1, l3, ruleset)

                elem_result = {
                    "element_id": wall.id(),
                    "element_name": wall_name,
                    "source_file": name,
                    "level1": l1,
                    "level2": l2,
                    "level3": l3,
                    "level4": l4,
                }
                all_results.append(elem_result)

                # Print summary
                vol = l1["volume"]
                bbox = l1["bbox"]["size"]
                wt = l1["is_watertight"]
                cw = l3.get("crown_width_mm", "N/A")
                cs = l3.get("crown_slope_percent", "N/A")
                wth = l3.get("min_wall_thickness_mm", "N/A")
                inc_r = l3.get("front_inclination_ratio")
                inc_str = "vertical" if inc_r and math.isinf(inc_r) else f"{inc_r:.1f}:1" if inc_r else "N/A"

                l4s = l4["summary"]

                print(f"  Volume:      {vol:.3f} m³")
                print(f"  BBox:        {bbox[0]:.1f} × {bbox[1]:.1f} × {bbox[2]:.1f} m")
                print(f"  Watertight:  {wt}")
                print(f"  Groups:      {l2['num_groups']}")
                print(f"  Crown width: {cw:.0f} mm" if isinstance(cw, float) else f"  Crown width: {cw}")
                print(f"  Crown slope: {cs:.2f} %" if isinstance(cs, float) else f"  Crown slope: {cs}")
                print(f"  Thickness:   {wth:.0f} mm" if isinstance(wth, float) else f"  Thickness: {wth}")
                print(f"  Inclination: {inc_str}")
                print(f"  Rules:       {l4s['passed']}/{l4s['total']} passed, "
                      f"{l4s['errors']} errors, {l4s['warnings']} warnings")

                # Verify against expected values
                if name in EXPECTED:
                    exp = EXPECTED[name]
                    checks = []
                    if "volume" in exp:
                        ok = abs(vol - exp["volume"]) < 0.1
                        checks.append(f"vol={'OK' if ok else 'MISMATCH'}")
                    if "crown_width_mm" in exp and isinstance(cw, float):
                        ok = abs(cw - exp["crown_width_mm"]) < 5
                        checks.append(f"cw={'OK' if ok else 'MISMATCH'}")
                    if "inclination" in exp and inc_r and not math.isinf(inc_r):
                        ok = abs(inc_r - exp["inclination"]) < 1
                        checks.append(f"inc={'OK' if ok else 'MISMATCH'}")
                    print(f"  Verify:      {', '.join(checks)}")

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "element_id": wall.id(),
                    "element_name": wall_name,
                    "source_file": name,
                    "error": str(e),
                })

    # Generate and write report
    print(f"\n{'='*70}")
    report = generate_report("batch_validation", all_results, ruleset)
    output = Path(__file__).parent / "validation_report.json"
    write_report(report, str(output))
    print(f"Report written: {output}")

    # Print summary table
    print(f"\n{'─'*70}")
    print(f"{'Model':<25} {'Vol (m³)':>10} {'Crown':>8} {'Thick':>8} {'Incl':>10} {'Rules':>10}")
    print(f"{'─'*70}")
    for r in all_results:
        if "error" in r:
            print(f"{r.get('source_file', '?'):<25} {'ERROR':>10}")
            continue
        l1 = r["level1"]
        l3 = r["level3"]
        l4 = r["level4"]
        inc_r = l3.get("front_inclination_ratio")
        inc = "vert" if inc_r and math.isinf(inc_r) else f"{inc_r:.1f}:1" if inc_r else "—"
        print(f"{r.get('source_file', '?'):<25} "
              f"{l1['volume']:>10.3f} "
              f"{l3.get('crown_width_mm', 0):>7.0f}mm "
              f"{l3.get('min_wall_thickness_mm', 0):>7.0f}mm "
              f"{inc:>10} "
              f"{l4['summary']['passed']}/{l4['summary']['total']:>3}")


if __name__ == "__main__":
    main()
