"""Run full validation pipeline on all test models and generate report.

Usage:
    py -3 validate_all_models.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import math
from pathlib import Path

from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.validation.level5 import validate_level5
from ifc_geo_validator.validation.level6 import validate_level6
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
    "T8_curved_wall.ifc": {"volume": 19.27, "crown_width_mm": 400.0},
    "T9_stepped_wall.ifc": {"volume": 3.6},
    "T10_complex_curved.ifc": {"volume": 15.53, "crown_width_mm": 300.0},
    "T11_s_curved.ifc": {"volume": 21.83},
    "T12_semicircle.ifc": {"volume": 17.37},
    "T13_polygonal.ifc": {"volume": 12.99},
    "T14_curved_l_profile.ifc": {"volume": 7.15},
    "T15_variable_height.ifc": {"volume": 14.0},
    "T16_height_step.ifc": {"volume": 13.0},
    "T17_curved_variable.ifc": {"volume": 13.64},
    "T18_buttressed.ifc": {"volume": 10.8},
    "T20_triangulated.ifc": {"volume": 9.6, "crown_width_mm": 400.0},
    "T21_extruded_trapezoid.ifc": {"volume": 10.8},
    "T22_with_terrain.ifc": {"volume": 9.6, "crown_width_mm": 400.0},
    "T23_astra_compliant_curved.ifc": {"volume": 24.0},
    "T24_highway_with_terrain.ifc": {"volume": 22.1},
    "T25_multi_failure.ifc": {"volume": 2.4},
    "T26_extruded_curved.ifc": {"volume": 7.3},
    "T27_long_curved_slope.ifc": {"volume": 57.0},
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

        terrain = get_terrain_mesh(model)
        print(f"  Elements: {len(walls)}" + (f", Terrain: Yes" if terrain else ""))

        # Phase 1: Per-element L1-L3
        model_results = []
        for wall in walls:
            wall_name = getattr(wall, "Name", None) or "Unnamed"

            try:
                mesh = extract_mesh(wall)
                l1 = validate_level1(mesh)
                l2 = validate_level2(mesh)
                l3 = validate_level3(mesh, l2)

                elem_result = {
                    "element_id": wall.id(),
                    "element_name": wall_name,
                    "source_file": name,
                    "level1": l1,
                    "level2": l2,
                    "level3": l3,
                    "mesh_data": mesh,
                }
                model_results.append(elem_result)

            except Exception as e:
                print(f"  ERROR ({wall_name}): {e}")
                model_results.append({
                    "element_id": wall.id(),
                    "element_name": wall_name,
                    "source_file": name,
                    "error": str(e),
                })

        # Phase 2: L5 inter-element (per model)
        valid_results = [r for r in model_results if "error" not in r]
        l5 = None
        if len(valid_results) > 1:
            l5 = validate_level5(valid_results)

        # Phase 3: L6 terrain context (per model)
        l6 = None
        if terrain:
            l6 = validate_level6(valid_results, terrain_mesh=terrain)

        # Phase 4: L4 rule evaluation with full context
        for elem_result in valid_results:
            eid = elem_result.get("element_id")
            l1 = elem_result["level1"]
            l3 = elem_result["level3"]

            # Build L5 context
            l5_ctx = {}
            if l5:
                for p in l5.get("pairs", []):
                    if p.get("element_a_id") == eid or p.get("element_b_id") == eid:
                        if p["pair_type"] == "stacked":
                            l5_ctx["foundation_extends_beyond_wall"] = p.get("foundation_extends_beyond_wall", False)
                            l5_ctx["wall_foundation_gap_mm"] = p.get("vertical_gap_mm", 0)

            # Build L6 context
            l6_ctx = {}
            if l6:
                l6_ctx["earth_side_determined"] = eid in l6.get("terrain_side", {})
                for emb in l6.get("embedments", []):
                    if emb.get("element_id") == eid:
                        l6_ctx["foundation_embedment_m"] = emb["foundation_embedment_m"]
                        break

            if l5_ctx:
                elem_result["level5_context"] = l5_ctx
            if l6_ctx:
                elem_result["level6_context"] = l6_ctx
            l4 = validate_level4(l1, l3, ruleset, level5_context=l5_ctx, level6_context=l6_ctx)
            elem_result["level4"] = l4

        # Print results
        for elem_result in model_results:
            if "error" in elem_result:
                continue
            l1 = elem_result["level1"]
            l3 = elem_result["level3"]
            l4 = elem_result.get("level4", {})

            vol = l1["volume"]
            bbox = l1["bbox"]["size"]
            wt = l1["is_watertight"]
            cw = l3.get("crown_width_mm", "N/A")
            cs = l3.get("crown_slope_percent", "N/A")
            wth = l3.get("min_wall_thickness_mm", "N/A")
            inc_r = l3.get("front_inclination_ratio")
            inc_str = "vertical" if inc_r and math.isinf(inc_r) else f"{inc_r:.1f}:1" if inc_r else "N/A"

            l4s = l4.get("summary", {})

            if len(walls) > 1:
                print(f"  [{elem_result['element_name']}]")
            print(f"  Volume:      {vol:.3f} m³")
            print(f"  BBox:        {bbox[0]:.1f} × {bbox[1]:.1f} × {bbox[2]:.1f} m")
            print(f"  Watertight:  {wt}")
            is_c = l3.get("is_curved", False)
            wlen = l3.get("wall_length_m")
            print(f"  Groups:      {elem_result['level2']['num_groups']}")
            if is_c:
                print(f"  Curved:      Yes (length {wlen:.1f} m)" if wlen else f"  Curved:      Yes")
            print(f"  Crown width: {cw:.0f} mm" if isinstance(cw, float) else f"  Crown width: {cw}")
            print(f"  Crown slope: {cs:.2f} %" if isinstance(cs, float) else f"  Crown slope: {cs}")
            print(f"  Thickness:   {wth:.0f} mm" if isinstance(wth, float) else f"  Thickness: {wth}")
            print(f"  Inclination: {inc_str}")
            if l4s:
                print(f"  Rules:       {l4s['passed']}/{l4s['total']} passed, "
                      f"{l4s.get('errors', 0)} errors, {l4s.get('warnings', 0)} warnings")

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

        # Print L5 summary
        if l5 and l5["pairs"]:
            print(f"  L5 Pairs:    {l5['summary']['num_pairs']} "
                  f"({l5['summary']['num_stacked']} stacked, "
                  f"{l5['summary']['num_side_by_side']} side-by-side)")

        # Print L6 summary
        if l6:
            if l6.get("terrain_side"):
                print(f"  L6 Terrain:  {len(l6['terrain_side'])} elements classified")
            for cl in l6.get("clearances", []):
                if cl.get("min_m") is not None:
                    print(f"  Clearance:   {cl['element_name']}: {cl['min_m']:.2f}–{cl['max_m']:.2f} m")
            for emb in l6.get("embedments", []):
                if emb.get("foundation_embedment_m") is not None:
                    print(f"  Embedment:   {emb['element_name']}: {emb['foundation_embedment_m']:.2f} m")

        # Clean up mesh_data before storing
        for r in model_results:
            r.pop("mesh_data", None)

        all_results.extend(model_results)

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
