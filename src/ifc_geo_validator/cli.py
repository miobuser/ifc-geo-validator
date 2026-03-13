"""Command-line interface for IFC Geometry Validator."""

import argparse
import json
import math
import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="ifc-geo-validator",
        description="Geometric validation of IFC infrastructure models",
    )
    parser.add_argument("ifc_file", help="Path to IFC file")
    parser.add_argument(
        "-r", "--ruleset",
        help="Path to YAML ruleset file",
        default=None,
    )
    parser.add_argument(
        "-o", "--output",
        help="Output report file (JSON)",
        default=None,
    )
    parser.add_argument(
        "--levels",
        help="Validation levels to run (e.g., 1,2,3,4)",
        default="1,2,3,4",
    )
    parser.add_argument(
        "--filter-type",
        help="IFC entity type to filter (default: IfcWall)",
        default="IfcWall",
    )
    parser.add_argument(
        "--filter-predefined",
        help="Predefined type filter (e.g., RETAININGWALL)",
        default=None,
    )
    parser.add_argument(
        "--enrich",
        help="Write enriched IFC with validation properties to this path",
        default=None,
    )
    parser.add_argument(
        "--bcf",
        help="Export failed checks as BCF issues to this path",
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    from ifc_geo_validator.core.ifc_parser import load_model, get_elements
    from ifc_geo_validator.core.mesh_converter import extract_mesh
    from ifc_geo_validator.validation.level1 import validate_level1
    from ifc_geo_validator.validation.level2 import validate_level2
    from ifc_geo_validator.validation.level3 import validate_level3
    from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset

    print(f"IFC Geometry Validator v0.1.0")
    print(f"File: {args.ifc_file}")
    print(f"Filter: {args.filter_type}"
          + (f" ({args.filter_predefined})" if args.filter_predefined else ""))
    print(f"Levels: {args.levels}")
    print()

    # Load ruleset if provided or if Level 4 requested
    ruleset = None
    if args.ruleset:
        ruleset = load_ruleset(args.ruleset)
    elif 4 in levels:
        # Auto-detect bundled ASTRA ruleset
        default_rs = Path(__file__).parent / "rules" / "rulesets" / "astra_fhb_stuetzmauer.yaml"
        if default_rs.exists():
            ruleset = load_ruleset(str(default_rs))
            print(f"Ruleset: {ruleset['metadata']['name']} v{ruleset['metadata']['version']}")

    # Load model
    model = load_model(args.ifc_file)
    elements = get_elements(model, args.filter_type, args.filter_predefined)
    print(f"Found {len(elements)} {args.filter_type} elements"
          + (f" (PredefinedType={args.filter_predefined})" if args.filter_predefined else ""))

    if not elements:
        print("No elements to validate.")
        return

    all_results = []

    for elem in elements:
        name = getattr(elem, "Name", None) or "Unnamed"
        print(f"\n{'='*60}")
        print(f"Element: {name} (#{elem.id()})")
        print(f"{'='*60}")

        try:
            mesh_data = extract_mesh(elem)
            elem_result = {
                "element_id": elem.id(),
                "element_name": name,
            }

            l1 = l2 = l3 = l4 = None

            # ── Level 1 ──
            if 1 in levels:
                l1 = validate_level1(mesh_data)
                elem_result["level1"] = l1

                print(f"  Volume:      {l1['volume']:.3f} m3")
                print(f"  Total area:  {l1['total_area']:.3f} m2")
                bbox = l1["bbox"]
                print(f"  Bounding box: {bbox['size'][0]:.3f} x {bbox['size'][1]:.3f} x {bbox['size'][2]:.3f} m")
                c = l1["centroid"]
                print(f"  Centroid:    ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
                print(f"  Watertight:  {l1['is_watertight']}")
                print(f"  Triangles:   {l1['num_triangles']}")

            # ── Level 2 ──
            if 2 in levels:
                l2 = validate_level2(mesh_data)
                elem_result["level2"] = l2

                print(f"\n  Face Classification ({l2['num_groups']} groups):")
                axis = l2["wall_axis"]
                print(f"  Wall axis:   ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")

                for g in l2["face_groups"]:
                    n = g["normal"]
                    print(f"    {g['category']:12s}  area={g['area']:.3f} m2  "
                          f"normal=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})  "
                          f"tris={g['num_triangles']}")

            # ── Level 3 ──
            if 3 in levels:
                if l2 is None:
                    l2 = validate_level2(mesh_data)
                l3 = validate_level3(mesh_data, l2)
                elem_result["level3"] = l3

                print(f"\n  Face Measurements:")
                if "crown_width_mm" in l3:
                    print(f"    Crown width:      {l3['crown_width_mm']:.1f} mm")
                if "crown_slope_percent" in l3:
                    print(f"    Crown slope:      {l3['crown_slope_percent']:.2f} %")
                if "min_wall_thickness_mm" in l3:
                    print(f"    Wall thickness:   {l3['min_wall_thickness_mm']:.1f} mm (min)")
                if "front_inclination_deg" in l3:
                    ratio = l3.get("front_inclination_ratio", 0)
                    if math.isinf(ratio):
                        print(f"    Front inclination: {l3['front_inclination_deg']:.2f} deg (vertical)")
                    else:
                        print(f"    Front inclination: {l3['front_inclination_deg']:.2f} deg ({ratio:.1f}:1)")

            # ── Level 4 ──
            if 4 in levels and ruleset:
                if l1 is None:
                    l1 = validate_level1(mesh_data)
                if l3 is None:
                    if l2 is None:
                        l2 = validate_level2(mesh_data)
                    l3 = validate_level3(mesh_data, l2)
                l4 = validate_level4(l1, l3, ruleset)
                elem_result["level4"] = l4

                print(f"\n  Rule Checks ({l4['summary']['total']} rules):")
                for chk in l4["checks"]:
                    icon = "PASS" if chk["status"] == "PASS" else (
                        "FAIL" if chk["status"] == "FAIL" else "SKIP")
                    print(f"    [{icon}] {chk['rule_id']}  {chk['name']}  "
                          f"({chk['severity']})")
                    if chk["status"] == "FAIL" and args.verbose:
                        print(f"           {chk['message']}")

                s = l4["summary"]
                print(f"\n  Summary: {s['passed']} passed, {s['failed']} failed, "
                      f"{s['skipped']} skipped "
                      f"({s['errors']} errors, {s['warnings']} warnings)")

            all_results.append(elem_result)

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "element_id": elem.id(),
                "element_name": name,
                "error": str(e),
            })

    # Export BCF issues for failed checks
    if args.bcf:
        from ifc_geo_validator.report.bcf_export import export_bcf
        ifc_name = Path(args.ifc_file).name
        export_bcf(all_results, args.bcf, ifc_name=ifc_name)
        print(f"\nBCF issues written to: {args.bcf}")

    # Write enriched IFC with validation properties
    if args.enrich:
        from ifc_geo_validator.report.ifc_property_writer import inject_all
        inject_all(model, elements, all_results, args.enrich)
        print(f"\nEnriched IFC written to: {args.enrich}")

    # Write JSON report if requested
    if args.output:
        # Handle non-serialisable types (numpy, inf)
        def _default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, float) and math.isinf(obj):
                return "Infinity"
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=_default)
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
