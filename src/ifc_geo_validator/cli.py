"""Command-line interface for IFC Geometry Validator."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _get_version() -> str:
    """Read version from pyproject.toml (avoids hardcoding)."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                # version = "0.1.0"  →  0.1.0
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "unknown"


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
        help="Validation levels to run (e.g., 1,2,3,4,5,6)",
        default="1,2,3,4,5,6",
    )
    parser.add_argument(
        "--filter-type",
        help="IFC entity types, comma-separated (default: IfcWall). "
             "Example: IfcWall,IfcSlab,IfcFooting",
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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print one-line summary per element (for CI/CD)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan model for all entity types with geometry and exit",
    )
    parser.add_argument(
        "--heatmap",
        choices=["cross", "long", "total"],
        default=None,
        help="Show slope heatmap: cross (Quergefälle), long (Längsgefälle), total",
    )
    parser.add_argument(
        "--heatmap-categories",
        default="crown",
        help="Face categories for heatmap, comma-separated (default: crown). "
             "Use 'all' for entire mesh.",
    )

    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    entity_types = [t.strip() for t in args.filter_type.split(",")]

    from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
    from ifc_geo_validator.core.mesh_converter import extract_mesh
    from ifc_geo_validator.validation.level5 import validate_level5
    from ifc_geo_validator.validation.level6 import validate_level6
    from ifc_geo_validator.validation.level1 import validate_level1
    from ifc_geo_validator.validation.level2 import validate_level2
    from ifc_geo_validator.validation.level3 import validate_level3
    from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset

    version = _get_version()
    print(f"IFC Geometry Validator v{version}")
    print(f"File: {args.ifc_file}")
    print(f"Filter: {', '.join(entity_types)}"
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

    # --scan: discover all entity types with geometry, then exit
    if args.scan:
        _scan_model(model)
        return

    # Collect elements from all specified entity types
    elements = []
    for etype in entity_types:
        found = get_elements(model, etype, args.filter_predefined)
        if found:
            print(f"Found {len(found)} {etype} elements"
                  + (f" (PredefinedType={args.filter_predefined})" if args.filter_predefined else ""))
            elements.extend(found)
    if len(entity_types) > 1:
        print(f"Total: {len(elements)} elements")

    # Terrain detection (print early so user knows)
    terrain = None
    if 6 in levels:
        terrain = get_terrain_mesh(model)
        print(f"Terrain: {'Detected (IfcSite geometry)' if terrain else 'Not found'}")

    if not elements:
        print("\nNo elements to validate.")
        print(f"  Hint: Try a different --filter-type (current: {args.filter_type}).")
        print(f"  Hint: Use comma-separated types: --filter-type IfcWall,IfcSlab,IfcFooting")
        if args.filter_predefined:
            print(f"  Hint: Remove --filter-predefined {args.filter_predefined} to widen the search.")
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

                # Mesh quality warnings
                n_degen = l1.get("n_degenerate_filtered", 0)
                if n_degen > 0:
                    print(f"  WARNING: {n_degen} degenerate triangles filtered")
                q = l1.get("mesh_quality", {})
                if q.get("non_manifold_edges", 0) > 0:
                    print(f"  WARNING: {q['non_manifold_edges']} non-manifold edges")

            # ── Level 2 ──
            if 2 in levels:
                # Use classification thresholds from ruleset if available
                cl_thresh = ruleset.get("classification_thresholds") if ruleset else None
                l2 = validate_level2(mesh_data, thresholds=cl_thresh)
                elem_result["level2"] = l2

                n_bodies = l2.get("n_bodies", 1)
                confidence = l2.get("confidence", 0)
                print(f"\n  Face Classification ({l2['num_groups']} groups, confidence={confidence:.0%}):")
                if n_bodies > 1:
                    print(f"  WARNING: {n_bodies} disconnected bodies — using largest")
                geo = l2.get("geometry_check", {})
                if not geo.get("is_wall_like", True):
                    print(f"  WARNING: {geo.get('reason', 'not wall-like')}")
                for diag in l2.get("diagnostics", []):
                    print(f"  NOTE: {diag}")
                axis = l2["wall_axis"]
                print(f"  Wall axis:   ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
                cinfo = l2.get("centerline_info")
                if cinfo:
                    curved = cinfo.get("is_curved", False)
                    if curved:
                        print(f"  Curved:      Yes (length {cinfo.get('length_m', 0):.2f} m, {cinfo.get('n_slices', 0)} slices)")
                    elif args.verbose:
                        print(f"  Curved:      No")

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
                if "foundation_width_mm" in l3:
                    print(f"    Foundation width:  {l3['foundation_width_mm']:.1f} mm")
                if "crown_width_cv" in l3:
                    print(f"    Profile CV:       {l3['crown_width_cv']:.4f}")

            # ── Slope heatmap ──────────────────────────────────────
            if args.heatmap and l2 is not None:
                from ifc_geo_validator.viz.slope_heatmap import (
                    compute_surface_slopes,
                    plot_slope_heatmap,
                    plot_slope_profile,
                )

                hm_cats = (
                    None if args.heatmap_categories == "all"
                    else [c.strip() for c in args.heatmap_categories.split(",")]
                )
                slopes = compute_surface_slopes(
                    mesh_data, l2["face_groups"],
                    categories=hm_cats,
                    axis=np.array(l2["wall_axis"]) if l2 else None,
                    centerline=l2.get("centerline"),
                )
                if slopes is not None:
                    n_sel = int(slopes["face_mask"].sum())
                    print(f"\n  Slope Heatmap ({args.heatmap}, {n_sel} faces):")
                    print(f"    Cross-slope:  avg={slopes['area_weighted_cross_pct']:.2f}%  "
                          f"min={slopes['min_cross_pct']:.2f}%  max={slopes['max_cross_pct']:.2f}%")
                    print(f"    Long. slope:  avg={slopes['area_weighted_long_pct']:.2f}%  "
                          f"min={slopes['min_long_pct']:.2f}%  max={slopes['max_long_pct']:.2f}%")

                    hm_title = f"{name} — {args.heatmap} slope"
                    try:
                        plot_slope_heatmap(
                            mesh_data, slopes, mode=args.heatmap,
                            title=hm_title, show=True,
                        )
                    except ImportError:
                        print("    (PyVista not installed — skipping 3D view)")
                else:
                    print(f"\n  Slope Heatmap: no faces in category '{args.heatmap_categories}'")

            # Store mesh_data for L5/L6 (needed after the per-element loop)
            elem_result["mesh_data"] = mesh_data

            all_results.append(elem_result)

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "element_id": elem.id(),
                "element_name": name,
                "error": str(e),
            })

    # ── Level 5: Inter-element context ─────────────────────────
    if 5 in levels and len(all_results) > 1:
        l5 = validate_level5(all_results)
        if l5["pairs"]:
            print(f"\n{'='*60}")
            print(f"Level 5: Inter-Element Analysis ({l5['summary']['num_pairs']} pairs)")
            print(f"{'='*60}")

            for p in l5["pairs"]:
                print(f"\n  {p['element_a_name']} <-> {p['element_b_name']} ({p['pair_type']})")
                if p["pair_type"] == "stacked":
                    print(f"    Upper: {p['upper_name']}, Lower: {p['lower_name']}")
                    print(f"    Vertical gap:   {p['vertical_gap_mm']:.1f} mm")
                    print(f"    Foundation overhang: {'Yes' if p['foundation_extends_beyond_wall'] else 'No'}"
                          f" ({p['overhang_mm']:.0f} mm)")
                    print(f"    Center offset:  {p['center_offset_mm']:.1f} mm")
                elif p["pair_type"] == "side_by_side":
                    print(f"    Horizontal gap: {p['horizontal_gap_mm']:.1f} mm")

    # ── Level 6: Distance checks & terrain context ────────────
    if 6 in levels:
        if terrain is None:
            terrain = get_terrain_mesh(model)
        l6 = validate_level6(all_results, terrain_mesh=terrain)

        print(f"\n{'='*60}")
        print(f"Level 6: Distance Checks (terrain={'Yes' if terrain else 'No'})")
        print(f"{'='*60}")

        if l6["terrain_side"]:
            print(f"\n  Terrain-based Front/Back:")
            for eid, info in l6["terrain_side"].items():
                print(f"    {info['element_name']}: {info['assignments']}")

        for cl in l6["clearances"]:
            if cl["min_m"] is not None:
                print(f"\n  Crown-Terrain Clearance ({cl['element_name']}):")
                print(f"    Min: {cl['min_m']:.2f} m, Max: {cl['max_m']:.2f} m, Avg: {cl['avg_m']:.2f} m")

        for emb in l6.get("embedments", []):
            print(f"\n  Foundation Embedment ({emb['element_name']}):")
            print(f"    Embedment: {emb['foundation_embedment_m']:.2f} m "
                  f"(terrain z={emb['terrain_z']:.2f}, foundation min z={emb['foundation_min_z']:.2f})")

        for d in l6["distances"]:
            print(f"\n  {d['element_a_name']} <-> {d['element_b_name']}:")
            print(f"    Min distance:  {d['min_distance_mm']:.1f} mm")
            print(f"    XY distance:   {d['horizontal_distance_mm']:.1f} mm")

    # ── Level 4: Rule evaluation (after L5/L6 for full context) ──
    # Pre-compute L5 once (not per element)
    l5_global = None
    if 5 in levels and len(all_results) > 1:
        l5_global = validate_level5(all_results)

    # Pre-compute L6 once (not per element)
    l6_global = None
    if 6 in levels:
        if terrain is None:
            terrain = get_terrain_mesh(model)
        if terrain:
            l6_global = validate_level6(all_results, terrain_mesh=terrain)

    if 4 in levels and ruleset:
        for elem_result in all_results:
            if "error" in elem_result:
                continue
            l1 = elem_result.get("level1")
            l3 = elem_result.get("level3")
            if l1 is None or l3 is None:
                continue

            eid = elem_result.get("element_id")

            # Build L5 context for this element from pre-computed global result
            l5_ctx = {}
            if l5_global:
                for p in l5_global.get("pairs", []):
                    if p.get("element_a_id") == eid or p.get("element_b_id") == eid:
                        if p["pair_type"] == "stacked":
                            l5_ctx["foundation_extends_beyond_wall"] = p.get("foundation_extends_beyond_wall", False)
                            l5_ctx["wall_foundation_gap_mm"] = p.get("vertical_gap_mm", 0)

            # Build L6 context for this element from pre-computed global result
            l6_ctx = {}
            if l6_global:
                l6_ctx["earth_side_determined"] = eid in l6_global.get("terrain_side", {})
                # Compute crown_slope_towards_earth_side
                if l6_ctx["earth_side_determined"]:
                    l6_ctx["crown_slope_towards_earth_side"] = _check_crown_slope_direction(
                        elem_result, l6_global["terrain_side"].get(eid, {})
                    )
                # Foundation embedment from L6 embedments
                for emb in l6_global.get("embedments", []):
                    if emb.get("element_id") == eid:
                        l6_ctx["foundation_embedment_m"] = emb["foundation_embedment_m"]
                        break

            # Store context for property writer / report
            if l5_ctx:
                elem_result["level5_context"] = l5_ctx
            if l6_ctx:
                elem_result["level6_context"] = l6_ctx

            l4 = validate_level4(l1, l3, ruleset, level5_context=l5_ctx, level6_context=l6_ctx)
            elem_result["level4"] = l4

            name = elem_result.get("element_name", "?")
            print(f"\n  Rule Checks for {name} ({l4['summary']['total']} rules):")
            for chk in l4["checks"]:
                icon = "PASS" if chk["status"] == "PASS" else (
                    "FAIL" if chk["status"] == "FAIL" else "SKIP")
                print(f"    [{icon}] {chk['rule_id']}  {chk['name']}  ({chk['severity']})")

            s = l4["summary"]
            print(f"\n  Summary: {s['passed']} passed, {s['failed']} failed, "
                  f"{s['skipped']} skipped")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} elements)")
    print(f"{'='*60}")
    # Header
    print(f"  {'Name':<25s} {'ID':>6s}  {'Vol(m3)':>8s}  {'Area(m2)':>8s}  {'WT':>3s}  {'L4':>12s}")
    print(f"  {'-'*25} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*3}  {'-'*12}")
    for r in all_results:
        name = r.get("element_name", "?")[:25]
        eid = r.get("element_id", "?")
        if "error" in r:
            print(f"  {name:<25s} {eid:>6}  {'ERROR':>8s}  {'':>8s}  {'':>3s}  {'':>12s}")
            continue
        l1 = r.get("level1", {})
        vol = f"{l1.get('volume', 0):.3f}" if l1 else ""
        area = f"{l1.get('total_area', 0):.3f}" if l1 else ""
        wt = "Yes" if l1.get("is_watertight") else "No"
        l4 = r.get("level4")
        if l4:
            s = l4["summary"]
            l4_str = f"{s['passed']}P/{s['failed']}F/{s['skipped']}S"
        else:
            l4_str = "-"
        print(f"  {name:<25s} {eid:>6}  {vol:>8s}  {area:>8s}  {wt:>3s}  {l4_str:>12s}")

    # Overall pass/fail
    has_l4 = any("level4" in r for r in all_results if "error" not in r)
    overall = "PASS"
    if has_l4:
        total_f = sum(r["level4"]["summary"]["failed"]
                      for r in all_results if "level4" in r)
        total_p = sum(r["level4"]["summary"]["passed"]
                      for r in all_results if "level4" in r)
        total_s = sum(r["level4"]["summary"]["skipped"]
                      for r in all_results if "level4" in r)
        total_err = sum(r["level4"]["summary"].get("errors", 0)
                        for r in all_results if "level4" in r)
        overall = "PASS" if total_err == 0 else "FAIL"
        print(f"\n  Overall: {overall}  ({total_p} passed, {total_f} failed, {total_s} skipped)")
    print()

    # Machine-readable summary (for CI/CD integration)
    if args.summary:
        print("# SUMMARY (machine-readable)")
        for r in all_results:
            name = r.get("element_name", "?")
            eid = r.get("element_id", "?")
            if "error" in r:
                print(f"ERROR\t#{eid}\t{name}\t{r['error']}")
                continue
            l4 = r.get("level4")
            if l4:
                s = l4["summary"]
                status = "FAIL" if s.get("errors", 0) > 0 else "PASS"
                print(f"{status}\t#{eid}\t{name}\tL4({s['passed']}P/{s['failed']}F/{s['skipped']}S)")
            else:
                print(f"SKIP\t#{eid}\t{name}\tno L4")
        # Exit with non-zero code if any mandatory rule (ERROR severity) failed
        if overall == "FAIL":
            sys.exit(1)

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
        # Handle non-serialisable types (numpy, inf, WallCenterline)
        def _default(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()  # WallCenterline → dict
            if hasattr(obj, "item"):
                return obj.item()  # numpy scalar
            if isinstance(obj, float) and math.isinf(obj):
                return "Infinity"
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Remove non-serializable objects (numpy arrays, WallCenterline)
        for r in all_results:
            r.pop("mesh_data", None)  # numpy arrays
            if "level2" in r:
                r["level2"].pop("centerline", None)  # WallCenterline object

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=_default)
        print(f"\nReport written to: {args.output}")


def _scan_model(model):
    """Scan an IFC model for all entity types with geometry.

    Groups elements by entity type and PredefinedType, reports counts
    and a sample element for each group. Helps users discover what
    --filter-type to use for unknown models.
    """
    from collections import Counter

    # Common structural entity types to check
    check_types = [
        "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcMember",
        "IfcFooting", "IfcPile", "IfcPlate", "IfcRailing",
        "IfcBuildingElementProxy", "IfcCivilElement",
        "IfcBearing", "IfcDeepFoundation", "IfcCourse",
    ]

    print("Entity Type Scan")
    print("=" * 60)

    found_any = False
    for etype in check_types:
        try:
            elems = model.by_type(etype)
        except Exception:
            continue
        if not elems:
            continue

        # Count by PredefinedType
        ptype_counts = Counter()
        has_repr = 0
        sample_name = None
        for e in elems:
            pt = getattr(e, "PredefinedType", None) or "—"
            ptype_counts[pt] += 1
            if e.Representation is not None:
                has_repr += 1
                if sample_name is None:
                    sample_name = getattr(e, "Name", None) or f"#{e.id()}"

        if has_repr == 0:
            continue

        found_any = True
        print(f"\n  {etype}: {len(elems)} elements ({has_repr} with geometry)")
        for pt, count in ptype_counts.most_common():
            print(f"    PredefinedType={pt}: {count}")
        if sample_name:
            print(f"    Sample: \"{sample_name}\"")

    if not found_any:
        print("\n  No elements with geometry found.")
        print("  This model may use non-standard entity types or lack geometry.")

    print(f"\n{'=' * 60}")
    print("Usage example:")
    if found_any:
        found_types = []
        for t in check_types:
            try:
                elems = model.by_type(t)
                if elems and any(e.Representation is not None for e in elems):
                    found_types.append(t)
            except Exception:
                pass
        if found_types:
            print(f"  ifc-geo-validator model.ifc --filter-type {','.join(found_types)}")
    print()


def _check_crown_slope_direction(elem_result, terrain_side_info):
    """Check if crown slope direction points towards the earth side.

    Compares the horizontal component of the crown slope direction
    (from L3) against the earth-side face normal (from L6 terrain
    classification). Returns True if the crown is tilted towards
    the earth (back) side, as required by ASTRA FHB T/G.

    The terrain_side_info contains 'assignments': a dict mapping
    face_group index → "front" or "back" (terrain-relative).
    Face groups with assignment "back" have normals pointing toward
    the terrain (earth side).
    """
    import numpy as np

    l2 = elem_result.get("level2", {})
    l3 = elem_result.get("level3", {})
    slope_dir = l3.get("crown_slope_direction")
    if not slope_dir:
        return False

    slope_dir = np.array(slope_dir[:2])  # XY only
    if np.linalg.norm(slope_dir) < 1e-6:
        return False  # No measurable slope direction

    # Find the earth-side face normal from terrain assignments
    # assignments: {face_group_index: "front" or "back"}
    assignments = terrain_side_info.get("assignments", {})
    face_groups = l2.get("face_groups", [])

    earth_normal = None
    for idx_str, side in assignments.items():
        idx = int(idx_str) if isinstance(idx_str, str) else idx_str
        if side == "back" and idx < len(face_groups):
            # "back" in terrain context = earth side (normal points toward terrain)
            earth_normal = np.array(face_groups[idx].get("normal", [0, 0, 0])[:2])
            break

    if earth_normal is None or np.linalg.norm(earth_normal) < 1e-6:
        return False

    # Crown slope should point towards the earth side
    # The earth-side face normal points toward terrain, so the slope
    # direction should align with it (positive dot product)
    cos_angle = float(np.dot(slope_dir, earth_normal) /
                       (np.linalg.norm(slope_dir) * np.linalg.norm(earth_normal)))
    return cos_angle > 0


if __name__ == "__main__":
    main()
