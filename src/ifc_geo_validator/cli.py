"""Command-line interface for IFC Geometry Validator."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


# ── ANSI color helpers ────────────────────────────────────────────

def _supports_color():
    """Check if terminal supports ANSI colors."""
    if sys.platform == "win32":
        return "ANSICON" in __import__("os").environ or "WT_SESSION" in __import__("os").environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()

def _green(s): return f"\033[32m{s}\033[0m" if _COLOR else s
def _red(s): return f"\033[31m{s}\033[0m" if _COLOR else s
def _yellow(s): return f"\033[33m{s}\033[0m" if _COLOR else s
def _dim(s): return f"\033[2m{s}\033[0m" if _COLOR else s
def _bold(s): return f"\033[1m{s}\033[0m" if _COLOR else s


from ifc_geo_validator import get_version as _get_version


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="ifc-geo-validator",
        description="Geometric validation of IFC infrastructure models",
    )
    parser.add_argument("ifc_file", nargs="+", help="Path to IFC file(s) or directory")
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
        "--html",
        help="Generate HTML validation report to this path",
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
        "--centerline",
        choices=["auto", "geometry", "alignment"],
        default="auto",
        help="Centerline source: auto (alignment if available, else geometry), "
             "geometry (from crown faces), alignment (from IfcAlignment)",
    )
    parser.add_argument(
        "--heatmap",
        choices=["cross", "long", "total"],
        default=None,
        help="Show slope heatmap: cross (Quergefälle), long (Längsgefälle), total",
    )
    parser.add_argument(
        "--cross-section",
        type=float, nargs="?", const=0.5, default=None,
        metavar="FRACTION",
        help="Show cross-section at position (0.0=start, 0.5=middle, 1.0=end)",
    )
    parser.add_argument(
        "--clearance",
        type=float, nargs=2, default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Check clearance profile (width height in meters, e.g. --clearance 8.0 4.5)",
    )
    parser.add_argument(
        "--distances",
        action="store_true",
        help="Compute pairwise distances between ALL elements (min vertex, horizontal, vertical)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Export all measurements as CSV spreadsheet (for Excel/Power BI)",
    )
    parser.add_argument(
        "--filter-name",
        default=None,
        help="Filter elements by name pattern (e.g. '*Stütz*' or 'Mauer')",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="REFERENCE_IFC",
        help="Compare against reference IFC (as-designed vs as-built)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-configure: detect entity types and best ruleset automatically",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name/number for report header (e.g. 'A1 Bern-Zürich, Los 3')",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Author/validator name for report signature",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create a default .igv.yaml config file in the current directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick summary: one line per element with key metrics and PASS/FAIL",
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

    # --init: create default config and exit
    if args.init:
        from ifc_geo_validator.core.project_config import create_default_config
        path = create_default_config()
        print(f"Created {path}")
        print("Edit this file to configure your project settings.")
        return

    # Load project config if available
    from ifc_geo_validator.core.project_config import find_config, load_config
    config_path = find_config()
    if config_path:
        config = load_config(config_path)
        print(f"Config: {config_path}")
        # Apply config as defaults (CLI flags override)
        if not args.project and config.get("project"):
            args.project = config["project"]
        if not args.author and config.get("author"):
            args.author = config["author"]
        if args.filter_type == "IfcWall" and config.get("filter_type"):
            ft = config["filter_type"]
            if isinstance(ft, list):
                args.filter_type = ",".join(ft)
            entity_types = [t.strip() for t in args.filter_type.split(",")]
        if config.get("auto") and not args.auto:
            args.auto = True
        if config.get("distances") and not args.distances:
            args.distances = True

    # Resolve input files (support directories and globs)
    import glob as globmod
    ifc_files = []
    for path_arg in args.ifc_file:
        p = Path(path_arg)
        if p.is_dir():
            ifc_files.extend(sorted(p.glob("*.ifc")))
        elif "*" in path_arg or "?" in path_arg:
            ifc_files.extend(sorted(Path(f) for f in globmod.glob(path_arg)))
        else:
            ifc_files.append(p)

    if len(ifc_files) > 1:
        # Batch mode: process each file separately
        print(f"IFC Geometry Validator v{_get_version()} — Batch Mode")
        print(f"Processing {len(ifc_files)} files...")
        batch_summary = []
        for i, fpath in enumerate(ifc_files, 1):
            print(f"\n{'#'*60}")
            print(f"# [{i}/{len(ifc_files)}] {fpath.name}")
            print(f"{'#'*60}")
            # Run single file (recursive call via subprocess to isolate state)
            batch_args = [str(fpath)]
            for flag in ["--filter-type", "--filter-predefined", "--levels",
                         "--ruleset", "--enrich", "--bcf", "--html"]:
                val = getattr(args, flag.lstrip("-").replace("-", "_"), None)
                if val:
                    batch_args.extend([flag, str(val)])
            if args.verbose:
                batch_args.append("-v")
            if args.summary:
                batch_args.append("--summary")
            # Save original argv and re-run
            orig_argv = sys.argv
            sys.argv = ["ifc-geo-validator"] + batch_args
            try:
                main()
            except SystemExit:
                pass
            finally:
                sys.argv = orig_argv
        return

    # Single file mode
    ifc_file = str(ifc_files[0])

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
    print(f"File: {ifc_file}")
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
    model = load_model(ifc_file)

    # --auto: auto-configure entity types and ruleset
    if args.auto:
        from ifc_geo_validator.core.auto_config import auto_configure
        config = auto_configure(model)
        print(f"Auto-Config: {config['description']}")
        print(f"  Schema:     {config['schema']}")
        print(f"  Elements:   {config['element_count']} ({', '.join(f'{t}:{n}' for t,n in config['found_types'].items())})")
        print(f"  Types:      {', '.join(config['entity_types'])}")
        print(f"  Ruleset:    {config['ruleset']}")
        print(f"  Terrain:    {'Ja' if config['has_terrain'] else 'Nein'}")
        print(f"  Alignment:  {'Ja' if config['has_alignment'] else 'Nein'}")
        print()

        # Apply auto-config
        entity_types = config["entity_types"]
        if not args.ruleset:
            rs_path = Path(__file__).parent / "rules" / "rulesets" / config["ruleset"]
            if rs_path.exists():
                ruleset = load_ruleset(str(rs_path))
                print(f"Ruleset: {ruleset['metadata']['name']} v{ruleset['metadata'].get('version', '?')}")

    # --compare: compare two models and exit
    if args.compare:
        from ifc_geo_validator.core.ifc_compare import compare_models
        print(f"Comparing: {ifc_file}")
        print(f"Reference: {args.compare}")
        print()
        result = compare_models(args.compare, ifc_file,
                                entity_type=entity_types[0])
        s = result["summary"]
        print(f"Matched: {s['total_matched']} elements")
        print(f"Deviations: {s['with_deviations']} elements exceed tolerance ({s['tolerance_mm']}mm)")
        if s["unmatched_a"]:
            print(f"Only in reference: {s['unmatched_a']}")
        if s["unmatched_b"]:
            print(f"Only in comparison: {s['unmatched_b']}")
        print()
        for m in result["matched"]:
            if "error" in m:
                print(f"  {m['name']}: ERROR {m['error']}")
                continue
            status = _red("DEVIATION") if m["has_deviation"] else _green("OK")
            print(f"  {m['name']}: {status}")
            for d in m.get("deviations", []):
                if d["exceeds_tolerance"]:
                    print(f"    {d['property']:>25s}: {d['value_a']} → {d['value_b']} "
                          f"(Δ{d['difference']:.2f} {d['unit']}, tol={d['tolerance']})")
        return

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

    # Name-based filtering (glob pattern on element name)
    if args.filter_name:
        import fnmatch
        pattern = args.filter_name
        before = len(elements)
        elements = [e for e in elements
                    if fnmatch.fnmatch(getattr(e, "Name", "") or "", pattern)]
        print(f"Name filter '{pattern}': {len(elements)}/{before} elements match")

    # Alignment detection
    alignment_centerline = None
    if args.centerline in ("auto", "alignment"):
        from ifc_geo_validator.core.ifc_parser import get_alignments
        from ifc_geo_validator.core.face_classifier import WallCenterline
        aligns = get_alignments(model)
        if aligns:
            al = aligns[0]  # Use first alignment
            alignment_centerline = WallCenterline.from_polyline(
                al["points_xy"], source="alignment"
            )
            print(f"Alignment: {al['name']} ({len(al['points_xy'])} points, "
                  f"{alignment_centerline.length:.1f}m)")
        elif args.centerline == "alignment":
            print("WARNING: No IfcAlignment found — falling back to geometric centerline")

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
    t_start = time.time()

    for elem in elements:
        name = getattr(elem, "Name", None) or "Unnamed"
        print(f"\n{'='*60}")
        print(f"Element: {_bold(name)} (#{elem.id()})")
        print(f"{'='*60}")

        t_elem = time.time()
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
                role = l2.get("element_role", "unknown")
                role_labels = {"wall_stem": "Mauerstiel", "foundation": "Fundament",
                               "parapet": "Brüstung", "column": "Stütze",
                               "slab": "Platte", "unknown": "unbekannt"}
                print(f"\n  Face Classification ({l2['num_groups']} groups, confidence={confidence:.0%}, role={role_labels.get(role, role)}):")
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
                        min_r = cinfo.get("min_radius_m", "?")
                        r_str = f", R_min={min_r}m" if min_r != "?" and min_r != float("inf") else ""
                        print(f"  Curved:      Yes (length {cinfo.get('length_m', 0):.2f} m, "
                              f"{cinfo.get('n_slices', 0)} slices{r_str})")
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

            # ── Anomaly detection ─────────────────────────────────
            if l2 is not None and l3 is not None:
                from ifc_geo_validator.core.anomaly_detection import detect_anomalies
                anomalies = detect_anomalies(mesh_data, l2, l3)
                if anomalies:
                    print(f"\n  Anomalies ({len(anomalies)}):")
                    for a in anomalies:
                        sev = _yellow("WARN") if a["severity"] == "warning" else _dim("INFO")
                        print(f"    [{sev}] {a['message'][:80]}")
                    elem_result["anomalies"] = anomalies

            # ── Advanced geometry analysis ────────────────────────
            if l2 is not None and l3 is not None:
                from ifc_geo_validator.core.advanced_geometry import (
                    compute_taper_profile, compute_planarity, check_plumbness,
                )
                taper = compute_taper_profile(
                    mesh_data, l2["face_groups"], np.array(l2["wall_axis"])
                )
                if taper.get("is_tapered"):
                    l3["taper_ratio"] = taper["taper_ratio"]
                    l3["thickness_min_mm"] = taper["min_thickness_mm"]
                    l3["thickness_max_mm"] = taper["max_thickness_mm"]
                    l3["thickness_at_crown_mm"] = taper["min_thickness_mm"]
                    l3["thickness_at_base_mm"] = taper["max_thickness_mm"]

                plumb = check_plumbness(l2["face_groups"])
                l3["front_plumbness_deg"] = plumb.get("front_plumbness_deg")
                l3["is_plumb"] = plumb.get("is_plumb", True)

                if args.verbose:
                    if taper.get("is_tapered"):
                        print(f"    Taper ratio:      {taper['taper_ratio']:.1f}:1 "
                              f"({taper['min_thickness_mm']:.0f}–{taper['max_thickness_mm']:.0f}mm)")
                    front_p = plumb.get("front_plumbness_deg")
                    if front_p is not None and front_p > 0.1:
                        print(f"    Plumbness:        {front_p:.1f}° from vertical")

            # ── Curvature data (inject into L3 for rule context) ──
            if l2 is not None:
                cl_obj = l2.get("centerline")
                if cl_obj and hasattr(cl_obj, "curvature_profile"):
                    curv = cl_obj.curvature_profile()
                    l3["min_radius_m"] = curv["min_radius_m"]
                    l3["max_curvature"] = curv["max_kappa"]

            # ── Slope analysis (always computed for crown) ───────
            if l2 is not None:
                try:
                    from ifc_geo_validator.viz.slope_heatmap import compute_surface_slopes as _css
                    _sl = _css(
                        mesh_data, l2["face_groups"], categories=["crown"],
                        axis=np.array(l2["wall_axis"]),
                        centerline=l2.get("centerline"),
                    )
                    if _sl is not None:
                        slope_data = {
                            "area_weighted_cross_pct": _sl["area_weighted_cross_pct"],
                            "max_cross_pct": _sl["max_cross_pct"],
                            "min_cross_pct": _sl["min_cross_pct"],
                            "area_weighted_long_pct": _sl["area_weighted_long_pct"],
                            "max_long_pct": _sl["max_long_pct"],
                            "min_long_pct": _sl["min_long_pct"],
                            "uses_local_frame": _sl.get("uses_local_frame", False),
                        }
                        elem_result["slope_analysis"] = slope_data
                        # Inject into L3 so L4 rule context can see them
                        if l3 is not None:
                            l3["cross_slope_avg_pct"] = slope_data["area_weighted_cross_pct"]
                            l3["cross_slope_max_pct"] = slope_data["max_cross_pct"]
                            l3["long_slope_avg_pct"] = slope_data["area_weighted_long_pct"]
                            l3["long_slope_max_pct"] = slope_data["max_long_pct"]
                except Exception:
                    pass

            # ── Slope heatmap (interactive, only if requested) ─────
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
                # Use alignment centerline if available, else geometric
                hm_centerline = alignment_centerline or l2.get("centerline")
                slopes = compute_surface_slopes(
                    mesh_data, l2["face_groups"],
                    categories=hm_cats,
                    axis=np.array(l2["wall_axis"]) if l2 else None,
                    centerline=hm_centerline,
                )
                if slopes is not None:
                    n_sel = int(slopes["face_mask"].sum())
                    cl_src = "alignment" if alignment_centerline else ("local" if slopes.get("uses_local_frame") else "global")
                    print(f"\n  Slope Heatmap ({args.heatmap}, {n_sel} faces, centerline={cl_src}):")
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

            # ── Cross-section (if requested) ──────────────────────
            if args.cross_section is not None and l2 is not None:
                from ifc_geo_validator.viz.cross_section import (
                    extract_cross_section, plot_cross_section,
                )
                cs_cl = alignment_centerline or l2.get("centerline")
                section = extract_cross_section(
                    mesh_data, cs_cl, position_fraction=args.cross_section,
                )
                if section is not None:
                    print(f"\n  Cross-Section at {section['position_m']:.1f}m "
                          f"({args.cross_section:.0%}):")
                    print(f"    Width:  {section['width_mm']:.0f} mm")
                    print(f"    Height: {section['height_m']:.2f} m")
                    print(f"    Vertices in slice: {section['n_vertices']}")
                    try:
                        plot_cross_section(section, title=f"{name} — Querschnitt")
                    except ImportError:
                        print("    (Matplotlib not installed — skipping plot)")
                else:
                    print(f"\n  Cross-Section: not enough vertices at position {args.cross_section:.0%}")

            # ── Clearance check (if requested) ──────────────────────
            if args.clearance and l2 is not None:
                from ifc_geo_validator.validation.clearance import (
                    check_clearance, astra_road_clearance,
                )
                cl_obj = alignment_centerline or l2.get("centerline")
                cl_width, cl_height = args.clearance
                profile = astra_road_clearance(cl_width, cl_height)
                cl_result = check_clearance(
                    mesh_data, cl_obj, profile, n_slices=20,
                )
                elem_result["clearance"] = cl_result
                if cl_result["clear"]:
                    print(f"\n  Clearance ({cl_width}×{cl_height}m): {_green('CLEAR')} "
                          f"({cl_result['n_slices_checked']} slices checked)")
                else:
                    print(f"\n  Clearance ({cl_width}×{cl_height}m): {_red('VIOLATION')}")
                    print(f"    {cl_result['n_violations']} vertices inside envelope")
                    print(f"    Max penetration: {cl_result['max_penetration_mm']:.0f} mm")
                    for v in cl_result["violations"][:5]:
                        print(f"    At {v['position_m']:.1f}m: "
                              f"{v['n_vertices']} vertices, {v['max_penetration_mm']:.0f}mm")

            # Store mesh_data for L5/L6 (needed after the per-element loop)
            elem_result["mesh_data"] = mesh_data

            dt = time.time() - t_elem
            if args.verbose:
                print(f"\n  {_dim(f'Processed in {dt:.2f}s')}")

            all_results.append(elem_result)

        except Exception as e:
            print(f"  {_red('ERROR')}: {e}")
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

    # ── Pairwise element distances (if requested) ──────────
    if args.distances and len(all_results) >= 2:
        from ifc_geo_validator.core.advanced_geometry import (
            compute_element_distances, find_nearby_pairs,
        )

        valid = [r for r in all_results if "error" not in r and "mesh_data" in r]
        if len(valid) >= 2:
            # Use spatial grid to find nearby pairs (O(N) instead of O(N²))
            t_dist = time.time()
            meshes = [r["mesh_data"] for r in valid]
            nearby = find_nearby_pairs(meshes, max_gap_m=10.0)

            print(f"\n{'='*60}")
            print(f"Element Distances ({len(valid)} elements, "
                  f"{len(nearby)} nearby pairs of {len(valid)*(len(valid)-1)//2})")
            print(f"{'='*60}")

            for i, j in nearby:
                a, b = valid[i], valid[j]
                d = compute_element_distances(a["mesh_data"], b["mesh_data"])
                a_name = a.get("element_name", "?")[:25]
                b_name = b.get("element_name", "?")[:25]

                # Store min distance for L4 context
                for elem in [a, b]:
                    prev = elem.get("_min_distance_mm", float("inf"))
                    elem["_min_distance_mm"] = min(prev, d["min_vertex_mm"])

                print(f"\n  {a_name} <-> {b_name}:")
                print(f"    Min vertex:  {d['min_vertex_mm']:>8.0f} mm")
                print(f"    Horizontal:  {d['horizontal_mm']:>8.0f} mm")
                print(f"    Vertical:    {d['vertical_mm']:>8.0f} mm")

            # Inject min_distance into each element's L3 for L4 rules
            for r in valid:
                min_d = r.pop("_min_distance_mm", None)
                if min_d is not None and "level3" in r:
                    r["level3"]["min_distance_to_nearest_mm"] = min_d

            dt_dist = time.time() - t_dist
            print(f"\n  {_dim(f'Distance analysis: {dt_dist*1000:.0f}ms')}")

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
                        # Foundation depth below terrain
                        if l3:
                            l3["foundation_depth_below_terrain_m"] = emb["foundation_embedment_m"]
                        break

                # Crown height above terrain from L6 clearances
                for cl in l6_global.get("clearances", []):
                    if cl.get("element_id") == eid and cl.get("max_m") is not None:
                        if l3:
                            l3["crown_height_above_terrain_m"] = round(cl["max_m"], 3)
                            # Wall exposure = crown height above terrain
                            # (how much of the wall is visible above ground)
                            l3["wall_exposure_height_m"] = round(cl["max_m"], 3)
                        break

            # Store context for property writer / report
            if l5_ctx:
                elem_result["level5_context"] = l5_ctx
            if l6_ctx:
                elem_result["level6_context"] = l6_ctx

            l2_for_l4 = elem_result.get("level2")
            l4 = validate_level4(l1, l3, ruleset, level5_context=l5_ctx,
                                  level6_context=l6_ctx, level2_result=l2_for_l4)
            elem_result["level4"] = l4

            name = elem_result.get("element_name", "?")
            print(f"\n  Rule Checks for {name} ({l4['summary']['total']} rules):")
            for chk in l4["checks"]:
                status = chk["status"]
                if status == "PASS":
                    icon = _green("PASS")
                elif status == "FAIL":
                    icon = _red("FAIL")
                else:
                    icon = _dim("SKIP")
                sev = chk["severity"]
                sev_str = _red(sev) if sev == "ERROR" else (_yellow(sev) if sev == "WARNING" else _dim(sev))
                print(f"    [{icon}] {chk['rule_id']}  {chk['name']}  ({sev_str})")
                if status == "FAIL" and chk.get("message", "").startswith("FAIL:"):
                    # Show actionable message for failures
                    msg_parts = chk["message"].split(" | ")
                    for part in msg_parts[1:]:
                        print(f"           {_yellow(part)}")

            s = l4["summary"]
            passed_str = _green(f"{s['passed']} passed")
            failed_str = _red(f"{s['failed']} failed") if s["failed"] > 0 else f"{s['failed']} failed"
            print(f"\n  Summary: {passed_str}, {failed_str}, {s['skipped']} skipped")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY ({len(all_results)} elements)")
    print(f"{'='*80}")
    if args.quick:
        # Quick mode: compact one-liner with key metrics
        print(f"  {'Name':<25s} {'Role':>10s} {'CW mm':>6s} {'Th mm':>6s} {'S%':>5s} {'Conf':>5s} {'Result':>12s}")
        print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*12}")
    else:
        print(f"  {'Name':<25s} {'ID':>6s}  {'Vol(m3)':>8s}  {'Area(m2)':>8s}  {'WT':>3s}  {'L4':>12s}")
        print(f"  {'-'*25} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*3}  {'-'*12}")

    for r in all_results:
        name = r.get("element_name", "?")[:25]
        eid = r.get("element_id", "?")
        if "error" in r:
            print(f"  {name:<25s} {'ERROR':>12s}")
            continue

        l1 = r.get("level1", {})
        l2 = r.get("level2", {})
        l3 = r.get("level3", {})
        l4 = r.get("level4")

        if args.quick:
            role = l2.get("element_role", "?")[:10]
            cw = f"{l3.get('crown_width_mm', 0):.0f}" if l3.get("crown_width_mm") else "-"
            th = f"{l3.get('min_wall_thickness_mm', 0):.0f}" if l3.get("min_wall_thickness_mm") else "-"
            sl = f"{l3.get('crown_slope_percent', 0):.1f}" if l3.get("crown_slope_percent") is not None else "-"
            conf = f"{l2.get('confidence', 0):.0%}"
            if l4:
                s = l4["summary"]
                if s.get("failed", 0) == 0:
                    result = _green(f"{s['passed']}P/{s['skipped']}S")
                else:
                    result = _red(f"{s['passed']}P/{s['failed']}F")
            else:
                result = "-"
            print(f"  {name:<25s} {role:>10s} {cw:>6s} {th:>6s} {sl:>5s} {conf:>5s} {result:>12s}")
        else:
            vol = f"{l1.get('volume', 0):.3f}" if l1 else ""
            area = f"{l1.get('total_area', 0):.3f}" if l1 else ""
            wt = "Yes" if l1.get("is_watertight") else "No"
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
        overall_str = _green("PASS") if overall == "PASS" else _red("FAIL")
        print(f"\n  Overall: {overall_str}  ({total_p} passed, {total_f} failed, {total_s} skipped)")

    dt_total = time.time() - t_start
    print(f"\n  {_dim(f'Total time: {dt_total:.2f}s for {len(all_results)} elements')}")

    # Statistical summary (for multi-element models)
    if len(all_results) >= 3 and args.verbose:
        from ifc_geo_validator.report.summary_stats import compute_summary_stats, format_summary
        stats = compute_summary_stats(all_results)
        if stats.get("outliers"):
            print(f"\n  {_yellow('Outliers detected:')}")
            for o in stats["outliers"]:
                print(f"    {o['element']}: {o['property']}={o['value']} "
                      f"(z={o['z_score']:.1f}, group median={o['group_median']})")

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
        ifc_name = Path(ifc_file).name
        export_bcf(all_results, args.bcf, ifc_name=ifc_name)
        print(f"\nBCF issues written to: {args.bcf}")

    # Write enriched IFC with validation properties
    if args.enrich:
        from ifc_geo_validator.report.ifc_property_writer import inject_all
        inject_all(model, elements, all_results, args.enrich)
        print(f"\nEnriched IFC written to: {args.enrich}")

    # Write HTML report if requested
    if args.html:
        from ifc_geo_validator.report.html_report import generate_html_report
        rs_name = ruleset["metadata"]["name"] if ruleset else "—"
        html = generate_html_report(
            all_results, ifc_filename=Path(ifc_file).name,
            ruleset_name=rs_name,
            project_name=args.project or "",
            author=args.author or "",
        )
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nHTML report written to: {args.html}")

    # Export measurements as CSV (for Excel/Power BI)
    if args.csv:
        import csv as csv_mod
        csv_rows = []
        for r in all_results:
            if "error" in r:
                continue
            l1 = r.get("level1", {})
            l2 = r.get("level2", {})
            l3 = r.get("level3", {})
            l4 = r.get("level4", {})
            slope = r.get("slope_analysis", {})
            row = {
                "element_id": r.get("element_id"),
                "element_name": r.get("element_name"),
                "role": l2.get("element_role", ""),
                "confidence": l2.get("confidence", ""),
                "volume_m3": l1.get("volume"),
                "total_area_m2": l1.get("total_area"),
                "num_triangles": l1.get("num_triangles"),
                "watertight": l1.get("is_watertight"),
                "crown_width_mm": l3.get("crown_width_mm"),
                "crown_slope_pct": l3.get("crown_slope_percent"),
                "min_thickness_mm": l3.get("min_wall_thickness_mm"),
                "avg_thickness_mm": l3.get("avg_wall_thickness_mm"),
                "wall_height_m": l3.get("wall_height_m"),
                "inclination_ratio": l3.get("front_inclination_ratio"),
                "foundation_width_mm": l3.get("foundation_width_mm"),
                "is_curved": l3.get("is_curved"),
                "min_radius_m": l3.get("min_radius_m"),
                "cross_slope_avg_pct": slope.get("area_weighted_cross_pct"),
                "cross_slope_max_pct": slope.get("max_cross_pct"),
                "long_slope_max_pct": slope.get("max_long_pct"),
                "taper_ratio": l3.get("taper_ratio"),
                "plumbness_deg": l3.get("front_plumbness_deg"),
                "uncertainty_mm": l3.get("measurement_uncertainty_mm"),
                "min_distance_mm": l3.get("min_distance_to_nearest_mm"),
            }
            if l4:
                s = l4.get("summary", {})
                row["rules_passed"] = s.get("passed")
                row["rules_failed"] = s.get("failed")
                row["rules_skipped"] = s.get("skipped")
            csv_rows.append(row)

        if csv_rows:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv_mod.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nCSV export written to: {args.csv} ({len(csv_rows)} elements)")

    # Write JSON report if requested
    if args.output:
        from ifc_geo_validator.report.json_report import json_default

        def _default(obj):
            # CLI extends the shared default with WallCenterline support
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return json_default(obj)

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
