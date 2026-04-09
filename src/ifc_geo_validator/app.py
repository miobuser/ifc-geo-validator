"""Streamlit web interface for IFC Geometry Validator."""

import json
import math
import tempfile
from pathlib import Path

import streamlit as st

from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
from ifc_geo_validator.core.mesh_converter import extract_mesh, MeshExtractionError
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.validation.level5 import validate_level5
from ifc_geo_validator.validation.level6 import validate_level6
from ifc_geo_validator.report.json_report import generate_report


# ── Defaults ─────────────────────────────────────────────────────────

RULESETS_DIR = Path(__file__).parent / "rules" / "rulesets"
BUILTIN_RULESETS = {
    "ASTRA FHB T/G — Stützmauern": RULESETS_DIR / "astra_fhb_stuetzmauer.yaml",
    "SIA 262 — Stützmauern": RULESETS_DIR / "sia_262_stuetzmauer.yaml",
    "ASTRA FHB T/G — Tunnel": RULESETS_DIR / "astra_fhb_tunnel.yaml",
}
DEFAULT_RULESET = RULESETS_DIR / "astra_fhb_stuetzmauer.yaml"


# ── Variable catalog (shared between editor and reference) ──────────

VARIABLE_CATALOG = {
    "Geometrie (L1)": {
        "volume": ("float", "m³", "Volumen (Divergenztheorem)"),
        "total_area": ("float", "m²", "Gesamtoberfläche"),
        "mesh_is_watertight": ("bool", "—", "Mesh geschlossen"),
        "bbox_dim_min_m": ("float", "m", "Kürzeste BBox-Achse"),
        "bbox_height_m": ("float", "m", "Vertikale BBox-Ausdehnung"),
        "volume_fill_ratio": ("float", "—", "V/V_bbox (1.0 = massiv)"),
        "slenderness_ratio": ("float", "—", "Höhe/Dicke"),
    },
    "Krone": {
        "crown_width_mm": ("float", "mm", "Kronenbreite (Oberfläche)"),
        "crown_slope_percent": ("float", "%", "Kronenneigung"),
        "cross_slope_max_pct": ("float", "%", "Quergefälle (max)"),
        "long_slope_max_pct": ("float", "%", "Längsgefälle (max)"),
    },
    "Wand": {
        "min_wall_thickness_mm": ("float", "mm", "Wandstärke senkrecht (min)"),
        "wall_height_m": ("float", "m", "Wandhöhe (max)"),
        "front_inclination_ratio": ("float", "n:1", "Neigung Ansichtsfläche"),
        "min_radius_m": ("float", "m", "Min. Krümmungsradius"),
        "taper_ratio": ("float", "n:1", "Anzug-Verhältnis"),
        "front_plumbness_deg": ("float", "°", "Lotabweichung"),
    },
    "Fundament / Terrain": {
        "foundation_width_mm": ("float", "mm", "Fundamentbreite"),
        "foundation_embedment_m": ("float", "m", "Einbindetiefe (Oberfläche)"),
        "crown_height_above_terrain_m": ("float", "m", "Kronenhöhe über Terrain"),
        "min_distance_to_nearest_mm": ("float", "mm", "Min. Abstand Nachbar"),
    },
}


def _run_variable_reference():
    """Show all available YAML variables."""
    st.title("📖 Variablen-Referenz")
    st.caption("Alle verfügbaren Variablen für YAML-Regeln")
    for category, vars_dict in VARIABLE_CATALOG.items():
        st.subheader(category)
        rows = [{"Variable": f"`{v}`", "Typ": t, "Einheit": u, "Beschreibung": d}
                for v, (t, u, d) in vars_dict.items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _run_ruleset_editor():
    """Visual ruleset editor."""
    import yaml as yaml_mod
    st.title("📐 Ruleset Editor")
    st.caption("Validierungsregeln visuell erstellen")

    rs_name = st.text_input("Ruleset-Name", "Mein Ruleset")

    if "editor_rules" not in st.session_state:
        st.session_state.editor_rules = []

    # Variable reference
    with st.expander("📖 Verfügbare Variablen"):
        for cat, vars_dict in VARIABLE_CATALOG.items():
            st.markdown(f"**{cat}**")
            for v, (t, u, d) in vars_dict.items():
                st.text(f"  {v} ({t}, {u}) — {d}")

    # Add rule
    st.subheader("Neue Regel")
    all_vars = {}
    for cat, vd in VARIABLE_CATALOG.items():
        for v, info in vd.items():
            all_vars[f"{v} — {info[2]}"] = v

    c1, c2, c3 = st.columns(3)
    with c1:
        sel = st.selectbox("Variable", list(all_vars.keys()))
        var = all_vars[sel]
    with c2:
        op = st.selectbox("Operator", [">=", "<=", ">", "<", "=="])
        val = st.number_input("Schwellwert", value=300.0)
    with c3:
        sev = st.selectbox("Severity", ["ERROR", "WARNING", "INFO"])
        name = st.text_input("Name", f"{var} Check")

    check = f"{var} {op} {val}"
    st.code(f'check: "{check}"')

    if st.button("Regel hinzufügen", type="primary"):
        st.session_state.editor_rules.append({
            "id": f"CUSTOM-{len(st.session_state.editor_rules)+1:03d}",
            "name": name, "check": check, "severity": sev,
        })

    # Show rules
    if st.session_state.editor_rules:
        st.subheader(f"Regeln ({len(st.session_state.editor_rules)})")
        for i, r in enumerate(st.session_state.editor_rules):
            cols = st.columns([8, 1])
            cols[0].text(f"[{r['severity']}] {r['id']}: {r['check']}")
            if cols[1].button("🗑", key=f"d{i}"):
                st.session_state.editor_rules.pop(i)
                st.rerun()

    # YAML output
    rs = {
        "metadata": {"name": rs_name, "version": "1.0.0"},
        "classification_thresholds": {"horizontal_deg": 45.0, "coplanar_deg": 5.0, "lateral_deg": 45.0},
        "level_3": st.session_state.editor_rules,
    }
    yaml_out = yaml_mod.dump(rs, default_flow_style=False, allow_unicode=True, sort_keys=False)
    st.code(yaml_out, language="yaml")
    st.download_button("📥 YAML herunterladen", yaml_out,
                        f"{rs_name.replace(' ','_')}.yaml", "text/yaml",
                        use_container_width=True)


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="IFC Geometry Validator",
    page_icon="🏗️",
    layout="wide",
)


# ── Language selector ────────────────────────────────────────────────

from ifc_geo_validator.i18n import t, set_language, get_language

lang = st.sidebar.selectbox(
    "🌐 Sprache / Langue / Lingua",
    ["Deutsch", "Français", "Italiano"],
    index=0,
)
set_language({"Deutsch": "de", "Français": "fr", "Italiano": "it"}[lang])

# ── Navigation ──────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigation",
    [f"🏗️ {t('validation')}", f"📐 {t('ruleset_editor')}", f"📖 {t('variable_reference')}"],
    index=0,
)

if t("ruleset_editor") in page:
    _run_ruleset_editor()
    st.stop()
elif t("variable_reference") in page:
    _run_variable_reference()
    st.stop()

# ── Sidebar (Validierung) ────────────────────────────────────────────

st.sidebar.title("IFC Geometry Validator")
st.sidebar.caption("Dateien werden nicht gespeichert. Verarbeitung nur im Arbeitsspeicher.")
def _get_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "unknown"

st.sidebar.caption(f"v{_get_version()} — BSc Thesis BFH")

uploaded_file = st.sidebar.file_uploader(
    "Upload IFC file",
    type=["ifc"],
    help="IFC 4x3 model with geometric elements",
)

entity_types = st.sidebar.multiselect(
    "Entity types",
    ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcMember",
     "IfcFooting", "IfcBuildingElementProxy", "IfcPlate", "IfcRailing"],
    default=["IfcWall"],
    help="Select one or more IFC entity types to validate",
)

predefined_type = st.sidebar.text_input(
    "Predefined type filter (optional)",
    value="",
    help="e.g. RETAININGWALL",
)

# Ruleset selector: built-in or custom
ruleset_choice = st.sidebar.selectbox(
    "Ruleset",
    list(BUILTIN_RULESETS.keys()) + ["Custom (upload)"],
    index=0,
)
ruleset_file = None
if ruleset_choice == "Custom (upload)":
    ruleset_file = st.sidebar.file_uploader(
        "Custom ruleset (YAML)",
        type=["yaml", "yml"],
    )

run_button = st.sidebar.button("Validate", type="primary", use_container_width=True)


# ── Main area ────────────────────────────────────────────────────────

if not uploaded_file and not run_button:
    st.title("IFC Geometry Validator")
    st.markdown(
        """
        Geometric validation of IFC infrastructure models against configurable
        requirements (ASTRA FHB T/G — Stützmauern).

        **Validation pipeline:**

        | Level | Description | Output |
        |-------|-------------|--------|
        | L1 | Mesh metrics | Volume, area, bbox, watertight |
        | L2 | Face classification | Coplanar clustering, semantic groups |
        | L3 | Measurements | Crown width, slope, thickness, inclination |
        | L4 | Rule checks | PASS / FAIL against ruleset |
        | L5 | Inter-element | Stacking, gaps, offsets |
        | L6 | Terrain context | Terrain clearance, element distances |

        Upload an IFC file in the sidebar to begin.
        """
    )
    st.stop()

if not uploaded_file:
    st.warning("Please upload an IFC file first.")
    st.stop()


# ── Run validation ───────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading IFC model...")
def run_validation(_file_bytes, file_name, entity_types_str, predefined_type, _ruleset_bytes, ruleset_choice):
    """Run the full validation pipeline and return structured results."""
    # Write uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name

    model = load_model(tmp_path)
    pred_filter = predefined_type if predefined_type else None

    # Collect elements from all selected entity types
    e_types = entity_types_str.split(",") if entity_types_str else ["IfcWall"]
    elements = []
    for etype in e_types:
        elements.extend(get_elements(model, etype.strip(), pred_filter))

    # Load ruleset (built-in or custom upload)
    if _ruleset_bytes:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="wb") as rs_tmp:
            rs_tmp.write(_ruleset_bytes)
            rs_path = rs_tmp.name
        ruleset = load_ruleset(rs_path)
    elif ruleset_choice in BUILTIN_RULESETS and BUILTIN_RULESETS[ruleset_choice].exists():
        ruleset = load_ruleset(str(BUILTIN_RULESETS[ruleset_choice]))
    elif DEFAULT_RULESET.exists():
        ruleset = load_ruleset(str(DEFAULT_RULESET))
    else:
        ruleset = None

    all_results = []
    for elem in elements:
        name = getattr(elem, "Name", None) or "Unnamed"
        try:
            mesh = extract_mesh(elem)
            l1 = validate_level1(mesh)
            l2 = validate_level2(mesh)
            l3 = validate_level3(mesh, l2)
            # L4 evaluated after L5/L6 below; store None for now
            l4 = None

            result = {
                "element_id": elem.id(),
                "element_name": name,
                "level1": l1,
                "level2": l2,
                "level3": l3,
                "mesh_data": mesh,
            }
            if l4:
                result["level4"] = l4
            all_results.append(result)
        except Exception as e:
            all_results.append({
                "element_id": elem.id(),
                "element_name": name,
                "error": str(e),
            })

    # L5: Inter-element analysis
    l5_result = None
    if len(all_results) > 1:
        valid = [r for r in all_results if "error" not in r and "mesh_data" in r]
        if len(valid) > 1:
            l5_result = validate_level5(valid)

    # L6: Terrain context & distances
    terrain = get_terrain_mesh(model)
    l6_result = None
    valid_for_l6 = [r for r in all_results if "error" not in r and "mesh_data" in r]
    if valid_for_l6:
        l6_result = validate_level6(valid_for_l6, terrain_mesh=terrain)

    # L4: Rule evaluation WITH L5/L6 context
    if ruleset:
        for r in all_results:
            if "error" in r:
                continue
            l1 = r.get("level1")
            l3 = r.get("level3")
            if l1 and l3:
                l5_ctx = {}
                if l5_result:
                    for p in l5_result.get("pairs", []):
                        if p.get("element_a_id") == r.get("element_id") or \
                           p.get("element_b_id") == r.get("element_id"):
                            if p["pair_type"] == "stacked":
                                l5_ctx["foundation_extends_beyond_wall"] = p.get("foundation_extends_beyond_wall", False)
                                l5_ctx["wall_foundation_gap_mm"] = p.get("vertical_gap_mm", 0)
                l6_ctx = {}
                if terrain:
                    l6_ctx["earth_side_determined"] = bool(l6_result and l6_result.get("terrain_side"))
                    # Foundation embedment from L6 embedments
                    if l6_result:
                        for emb in l6_result.get("embedments", []):
                            if emb.get("element_id") == r.get("element_id"):
                                l6_ctx["foundation_embedment_m"] = emb["foundation_embedment_m"]
                                break
                # Store context for property writer / report
                if l5_ctx:
                    r["level5_context"] = l5_ctx
                if l6_ctx:
                    r["level6_context"] = l6_ctx
                r["level4"] = validate_level4(l1, l3, ruleset, level5_context=l5_ctx,
                                              level6_context=l6_ctx, level2_result=r.get("level2"))

    report = generate_report(file_name, all_results, ruleset)
    return all_results, report, ruleset, l5_result, l6_result, terrain is not None


if run_button or uploaded_file:
    file_bytes = uploaded_file.getvalue()
    rs_bytes = ruleset_file.getvalue() if ruleset_file else None
    pred = predefined_type.strip() or ""
    etypes_str = ",".join(entity_types) if entity_types else "IfcWall"

    results, report, ruleset, l5_result, l6_result, has_terrain = run_validation(
        file_bytes, uploaded_file.name, etypes_str, pred, rs_bytes, ruleset_choice
    )

    if not results:
        st.error(f"No elements found for types: {etypes_str}. "
                 f"Try different entity types in the sidebar.")
        st.stop()

    st.title(f"Validation: {uploaded_file.name}")
    if ruleset:
        st.caption(
            f"Ruleset: {ruleset['metadata']['name']} "
            f"v{ruleset['metadata'].get('version', '?')}"
        )

    # ── Summary metrics ──────────────────────────────────────────
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Elements", len(results))
    col2.metric("Validated", len(valid_results))
    col3.metric("Errors", len(error_results))
    col4.metric("Terrain", "Detected" if has_terrain else "None")

    if valid_results and "level4" in valid_results[0]:
        total_checks = sum(
            r["level4"]["summary"]["total"]
            for r in valid_results if "level4" in r
        )
        total_passed = sum(
            r["level4"]["summary"]["passed"]
            for r in valid_results if "level4" in r
        )
        col5.metric("Rules passed", f"{total_passed}/{total_checks}")

    # ── Summary table (all elements at a glance) ────────────────
    st.subheader("Element Overview")
    summary_rows = []
    for r in results:
        row = {
            "ID": r.get("element_id", "?"),
            "Name": r.get("element_name", "?"),
        }
        if "error" in r:
            row["Volume (m3)"] = None
            row["Area (m2)"] = None
            row["Watertight"] = None
            row["L4 Result"] = "ERROR"
        else:
            l1 = r.get("level1", {})
            row["Volume (m3)"] = round(l1.get("volume", 0), 3) if l1 else None
            row["Area (m2)"] = round(l1.get("total_area", 0), 3) if l1 else None
            row["Watertight"] = "Yes" if l1.get("is_watertight") else "No"
            l4 = r.get("level4")
            if l4:
                s = l4["summary"]
                row["L4 Result"] = f"{s['passed']}P / {s['failed']}F / {s['skipped']}S"
            else:
                row["L4 Result"] = "-"
        summary_rows.append(row)
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    # ── Element selector (when multiple elements) ───────────────
    if len(valid_results) > 1:
        elem_options = {
            f"#{r['element_id']} {r.get('element_name', '?')}": i
            for i, r in enumerate(results)
            if "error" not in r
        }
        selected_label = st.selectbox(
            "Select element for detail view",
            options=list(elem_options.keys()),
        )
        detail_results = [results[elem_options[selected_label]]]
    else:
        detail_results = results

    # ── Per-element results ──────────────────────────────────────
    for r in detail_results:
        name = r.get("element_name", "?")
        eid = r.get("element_id", "?")

        if "error" in r:
            with st.expander(f"#{eid} {name} — ERROR", expanded=False):
                st.error(r["error"])
            continue

        l1 = r["level1"]
        l2 = r["level2"]
        l3 = r["level3"]
        l4 = r.get("level4")

        # Status indicator
        if l4:
            s = l4["summary"]
            if s["failed"] == 0 and s["errors"] == 0:
                status = "PASS"
            elif s["errors"] > 0:
                status = "FAIL"
            else:
                status = "WARN"
        else:
            status = "—"

        with st.expander(
            f"#{eid} {name} — {status}",
            expanded=len(results) == 1,
        ):
            # ── L1: Geometry ─────────────────────────────────
            st.subheader("Geometry (L1)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Volume", f"{l1['volume']:.3f} m³")
            c2.metric("Surface area", f"{l1['total_area']:.3f} m²")
            bbox = l1["bbox"]["size"]
            c3.metric(
                "Bounding box",
                f"{bbox[0]:.2f} × {bbox[1]:.2f} × {bbox[2]:.2f} m",
            )

            c4, c5, c6 = st.columns(3)
            c4.metric("Triangles", l1["num_triangles"])
            c5.metric("Vertices", l1["num_vertices"])
            wt_label = "Yes" if l1["is_watertight"] else "No"
            c6.metric("Watertight", wt_label)

            # Mesh quality warnings
            n_degen = l1.get("n_degenerate_filtered", 0)
            n_bodies = l2.get("n_bodies", 1)
            q = l1.get("mesh_quality", {})
            nm_edges = q.get("non_manifold_edges", 0)
            warnings = []
            if n_degen > 0:
                warnings.append(f"{n_degen} degenerate triangles filtered")
            if n_bodies > 1:
                warnings.append(f"{n_bodies} disconnected bodies (largest used)")
            if nm_edges > 0:
                warnings.append(f"{nm_edges} non-manifold edges")
            if warnings:
                st.warning(" | ".join(warnings))

            # ── L2: Face classification ──────────────────────
            st.subheader("Face Classification (L2)")
            st.caption(f"{l2['num_groups']} groups detected")

            group_data = []
            for g in l2["face_groups"]:
                n = g["normal"]
                group_data.append({
                    "Category": g["category"],
                    "Area (m²)": round(g["area"], 3),
                    "Triangles": g["num_triangles"],
                    "Normal": f"({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})",
                })
            st.dataframe(group_data, use_container_width=True, hide_index=True)

            # ── 3D Viewer (web-ifc + Three.js) ─────────────
            try:
                from ifc_geo_validator.viz.webifc_viewer import render_ifc_viewer

                # Build classification data for element coloring
                class_data = {}
                for g in l2.get("face_groups", []):
                    cat = g.get("category", "unclassified")
                    # Map element expressID to category (for web-ifc overlay)
                    class_data[str(eid)] = {
                        "category": l2.get("element_role", "wall_stem"),
                        "has_errors": any(c["status"] == "FAIL" and c["severity"] == "ERROR"
                                         for c in l4.get("checks", [])) if l4 else False,
                    }

                render_ifc_viewer(
                    file_bytes, height=500,
                    classification_data=class_data,
                )
            except Exception:
                pass  # Viewer is optional

            # ── Curved wall info ────────────────────────────
            is_c = l3.get("is_curved")
            if is_c is not None:
                cinfo_cols = st.columns(3)
                cinfo_cols[0].metric("Curved", "Yes" if is_c else "No")
                wlen = l3.get("wall_length_m")
                if wlen:
                    cinfo_cols[1].metric("Wall length", f"{wlen:.2f} m")
                cmethod = l3.get("crown_width_method", "")
                if cmethod:
                    cinfo_cols[2].metric("Measurement", cmethod)

            # ── L3: Measurements ─────────────────────────────
            st.subheader("Measurements (L3)")
            mc1, mc2, mc3, mc4 = st.columns(4)

            cw = l3.get("crown_width_mm")
            if cw is not None:
                mc1.metric("Crown width", f"{cw:.0f} mm")
            cs = l3.get("crown_slope_percent")
            if cs is not None:
                mc2.metric("Crown slope", f"{cs:.2f} %")
            wth = l3.get("min_wall_thickness_mm")
            if wth is not None:
                mc3.metric("Wall thickness", f"{wth:.0f} mm")
            inc = l3.get("front_inclination_deg")
            ratio = l3.get("front_inclination_ratio")
            if inc is not None:
                if ratio and math.isinf(ratio):
                    mc4.metric("Inclination", "vertical")
                elif ratio:
                    mc4.metric("Inclination", f"{ratio:.1f}:1 ({inc:.1f}°)")

            # Second row: foundation + height
            fw = l3.get("foundation_width_mm")
            wh = l3.get("wall_height_m")
            if fw is not None or wh is not None:
                mc5, mc6, _, _ = st.columns(4)
                if fw is not None:
                    mc5.metric("Foundation width", f"{fw:.0f} mm")
                if wh is not None:
                    mc6.metric("Wall height", f"{wh:.2f} m")

            # ── Profile consistency (curved walls) ─────────
            cv = l3.get("crown_width_cv")
            if cv is not None:
                st.metric("Profile consistency (CV)", f"{cv:.4f}",
                          delta="uniform" if cv <= 0.1 else "variable",
                          delta_color="normal" if cv <= 0.1 else "inverse")

            # ── Slope Analysis ────────────────────────────────
            try:
                import numpy as np_app
                from ifc_geo_validator.viz.slope_heatmap import compute_surface_slopes

                centerline_obj = l2.get("centerline")
                wall_axis = l2.get("wall_axis")
                slope_cats = st.multiselect(
                    "Slope analysis categories",
                    ["crown", "foundation", "front", "back"],
                    default=["crown"],
                    key=f"slope_cats_{eid}",
                )
                if slope_cats:
                    slopes = compute_surface_slopes(
                        r.get("mesh_data") or {},
                        l2["face_groups"],
                        categories=slope_cats,
                        axis=np_app.array(wall_axis) if wall_axis else None,
                        centerline=centerline_obj,
                    )
                    if slopes is not None:
                        st.subheader("Slope Analysis")
                        local_label = "local" if slopes.get("uses_local_frame") else "global"
                        st.caption(f"Centerline: {local_label} frame, {int(slopes['face_mask'].sum())} faces")

                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Cross-slope (avg)", f"{slopes['area_weighted_cross_pct']:.2f} %")
                        sc2.metric("Cross-slope (max)", f"{slopes['max_cross_pct']:.2f} %")
                        sc3.metric("Long. slope (avg)", f"{slopes['area_weighted_long_pct']:.2f} %")
                        sc4.metric("Long. slope (max)", f"{slopes['max_long_pct']:.2f} %")

                        # ASTRA threshold warning
                        if slopes["max_cross_pct"] > 5.0:
                            st.warning(
                                f"Cross-slope {slopes['max_cross_pct']:.1f}% exceeds "
                                f"ASTRA maximum of 5%"
                            )

                        # Per-triangle slope table (downloadable)
                        sel_cross = slopes["selected_cross_pct"]
                        sel_long = slopes["selected_long_pct"]
                        if len(sel_cross) > 0:
                            slope_df = {
                                "Cross-slope (%)": [round(v, 2) for v in sel_cross],
                                "Long. slope (%)": [round(v, 2) for v in sel_long],
                            }
                            with st.expander("Per-triangle slope data"):
                                st.dataframe(slope_df, use_container_width=True, hide_index=True)
            except Exception:
                pass  # Slope analysis is optional

            # ── L4: Rule checks ──────────────────────────────
            if l4:
                st.subheader("Rule Checks (L4)")
                s = l4["summary"]
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Passed", s["passed"])
                rc2.metric("Failed", s["failed"])
                rc3.metric("Skipped", s["skipped"])
                rc4.metric(
                    "Result",
                    "PASS" if s["failed"] == 0 and s["errors"] == 0 else "FAIL",
                )

                checks_data = []
                for chk in l4["checks"]:
                    checks_data.append({
                        "Status": chk["status"],
                        "Rule": chk["rule_id"],
                        "Name": chk["name"],
                        "Severity": chk["severity"],
                        "Message": chk.get("message", ""),
                    })
                st.dataframe(checks_data, use_container_width=True, hide_index=True)

    # ── L5: Inter-element analysis ─────────────────────────────
    if l5_result and l5_result.get("pairs"):
        st.subheader("Inter-Element Analysis (L5)")
        st.caption(f"{l5_result['summary']['num_pairs']} pair(s) analysed")
        l5_rows = []
        for p in l5_result["pairs"]:
            row = {
                "Element A": p.get("element_a_name", "?"),
                "Element B": p.get("element_b_name", "?"),
                "Type": p.get("pair_type", "?"),
            }
            if p.get("pair_type") == "stacked":
                row["Vertical gap (mm)"] = round(p.get("vertical_gap_mm", 0), 1)
                row["Overhang (mm)"] = round(p.get("overhang_mm", 0), 0)
                row["Center offset (mm)"] = round(p.get("center_offset_mm", 0), 1)
            elif p.get("pair_type") == "side_by_side":
                row["Horizontal gap (mm)"] = round(p.get("horizontal_gap_mm", 0), 1)
            l5_rows.append(row)
        st.dataframe(l5_rows, use_container_width=True, hide_index=True)

    # ── L6: Terrain context & distances ─────────────────────────
    if l6_result:
        st.subheader("Terrain & Distance Checks (L6)")
        st.caption(f"Terrain: {'Detected' if has_terrain else 'Not found'}")

        # Terrain side assignments
        if l6_result.get("terrain_side"):
            st.markdown("**Terrain-based front/back assignment:**")
            ts_rows = []
            for eid, info in l6_result["terrain_side"].items():
                ts_rows.append({
                    "Element": info.get("element_name", eid),
                    "Assignments": str(info.get("assignments", "")),
                })
            st.dataframe(ts_rows, use_container_width=True, hide_index=True)

        # Clearances
        clearances = [cl for cl in l6_result.get("clearances", []) if cl.get("min_m") is not None]
        if clearances:
            st.markdown("**Crown-terrain clearance:**")
            cl_rows = []
            for cl in clearances:
                cl_rows.append({
                    "Element": cl.get("element_name", "?"),
                    "Min (m)": round(cl["min_m"], 2),
                    "Max (m)": round(cl["max_m"], 2),
                    "Avg (m)": round(cl["avg_m"], 2),
                })
            st.dataframe(cl_rows, use_container_width=True, hide_index=True)

        # Foundation embedments
        embedments = [e for e in l6_result.get("embedments", []) if e.get("foundation_embedment_m") is not None]
        if embedments:
            st.markdown("**Foundation embedment depth:**")
            emb_rows = []
            for e in embedments:
                emb_rows.append({
                    "Element": e.get("element_name", "?"),
                    "Embedment (m)": round(e["foundation_embedment_m"], 2),
                    "Terrain Z (m)": round(e["terrain_z"], 2),
                    "Foundation min Z (m)": round(e["foundation_min_z"], 2),
                })
            st.dataframe(emb_rows, use_container_width=True, hide_index=True)

        # Element distances
        if l6_result.get("distances"):
            st.markdown("**Inter-element distances:**")
            d_rows = []
            for d in l6_result["distances"]:
                d_rows.append({
                    "Element A": d.get("element_a_name", "?"),
                    "Element B": d.get("element_b_name", "?"),
                    "Min dist (mm)": round(d.get("min_distance_mm", 0), 1),
                    "XY dist (mm)": round(d.get("horizontal_distance_mm", 0), 1),
                })
            st.dataframe(d_rows, use_container_width=True, hide_index=True)

    # ── Anomaly detection ──────────────────────────────────────
    try:
        from ifc_geo_validator.core.anomaly_detection import detect_anomalies
        for r in [rr for rr in results if "error" not in rr]:
            anomalies = detect_anomalies(
                r.get("mesh_data", {}),
                r.get("level2", {}),
                r.get("level3", {}),
            )
            if anomalies:
                with st.expander(f"⚠️ Anomalien: {r.get('element_name', '?')} ({len(anomalies)})"):
                    for a in anomalies:
                        if a["severity"] == "warning":
                            st.warning(a["message"])
                        else:
                            st.info(a["message"])
    except Exception:
        pass

    # ── Project metadata + HTML report ──────────────────────────
    st.divider()
    st.subheader("Prüfprotokoll")
    pcol1, pcol2 = st.columns(2)
    project_name = pcol1.text_input("Projektname", "", key="proj_name")
    author_name = pcol2.text_input("Prüfer/in", "", key="author_name")

    if project_name or author_name:
        try:
            from ifc_geo_validator.report.html_report import generate_html_report
            rs_name = ruleset["metadata"]["name"] if ruleset else "—"
            html_report = generate_html_report(
                results, ifc_filename=uploaded_file.name,
                ruleset_name=rs_name,
                project_name=project_name,
                author=author_name,
            )
            st.download_button(
                "📄 HTML-Prüfprotokoll herunterladen",
                data=html_report,
                file_name=f"{Path(uploaded_file.name).stem}_pruefprotokoll.html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception:
            pass

    # ── CSV export ───────────────────────────────────────────────
    try:
        import csv
        import io
        csv_buf = io.StringIO()
        csv_rows = []
        for r in results:
            if "error" in r:
                continue
            l1r = r.get("level1", {})
            l3r = r.get("level3", {})
            l4r = r.get("level4", {})
            l2r = r.get("level2", {})
            row = {
                "element_id": r.get("element_id"),
                "element_name": r.get("element_name"),
                "role": l2r.get("element_role", ""),
                "confidence": l2r.get("confidence", ""),
                "volume_m3": l1r.get("volume"),
                "crown_width_mm": l3r.get("crown_width_mm"),
                "crown_slope_pct": l3r.get("crown_slope_percent"),
                "min_thickness_mm": l3r.get("min_wall_thickness_mm"),
                "wall_height_m": l3r.get("wall_height_m"),
                "is_curved": l3r.get("is_curved"),
                "min_radius_m": l3r.get("min_radius_m"),
                "uncertainty_mm": l3r.get("measurement_uncertainty_mm"),
            }
            if l4r:
                s = l4r.get("summary", {})
                row["rules_passed"] = s.get("passed")
                row["rules_failed"] = s.get("failed")
            csv_rows.append(row)

        if csv_rows:
            writer = csv.DictWriter(csv_buf, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
            st.download_button(
                f"📊 CSV {t('download_report') if 'download' in t('download_report').lower() else 'Export'}",
                data=csv_buf.getvalue(),
                file_name=f"{Path(uploaded_file.name).stem}_measurements.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception:
        pass

    # ── Download report ──────────────────────────────────────────
    st.divider()

    def _json_default(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()  # WallCenterline → dict
        if hasattr(obj, "item"):
            return obj.item()  # numpy scalar
        if isinstance(obj, float) and math.isinf(obj):
            return "Infinity"
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    # Remove non-serializable objects from report (numpy arrays, WallCenterline)
    for elem in report.get("elements", []):
        l2_data = elem.get("level2") or elem.get("face_classification")
        if l2_data and "centerline" in l2_data:
            l2_data.pop("centerline", None)
    # Also clean mesh_data from all_results if present
    for r in results:
        if isinstance(r, dict):
            r.pop("mesh_data", None)

    report_json = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)

    dl1, dl2, dl3 = st.columns(3)

    dl1.download_button(
        "Download JSON Report",
        data=report_json,
        file_name=f"{Path(uploaded_file.name).stem}_report.json",
        mime="application/json",
        use_container_width=True,
    )

    # Enriched IFC download
    try:
        from ifc_geo_validator.report.ifc_property_writer import inject_all

        with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
            tmp.write(file_bytes)
            enrich_tmp = tmp.name

        enrich_model = load_model(enrich_tmp)
        enrich_elems = []
        for etype in (entity_types if entity_types else ["IfcWall"]):
            enrich_elems.extend(get_elements(
                enrich_model, etype, predefined_type.strip() or None
            ))
        inject_all(enrich_model, enrich_elems, results, enrich_tmp)

        with open(enrich_tmp, "rb") as f:
            enriched_bytes = f.read()

        dl2.download_button(
            "Download Enriched IFC",
            data=enriched_bytes,
            file_name=f"{Path(uploaded_file.name).stem}_validated.ifc",
            mime="application/octet-stream",
            use_container_width=True,
        )
    except Exception:
        dl2.button("Enriched IFC (unavailable)", disabled=True,
                   use_container_width=True)

    # BCF download
    try:
        from ifc_geo_validator.report.bcf_export import export_bcf

        bcf_tmp = tempfile.mktemp(suffix=".bcf")
        export_bcf(results, bcf_tmp, ifc_name=uploaded_file.name)

        with open(bcf_tmp, "rb") as f:
            bcf_bytes = f.read()

        dl3.download_button(
            "Download BCF Issues",
            data=bcf_bytes,
            file_name=f"{Path(uploaded_file.name).stem}_issues.bcf",
            mime="application/octet-stream",
            use_container_width=True,
        )
    except Exception:
        dl3.button("BCF Issues (unavailable)", disabled=True,
                   use_container_width=True)
