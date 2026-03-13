"""Streamlit web interface for IFC Geometry Validator."""

import json
import math
import tempfile
from pathlib import Path

import streamlit as st

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.report.json_report import generate_report


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_RULESET = Path(__file__).parent / "rules" / "rulesets" / "astra_fhb_stuetzmauer.yaml"


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="IFC Geometry Validator",
    page_icon="🏗️",
    layout="wide",
)


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("IFC Geometry Validator")
st.sidebar.caption("v0.1.0 — BSc Thesis BFH")

uploaded_file = st.sidebar.file_uploader(
    "Upload IFC file",
    type=["ifc"],
    help="IFC 4x3 model with IfcWall elements",
)

entity_type = st.sidebar.selectbox(
    "Entity type",
    ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam"],
    index=0,
)

predefined_type = st.sidebar.text_input(
    "Predefined type filter (optional)",
    value="",
    help="e.g. RETAININGWALL",
)

ruleset_file = st.sidebar.file_uploader(
    "Custom ruleset (YAML)",
    type=["yaml", "yml"],
    help="Leave empty to use built-in ASTRA FHB T/G ruleset",
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

        Upload an IFC file in the sidebar to begin.
        """
    )
    st.stop()

if not uploaded_file:
    st.warning("Please upload an IFC file first.")
    st.stop()


# ── Run validation ───────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading IFC model...")
def run_validation(_file_bytes, file_name, entity_type, predefined_type, _ruleset_bytes):
    """Run the full validation pipeline and return structured results."""
    # Write uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name

    model = load_model(tmp_path)
    pred_filter = predefined_type if predefined_type else None
    elements = get_elements(model, entity_type, pred_filter)

    # Load ruleset
    if _ruleset_bytes:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="wb") as rs_tmp:
            rs_tmp.write(_ruleset_bytes)
            rs_path = rs_tmp.name
        ruleset = load_ruleset(rs_path)
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
            l4 = validate_level4(l1, l3, ruleset) if ruleset else None

            result = {
                "element_id": elem.id(),
                "element_name": name,
                "level1": l1,
                "level2": l2,
                "level3": l3,
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

    report = generate_report(file_name, all_results, ruleset)
    return all_results, report, ruleset


if run_button or uploaded_file:
    file_bytes = uploaded_file.getvalue()
    rs_bytes = ruleset_file.getvalue() if ruleset_file else None
    pred = predefined_type.strip() or ""

    results, report, ruleset = run_validation(
        file_bytes, uploaded_file.name, entity_type, pred, rs_bytes
    )

    if not results:
        st.error(f"No {entity_type} elements found in the model.")
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Elements", len(results))
    col2.metric("Validated", len(valid_results))
    col3.metric("Errors", len(error_results))

    if valid_results and "level4" in valid_results[0]:
        total_checks = sum(
            r["level4"]["summary"]["total"]
            for r in valid_results if "level4" in r
        )
        total_passed = sum(
            r["level4"]["summary"]["passed"]
            for r in valid_results if "level4" in r
        )
        col4.metric("Rules passed", f"{total_passed}/{total_checks}")

    # ── Per-element results ──────────────────────────────────────
    for r in results:
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

    # ── Download report ──────────────────────────────────────────
    st.divider()

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, float) and math.isinf(obj):
            return "Infinity"
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    report_json = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    st.download_button(
        "Download JSON Report",
        data=report_json,
        file_name=f"{Path(uploaded_file.name).stem}_report.json",
        mime="application/json",
        use_container_width=True,
    )
