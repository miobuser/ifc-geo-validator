"""Interactive Review App — 3D visualization with classification explanation.

Usage:
    streamlit run src/ifc_geo_validator/review_app.py

Upload an IFC model to see:
  - Interactive 3D view with color-coded face classification
  - Why each face group was classified (normal analysis, thresholds)
  - Centerline visualization for curved walls
  - Step-by-step algorithm trace
"""
import sys
import math
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="IFC Geo Validator — Review", page_icon="🔍", layout="wide")

from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.core.face_classifier import (
    classify_faces, WallCenterline, CROWN, FOUNDATION, FRONT, BACK,
    END_LEFT, END_RIGHT, UNCLASSIFIED, DEFAULT_THRESHOLDS,
)
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3

CATEGORY_COLORS = {
    CROWN:        "#2196F3",
    FOUNDATION:   "#795548",
    FRONT:        "#F44336",
    BACK:         "#FF9800",
    END_LEFT:     "#4CAF50",
    END_RIGHT:    "#8BC34A",
    UNCLASSIFIED: "#9E9E9E",
}

CATEGORY_LABELS_DE = {
    CROWN:        "Krone (Mauerkrone)",
    FOUNDATION:   "Fundament (Sohle)",
    FRONT:        "Ansichtsfläche (Luftseite)",
    BACK:         "Rückseite (Erdseite)",
    END_LEFT:     "Stirnfläche Links",
    END_RIGHT:    "Stirnfläche Rechts",
    UNCLASSIFIED: "Unklassifiziert (intern)",
}


def _hex_to_rgb(h):
    return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]


def build_plotly_mesh(vertices, faces, face_groups):
    """Build a Plotly Mesh3d figure with per-face category colors."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    # Build per-face color array
    n_faces = len(faces)
    face_colors = np.zeros((n_faces, 3), dtype=int)
    face_categories = [""] * n_faces

    for g in face_groups:
        cat = g["category"]
        rgb = _hex_to_rgb(CATEGORY_COLORS.get(cat, "#9E9E9E"))
        for fi in g["face_indices"]:
            face_colors[fi] = rgb
            face_categories[fi] = cat

    # Convert to plotly format
    color_strings = [f"rgb({r},{g},{b})" for r, g, b in face_colors]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            facecolor=color_strings,
            flatshading=True,
            hoverinfo="text",
            text=[f"Face {i}: {face_categories[i]}" for i in range(n_faces)],
        )
    ])
    fig.update_layout(
        scene=dict(aspectmode="data", bgcolor="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    return fig


def explain_classification(face_groups, centerline, thresholds):
    """Generate human-readable explanation of why each group was classified."""
    explanations = []
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_horiz = np.cos(np.radians(thresholds.get("horizontal_deg", 45.0)))

    for g in face_groups:
        cat = g["category"]
        normal = np.array(g["normal"])
        centroid = np.array(g["centroid"])
        area = g["area"]
        dot_z = abs(np.dot(normal, z_axis))

        exp = {
            "category": cat,
            "label": CATEGORY_LABELS_DE.get(cat, cat),
            "color": CATEGORY_COLORS.get(cat, "#9E9E9E"),
            "area_m2": round(area, 3),
            "normal": [round(n, 4) for n in normal],
            "centroid": [round(c, 3) for c in centroid],
            "dot_z": round(float(dot_z), 4),
            "triangles": g["num_triangles"],
            "reason": "",
        }

        if cat in (CROWN, FOUNDATION):
            direction = "oben (+Z)" if cat == CROWN else "unten (-Z)"
            exp["reason"] = (
                f"|n·e_z| = {dot_z:.4f} > cos({thresholds.get('horizontal_deg', 45):.0f}°) = {cos_horiz:.4f} "
                f"→ Horizontale Fläche → Normale zeigt nach {direction} → {CATEGORY_LABELS_DE[cat]}"
            )
        elif cat in (FRONT, BACK):
            exp["reason"] = (
                f"|n·e_z| = {dot_z:.4f} < {cos_horiz:.4f} → Vertikale Fläche → "
                f"Normale ≈ senkrecht zur Wandachse → {CATEGORY_LABELS_DE[cat]} "
                f"(grössere Fläche = Ansichtsfläche)"
            )
        elif cat in (END_LEFT, END_RIGHT):
            exp["reason"] = (
                f"|n·e_z| = {dot_z:.4f} < {cos_horiz:.4f} → Vertikale Fläche → "
                f"Normale ≈ parallel zur Wandachse → {CATEGORY_LABELS_DE[cat]} "
                f"(an Wandextremität)"
            )
        elif cat == UNCLASSIFIED:
            exp["reason"] = (
                "Normale parallel zur Wandachse, aber NICHT an Wandextremität → "
                "Interne Fläche (z.B. Strebepfeiler-Seite)"
            )

        explanations.append(exp)
    return explanations


# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.title("🔍 Review App")
st.sidebar.caption("Interaktive Prüfung der Flächenklassifikation")

uploaded = st.sidebar.file_uploader("IFC-Datei hochladen", type=["ifc"])

if not uploaded:
    st.title("IFC Geometry Validator — Interaktive Review")
    st.info("Lade eine IFC-Datei in der Sidebar hoch um die Klassifikation zu prüfen.")

    # Show test models for quick access
    models_dir = Path(__file__).resolve().parents[2] / "tests" / "test_models"
    if models_dir.exists():
        st.subheader("Verfügbare Testmodelle")
        models = sorted(models_dir.glob("T*.ifc"))
        cols = st.columns(4)
        for i, m in enumerate(models):
            cols[i % 4].code(m.name, language=None)
    st.stop()

# ── Load model ──────────────────────────────────────────────────

with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
    f.write(uploaded.getvalue())
    tmp_path = f.name

model = load_model(tmp_path)
walls = get_elements(model, "IfcWall")
terrain = get_terrain_mesh(model)

if not walls:
    st.error("Keine IfcWall-Elemente gefunden.")
    st.stop()

st.title(f"Review: {uploaded.name}")
st.caption(f"{len(walls)} IfcWall Element(e)" + (", Terrain erkannt" if terrain else ""))

# ── Element selector ─────────────────────────────────────────────

if len(walls) > 1:
    wall_names = [f"#{w.id()} {getattr(w, 'Name', '?')}" for w in walls]
    selected_idx = st.sidebar.selectbox("Element auswählen", range(len(walls)),
                                         format_func=lambda i: wall_names[i])
else:
    selected_idx = 0

wall = walls[selected_idx]
wall_name = getattr(wall, "Name", "Unnamed")
st.subheader(f"Element: {wall_name} (#{wall.id()})")

# ── Run pipeline ─────────────────────────────────────────────────

mesh = extract_mesh(wall)
l1 = validate_level1(mesh)
result = classify_faces(mesh)
l2 = validate_level2(mesh)
l3 = validate_level3(mesh, l2)

face_groups = l2["face_groups"]
centerline = result.get("centerline")
thresholds = result.get("thresholds_used", DEFAULT_THRESHOLDS)

# ── Tab layout ───────────────────────────────────────────────────

tab_3d, tab_explain, tab_metrics, tab_centerline, tab_raw = st.tabs([
    "3D Ansicht", "Klassifikation erklärt", "Messwerte", "Centerline", "Rohdaten"
])

# ── Tab 1: 3D View ──────────────────────────────────────────────

with tab_3d:
    fig = build_plotly_mesh(mesh["vertices"], mesh["faces"], face_groups)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly nicht installiert. `pip install plotly`")

    # Legend
    st.markdown("**Legende:**")
    cols = st.columns(len(CATEGORY_COLORS))
    present = {g["category"] for g in face_groups}
    for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
        if cat in present:
            cols[i].markdown(
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px">{CATEGORY_LABELS_DE.get(cat, cat)}</span>',
                unsafe_allow_html=True,
            )

# ── Tab 2: Classification Explained ─────────────────────────────

with tab_explain:
    explanations = explain_classification(face_groups, centerline, thresholds)

    st.markdown(f"**Schwellwerte:** horizontal={thresholds['horizontal_deg']}° "
                f"(cos={np.cos(np.radians(thresholds['horizontal_deg'])):.4f}), "
                f"coplanar={thresholds['coplanar_deg']}°, "
                f"lateral={thresholds['lateral_deg']}°")
    st.markdown(f"**Wandachse:** {result['wall_axis']}")
    if centerline:
        st.markdown(f"**Centerline:** {centerline.length:.2f}m, "
                    f"{'gekrümmt' if centerline.is_curved else 'gerade'}, "
                    f"{len(centerline.points_2d)} Slices")

    for exp in explanations:
        with st.expander(
            f"{'🟦' if 'Krone' in exp['label'] else '🟫' if 'Fund' in exp['label'] else '🟥' if 'Ansicht' in exp['label'] else '🟧' if 'Rück' in exp['label'] else '🟩' if 'Stirn' in exp['label'] else '⬜'} "
            f"{exp['label']} — {exp['area_m2']} m², {exp['triangles']} Dreiecke",
            expanded=False,
        ):
            st.markdown(f"**Warum?** {exp['reason']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Normale", f"({exp['normal'][0]}, {exp['normal'][1]}, {exp['normal'][2]})")
            col2.metric("|n·Z|", f"{exp['dot_z']:.4f}")
            col3.metric("Schwerpunkt", f"({exp['centroid'][0]}, {exp['centroid'][1]}, {exp['centroid'][2]})")

# ── Tab 3: Measurements ─────────────────────────────────────────

with tab_metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("Volumen", f"{l1['volume']:.3f} m³")
    col2.metric("Oberfläche", f"{l1['total_area']:.3f} m²")
    col3.metric("Watertight", "Ja" if l1["is_watertight"] else "Nein")

    col4, col5, col6 = st.columns(3)
    col4.metric("Kronenbreite", f"{l3.get('crown_width_mm', 'N/A')} mm")
    col5.metric("Wandstärke", f"{l3.get('min_wall_thickness_mm', 'N/A')} mm")
    col6.metric("Wandhöhe", f"{l3.get('wall_height_m', 'N/A')} m")

    col7, col8, col9 = st.columns(3)
    col7.metric("Kronenneigung", f"{l3.get('crown_slope_percent', 0):.2f} %")
    inc = l3.get("front_inclination_ratio")
    col8.metric("Neigung", "vertikal" if inc and math.isinf(inc) else f"{inc:.1f}:1" if inc else "N/A")
    col9.metric("Gekrümmt", "Ja" if l3.get("is_curved") else "Nein")

# ── Tab 4: Centerline ───────────────────────────────────────────

with tab_centerline:
    if centerline and len(centerline.points_2d) > 2:
        try:
            import plotly.graph_objects as go

            pts = centerline.points_2d
            fig_cl = go.Figure()

            # Crown vertices projected to XY
            crown_groups = [g for g in face_groups if g["category"] == CROWN]
            if crown_groups:
                crown_fi = []
                for g in crown_groups:
                    crown_fi.extend(g["face_indices"])
                if crown_fi:
                    crown_vi = np.unique(mesh["faces"][np.array(crown_fi)].ravel())
                    crown_xy = mesh["vertices"][crown_vi, :2]
                    fig_cl.add_trace(go.Scatter(
                        x=crown_xy[:, 0], y=crown_xy[:, 1],
                        mode="markers", name="Kronen-Vertices",
                        marker=dict(size=4, color="#2196F3", opacity=0.5),
                    ))

            # Centerline
            fig_cl.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode="lines+markers", name="Centerline",
                line=dict(color="red", width=3),
                marker=dict(size=6),
            ))

            # Tangent arrows
            for i in range(0, len(pts), max(1, len(pts) // 8)):
                t = centerline.tangents[i]
                n = centerline.normals[i]
                scale = centerline.length * 0.05
                fig_cl.add_annotation(
                    x=pts[i, 0] + t[0] * scale, y=pts[i, 1] + t[1] * scale,
                    ax=pts[i, 0], ay=pts[i, 1],
                    arrowhead=2, arrowcolor="red", arrowwidth=2,
                    showarrow=True, text="",
                )
                fig_cl.add_annotation(
                    x=pts[i, 0] + n[0] * scale * 0.5, y=pts[i, 1] + n[1] * scale * 0.5,
                    ax=pts[i, 0], ay=pts[i, 1],
                    arrowhead=2, arrowcolor="blue", arrowwidth=1,
                    showarrow=True, text="",
                )

            fig_cl.update_layout(
                xaxis=dict(scaleanchor="y", title="X"),
                yaxis=dict(title="Y"),
                height=500,
                title="Centerline (rot) mit Tangenten (rot) und Normalen (blau)",
            )
            st.plotly_chart(fig_cl, use_container_width=True)

            st.dataframe([{
                "Slice": i,
                "X": round(pts[i, 0], 3),
                "Y": round(pts[i, 1], 3),
                "Width (m)": round(float(centerline.widths[i]), 4) if i < len(centerline.widths) else None,
            } for i in range(len(pts))], use_container_width=True)

        except ImportError:
            st.warning("Plotly nicht installiert für Centerline-Visualisierung.")
    else:
        st.info("Gerade Wand — keine Centerline-Visualisierung nötig (2-Punkt Degenerat).")

# ── Tab 5: Raw Data ──────────────────────────────────────────────

with tab_raw:
    st.json({
        "thresholds": thresholds,
        "wall_axis": result["wall_axis"],
        "num_groups": result["num_groups"],
        "centerline": centerline.to_dict() if centerline else None,
        "face_groups": [
            {
                "category": g["category"],
                "area": round(g["area"], 4),
                "normal": [round(n, 6) for n in g["normal"]],
                "centroid": [round(c, 4) for c in g["centroid"]],
                "num_triangles": g["num_triangles"],
                "face_indices": g["face_indices"][:10],  # truncate for display
            }
            for g in face_groups
        ],
    })

# Cleanup
import os
try:
    os.unlink(tmp_path)
except Exception:
    pass
