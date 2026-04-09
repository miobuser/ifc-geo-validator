"""Interactive 3D viewer using Plotly (embeddable in Streamlit).

Renders classified meshes with face-category colors, optional slope
heatmap overlay, and measurement annotations. No external dependencies
beyond plotly (included with Streamlit).

Usage in Streamlit:
    from ifc_geo_validator.viz.plotly_viewer import create_3d_figure
    fig = create_3d_figure(mesh_data, l2_result)
    st.plotly_chart(fig, use_container_width=True)
"""

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# Category colors (matching the existing viewer)
CATEGORY_COLORS = {
    "crown":        "rgb(33,150,243)",    # blue
    "foundation":   "rgb(121,85,72)",     # brown
    "front":        "rgb(244,67,54)",     # red
    "back":         "rgb(255,152,0)",     # orange
    "end_left":     "rgb(76,175,80)",     # green
    "end_right":    "rgb(139,195,74)",    # light green
    "unclassified": "rgb(158,158,158)",   # grey
}

CATEGORY_LABELS = {
    "crown": "Krone",
    "foundation": "Fundament",
    "front": "Ansichtsfläche",
    "back": "Rückseite",
    "end_left": "Stirnfläche L",
    "end_right": "Stirnfläche R",
    "unclassified": "Unklassifiziert",
}


def create_3d_figure(
    mesh_data: dict,
    l2_result: dict = None,
    mode: str = "classification",
    slope_data: dict = None,
    title: str = "",
    height: int = 600,
) -> "go.Figure":
    """Create an interactive 3D plotly figure of a classified mesh.

    Args:
        mesh_data: dict with vertices, faces.
        l2_result: dict with face_groups (for classification coloring).
        mode: "classification" or "slope" (heatmap).
        slope_data: dict from compute_triangle_slopes (for slope mode).
        title: figure title.
        height: figure height in pixels.

    Returns:
        plotly Figure object (use st.plotly_chart to display).
    """
    if go is None:
        raise ImportError("plotly required: pip install plotly")

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]

    fig = go.Figure()

    if mode == "classification" and l2_result is not None:
        # One trace per face category
        face_groups = l2_result.get("face_groups", [])
        for cat in CATEGORY_COLORS:
            cat_faces = []
            for g in face_groups:
                if g.get("category") == cat:
                    cat_faces.extend(g["face_indices"])

            if not cat_faces:
                continue

            fi = np.array(cat_faces)
            # Build mesh for this category
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[fi, 0],
                j=faces[fi, 1],
                k=faces[fi, 2],
                color=CATEGORY_COLORS[cat],
                opacity=0.85,
                name=CATEGORY_LABELS.get(cat, cat),
                showlegend=True,
                flatshading=True,
            ))

    elif mode == "slope" and slope_data is not None:
        # Slope heatmap: color per face
        values = np.clip(slope_data.get("cross_slope_pct", np.zeros(len(faces))), 0, 10)

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=values,
            colorscale="RdYlGn_r",
            cmin=0, cmax=6,
            colorbar=dict(title="Quergefälle (%)"),
            opacity=0.9,
            name="Quergefälle",
            flatshading=True,
        ))

    else:
        # Simple mesh without classification
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="rgb(100,149,237)",
            opacity=0.8,
            flatshading=True,
        ))

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        ),
        height=height,
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    return fig
