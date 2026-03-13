"""3D visualization of face classification results using PyVista.

Generates publication-quality figures for the thesis showing:
- Color-coded face groups by semantic category
- Wall axis and perpendicular direction indicators
- Measurement annotations (crown width, inclination)
"""

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

from ifc_geo_validator.core.face_classifier import (
    CROWN,
    FOUNDATION,
    FRONT,
    BACK,
    END_LEFT,
    END_RIGHT,
    UNCLASSIFIED,
)


# Thesis-friendly color palette (colorblind-safe)
CATEGORY_COLORS = {
    CROWN:        "#2196F3",  # blue
    FOUNDATION:   "#795548",  # brown
    FRONT:        "#F44336",  # red
    BACK:         "#FF9800",  # orange
    END_LEFT:     "#4CAF50",  # green
    END_RIGHT:    "#8BC34A",  # light green
    UNCLASSIFIED: "#9E9E9E",  # grey
}

CATEGORY_LABELS = {
    CROWN:        "Krone",
    FOUNDATION:   "Fundament",
    FRONT:        "Ansichtsfläche",
    BACK:         "Rückseite",
    END_LEFT:     "Stirnfläche L",
    END_RIGHT:    "Stirnfläche R",
    UNCLASSIFIED: "Unklassifiziert",
}


def plot_classified_faces(
    mesh_data: dict,
    level2_result: dict,
    title: str = "Face Classification",
    show: bool = True,
    screenshot: str = None,
    window_size: tuple = (1600, 900),
):
    """Render 3D mesh with faces colored by classification category.

    Args:
        mesh_data: dict from mesh_converter.extract_mesh().
        level2_result: dict from validate_level2().
        title: Plot title.
        show: If True, display interactive window.
        screenshot: If given, save PNG to this path.
        window_size: Window size in pixels.

    Returns:
        pv.Plotter instance (for further customization).
    """
    if pv is None:
        raise ImportError("pyvista is required for visualization: pip install pyvista")

    vertices = mesh_data["vertices"]
    faces_arr = mesh_data["faces"]
    groups = level2_result["face_groups"]

    # Build per-triangle category labels
    n_tris = len(faces_arr)
    tri_categories = ["UNCLASSIFIED"] * n_tris
    for g in groups:
        cat = g["category"]
        for fi in g["face_indices"]:
            tri_categories[fi] = cat

    # Build PyVista PolyData
    pv_faces = np.column_stack([
        np.full(n_tris, 3, dtype=int),
        faces_arr,
    ]).ravel()
    mesh = pv.PolyData(vertices, pv_faces)

    # Assign scalars for coloring
    cat_to_int = {cat: i for i, cat in enumerate(CATEGORY_COLORS)}
    cat_ids = np.array([cat_to_int.get(c, len(cat_to_int) - 1) for c in tri_categories])
    mesh.cell_data["category"] = cat_ids

    # Build color map aligned with category IDs
    colors = list(CATEGORY_COLORS.values())

    # Create plotter
    pl = pv.Plotter(window_size=window_size, off_screen=not show)
    pl.set_background("white")

    # Add mesh with categorical coloring
    pl.add_mesh(
        mesh,
        scalars="category",
        cmap=colors,
        clim=[0, len(colors) - 1],
        show_scalar_bar=False,
        show_edges=True,
        edge_color="#cccccc",
        line_width=0.5,
    )

    # Add legend
    legend_entries = []
    present_categories = set(tri_categories)
    for cat, color in CATEGORY_COLORS.items():
        if cat in present_categories:
            label = CATEGORY_LABELS.get(cat, cat)
            legend_entries.append([label, color])

    if legend_entries:
        pl.add_legend(
            legend_entries,
            bcolor="white",
            border=True,
            size=(0.15, 0.25),
        )

    # Add wall axis arrow
    wall_axis = np.array(level2_result["wall_axis"])
    centroid = vertices.mean(axis=0)
    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    arrow_len = max(bbox_size) * 0.3

    pl.add_arrows(
        centroid.reshape(1, 3),
        wall_axis.reshape(1, 3),
        mag=arrow_len,
        color="#000000",
    )
    pl.add_point_labels(
        [centroid + wall_axis * arrow_len * 1.1],
        ["Mauerachse"],
        font_size=12,
        text_color="black",
        shape=None,
        render_points_as_spheres=False,
    )

    pl.add_title(title, font_size=14, color="black")

    if screenshot:
        pl.show(auto_close=False)
        pl.screenshot(screenshot)
        pl.close()
    elif show:
        pl.show()

    return pl


def plot_model_from_ifc(
    ifc_path: str,
    element_index: int = 0,
    entity_type: str = "IfcWall",
    show: bool = True,
    screenshot: str = None,
):
    """Convenience function: load IFC, classify faces, and visualize.

    Args:
        ifc_path: Path to IFC file.
        element_index: Which element to visualize (default: first).
        entity_type: IFC entity type filter.
        show: If True, display interactive window.
        screenshot: If given, save PNG to this path.

    Returns:
        Tuple of (level2_result, plotter).
    """
    from ifc_geo_validator.core.ifc_parser import load_model, get_elements
    from ifc_geo_validator.core.mesh_converter import extract_mesh
    from ifc_geo_validator.validation.level2 import validate_level2

    model = load_model(ifc_path)
    elements = get_elements(model, entity_type)

    if element_index >= len(elements):
        raise IndexError(f"Element index {element_index} out of range (found {len(elements)})")

    elem = elements[element_index]
    name = getattr(elem, "Name", None) or "Unnamed"
    mesh_data = extract_mesh(elem)
    l2 = validate_level2(mesh_data)

    title = f"{name} — Face Classification"
    pl = plot_classified_faces(mesh_data, l2, title=title, show=show, screenshot=screenshot)
    return l2, pl
