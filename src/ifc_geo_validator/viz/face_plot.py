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
    element_index: int = None,
    entity_type: str = "IfcWall",
    show: bool = True,
    screenshot: str = None,
):
    """Convenience function: load IFC, classify faces, and visualize.

    When element_index is None (default) and the model contains multiple
    elements, ALL elements are rendered in a single scene. When element_index
    is set, only that element is rendered.

    Args:
        ifc_path: Path to IFC file.
        element_index: Which element to visualize (None = all elements).
        entity_type: IFC entity type filter.
        show: If True, display interactive window.
        screenshot: If given, save PNG to this path.

    Returns:
        Tuple of (level2_result_or_list, plotter).
    """
    from ifc_geo_validator.core.ifc_parser import load_model, get_elements
    from ifc_geo_validator.core.mesh_converter import extract_mesh
    from ifc_geo_validator.validation.level2 import validate_level2

    model = load_model(ifc_path)
    elements = get_elements(model, entity_type)

    if not elements:
        raise ValueError(f"No {entity_type} elements found in {ifc_path}")

    # Single element mode (backward compatible)
    if element_index is not None:
        if element_index >= len(elements):
            raise IndexError(f"Element index {element_index} out of range (found {len(elements)})")
        elem = elements[element_index]
        name = getattr(elem, "Name", None) or "Unnamed"
        mesh_data = extract_mesh(elem)
        l2 = validate_level2(mesh_data)
        title = f"{name} — Face Classification"
        pl = plot_classified_faces(mesh_data, l2, title=title, show=show, screenshot=screenshot)
        return l2, pl

    # Multi-element mode: render all elements in one scene
    return plot_all_elements(elements, extract_mesh, validate_level2,
                             ifc_path, show=show, screenshot=screenshot)


def _lighten_hex(hex_color, factor=0.45):
    """Lighten a hex color toward white by the given factor (0=unchanged, 1=white)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# Per-element color palettes: element 0 = saturated, 1+ = progressively lighter
def _element_palette(element_idx, n_elements):
    """Generate a per-element color palette.

    Element 0 (main wall): original saturated colors.
    Element 1+ (foundation, buttresses): lightened versions so they're
    visually distinct from the main wall while preserving category meaning.
    """
    if n_elements <= 1 or element_idx == 0:
        return CATEGORY_COLORS

    # Lighten factor: strong contrast between element 0 and 1+
    factor = min(0.5 + 0.15 * (element_idx - 1), 0.85)
    return {cat: _lighten_hex(color, factor) for cat, color in CATEGORY_COLORS.items()}


def plot_all_elements(elements, extract_mesh_fn, validate_l2_fn,
                      ifc_path="", show=True, screenshot=None,
                      window_size=(1600, 900)):
    """Render all wall elements in a single scene with face classification.

    Each element gets a distinct color palette: the first element (main wall)
    uses saturated colors, subsequent elements (foundation, buttresses) use
    progressively lighter pastel versions. Face categories are preserved.
    """
    if pv is None:
        raise ImportError("pyvista is required for visualization: pip install pyvista")

    from pathlib import Path
    file_stem = Path(ifc_path).stem if ifc_path else "Model"

    pl = pv.Plotter(window_size=window_size, off_screen=not show)
    pl.set_background("white")

    all_results = []
    legend_entries = []
    legend_added = set()
    n_elements = len(elements)

    for ei, elem in enumerate(elements):
        name = getattr(elem, "Name", None) or "Unnamed"
        mesh_data = extract_mesh_fn(elem)
        l2 = validate_l2_fn(mesh_data)
        all_results.append((name, mesh_data, l2))

        vertices = mesh_data["vertices"]
        faces_arr = mesh_data["faces"]
        groups = l2["face_groups"]
        palette = _element_palette(ei, n_elements)

        # Build per-triangle RGB colors directly
        n_tris = len(faces_arr)
        tri_colors = np.zeros((n_tris, 3), dtype=np.uint8)
        tri_categories = [UNCLASSIFIED] * n_tris

        for g in groups:
            cat = g["category"]
            hex_c = palette.get(cat, palette[UNCLASSIFIED])
            rgb = (int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16))
            for fi in g["face_indices"]:
                tri_colors[fi] = rgb
                tri_categories[fi] = cat

        # Build PyVista mesh
        pv_faces = np.column_stack([
            np.full(n_tris, 3, dtype=int),
            faces_arr,
        ]).ravel()
        mesh = pv.PolyData(vertices, pv_faces)
        mesh.cell_data["colors"] = tri_colors

        pl.add_mesh(
            mesh,
            scalars="colors",
            rgb=True,
            show_scalar_bar=False,
            show_edges=True,
            edge_color="#cccccc",
            line_width=0.5,
        )

        # Build legend entries (categories for this element)
        present = set(tri_categories)
        for cat in CATEGORY_COLORS:
            if cat in present:
                key = (cat, ei)
                if key not in legend_added:
                    legend_added.add(key)
                    label = CATEGORY_LABELS.get(cat, cat)
                    if n_elements > 1:
                        label = f"{label} ({name})"
                    legend_entries.append([label, palette[cat]])

    if legend_entries:
        pl.add_legend(legend_entries, bcolor="white", border=True,
                      size=(0.2, min(0.05 * len(legend_entries) + 0.05, 0.5)))

    title = f"{file_stem} — Face Classification ({n_elements} elements)"
    pl.add_title(title, font_size=14, color="black")

    if screenshot:
        pl.show(auto_close=False)
        pl.screenshot(screenshot)
        pl.close()
    elif show:
        pl.show()

    return all_results, pl
