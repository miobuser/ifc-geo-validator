"""Slope heatmap visualization for road and crown surfaces.

Computes per-triangle slope decomposed into:
  - Total slope (Gesamtneigung): angle from horizontal
  - Cross-slope (Quergefälle): component perpendicular to road/wall axis
  - Longitudinal slope (Längsgefälle): component along road/wall axis

Color-codes each triangle and renders as interactive 3D heatmap.

Mathematical basis:
  For a triangle with unit normal n = (nx, ny, nz):
    total_slope = arctan(sqrt(nx² + ny²) / |nz|)

  Given road axis direction a (unit vector in XY plane):
    perp = [-ay, ax, 0]  (cross-slope direction)
    cross_component = |n · perp| / |n_horiz|
    long_component  = |n · a|   / |n_horiz|
    cross_slope  = arctan(cross_component / |nz|)
    long_slope   = arctan(long_component  / |nz|)

References:
  - ASTRA FHB T/G 24 001-10101: Quergefälle max 5% im Tunnel
  - SIA 197/2: Strassenquerschnitt im Tunnel
"""

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    plt = None


# ── Slope computation ─────────────────────────────────────────────

def compute_triangle_slopes(mesh_data: dict, axis: np.ndarray = None,
                            centerline=None) -> dict:
    """Compute per-triangle slope values for a mesh.

    For curved roads/walls, uses LOCAL tangent directions from the
    centerline at each triangle's position. This gives correct
    cross-slope (Quergefälle) decomposition even on curves — the
    "perpendicular to road" direction rotates with the curve.

    For straight roads or when no centerline is provided, uses a
    single global axis (PCA or provided).

    Args:
        mesh_data: dict with vertices, faces, normals, areas.
        axis: optional road/wall axis direction (3D unit vector).
              If None, determined via PCA on the mesh vertices.
        centerline: optional WallCenterline for local frame lookup.

    Returns:
        dict with:
            total_slope_pct:  np.array (M,) — total slope in percent per triangle
            cross_slope_pct:  np.array (M,) — cross-slope in percent (Quergefälle)
            long_slope_pct:   np.array (M,) — longitudinal slope in percent (Längsgefälle)
            axis:             np.array (3,) — global road/wall axis
            perp:             np.array (3,) — global perpendicular direction
            uses_local_frame: bool — True if local centerline frames were used
    """
    normals = mesh_data["normals"]
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    n_faces = len(normals)

    # Determine global axis from PCA if not provided
    if axis is None:
        xy = vertices[:, :2]
        centered = xy - xy.mean(axis=0)
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        ax_2d = eigvecs[:, -1]
        axis = np.array([ax_2d[0], ax_2d[1], 0.0])

    # Ensure float unit vector in XY
    axis = np.asarray(axis, dtype=float).copy()
    axis[2] = 0.0
    ax_mag = np.linalg.norm(axis)
    if ax_mag > 1e-10:
        axis /= ax_mag
    else:
        axis = np.array([1.0, 0.0, 0.0])

    perp = np.array([-axis[1], axis[0], 0.0])

    # Decide whether to use local frames (only for curved centerlines)
    use_local = (centerline is not None
                 and hasattr(centerline, "use_local_measurement")
                 and centerline.use_local_measurement)

    # Precompute face centroids for local frame lookups
    if use_local:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_centroids_xy = ((v0 + v1 + v2) / 3.0)[:, :2]

    total_slope = np.zeros(n_faces)
    cross_slope = np.zeros(n_faces)
    long_slope = np.zeros(n_faces)

    for i in range(n_faces):
        n = normals[i]
        nz = abs(n[2])

        if nz < 1e-10:
            total_slope[i] = 9000.0
            cross_slope[i] = 9000.0
            long_slope[i] = 9000.0
            continue

        n_horiz = np.sqrt(n[0]**2 + n[1]**2)
        total_slope[i] = (n_horiz / nz) * 100.0

        # Get local axis/perp for this triangle
        if use_local:
            local_tang, local_norm, _ = centerline.get_local_frame(
                face_centroids_xy[i]
            )
            # local_tang = tangent along road, local_norm = perpendicular
            local_perp_2d = np.array([local_norm[0], local_norm[1]])
            local_axis_2d = np.array([local_tang[0], local_tang[1]])
            # Normalize (may be needed if tangent has Z component)
            pm = np.linalg.norm(local_perp_2d)
            am = np.linalg.norm(local_axis_2d)
            if pm > 1e-12:
                local_perp_2d /= pm
            if am > 1e-12:
                local_axis_2d /= am
        else:
            local_perp_2d = perp[:2]
            local_axis_2d = axis[:2]

        cross_comp = abs(n[0] * local_perp_2d[0] + n[1] * local_perp_2d[1])
        long_comp = abs(n[0] * local_axis_2d[0] + n[1] * local_axis_2d[1])

        cross_slope[i] = (cross_comp / nz) * 100.0
        long_slope[i] = (long_comp / nz) * 100.0

    return {
        "total_slope_pct": total_slope,
        "cross_slope_pct": cross_slope,
        "long_slope_pct": long_slope,
        "axis": axis,
        "perp": perp,
        "uses_local_frame": use_local,
    }


def compute_surface_slopes(
    mesh_data: dict,
    face_groups: list,
    categories: list[str] = None,
    axis: np.ndarray = None,
    centerline=None,
) -> dict:
    """Compute slopes only for faces in specified categories.

    Args:
        mesh_data: dict from extract_mesh().
        face_groups: list of face group dicts (from L2).
        categories: list of categories to include (default: ["crown"]).
        axis: optional road/wall axis.
        centerline: optional WallCenterline for local frame on curves.

    Returns:
        Same as compute_triangle_slopes, but only for selected faces.
        Includes face_mask (bool array) for indexing into the full mesh.
    """
    if categories is None:
        categories = ["crown"]

    # Collect face indices from selected categories
    selected_indices = []
    for g in face_groups:
        if g.get("category") in categories:
            selected_indices.extend(g["face_indices"])

    if not selected_indices:
        return None

    # Create mask for selected faces
    n_faces = len(mesh_data["faces"])
    mask = np.zeros(n_faces, dtype=bool)
    mask[selected_indices] = True

    # Compute slopes for all faces (with local frames for curves)
    slopes = compute_triangle_slopes(mesh_data, axis, centerline=centerline)

    # Add mask and statistics for selected faces only
    slopes["face_mask"] = mask
    slopes["selected_total_pct"] = slopes["total_slope_pct"][mask]
    slopes["selected_cross_pct"] = slopes["cross_slope_pct"][mask]
    slopes["selected_long_pct"] = slopes["long_slope_pct"][mask]

    areas = mesh_data["areas"][mask]
    total_area = float(areas.sum())
    if total_area > 0:
        slopes["area_weighted_cross_pct"] = float(
            (slopes["selected_cross_pct"] * areas).sum() / total_area
        )
        slopes["area_weighted_long_pct"] = float(
            (slopes["selected_long_pct"] * areas).sum() / total_area
        )
    else:
        slopes["area_weighted_cross_pct"] = 0.0
        slopes["area_weighted_long_pct"] = 0.0

    slopes["min_cross_pct"] = float(slopes["selected_cross_pct"].min())
    slopes["max_cross_pct"] = float(slopes["selected_cross_pct"].max())
    slopes["min_long_pct"] = float(slopes["selected_long_pct"].min())
    slopes["max_long_pct"] = float(slopes["selected_long_pct"].max())

    return slopes


# ── PyVista 3D Heatmap ────────────────────────────────────────────

def plot_slope_heatmap(
    mesh_data: dict,
    slopes: dict,
    mode: str = "cross",
    title: str = None,
    clim: tuple = None,
    screenshot: str = None,
    show: bool = True,
) -> None:
    """Render interactive 3D slope heatmap using PyVista.

    Args:
        mesh_data: dict with vertices, faces.
        slopes: dict from compute_triangle_slopes or compute_surface_slopes.
        mode: "total", "cross", or "long" — which slope component to show.
        title: plot title.
        clim: color limits (min_pct, max_pct). Default: (0, 5) for cross-slope.
        screenshot: if provided, save image to this path.
        show: if True, display interactive window.
    """
    if pv is None:
        raise ImportError("PyVista is required for 3D heatmap: pip install pyvista")

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]

    # Select slope component
    key = f"{mode}_slope_pct"
    if key not in slopes:
        raise ValueError(f"Unknown mode '{mode}'. Use 'total', 'cross', or 'long'.")
    values = slopes[key]

    # Clamp vertical faces
    values = np.clip(values, 0, 100)

    # Apply face mask if available (only color selected faces)
    mask = slopes.get("face_mask")

    # Build PyVista mesh
    n_faces = len(faces)
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=int),
        faces,
    ]).ravel()
    mesh_pv = pv.PolyData(vertices, pv_faces)
    mesh_pv.cell_data["slope_pct"] = values

    if mask is not None:
        mesh_pv.cell_data["selected"] = mask.astype(float)

    # Color map
    if clim is None:
        if mode == "cross":
            clim = (0, 6)
        elif mode == "long":
            clim = (0, 10)
        else:
            clim = (0, 10)

    mode_labels = {
        "total": "Gesamtneigung",
        "cross": "Quergefälle",
        "long": "Längsgefälle",
    }

    if title is None:
        title = f"Slope Heatmap — {mode_labels.get(mode, mode)} (%)"

    pl = pv.Plotter()
    pl.add_mesh(
        mesh_pv,
        scalars="slope_pct",
        cmap="RdYlGn_r",  # Green=flat → Red=steep
        clim=clim,
        scalar_bar_args={
            "title": f"{mode_labels.get(mode, mode)} (%)",
            "fmt": "%.1f",
        },
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
    )

    # Add ASTRA threshold lines as annotation
    if mode == "cross":
        pl.add_text("ASTRA max: 5%", position="upper_right", font_size=10)

    pl.add_text(title, position="upper_left", font_size=12)
    pl.add_axes()
    pl.set_background("white")

    if screenshot:
        pl.screenshot(screenshot, transparent_background=False)

    if show:
        pl.show()
    else:
        pl.close()


# ── Matplotlib 2D profile plot ────────────────────────────────────

def plot_slope_profile(
    mesh_data: dict,
    slopes: dict,
    mode: str = "cross",
    axis_label: str = "Position entlang Achse (m)",
    output: str = None,
    show: bool = True,
) -> None:
    """Plot slope values along the wall/road axis as a 2D profile.

    Shows per-triangle slope vs position along the axis, with ASTRA
    threshold lines for reference.
    """
    if plt is None:
        raise ImportError("Matplotlib required: pip install matplotlib")

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    ax_dir = slopes["axis"]

    key = f"{mode}_slope_pct"
    values = slopes[key]

    mask = slopes.get("face_mask")
    if mask is not None:
        sel_faces = faces[mask]
        sel_values = values[mask]
    else:
        sel_faces = faces
        sel_values = values

    # Compute face centroids and project onto axis
    v0 = vertices[sel_faces[:, 0]]
    v1 = vertices[sel_faces[:, 1]]
    v2 = vertices[sel_faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    positions = centroids @ ax_dir

    mode_labels = {"total": "Gesamtneigung", "cross": "Quergefälle", "long": "Längsgefälle"}

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(positions, sel_values, s=8, alpha=0.7, c=sel_values,
               cmap="RdYlGn_r", vmin=0, vmax=6)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(f"{mode_labels.get(mode, mode)} (%)")
    ax.set_title(f"Neigungsprofil — {mode_labels.get(mode, mode)}")

    # ASTRA threshold
    if mode == "cross":
        ax.axhline(y=5.0, color="red", linestyle="--", linewidth=1, label="ASTRA max 5%")
        ax.axhline(y=3.0, color="orange", linestyle="--", linewidth=1, label="Empfohlen 3%")
        ax.legend()

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
