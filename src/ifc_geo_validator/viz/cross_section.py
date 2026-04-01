"""Cross-section profile extraction and visualization.

Slices the mesh at a given position along the centerline and produces
a 2D polygon outline of the wall profile with dimension annotations.

Algorithm:
  1. Project mesh vertices onto the local tangent at the slice position
  2. Select vertices within ±tolerance of the slice plane
  3. Project selected vertices onto the local (normal, Z) plane
  4. Compute convex hull → cross-section outline
  5. Annotate with crown width, wall thickness, height

References:
  - de Berg, M. et al. (2008). Computational Geometry, Ch. 3 (convex hull).
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    plt = None


def extract_cross_section(
    mesh_data: dict,
    centerline,
    position_fraction: float = 0.5,
    tolerance: float = None,
) -> dict | None:
    """Extract a 2D cross-section at a position along the centerline.

    Args:
        mesh_data: dict with vertices, faces.
        centerline: WallCenterline with local frames.
        position_fraction: 0.0=start, 0.5=middle, 1.0=end of wall.
        tolerance: slice thickness in meters. If None, auto-computed
                   from centerline point spacing.

    Returns:
        dict with:
            outline_2d: (N, 2) array — cross-section polygon in (perp, Z)
            position_m: float — position along centerline in meters
            local_tangent: (3,) — tangent at slice position
            local_normal: (3,) — normal at slice position
            width_mm: float — horizontal extent of section
            height_m: float — vertical extent of section
            n_vertices: int — number of vertices in slice
        Or None if not enough vertices found.
    """
    if centerline is None or not hasattr(centerline, "points_2d"):
        return None

    vertices = mesh_data["vertices"]
    n_pts = len(centerline.points_2d)

    # Find the centerline point at the requested fraction
    idx = int(np.clip(position_fraction * (n_pts - 1), 0, n_pts - 1))
    slice_pt = centerline.points_2d[idx]
    tangent = centerline.tangents[idx]
    normal = centerline.normals[idx]

    # Auto-compute tolerance from centerline spacing
    if tolerance is None:
        if n_pts > 1:
            diffs = np.diff(centerline.points_2d, axis=0)
            spacing = np.median(np.linalg.norm(diffs, axis=1))
            tolerance = float(spacing) * 0.6
        else:
            tolerance = 0.5

    # Project all vertices onto tangent direction relative to slice point
    tang_2d = tangent[:2]
    tang_mag = np.linalg.norm(tang_2d)
    if tang_mag < 1e-10:
        return None
    tang_2d = tang_2d / tang_mag

    verts_xy = vertices[:, :2]
    t_proj = (verts_xy - slice_pt) @ tang_2d

    # Select vertices within tolerance of slice plane
    mask = np.abs(t_proj) < tolerance
    if mask.sum() < 3:
        return None

    selected = vertices[mask]

    # Project onto local (normal, Z) plane
    norm_2d = normal[:2]
    norm_mag = np.linalg.norm(norm_2d)
    if norm_mag < 1e-10:
        return None
    norm_2d = norm_2d / norm_mag

    perp_proj = (selected[:, :2] - slice_pt) @ norm_2d
    z_vals = selected[:, 2]

    # 2D points in (perpendicular, Z) space
    pts_2d = np.column_stack([perp_proj, z_vals])

    # Convex hull for clean outline
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_2d)
        outline = pts_2d[hull.vertices]
        # Close the polygon
        outline = np.vstack([outline, outline[0]])
    except Exception:
        # Fallback: sort by angle from centroid
        center = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - center[1], pts_2d[:, 0] - center[0])
        order = np.argsort(angles)
        outline = pts_2d[order]
        outline = np.vstack([outline, outline[0]])

    # Compute dimensions
    width_m = float(perp_proj.max() - perp_proj.min())
    height_m = float(z_vals.max() - z_vals.min())

    # Position along centerline in meters
    if idx > 0:
        cum_dists = np.cumsum(np.linalg.norm(
            np.diff(centerline.points_2d[:idx + 1], axis=0), axis=1
        ))
        position_m = float(cum_dists[-1]) if len(cum_dists) > 0 else 0.0
    else:
        position_m = 0.0

    return {
        "outline_2d": outline,
        "points_2d": pts_2d,
        "position_m": position_m,
        "position_fraction": position_fraction,
        "local_tangent": tangent,
        "local_normal": normal,
        "width_mm": width_m * 1000,
        "height_m": height_m,
        "z_min": float(z_vals.min()),
        "z_max": float(z_vals.max()),
        "perp_min": float(perp_proj.min()),
        "perp_max": float(perp_proj.max()),
        "n_vertices": int(mask.sum()),
    }


def plot_cross_section(
    section: dict,
    title: str = None,
    show_dims: bool = True,
    output: str = None,
    show: bool = True,
) -> None:
    """Plot a 2D cross-section profile with dimensions.

    Args:
        section: dict from extract_cross_section().
        title: plot title.
        show_dims: annotate width and height dimensions.
        output: save figure to this path.
        show: display interactive window.
    """
    if plt is None:
        raise ImportError("Matplotlib required: pip install matplotlib")

    outline = section["outline_2d"]
    pts = section["points_2d"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot outline
    ax.fill(outline[:, 0], outline[:, 1], alpha=0.3, color="#2196F3", label="Querschnitt")
    ax.plot(outline[:, 0], outline[:, 1], color="#1565C0", linewidth=2)

    # Plot individual vertices
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#F44336", alpha=0.5, zorder=5)

    # Dimension annotations
    if show_dims:
        w_mm = section["width_mm"]
        h_m = section["height_m"]
        p_min = section["perp_min"]
        p_max = section["perp_max"]
        z_min = section["z_min"]
        z_max = section["z_max"]

        # Width annotation (horizontal arrow at top)
        y_arrow = z_max + h_m * 0.05
        ax.annotate("", xy=(p_max, y_arrow), xytext=(p_min, y_arrow),
                     arrowprops=dict(arrowstyle="<->", color="#E65100", lw=1.5))
        ax.text((p_min + p_max) / 2, y_arrow + h_m * 0.03,
                f"{w_mm:.0f} mm", ha="center", fontsize=10, color="#E65100", fontweight="bold")

        # Height annotation (vertical arrow on right)
        x_arrow = p_max + (p_max - p_min) * 0.08
        ax.annotate("", xy=(x_arrow, z_max), xytext=(x_arrow, z_min),
                     arrowprops=dict(arrowstyle="<->", color="#1B5E20", lw=1.5))
        ax.text(x_arrow + (p_max - p_min) * 0.03, (z_min + z_max) / 2,
                f"{h_m:.2f} m", ha="left", va="center", fontsize=10,
                color="#1B5E20", fontweight="bold", rotation=90)

    # Labels
    ax.set_xlabel("Querrichtung (m)", fontsize=11)
    ax.set_ylabel("Höhe (m)", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    pos_m = section["position_m"]
    frac = section["position_fraction"]
    if title is None:
        title = f"Querschnitt bei {pos_m:.1f} m ({frac:.0%})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Info box
    info = (f"Breite: {section['width_mm']:.0f} mm\n"
            f"Höhe: {section['height_m']:.2f} m\n"
            f"Vertices: {section['n_vertices']}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_sections(
    mesh_data: dict,
    centerline,
    n_sections: int = 5,
    title: str = None,
    output: str = None,
    show: bool = True,
) -> None:
    """Plot multiple cross-sections along the wall in a single figure."""
    if plt is None:
        raise ImportError("Matplotlib required: pip install matplotlib")

    fractions = np.linspace(0.1, 0.9, n_sections)
    sections = []
    for f in fractions:
        s = extract_cross_section(mesh_data, centerline, position_fraction=f)
        if s is not None:
            sections.append(s)

    if not sections:
        return

    fig, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 6),
                              sharey=True)
    if len(sections) == 1:
        axes = [axes]

    for ax, s in zip(axes, sections):
        outline = s["outline_2d"]
        ax.fill(outline[:, 0], outline[:, 1], alpha=0.3, color="#2196F3")
        ax.plot(outline[:, 0], outline[:, 1], color="#1565C0", linewidth=1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{s['position_m']:.1f} m\n{s['width_mm']:.0f}×{s['height_m']*1000:.0f} mm",
                      fontsize=9)
        ax.set_xlabel("Quer (m)", fontsize=8)

    axes[0].set_ylabel("Höhe (m)", fontsize=9)

    if title is None:
        title = f"Querschnitte entlang Centerline ({len(sections)} Schnitte)"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
