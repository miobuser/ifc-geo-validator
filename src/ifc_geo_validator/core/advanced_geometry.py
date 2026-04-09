"""Advanced geometric analysis for infrastructure elements.

Provides measurements beyond basic L1-L3: profile variation, planarity,
taper analysis, plumbness, overlap detection, and per-slice profiling.

All algorithms are vectorized numpy and operate on triangulated meshes.

References:
  - Botsch et al. (2010). Polygon Mesh Processing, §4 (planarity).
  - Ericson, C. (2004). Real-Time Collision Detection, §4 (AABB overlap).
"""

import numpy as np


# ── Wall Taper Profile ────────────────────────────────────────────

def compute_taper_profile(mesh_data: dict, face_groups: list,
                          wall_axis: np.ndarray, n_slices: int = 10) -> dict:
    """Compute how wall thickness varies with height (taper profile).

    Slices the wall at n_slices heights between foundation and crown,
    measuring the perpendicular extent (thickness) at each height.

    For a wall with 10:1 inclination (Anzug):
      thickness(z) = crown_thickness + (z_crown - z) / ratio

    Returns:
        dict with:
            heights_m:     (N,) array of slice heights above foundation
            thickness_mm:  (N,) array of thickness at each height
            taper_ratio:   float — Δthickness / Δheight (e.g. 10.0 for 10:1)
            is_tapered:    bool — True if thickness varies > 5%
            min_thickness_mm: float
            max_thickness_mm: float
    """
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]

    # Collect front and back face vertices
    front_verts = _collect_face_vertices(vertices, faces, face_groups, "front")
    back_verts = _collect_face_vertices(vertices, faces, face_groups, "back")

    if len(front_verts) == 0 or len(back_verts) == 0:
        return {"heights_m": np.array([]), "thickness_mm": np.array([]),
                "taper_ratio": float("inf"), "is_tapered": False,
                "min_thickness_mm": 0, "max_thickness_mm": 0}

    # Perpendicular axis
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.asarray(wall_axis, dtype=float)
    axis[2] = 0
    ax_mag = np.linalg.norm(axis)
    if ax_mag > 1e-10:
        axis /= ax_mag
    perp = np.array([-axis[1], axis[0], 0.0])

    # Z range from all vertical faces
    all_z = np.concatenate([front_verts[:, 2], back_verts[:, 2]])
    z_min, z_max = float(all_z.min()), float(all_z.max())
    height = z_max - z_min

    if height < 1e-6:
        return {"heights_m": np.array([0]), "thickness_mm": np.array([0]),
                "taper_ratio": float("inf"), "is_tapered": False,
                "min_thickness_mm": 0, "max_thickness_mm": 0}

    # Slice at regular heights
    slice_z = np.linspace(z_min + height * 0.05, z_max - height * 0.05, n_slices)
    dz = height / n_slices * 0.6  # slice tolerance

    thicknesses = []
    heights = []

    for z in slice_z:
        f_mask = np.abs(front_verts[:, 2] - z) < dz
        b_mask = np.abs(back_verts[:, 2] - z) < dz

        if f_mask.sum() < 1 or b_mask.sum() < 1:
            continue

        f_proj = front_verts[f_mask] @ perp
        b_proj = back_verts[b_mask] @ perp

        f_med = float(np.median(f_proj))
        b_med = float(np.median(b_proj))
        th = abs(f_med - b_med)

        thicknesses.append(th)
        heights.append(z - z_min)

    if not thicknesses:
        return {"heights_m": np.array([]), "thickness_mm": np.array([]),
                "taper_ratio": float("inf"), "is_tapered": False,
                "min_thickness_mm": 0, "max_thickness_mm": 0}

    th_arr = np.array(thicknesses) * 1000  # to mm
    h_arr = np.array(heights)

    # Taper ratio: linear fit thickness = a*h + b → ratio = 1/|a|
    if len(h_arr) >= 2:
        # Linear regression: thickness = slope * height + intercept
        coeffs = np.polyfit(h_arr, th_arr, 1)
        slope_mm_per_m = coeffs[0]  # mm per m height
        if abs(slope_mm_per_m) > 0.1:
            taper_ratio = abs(1000.0 / slope_mm_per_m)  # m per m → ratio
        else:
            taper_ratio = float("inf")
    else:
        taper_ratio = float("inf")

    is_tapered = (th_arr.max() - th_arr.min()) / max(th_arr.mean(), 1e-6) > 0.05

    return {
        "heights_m": h_arr,
        "thickness_mm": th_arr,
        "taper_ratio": round(taper_ratio, 1),
        "is_tapered": bool(is_tapered),
        "min_thickness_mm": round(float(th_arr.min()), 1),
        "max_thickness_mm": round(float(th_arr.max()), 1),
    }


# ── Surface Planarity ────────────────────────────────────────────

def compute_planarity(mesh_data: dict, face_groups: list,
                      category: str = "front") -> dict:
    """Measure how planar a face group is (deviation from best-fit plane).

    Uses least-squares plane fitting: minimize Σ (n·pᵢ - d)².
    The RMS residual quantifies the planarity.

    Returns:
        dict with:
            rms_deviation_mm: float — RMS distance from best-fit plane
            max_deviation_mm: float — maximum distance from plane
            plane_normal:     (3,) — unit normal of best-fit plane
            is_planar:        bool — True if RMS < 1mm (practical threshold)

    Reference: Botsch et al. (2010). Polygon Mesh Processing, §4.2.
    """
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    verts = _collect_face_vertices(vertices, faces, face_groups, category)

    if len(verts) < 3:
        return {"rms_deviation_mm": 0, "max_deviation_mm": 0,
                "plane_normal": [0, 0, 0], "is_planar": True}

    # Centroid
    centroid = verts.mean(axis=0)
    centered = verts - centroid

    # Best-fit plane via SVD (smallest singular value = plane normal)
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # last row = direction of least variance

    # Distance from plane: d = (p - centroid) · normal
    dists = centered @ normal
    rms = float(np.sqrt((dists ** 2).mean()))
    max_dev = float(np.abs(dists).max())

    return {
        "rms_deviation_mm": round(rms * 1000, 2),
        "max_deviation_mm": round(max_dev * 1000, 2),
        "plane_normal": normal.tolist(),
        "is_planar": rms * 1000 < 1.0,
    }


# ── Overlap / Collision Detection ─────────────────────────────────

def check_overlap(mesh_a: dict, mesh_b: dict) -> dict:
    """Check if two meshes overlap (AABB + vertex-in-mesh test).

    Phase 1: AABB overlap test (fast rejection).
    Phase 2: Sample vertices of A, check if any are inside B's
             bounding box (approximation for convex elements).

    Returns:
        dict with:
            aabb_overlap: bool — bounding boxes overlap
            n_penetrating: int — estimated penetrating vertices
            overlap_volume_m3: float — estimated overlap (AABB intersection)
            clear: bool — no overlap detected

    Reference: Ericson (2004). Real-Time Collision Detection, §4.
    """
    va, vb = mesh_a["vertices"], mesh_b["vertices"]

    min_a, max_a = va.min(axis=0), va.max(axis=0)
    min_b, max_b = vb.min(axis=0), vb.max(axis=0)

    # AABB overlap test
    overlap_per_axis = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    aabb_overlap = bool(np.all(overlap_per_axis > 0))

    if not aabb_overlap:
        return {"aabb_overlap": False, "n_penetrating": 0,
                "overlap_volume_m3": 0.0, "clear": True}

    # Overlap volume (AABB intersection)
    overlap_size = np.clip(overlap_per_axis, 0, None)
    overlap_vol = float(np.prod(overlap_size))

    # Vertex penetration test (check if A's vertices are inside B's AABB)
    inside_b = np.all((va >= min_b) & (va <= max_b), axis=1)
    n_pen = int(inside_b.sum())

    return {
        "aabb_overlap": True,
        "n_penetrating": n_pen,
        "overlap_volume_m3": round(overlap_vol, 4),
        "clear": n_pen == 0,
    }


# ── Profile Variation Along Centerline ────────────────────────────

def compute_profile_variation(mesh_data: dict, centerline,
                              face_groups: list, n_slices: int = 15) -> dict:
    """Measure how the cross-section profile varies along the centerline.

    At each slice: measures width (crown), height (vertical extent),
    and area (cross-section polygon area via shoelace formula).

    Returns:
        dict with:
            positions_m:  (N,) — position along centerline
            widths_mm:    (N,) — crown width at each position
            heights_m:    (N,) — wall height at each position
            width_cv:     float — coefficient of variation of width
            height_cv:    float — coefficient of variation of height
            is_variable:  bool — True if CV > 5%
    """
    if centerline is None or not hasattr(centerline, "points_2d"):
        return None

    vertices = mesh_data["vertices"]
    n_pts = len(centerline.points_2d)

    fractions = np.linspace(0.05, 0.95, n_slices)
    indices = np.clip((fractions * (n_pts - 1)).astype(int), 0, n_pts - 1)

    # Slice tolerance
    if n_pts > 1:
        diffs = np.diff(centerline.points_2d, axis=0)
        tol = float(np.median(np.linalg.norm(diffs, axis=1))) * 0.6
    else:
        tol = 0.5

    positions = []
    widths = []
    heights = []

    for idx in indices:
        pt = centerline.points_2d[idx]
        tangent = centerline.tangents[idx]
        normal = centerline.normals[idx]

        tang_2d = tangent[:2]
        tang_mag = np.linalg.norm(tang_2d)
        if tang_mag < 1e-10:
            continue
        tang_2d = tang_2d / tang_mag

        # Select vertices near this slice
        t_proj = (vertices[:, :2] - pt) @ tang_2d
        mask = np.abs(t_proj) < tol
        if mask.sum() < 3:
            continue

        selected = vertices[mask]

        # Width: extent perpendicular to wall axis
        norm_2d = normal[:2]
        norm_mag = np.linalg.norm(norm_2d)
        if norm_mag < 1e-10:
            continue
        norm_2d = norm_2d / norm_mag
        perp_proj = (selected[:, :2] - pt) @ norm_2d
        width = float(perp_proj.max() - perp_proj.min())

        # Height: Z extent
        height = float(selected[:, 2].max() - selected[:, 2].min())

        # Position along centerline
        if idx > 0:
            cum = float(np.linalg.norm(
                np.diff(centerline.points_2d[:idx + 1], axis=0), axis=1
            ).sum())
        else:
            cum = 0.0

        positions.append(cum)
        widths.append(width * 1000)  # mm
        heights.append(height)

    if not widths:
        return None

    w_arr = np.array(widths)
    h_arr = np.array(heights)
    p_arr = np.array(positions)

    w_cv = float(w_arr.std() / w_arr.mean()) if w_arr.mean() > 0 else 0
    h_cv = float(h_arr.std() / h_arr.mean()) if h_arr.mean() > 0 else 0

    return {
        "positions_m": p_arr,
        "widths_mm": w_arr,
        "heights_m": h_arr,
        "width_cv": round(w_cv, 4),
        "height_cv": round(h_cv, 4),
        "width_min_mm": round(float(w_arr.min()), 1),
        "width_max_mm": round(float(w_arr.max()), 1),
        "height_min_m": round(float(h_arr.min()), 3),
        "height_max_m": round(float(h_arr.max()), 3),
        "is_variable": w_cv > 0.05 or h_cv > 0.05,
    }


# ── Plumbness Check ──────────────────────────────────────────────

def check_plumbness(face_groups: list) -> dict:
    """Check how close vertical faces are to true vertical.

    Computes the angle between each front/back face normal and the
    horizontal plane. A perfectly vertical face has angle = 0°.

    Returns:
        dict with:
            front_plumbness_deg: float — angle from vertical (0° = plumb)
            back_plumbness_deg:  float — angle from vertical
            max_deviation_deg:   float — worst case
            is_plumb:            bool — True if deviation < 0.5°
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    results = {}

    for category in ["front", "back"]:
        groups = [g for g in face_groups if g.get("category") == category]
        if not groups:
            results[f"{category}_plumbness_deg"] = None
            continue

        # Area-weighted average normal
        total_area = sum(g["area"] for g in groups)
        if total_area <= 0:
            results[f"{category}_plumbness_deg"] = None
            continue

        avg_normal = np.zeros(3)
        for g in groups:
            avg_normal += np.array(g["normal"]) * g["area"]
        avg_normal /= total_area
        mag = np.linalg.norm(avg_normal)
        if mag > 0:
            avg_normal /= mag

        # Angle from vertical: asin(|n·z|)
        z_comp = abs(float(np.dot(avg_normal, z_axis)))
        angle_deg = float(np.degrees(np.arcsin(np.clip(z_comp, 0, 1))))
        results[f"{category}_plumbness_deg"] = round(angle_deg, 3)

    front = results.get("front_plumbness_deg")
    back = results.get("back_plumbness_deg")
    vals = [v for v in [front, back] if v is not None]
    max_dev = max(vals) if vals else 0.0

    results["max_deviation_deg"] = round(max_dev, 3)
    results["is_plumb"] = max_dev < 0.5

    return results


# ── Helpers ───────────────────────────────────────────────────────

def _collect_face_vertices(vertices, faces, face_groups, category):
    """Collect unique vertices from faces of a given category."""
    all_idx = []
    for g in face_groups:
        if g.get("category") == category:
            for fi in g["face_indices"]:
                all_idx.extend(faces[fi].tolist())
    if not all_idx:
        return np.array([]).reshape(0, 3)
    return vertices[list(set(all_idx))]
