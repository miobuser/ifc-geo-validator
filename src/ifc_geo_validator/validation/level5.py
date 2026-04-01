"""Level 5 Validation: Inter-element geometric context.

Analyzes geometric relationships between multiple IFC elements within
the same spatial structure (e.g. wall stem + foundation slab):
  - Pair detection via bounding box proximity
  - Pair classification via Contact Surface Normal Analysis
  - Gap measurement between adjacent elements
  - Foundation overhang check

Pair classification algorithm:
  1. Compute contact tolerance ε from mesh resolution (median edge length)
  2. Find proximity face pairs (centroid distance < ε)
  3. Compute area-weighted contact normal from proximity faces
  4. Classify using the verticality index κ = |N · e_z| / ||N||
     - κ > cos(π/4) ≈ 0.707 → stacked (contact surface horizontal)
     - κ ≤ cos(π/4)          → side_by_side (contact surface vertical)

The cos(π/4) threshold is not a heuristic — it is the geometric equator
where the angle to vertical equals the angle to horizontal (45°).

References:
  - Ericson, C. (2004). Real-Time Collision Detection. Morgan Kaufmann.
  - de Berg, M. et al. (2008). Computational Geometry. Springer.
"""

import numpy as np
from itertools import combinations

# Geometric constant: cos(45°) = √2/2 — the natural boundary between
# "more vertical than horizontal" and vice versa.
COS_45 = np.sqrt(2.0) / 2.0  # ≈ 0.7071
Z_AXIS = np.array([0.0, 0.0, 1.0])


def validate_level5(elements_data: list[dict]) -> dict:
    """Evaluate geometric relationships between elements.

    Args:
        elements_data: list of per-element dicts, each containing:
            element_id, element_name, level1 (with bbox), mesh_data

    Returns:
        dict with:
            pairs: list of pair analysis dicts
            summary: aggregate statistics
    """
    if len(elements_data) < 2:
        return {"pairs": [], "summary": {"num_pairs": 0, "num_elements": len(elements_data)}}

    # Extract element info
    elem_info = []
    for e in elements_data:
        l1 = e.get("level1", {})
        bbox = l1.get("bbox", {})
        mesh = e.get("mesh_data")
        if not bbox or mesh is None:
            continue
        elem_info.append({
            "id": e.get("element_id"),
            "name": e.get("element_name", "Unnamed"),
            "bbox_min": np.array(bbox.get("min", [0, 0, 0])),
            "bbox_max": np.array(bbox.get("max", [0, 0, 0])),
            "bbox_size": np.array(bbox.get("size", [0, 0, 0])),
            "volume": l1.get("volume", 0),
            "centroid": np.array(l1.get("centroid", [0, 0, 0])),
            "mesh_data": mesh,
        })

    # Find spatially close pairs
    pairs = []
    for i, j in combinations(range(len(elem_info)), 2):
        a, b = elem_info[i], elem_info[j]
        pair = _analyze_pair(a, b)
        if pair is not None:
            pairs.append(pair)

    # Build summary
    stacked = [p for p in pairs if p["pair_type"] == "stacked"]
    side_by_side = [p for p in pairs if p["pair_type"] == "side_by_side"]

    return {
        "pairs": pairs,
        "summary": {
            "num_elements": len(elem_info),
            "num_pairs": len(pairs),
            "num_stacked": len(stacked),
            "num_side_by_side": len(side_by_side),
        },
    }


# ── Pair analysis ──────────────────────────────────────────────────

def _analyze_pair(a: dict, b: dict) -> dict | None:
    """Analyze the geometric relationship between two elements."""
    # Quick rejection: AABB gap > 1m
    gap_3d = _bbox_gap(a["bbox_min"], a["bbox_max"], b["bbox_min"], b["bbox_max"])
    if max(gap_3d) > 1.0:
        return None

    # Classify using Contact Surface Normal Analysis
    pair_type, upper, lower = _classify_pair_by_contact_normal(a, b)

    result = {
        "element_a_id": a["id"],
        "element_a_name": a["name"],
        "element_b_id": b["id"],
        "element_b_name": b["name"],
        "pair_type": pair_type,
        "bbox_gap_xyz_m": [round(float(g), 4) for g in gap_3d],
    }

    if pair_type == "stacked":
        result["upper_name"] = upper["name"]
        result["lower_name"] = lower["name"]

        z_gap_m = float(upper["bbox_min"][2] - lower["bbox_max"][2])
        result["vertical_gap_mm"] = round(z_gap_m * 1000.0, 1)

        overhang = _compute_overhang(upper, lower)
        result["foundation_extends_beyond_wall"] = overhang["extends"]
        result["overhang_mm"] = overhang["overhang_mm"]

        upper_center_xy = (upper["bbox_min"][:2] + upper["bbox_max"][:2]) / 2
        lower_center_xy = (lower["bbox_min"][:2] + lower["bbox_max"][:2]) / 2
        offset = float(np.linalg.norm(upper_center_xy - lower_center_xy))
        result["center_offset_mm"] = round(offset * 1000.0, 1)

    elif pair_type == "side_by_side":
        min_dist = _min_bbox_distance_xy(a, b)
        result["horizontal_gap_mm"] = round(min_dist * 1000.0, 1)

    return result


# ── Contact Surface Normal Classification ──────────────────────────

def _classify_pair_by_contact_normal(a, b):
    """Classify pair using Contact Surface Normal Analysis.

    Algorithm:
      1. Compute contact tolerance ε from median edge length
      2. Find proximity face pairs (centroid distance < ε)
      3. Compute area-weighted contact normal N
      4. Verticality index κ = |N · e_z| / ||N||
      5. κ > cos(π/4) → stacked, else → side_by_side

    Fallback: If no proximity faces found, use centroid displacement.
    """
    mesh_a = a["mesh_data"]
    mesh_b = b["mesh_data"]

    # Step 1: Contact tolerance from mesh resolution
    epsilon = _compute_contact_tolerance(mesh_a, mesh_b)

    # Step 2: Find proximity face pairs
    centroids_a = _face_centroids(mesh_a["vertices"], mesh_a["faces"])
    centroids_b = _face_centroids(mesh_b["vertices"], mesh_b["faces"])
    proximity = _find_proximity_pairs(centroids_a, centroids_b, epsilon)

    # Step 3 & 4: Compute contact normal and verticality index
    if len(proximity) > 0:
        kappa = _compute_contact_kappa(
            proximity, mesh_a["normals"], mesh_a["areas"],
            mesh_b["normals"], mesh_b["areas"]
        )
    else:
        # Fallback: centroid displacement direction
        disp = a["centroid"] - b["centroid"]
        d_norm = np.linalg.norm(disp)
        if d_norm < 1e-10:
            # Identical centroids — geometrically underdetermined.
            # Compare Z-extents: if both elements span similar Z-ranges,
            # they are side-by-side; if Z-ranges are disjoint, stacked.
            a_z = (float(a["bbox_min"][2]), float(a["bbox_max"][2]))
            b_z = (float(b["bbox_min"][2]), float(b["bbox_max"][2]))
            z_overlap = max(0, min(a_z[1], b_z[1]) - max(a_z[0], b_z[0]))
            min_h = min(a_z[1] - a_z[0], b_z[1] - b_z[0])
            kappa = 0.0 if (min_h > 0 and z_overlap / min_h > 0.5) else 1.0
        else:
            kappa = abs(disp[2]) / d_norm

    # Step 5: Classify
    if kappa > COS_45:
        # Contact surface is predominantly horizontal → stacked
        if a["centroid"][2] > b["centroid"][2]:
            return "stacked", a, b  # a on top of b
        else:
            return "stacked", b, a  # b on top of a
    else:
        return "side_by_side", None, None


def _compute_contact_tolerance(mesh_a, mesh_b):
    """Derive contact tolerance from element geometry.

    For two retaining wall elements, the contact interface lies on a shared
    boundary plane. Face centroids on the contact side of each element are
    at most t/2 from this plane (where t is the element's thickness =
    smallest bounding box dimension). Two face centroids that belong to
    faces in actual contact are therefore separated by at most:

        ε = (t_A + t_B) / 2

    However, for coarse meshes (e.g. a single quad per face), face centroids
    may lie far from the contact region along the non-thickness axes. To
    prevent false positives, we use the tighter bound:

        ε = min(t_A, t_B) / 2

    This ensures the tolerance does not exceed half the thinner element's
    cross-section — geometrically, a face centroid on the contact side
    cannot be farther from the contact plane than half the element's
    thickness.

    Note: This is a centroid-based approximation. For production use with
    very coarse meshes, point-to-face distance would be more precise
    (Ericson 2004, Ch. 5).
    """
    def smallest_dim(mesh):
        verts = mesh["vertices"]
        if len(verts) == 0:
            return 1.0
        bbox_size = verts.max(axis=0) - verts.min(axis=0)
        valid = bbox_size[bbox_size > 1e-6]
        return float(valid.min()) if len(valid) > 0 else 1.0

    t_a = smallest_dim(mesh_a)
    t_b = smallest_dim(mesh_b)

    return min(t_a, t_b) / 2.0


def _find_proximity_pairs(centroids_a, centroids_b, epsilon):
    """Find face pairs with centroid distance < ε.

    Returns list of (idx_a, idx_b, distance) tuples.
    Complexity: O(M_A × M_B) — trivial for retaining wall meshes.
    """
    pairs = []
    for i, ca in enumerate(centroids_a):
        dists = np.linalg.norm(centroids_b - ca, axis=1)
        close = np.where(dists < epsilon)[0]
        for j in close:
            pairs.append((i, int(j), float(dists[j])))
    return pairs


def _compute_contact_kappa(proximity_pairs, normals_a, areas_a,
                           normals_b, areas_b):
    """Compute verticality index κ of the contact surface.

    κ = |N · e_z| / ||N||

    where N = Σ min(area_a[i], area_b[j]) · normal_a[i]
    for each proximity pair (i, j).

    κ close to 1.0: contact surface is horizontal → stacked
    κ close to 0.0: contact surface is vertical → side_by_side
    """
    weighted_sum = np.zeros(3)
    for idx_a, idx_b, dist in proximity_pairs:
        a_eff = min(float(areas_a[idx_a]), float(areas_b[idx_b]))
        weighted_sum += a_eff * normals_a[idx_a]

    magnitude = np.linalg.norm(weighted_sum)
    if magnitude < 1e-10:
        return 1.0  # degenerate: assume stacked

    n_contact = weighted_sum / magnitude
    return abs(float(np.dot(n_contact, Z_AXIS)))


# ── Geometric helpers ──────────────────────────────────────────────

def _face_centroids(vertices, faces):
    """Compute centroid of each triangle. Returns (M, 3) array."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return (v0 + v1 + v2) / 3.0


def _median_edge_length(vertices, faces):
    """Compute median edge length of the mesh."""
    edges = []
    for tri in faces:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            edges.append(np.linalg.norm(vertices[a] - vertices[b]))
    return float(np.median(edges)) if edges else 1.0


def _bbox_gap(min_a, max_a, min_b, max_b):
    """Compute gap between two AABBs per axis (0 if overlapping)."""
    gap = np.zeros(3)
    for i in range(3):
        if max_a[i] < min_b[i]:
            gap[i] = min_b[i] - max_a[i]
        elif max_b[i] < min_a[i]:
            gap[i] = min_a[i] - max_b[i]
    return gap


def _compute_overhang(upper, lower):
    """Check if lower element extends beyond upper in XY.

    Overhang = how much the lower element protrudes beyond the upper
    on each side. Positive = lower extends beyond, negative = doesn't.
    """
    u_min = upper["bbox_min"][:2]
    u_max = upper["bbox_max"][:2]
    l_min = lower["bbox_min"][:2]
    l_max = lower["bbox_max"][:2]

    # Scale-relative tolerance (1e-6 of max extent, not hardcoded mm)
    extent = max(float(np.linalg.norm(u_max - u_min)),
                 float(np.linalg.norm(l_max - l_min)), 1e-6)
    tol = extent * 1e-6

    extends_min = np.all(l_min <= u_min + tol)
    extends_max = np.all(l_max >= u_max - tol)
    extends = bool(extends_min and extends_max)

    # Overhang per side: how much lower extends beyond upper (positive = good)
    overhangs = np.concatenate([u_min - l_min, l_max - u_max])
    min_overhang = max(float(overhangs.min()) * 1000.0, 0.0)  # clamp ≥ 0

    return {"extends": extends, "overhang_mm": round(min_overhang, 1)}


def _min_bbox_distance_xy(a, b):
    """Minimum XY distance between two bounding boxes."""
    gap_x = max(0, max(a["bbox_min"][0], b["bbox_min"][0]) -
                    min(a["bbox_max"][0], b["bbox_max"][0]))
    gap_y = max(0, max(a["bbox_min"][1], b["bbox_min"][1]) -
                    min(a["bbox_max"][1], b["bbox_max"][1]))
    return float(max(gap_x, 0) + max(gap_y, 0))
