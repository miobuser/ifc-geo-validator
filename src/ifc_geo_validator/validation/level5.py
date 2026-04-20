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

# Default pair-candidacy thresholds. The 1.0 m cutoff rejects AABB pairs
# too far apart to be in structural contact; the 0.5 overlap ratio
# distinguishes stacked from lateral pairs by demanding at least half
# the shorter element's Z-extent be shared. Both are overridable via
# project_config's pair_candidacy section — see core/project_config.py.
DEFAULT_MAX_GAP_3D_M = 1.0
DEFAULT_MIN_Z_OVERLAP_RATIO = 0.5

# ASTRA FHB T/G recommends dilatation joints every ~15 m along a
# retaining-wall run. The default is conservative; project-specific
# overrides live in .igv.yaml under pair_candidacy.
DEFAULT_DILATATION_MAX_SPACING_M = 15.0


def validate_level5(elements_data: list[dict], config: dict | None = None) -> dict:
    """Evaluate geometric relationships between elements.

    Args:
        elements_data: list of per-element dicts, each containing:
            element_id, element_name, level1 (with bbox), mesh_data
        config: optional dict with pair_candidacy overrides. Keys:
            max_gap_3d_m (float, default 1.0) — AABB rejection cutoff.
            min_z_overlap_ratio (float, default 0.5) — stacked-pair
                discriminator.

    Returns:
        dict with:
            pairs: list of pair analysis dicts
            summary: aggregate statistics
    """
    cfg = config or {}
    max_gap = float(cfg.get("max_gap_3d_m", DEFAULT_MAX_GAP_3D_M))
    min_overlap = float(cfg.get("min_z_overlap_ratio", DEFAULT_MIN_Z_OVERLAP_RATIO))

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

    # Vectorised AABB prefilter via shared helper (see core/distance.
    # aabb_pair_candidates). Replaces the previous O(N²) Python loop.
    from ifc_geo_validator.core.distance import aabb_pair_candidates
    if len(elem_info) >= 2:
        bmin = np.array([e["bbox_min"] for e in elem_info])
        bmax = np.array([e["bbox_max"] for e in elem_info])
        ii, jj = aabb_pair_candidates(bmin, bmax, max_gap)
        candidate_pairs = list(zip(ii.tolist(), jj.tolist()))
    else:
        candidate_pairs = []

    # Detailed analysis only for the handful of pairs that passed the
    # prefilter — the expensive contact-normal step runs O(k) instead
    # of O(N²).
    pairs = []
    for i, j in candidate_pairs:
        a, b = elem_info[i], elem_info[j]
        pair = _analyze_pair(a, b, max_gap_3d_m=max_gap,
                             min_z_overlap_ratio=min_overlap)
        if pair is not None:
            pairs.append(pair)

    # Build summary
    stacked = [p for p in pairs if p["pair_type"] == "stacked"]
    side_by_side = [p for p in pairs if p["pair_type"] == "side_by_side"]

    # Dilatation-joint spacing check: adjacent retaining walls should
    # have a relief joint every ~15 m along the alignment. Findings
    # are reported as a separate list so downstream L4 rules can
    # consume `dilatation_spacing_max_m` as a single scalar.
    joints = _analyze_dilatation_joints(elem_info, max_spacing_m=cfg.get(
        "dilatation_max_spacing_m", DEFAULT_DILATATION_MAX_SPACING_M))

    return {
        "pairs": pairs,
        "dilatation_joints": joints,
        "summary": {
            "num_elements": len(elem_info),
            "num_pairs": len(pairs),
            "num_stacked": len(stacked),
            "num_side_by_side": len(side_by_side),
            "dilatation_violations": sum(1 for j in joints if j["exceeds_max"]),
            "dilatation_spacing_max_m": max(
                (j["spacing_m"] for j in joints), default=0.0),
        },
    }


# ── Pair analysis ──────────────────────────────────────────────────

def _analyze_pair(
    a: dict, b: dict,
    max_gap_3d_m: float = DEFAULT_MAX_GAP_3D_M,
    min_z_overlap_ratio: float = DEFAULT_MIN_Z_OVERLAP_RATIO,
) -> dict | None:
    """Analyze the geometric relationship between two elements."""
    # Quick rejection: AABB gap exceeds the configured cutoff
    gap_3d = _bbox_gap(a["bbox_min"], a["bbox_max"], b["bbox_min"], b["bbox_max"])
    if max(gap_3d) > max_gap_3d_m:
        return None

    # Classify using Contact Surface Normal Analysis
    pair_type, upper, lower = _classify_pair_by_contact_normal(
        a, b, min_z_overlap_ratio=min_z_overlap_ratio)

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

def _classify_pair_by_contact_normal(a, b, min_z_overlap_ratio: float = DEFAULT_MIN_Z_OVERLAP_RATIO):
    """Classify pair using Contact Surface Normal Analysis.

    Algorithm:
      1. Compute contact tolerance ε from median edge length
      2. Find proximity face pairs (centroid distance < ε)
      3. Compute area-weighted contact normal N
      4. Verticality index κ = |N · e_z| / ||N||
      5. κ > cos(π/4) → stacked, else → side_by_side

    Fallback: If no proximity faces found, use centroid displacement.
    If centroids coincide, use Z-overlap with the configured
    min_z_overlap_ratio as the side-by-side vs stacked discriminator.
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
            kappa = 0.0 if (min_h > 0 and z_overlap / min_h > min_z_overlap_ratio) else 1.0
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

    Uses fully vectorized numpy when the distance matrix fits in memory
    (M_A × M_B < 2M entries ≈ 48 MB for float64). Falls back to chunked
    computation for larger meshes.

    Complexity: O(M_A × M_B), but vectorized (no Python loop).
    """
    m_a, m_b = len(centroids_a), len(centroids_b)

    if m_a * m_b < 2_000_000:
        # Full vectorized distance matrix
        diff = centroids_a[:, np.newaxis, :] - centroids_b[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)  # (M_A, M_B)
        close_i, close_j = np.where(dist_matrix < epsilon)
        return [(int(i), int(j), float(dist_matrix[i, j]))
                for i, j in zip(close_i, close_j)]
    else:
        # Chunked: process rows in batches to limit memory
        pairs = []
        chunk_size = max(1, 2_000_000 // m_b)
        for start in range(0, m_a, chunk_size):
            end = min(start + chunk_size, m_a)
            chunk = centroids_a[start:end]
            diff = chunk[:, np.newaxis, :] - centroids_b[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            ci, cj = np.where(dists < epsilon)
            for i, j in zip(ci, cj):
                pairs.append((int(start + i), int(j), float(dists[i, j])))
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


def _analyze_dilatation_joints(elem_info: list, max_spacing_m: float) -> list[dict]:
    """Detect missing dilatation (expansion) joints along a wall run.

    Walls in a retaining-wall series are ordered along their dominant
    horizontal axis (the direction of largest extent across all walls)
    and consecutive centroid gaps are compared against the
    ``max_spacing_m`` threshold. A gap below the threshold is still
    reported so users see the actual series layout; the `exceeds_max`
    flag marks violations.

    Logic:
      1. Compute the overall bbox of all walls → pick the longest axis
         as the run direction.
      2. Project each wall centroid onto that axis → 1-D positions.
      3. Sort by position → series order.
      4. Emit one record per consecutive pair.

    This is the first L5 check that uses the *series structure* of the
    input rather than each pair independently. A more rigorous approach
    would project centroids onto the actual IfcAlignment curve — that's
    the planned v2.1 upgrade (see docs/roadmap). For a linear corridor
    the bbox-axis projection is within 1-2 % of the curved-alignment
    result.
    """
    if len(elem_info) < 2:
        return []

    centroids = np.array([e["centroid"] for e in elem_info])
    bmin = np.array([e["bbox_min"] for e in elem_info]).min(axis=0)
    bmax = np.array([e["bbox_max"] for e in elem_info]).max(axis=0)
    extent = bmax - bmin
    primary_axis = int(np.argmax(extent[:2]))  # pick X or Y, ignore Z

    positions = centroids[:, primary_axis]
    order = np.argsort(positions)

    out = []
    for k in range(len(order) - 1):
        i = int(order[k])
        j = int(order[k + 1])
        # Use centroid-to-centroid along the primary axis as the
        # dilatation spacing. Walls that overlap produce a small/
        # negative spacing — we report abs() so sign doesn't confuse
        # downstream rules.
        spacing = float(abs(positions[order[k + 1]] - positions[order[k]]))
        out.append({
            "element_a_id": elem_info[i]["id"],
            "element_a_name": elem_info[i]["name"],
            "element_b_id": elem_info[j]["id"],
            "element_b_name": elem_info[j]["name"],
            "spacing_m": round(spacing, 3),
            "exceeds_max": bool(spacing > max_spacing_m),
            "max_allowed_m": max_spacing_m,
        })
    return out


def _min_bbox_distance_xy(a, b):
    """Minimum Euclidean XY distance between two axis-aligned bounding boxes.

    For overlapping projections the gap is zero on that axis; otherwise it
    is the signed separation along that axis (positive). The Euclidean
    combination √(dx² + dy²) is the correct 2D free-space distance between
    the closest points of two AABBs (Arvo & Kirk, "A Survey of Ray Tracing
    Acceleration Techniques", 1989). Manhattan summation would overstate
    the gap by up to √2 on diagonal pairs and affect proximity-based rule
    outcomes.
    """
    import math
    gap_x = max(0.0, max(a["bbox_min"][0], b["bbox_min"][0]) -
                     min(a["bbox_max"][0], b["bbox_max"][0]))
    gap_y = max(0.0, max(a["bbox_min"][1], b["bbox_min"][1]) -
                     min(a["bbox_max"][1], b["bbox_max"][1]))
    return float(math.hypot(gap_x, gap_y))
