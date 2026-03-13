"""Face classification via surface normal analysis and coplanar clustering.

Classifies triangulated mesh faces into semantic groups:
  - crown:      top horizontal face (Mauerkrone)
  - foundation: bottom horizontal face (Fundament/Sohle)
  - front:      vertical face perpendicular to wall axis (Vorderfläche)
  - back:       opposite vertical face (Rückfläche)
  - end_left / end_right: vertical end faces (Stirnflächen)
  - unclassified: faces that don't fit any category

Algorithm:
  1. Weld duplicate vertices to establish edge adjacency
  2. Build face adjacency graph from shared edges
  3. Cluster adjacent faces with similar normals (coplanar threshold)
  4. Determine wall longitudinal axis via PCA on horizontal vertex projection
  5. Classify each cluster by comparing area-weighted normal to reference axes
"""

import numpy as np
from dataclasses import dataclass

# ── Category constants ──────────────────────────────────────────────

CROWN = "crown"
FOUNDATION = "foundation"
FRONT = "front"
BACK = "back"
END_LEFT = "end_left"
END_RIGHT = "end_right"
UNCLASSIFIED = "unclassified"

DEFAULT_THRESHOLDS = {
    "horizontal_deg": 30.0,
    "coplanar_deg": 5.0,
    "lateral_deg": 45.0,
}


# ── Data class ──────────────────────────────────────────────────────

@dataclass
class FaceGroup:
    """A group of coplanar triangles classified as a semantic face."""
    category: str
    face_indices: list
    normal: list          # area-weighted average normal [x, y, z]
    area: float           # total area (m²)
    centroid: list        # area-weighted centroid [x, y, z]
    num_triangles: int


# ── Public API ──────────────────────────────────────────────────────

def classify_faces(mesh_data: dict, thresholds: dict = None) -> dict:
    """Classify mesh faces into semantic groups.

    Args:
        mesh_data: dict from mesh_converter.extract_mesh() with keys
                   vertices, faces, normals, areas.
        thresholds: dict with horizontal_deg, coplanar_deg, lateral_deg.
                    Falls back to DEFAULT_THRESHOLDS for missing keys.

    Returns:
        dict with:
            face_groups:     list[FaceGroup]
            wall_axis:       [x, y, z] longitudinal direction
            num_groups:      int
            thresholds_used: dict
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    horiz_rad = np.radians(t["horizontal_deg"])
    coplanar_rad = np.radians(t["coplanar_deg"])
    lateral_rad = np.radians(t["lateral_deg"])

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    normals = mesh_data["normals"]
    areas = mesh_data["areas"]

    # Step 1: Weld vertices → adjacency → coplanar clustering
    _, welded_faces = _weld_vertices(vertices, faces)
    adj_pairs = _build_face_adjacency(welded_faces)
    clusters = _cluster_coplanar(len(faces), adj_pairs, normals, coplanar_rad)

    # Step 2: Compute properties per cluster
    raw_groups = _compute_group_properties(clusters, vertices, faces, normals, areas)

    # Step 3: Determine wall longitudinal axis
    wall_axis = _determine_wall_axis(vertices)

    # Step 4: Classify each group
    classified = _classify_groups(raw_groups, wall_axis, horiz_rad, lateral_rad)

    return {
        "face_groups": classified,
        "wall_axis": wall_axis.tolist(),
        "num_groups": len(classified),
        "thresholds_used": t,
    }


# ── Step 1: Vertex welding ──────────────────────────────────────────

def _weld_vertices(vertices, faces, precision=6):
    """Merge duplicate vertices by position, remap face indices.

    IfcOpenShell with weld-vertices=False produces separate vertex buffers
    per BRep face.  We merge by rounding to `precision` decimals (~µm).
    """
    n = len(vertices)
    pos_to_idx = {}
    old_to_new = np.empty(n, dtype=int)

    for i in range(n):
        key = (
            round(float(vertices[i, 0]), precision),
            round(float(vertices[i, 1]), precision),
            round(float(vertices[i, 2]), precision),
        )
        if key not in pos_to_idx:
            pos_to_idx[key] = len(pos_to_idx)
        old_to_new[i] = pos_to_idx[key]

    welded_verts = np.zeros((len(pos_to_idx), 3))
    for i in range(n):
        welded_verts[old_to_new[i]] = vertices[i]

    return welded_verts, old_to_new[faces]


# ── Step 1: Face adjacency ─────────────────────────────────────────

def _build_face_adjacency(faces):
    """Build face adjacency pairs from shared edges.

    Returns list of (face_i, face_j) tuples where the two faces share
    exactly one edge (2 vertices).
    """
    edge_to_faces: dict[tuple[int, int], list[int]] = {}

    for fi in range(len(faces)):
        tri = faces[fi]
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            edge = (min(a, b), max(a, b))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)

    pairs = []
    for face_list in edge_to_faces.values():
        if len(face_list) == 2:
            pairs.append((face_list[0], face_list[1]))
    return pairs


# ── Step 1: Coplanar clustering (Union-Find) ───────────────────────

def _cluster_coplanar(n_faces, adj_pairs, normals, threshold_rad):
    """Cluster adjacent faces whose normals differ by less than threshold.

    Uses Union-Find with path compression for O(n·α(n)) performance.
    Returns list of lists of face indices.
    """
    parent = list(range(n_faces))
    rank = [0] * n_faces

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for fi, fj in adj_pairs:
        dot = np.dot(normals[fi], normals[fj])
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.arccos(dot)
        if angle < threshold_rad:
            union(fi, fj)

    # Collect components
    groups: dict[int, list[int]] = {}
    for i in range(n_faces):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return list(groups.values())


# ── Step 2: Group properties ────────────────────────────────────────

def _compute_group_properties(clusters, vertices, faces, normals, areas):
    """Compute area-weighted normal, total area, and centroid per cluster."""
    raw_groups = []
    for cluster_indices in clusters:
        idx = np.array(cluster_indices)
        w_areas = areas[idx]
        w_normals = normals[idx]

        # Area-weighted average normal
        avg_normal = (w_normals * w_areas[:, np.newaxis]).sum(axis=0)
        mag = np.linalg.norm(avg_normal)
        if mag > 0:
            avg_normal /= mag

        total_area = float(w_areas.sum())

        # Area-weighted centroid
        v0 = vertices[faces[idx, 0]]
        v1 = vertices[faces[idx, 1]]
        v2 = vertices[faces[idx, 2]]
        tri_centroids = (v0 + v1 + v2) / 3.0
        if total_area > 0:
            centroid = (tri_centroids * w_areas[:, np.newaxis]).sum(axis=0) / total_area
        else:
            centroid = tri_centroids.mean(axis=0)

        raw_groups.append({
            "indices": cluster_indices,
            "normal": avg_normal,
            "area": total_area,
            "centroid": centroid,
        })

    return raw_groups


# ── Step 3: Wall axis via PCA ───────────────────────────────────────

def _determine_wall_axis(vertices):
    """Determine wall longitudinal axis via PCA on horizontal projection.

    Projects all vertices onto the XY plane, computes the covariance
    matrix, and returns the eigenvector with the largest eigenvalue
    (= direction of greatest horizontal spread = wall length direction).

    Falls back to bbox longest horizontal dimension if PCA is ambiguous
    (eigenvalue ratio < 2).
    """
    xy = vertices[:, :2]
    centered = xy - xy.mean(axis=0)

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order; last = largest
    ratio = eigenvalues[-1] / max(eigenvalues[-2], 1e-12)

    if ratio < 2.0:
        # PCA ambiguous (nearly square plan) → use bbox
        bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
        axis_idx = int(np.argmax(bbox_size[:2]))
        axis = np.zeros(3)
        axis[axis_idx] = 1.0
        return axis

    axis_2d = eigenvectors[:, -1]
    return np.array([axis_2d[0], axis_2d[1], 0.0])


# ── Step 4: Classification ──────────────────────────────────────────

def _classify_groups(raw_groups, wall_axis, horiz_rad, lateral_rad):
    """Classify each face group by comparing its normal to reference axes.

    Horizontal faces:  |dot(normal, Z)| > cos(horiz_rad)
      - dot > 0 → crown,  dot < 0 → foundation

    Vertical faces projected to XY:
      - |dot(n_horiz, wall_axis)| → angle_from_axis
      - angle < lateral_rad → end face (normal ∥ wall axis)
      - angle ≥ lateral_rad → front/back (normal ⊥ wall axis)

    Front vs. back:  larger total area = front (heuristic;
    definitive assignment requires terrain context in Level 6).
    """
    z_axis = np.array([0.0, 0.0, 1.0])

    # Perpendicular horizontal axis (wall_axis × Z)
    perp_axis = np.cross(wall_axis, z_axis)
    perp_norm = np.linalg.norm(perp_axis)
    if perp_norm > 1e-10:
        perp_axis /= perp_norm

    cos_horiz = np.cos(horiz_rad)

    horizontal = []
    vertical = []

    for g in raw_groups:
        normal = g["normal"]
        dot_z = np.dot(normal, z_axis)

        if abs(dot_z) > cos_horiz:
            g["category"] = CROWN if dot_z > 0 else FOUNDATION
            horizontal.append(g)
        else:
            vertical.append(g)

    # Sub-classify vertical faces
    front_candidates = []
    back_candidates = []

    for g in vertical:
        normal = g["normal"]
        n_horiz = np.array([normal[0], normal[1], 0.0])
        n_mag = np.linalg.norm(n_horiz)
        if n_mag < 1e-10:
            g["category"] = UNCLASSIFIED
            continue
        n_horiz /= n_mag

        cos_axis = abs(np.dot(n_horiz, wall_axis))
        angle_from_axis = np.arccos(np.clip(cos_axis, 0.0, 1.0))

        if angle_from_axis < lateral_rad:
            # Normal ≈ parallel to wall axis → end face
            sign = np.dot(n_horiz, wall_axis)
            g["category"] = END_RIGHT if sign > 0 else END_LEFT
        else:
            # Normal ≈ perpendicular to wall axis → front or back
            perp_dot = np.dot(n_horiz, perp_axis)
            g["_perp_dot"] = perp_dot

    # Separate front/back candidates by sign of perpendicular projection
    undecided = [g for g in vertical if "_perp_dot" in g]
    pos = [g for g in undecided if g["_perp_dot"] >= 0]
    neg = [g for g in undecided if g["_perp_dot"] < 0]

    pos_area = sum(g["area"] for g in pos)
    neg_area = sum(g["area"] for g in neg)

    # Heuristic: larger combined area = front (earth retention side)
    if pos_area >= neg_area:
        front_list, back_list = pos, neg
    else:
        front_list, back_list = neg, pos

    for g in front_list:
        g["category"] = FRONT
    for g in back_list:
        g["category"] = BACK

    # Clean up temp keys
    for g in vertical:
        g.pop("_perp_dot", None)

    # Any remaining unclassified
    for g in raw_groups:
        if "category" not in g:
            g["category"] = UNCLASSIFIED

    # Build FaceGroup dataclass instances
    result = []
    for g in raw_groups:
        result.append(FaceGroup(
            category=g["category"],
            face_indices=g["indices"],
            normal=g["normal"].tolist(),
            area=g["area"],
            centroid=g["centroid"].tolist(),
            num_triangles=len(g["indices"]),
        ))

    return result
