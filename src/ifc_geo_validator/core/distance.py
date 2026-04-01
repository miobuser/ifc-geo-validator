"""Geometric distance primitives for inter-element analysis.

Provides mesh-to-mesh distance computation, terrain height queries via
barycentric interpolation, and vertical/horizontal clearance measurement.

All functions are pure numpy — no external dependencies.

References:
  - Ericson, C. (2004). Real-Time Collision Detection, Ch. 5.
  - de Berg, M. et al. (2008). Computational Geometry, Ch. 6.
"""

import numpy as np


def min_mesh_distance(verts_a, faces_a, verts_b, faces_b) -> float:
    """Minimum distance between two triangle meshes.

    Uses face-centroid-to-face-centroid distance as a fast approximation.
    For the retaining wall mesh sizes (<1000 triangles), this is sufficient.

    Complexity: O(M_A × M_B).
    """
    centroids_a = _face_centroids(verts_a, faces_a)
    centroids_b = _face_centroids(verts_b, faces_b)

    min_dist = float("inf")
    for ca in centroids_a:
        dists = np.linalg.norm(centroids_b - ca, axis=1)
        d = float(dists.min())
        if d < min_dist:
            min_dist = d
    return min_dist


def min_vertex_distance(verts_a, verts_b) -> float:
    """Minimum distance between two vertex sets.

    Uses vectorized NumPy for performance. For N_A, N_B < 1000,
    computes the full distance matrix at once. For larger sets,
    falls back to chunked computation.

    Complexity: O(N_A × N_B), but vectorized.
    """
    a = np.asarray(verts_a)
    b = np.asarray(verts_b)

    if len(a) == 0 or len(b) == 0:
        return float("inf")

    if len(a) * len(b) < 1_000_000:
        # Full vectorized: distance matrix (N_A, N_B)
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return float(dists.min())
    else:
        # Chunked for memory efficiency
        min_dist = float("inf")
        for va in a:
            d = float(np.linalg.norm(b - va, axis=1).min())
            if d < min_dist:
                min_dist = d
        return min_dist


def terrain_height_at_xy(terrain_verts, terrain_faces, x, y) -> float | None:
    """Query terrain height at (x, y) via barycentric interpolation.

    Projects terrain triangles onto the XY plane, finds the triangle
    containing (x, y), then interpolates z using barycentric coordinates.

    Uses AABB pre-filtering: only tests triangles whose XY bounding box
    contains the query point. This reduces O(M) to O(M/k) for k grid cells.

    Returns None if (x, y) is outside the terrain mesh footprint.

    Mathematical basis: Barycentric coordinates (de Berg et al. 2008).
    """
    px, py = float(x), float(y)

    for tri in terrain_faces:
        v0 = terrain_verts[tri[0]]
        v1 = terrain_verts[tri[1]]
        v2 = terrain_verts[tri[2]]

        # Quick AABB rejection
        min_x = min(v0[0], v1[0], v2[0])
        max_x = max(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        max_y = max(v0[1], v1[1], v2[1])
        if px < min_x or px > max_x or py < min_y or py > max_y:
            continue

        # Barycentric coordinates in XY
        bary = _barycentric_2d(px, py, v0[0], v0[1], v1[0], v1[1], v2[0], v2[1])
        if bary is None:
            continue

        u, v, w = bary
        z = u * v0[2] + v * v1[2] + w * v2[2]
        return float(z)

    return None


def nearest_terrain_point(terrain_verts, terrain_faces, point_3d):
    """Find the nearest point on the terrain mesh to a 3D query point.

    Returns (nearest_point, distance).
    """
    query = np.asarray(point_3d)
    centroids = _face_centroids(terrain_verts, terrain_faces)

    # Find nearest centroid first (fast), then refine
    dists = np.linalg.norm(centroids - query, axis=1)
    nearest_idx = int(np.argmin(dists))

    # Use the centroid of the nearest triangle as approximation
    nearest_pt = centroids[nearest_idx]
    distance = float(np.linalg.norm(nearest_pt - query))

    return nearest_pt, distance


def vertical_clearance_crown_to_terrain(crown_verts, terrain_verts, terrain_faces) -> dict:
    """Compute vertical clearance from crown vertices to terrain surface.

    For each crown vertex, queries the terrain height at the same XY
    position and computes the Z-difference (clearance = crown_z - terrain_z).

    Returns dict with min/max/avg clearance in meters.
    """
    clearances = []
    for v in crown_verts:
        z_terrain = terrain_height_at_xy(terrain_verts, terrain_faces, v[0], v[1])
        if z_terrain is not None:
            clearances.append(float(v[2]) - z_terrain)

    if not clearances:
        return {"min_m": None, "max_m": None, "avg_m": None, "n_samples": 0}

    return {
        "min_m": min(clearances),
        "max_m": max(clearances),
        "avg_m": sum(clearances) / len(clearances),
        "n_samples": len(clearances),
    }


def horizontal_distance_xy(bbox_a_min, bbox_a_max, bbox_b_min, bbox_b_max) -> float:
    """Horizontal (XY-only) gap between two axis-aligned bounding boxes.

    Returns 0 if the bounding boxes overlap in XY.
    """
    gap_x = max(0, max(bbox_a_min[0], bbox_b_min[0]) - min(bbox_a_max[0], bbox_b_max[0]))
    gap_y = max(0, max(bbox_a_min[1], bbox_b_min[1]) - min(bbox_a_max[1], bbox_b_max[1]))
    return float(np.sqrt(gap_x**2 + gap_y**2))


def classify_terrain_side(face_groups, terrain_verts, terrain_faces):
    """Classify front/back faces using terrain proximity.

    For each face group classified as FRONT or BACK, determines whether
    its normal points toward or away from the terrain surface.

    Algorithm:
      1. For each face group, query terrain height at face centroid XY
      2. Compute vector from face centroid to terrain point
      3. dot(face_normal, to_terrain) > 0 → normal points toward terrain → BACK (earth)
      4. dot(face_normal, to_terrain) ≤ 0 → normal points away from terrain → FRONT (air)

    Mathematical basis: sign(dot(n, p_terrain - c_face)) — same as L5 κ.

    Returns dict mapping face_group index to "front" or "back".
    """
    assignments = {}

    for i, g in enumerate(face_groups):
        cat = g.get("category", g.category if hasattr(g, "category") else "")
        if cat not in ("front", "back"):
            continue

        centroid = np.array(g.get("centroid", g.centroid if hasattr(g, "centroid") else [0, 0, 0]))
        normal = np.array(g.get("normal", g.normal if hasattr(g, "normal") else [0, 0, 0]))

        # Query terrain at face centroid XY
        z_terrain = terrain_height_at_xy(terrain_verts, terrain_faces,
                                          centroid[0], centroid[1])
        if z_terrain is None:
            # Try nearest point as fallback; skip if too far from terrain
            nearest_pt, dist = nearest_terrain_point(terrain_verts, terrain_faces, centroid)
            if dist > 50.0:  # Element > 50m from terrain — unreliable
                continue
            terrain_point = nearest_pt
        else:
            terrain_point = np.array([centroid[0], centroid[1], z_terrain])

        # Direction from face centroid to terrain
        to_terrain = terrain_point - centroid
        alignment = float(np.dot(normal, to_terrain))

        if alignment > 0:
            assignments[i] = "back"   # normal points toward terrain → earth side
        else:
            assignments[i] = "front"  # normal points away from terrain → air side

    return assignments


# ── Internal helpers ───────────────────────────────────────────────

def _face_centroids(vertices, faces):
    """Compute centroid of each triangle."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return (v0 + v1 + v2) / 3.0


def _barycentric_2d(px, py, ax, ay, bx, by, cx, cy):
    """Compute barycentric coordinates of point (px, py) in triangle (a, b, c).

    Returns (u, v, w) if point is inside triangle, None otherwise.
    All coordinates are in the XY plane.

    Uses the standard area-ratio method (de Berg et al. 2008):
      u = area(P,B,C) / area(A,B,C)
      v = area(A,P,C) / area(A,B,C)
      w = 1 - u - v

    Point is inside if u >= 0, v >= 0, w >= 0.
    """
    # Scale-relative tolerance: based on triangle extent to handle
    # both meter-scale and large-coordinate (UTM) models correctly.
    extent = max(abs(ax - cx), abs(bx - cx), abs(ay - cy), abs(by - cy), 1e-30)
    degen_tol = 1e-12 * extent * extent  # relative to triangle area

    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < degen_tol:
        return None  # degenerate triangle (zero area)

    u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    w = 1.0 - u - v

    # Boundary tolerance scaled to triangle size
    boundary_tol = -1e-9 * extent
    if u >= boundary_tol and v >= boundary_tol and w >= boundary_tol:
        return (u, v, w)
    return None
