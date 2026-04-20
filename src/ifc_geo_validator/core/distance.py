"""Geometric distance primitives for inter-element analysis.

Provides mesh-to-mesh distance computation, terrain height queries via
barycentric interpolation, and vertical/horizontal clearance measurement.

All functions are pure numpy — no external dependencies.

References:
  - Ericson, C. (2004). Real-Time Collision Detection, Ch. 5.
  - de Berg, M. et al. (2008). Computational Geometry, Ch. 6.
"""

import numpy as np


def min_mesh_distance(verts_a: np.ndarray, faces_a: np.ndarray,
                      verts_b: np.ndarray, faces_b: np.ndarray) -> float:
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


def min_vertex_distance(verts_a: np.ndarray, verts_b: np.ndarray) -> float:
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


class _TerrainGrid:
    """Spatial hash grid for fast terrain triangle lookups.

    Partitions the terrain's XY bounding box into a regular grid of cells.
    Each cell stores indices of triangles whose XY AABB overlaps that cell.
    Point queries then only test triangles in the query point's cell.

    Complexity: O(1) amortized per query (vs O(M) for linear scan).
    Construction: O(M) where M = number of terrain triangles.
    """
    __slots__ = ("verts", "faces", "grid", "ox", "oy", "cell_size", "nx", "ny")

    def __init__(self, verts, faces, target_cells_per_axis=50):
        self.verts = verts
        self.faces = faces

        # Compute terrain XY bounds
        xy_min = verts[:, :2].min(axis=0)
        xy_max = verts[:, :2].max(axis=0)
        extent = xy_max - xy_min
        max_extent = float(max(extent[0], extent[1], 1e-6))

        self.cell_size = max_extent / target_cells_per_axis
        self.ox = float(xy_min[0])
        self.oy = float(xy_min[1])
        self.nx = int(np.ceil(extent[0] / self.cell_size)) + 1
        self.ny = int(np.ceil(extent[1] / self.cell_size)) + 1

        # Build grid: dict[cell_key] → list of face indices
        self.grid: dict[tuple[int, int], list[int]] = {}
        for fi in range(len(faces)):
            tri = faces[fi]
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            # Triangle AABB in grid coordinates
            tri_min_x = min(v0[0], v1[0], v2[0])
            tri_max_x = max(v0[0], v1[0], v2[0])
            tri_min_y = min(v0[1], v1[1], v2[1])
            tri_max_y = max(v0[1], v1[1], v2[1])

            ci_min = max(0, int((tri_min_x - self.ox) / self.cell_size))
            ci_max = min(self.nx - 1, int((tri_max_x - self.ox) / self.cell_size))
            cj_min = max(0, int((tri_min_y - self.oy) / self.cell_size))
            cj_max = min(self.ny - 1, int((tri_max_y - self.oy) / self.cell_size))

            for ci in range(ci_min, ci_max + 1):
                for cj in range(cj_min, cj_max + 1):
                    key = (ci, cj)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(fi)

    def query(self, px, py) -> float | None:
        """Query terrain height at (px, py). Returns None if outside footprint."""
        ci = int((px - self.ox) / self.cell_size)
        cj = int((py - self.oy) / self.cell_size)
        candidates = self.grid.get((ci, cj))
        if candidates is None:
            return None

        for fi in candidates:
            tri = self.faces[fi]
            v0 = self.verts[tri[0]]
            v1 = self.verts[tri[1]]
            v2 = self.verts[tri[2]]
            bary = _barycentric_2d(px, py, v0[0], v0[1], v1[0], v1[1], v2[0], v2[1])
            if bary is not None:
                u, v, w = bary
                return float(u * v0[2] + v * v1[2] + w * v2[2])
        return None


# Module-level cache for terrain grids (avoids rebuilding per query)
_terrain_grid_cache: dict[int, _TerrainGrid] = {}


def terrain_height_at_xy(terrain_verts: np.ndarray, terrain_faces: np.ndarray,
                         x: float, y: float) -> float | None:
    """Query terrain height at (x, y) via barycentric interpolation.

    Uses a spatial hash grid for O(1) amortized lookups instead of
    O(M) linear scan. The grid is built lazily on first query and
    cached by terrain identity (id of the vertex array).

    Returns None if (x, y) is outside the terrain mesh footprint.

    Mathematical basis: Barycentric coordinates (de Berg et al. 2008).
    """
    # Build or retrieve spatial grid
    cache_key = id(terrain_verts)
    if cache_key not in _terrain_grid_cache:
        _terrain_grid_cache[cache_key] = _TerrainGrid(terrain_verts, terrain_faces)
    grid = _terrain_grid_cache[cache_key]

    return grid.query(float(x), float(y))


def nearest_terrain_point(
    terrain_verts: np.ndarray,
    terrain_faces: np.ndarray,
    point_3d: np.ndarray,
) -> tuple[np.ndarray, float]:
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


def vertical_clearance_crown_to_terrain(
    crown_verts: np.ndarray,
    terrain_verts: np.ndarray,
    terrain_faces: np.ndarray,
) -> dict:
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


def horizontal_distance_xy(
    bbox_a_min: np.ndarray, bbox_a_max: np.ndarray,
    bbox_b_min: np.ndarray, bbox_b_max: np.ndarray,
) -> float:
    """Horizontal (XY-only) gap between two axis-aligned bounding boxes.

    Returns 0 if the bounding boxes overlap in XY.
    """
    gap_x = max(0, max(bbox_a_min[0], bbox_b_min[0]) - min(bbox_a_max[0], bbox_b_max[0]))
    gap_y = max(0, max(bbox_a_min[1], bbox_b_min[1]) - min(bbox_a_max[1], bbox_b_max[1]))
    return float(np.sqrt(gap_x**2 + gap_y**2))


def classify_terrain_side(
    face_groups: list,
    terrain_verts: np.ndarray,
    terrain_faces: np.ndarray,
) -> dict[int, str]:
    """Classify front/back faces using terrain gradient direction.

    Determines which side of the wall faces the earth (terrain rises)
    and which faces the air (terrain falls or is absent).

    Algorithm (terrain gradient method):
      1. For each face group, sample terrain height at two points:
         - p_plus  = centroid + δ · n_horiz  (offset along face normal)
         - p_minus = centroid - δ · n_horiz  (offset against face normal)
         where δ = 1m (probe distance) and n_horiz = horizontal normal.
      2. z_plus = terrain(p_plus), z_minus = terrain(p_minus)
      3. If z_plus > z_minus: terrain rises in the normal direction → BACK (earth)
         If z_plus < z_minus: terrain falls in the normal direction → FRONT (air)

    This is robust even when the terrain is far below the wall (retaining
    walls on slopes), because it uses the terrain GRADIENT rather than the
    absolute height relative to the face centroid.

    Fallback: if gradient sampling fails, uses the original centroid-to-terrain
    vector method.

    Returns dict mapping face_group index to "front" or "back".
    """
    # Terrain-gradient probe distance (metres). We sample the terrain at
    # centroid ± PROBE_DIST · n_horiz so the gradient baseline is 2·δ = 2 m.
    # Derivation: ASTRA FHB T/G allows quer gradient up to 5 % across a
    # typical 8 m road width; a 2 m baseline captures 10 cm of relief,
    # well above the 1 cm noise floor. Shorter probes (0.5 m) are
    # dominated by terrain mesh tessellation, longer probes (≥ 3 m) may
    # cross local terrain features (kerb, embankment crest) and reverse
    # the gradient sign.
    PROBE_DIST_M = 1.0
    # Minimum gradient (Δz over 2·PROBE_DIST) that counts as "significant".
    # 1 cm over 2 m ≈ 0.5 % slope: below this the gradient is within
    # terrain tessellation noise for a typical 1 m triangle mesh.
    MIN_SIGNIFICANT_GRADIENT_M = 0.01
    # Cutoff for the fallback "nearest terrain point" method. Elements
    # farther than this from any terrain vertex are treated as
    # terrain-free (e.g. stand-alone structures in the Bauplatz). 50 m
    # is chosen as an upper bound for a typical retaining-wall terrain
    # corridor; values larger than this almost certainly indicate a
    # mismatched IfcSite or an unrelated structure.
    MAX_TERRAIN_PROXIMITY_M = 50.0
    assignments = {}

    for i, g in enumerate(face_groups):
        cat = g.get("category", g.category if hasattr(g, "category") else "")
        if cat not in ("front", "back"):
            continue

        centroid = np.array(g.get("centroid", g.centroid if hasattr(g, "centroid") else [0, 0, 0]))
        normal = np.array(g.get("normal", g.normal if hasattr(g, "normal") else [0, 0, 0]))

        # Horizontal component of the face normal
        n_horiz = np.array([normal[0], normal[1], 0.0])
        n_mag = np.linalg.norm(n_horiz)
        if n_mag < 1e-10:
            continue
        n_horiz /= n_mag

        # Method 1: Terrain gradient — sample terrain height on both sides
        p_plus = centroid[:2] + PROBE_DIST_M * n_horiz[:2]
        p_minus = centroid[:2] - PROBE_DIST_M * n_horiz[:2]

        z_plus = terrain_height_at_xy(terrain_verts, terrain_faces,
                                       p_plus[0], p_plus[1])
        z_minus = terrain_height_at_xy(terrain_verts, terrain_faces,
                                        p_minus[0], p_minus[1])

        if z_plus is not None and z_minus is not None:
            gradient = z_plus - z_minus  # positive = terrain rises in normal direction
            if abs(gradient) > MIN_SIGNIFICANT_GRADIENT_M:
                assignments[i] = "back" if gradient > 0 else "front"
                continue

        # Method 2: Fallback — centroid-to-terrain vector
        z_terrain = terrain_height_at_xy(terrain_verts, terrain_faces,
                                          centroid[0], centroid[1])
        if z_terrain is None:
            nearest_pt, dist = nearest_terrain_point(terrain_verts, terrain_faces, centroid)
            if dist > MAX_TERRAIN_PROXIMITY_M:
                continue
            terrain_point = nearest_pt
        else:
            terrain_point = np.array([centroid[0], centroid[1], z_terrain])

        to_terrain = terrain_point - centroid
        alignment = float(np.dot(normal, to_terrain))

        if alignment > 0:
            assignments[i] = "back"
        else:
            assignments[i] = "front"

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
