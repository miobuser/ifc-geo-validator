"""Pure geometry computations on triangulated mesh data.

All algorithms are mathematically exact for triangulated meshes:
  - Volume: Divergence theorem (Gauss, 1813) on signed tetrahedra
  - Area: Cross product magnitude, A = ||e₁ × e₂|| / 2
  - BBox: Axis-aligned bounding box from vertex extrema
  - Centroid: Area-weighted surface centroid

No IFC dependency — operates on numpy arrays only.
"""

import numpy as np


def compute_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute volume of a closed mesh using the divergence theorem.

    V = (1/6) * |Σ v₀ · (v₁ × v₂)| for each triangle.

    Vertices are centered before computation to avoid catastrophic
    cancellation at large coordinates (e.g. LV95/UTM with offsets
    of 10⁶). Centering is mathematically equivalent: translating a
    closed mesh does not change its volume (translation invariance).

    Without centering, float64 precision is ~10⁻¹⁵ relative to
    coordinate magnitude. At offset=2.6×10⁶ with volume=9.6m³,
    this gives ~10⁻⁹ absolute precision — but the signed tetrahedra
    sum involves cancellation of large numbers, degrading to ~10⁻².
    Centering restores full precision.

    Reference: Ericson, C. (2004). Real-Time Collision Detection, §12.4.
    """
    return abs(compute_signed_volume(vertices, faces))


def compute_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume — positive if outward-normal convention holds.

    Same divergence-theorem formula as compute_volume but without the
    final absolute value. A positive result means the face winding
    order encodes outward-pointing normals (the standard convention
    assumed by the face classifier: crown = dot(normal, z) > 0).
    A negative result means the mesh has inverted winding; callers
    can detect this and flip the normal vectors before classification.

    This is the single authoritative check for normal orientation; no
    other module should duplicate the outward-normal heuristic.
    """
    center = vertices.mean(axis=0)
    v0 = vertices[faces[:, 0]] - center
    v1 = vertices[faces[:, 1]] - center
    v2 = vertices[faces[:, 2]] - center
    signed = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    return float(signed.sum() / 6.0)


def compute_total_area(areas: np.ndarray) -> float:
    """Sum of all triangle areas."""
    return float(areas.sum())


def compute_bbox(vertices: np.ndarray) -> dict:
    """Axis-aligned bounding box.

    Returns:
        dict with 'min', 'max', 'size' — each a list [x, y, z].
    """
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return {
        "min": vmin.tolist(),
        "max": vmax.tolist(),
        "size": (vmax - vmin).tolist(),
    }


def compute_mesh_quality(vertices: np.ndarray, faces: np.ndarray, areas: np.ndarray) -> dict:
    """Compute mesh quality metrics for diagnostics.

    Returns:
        dict with:
            n_degenerate:    int — triangles with area < 1e-10 × median_area
            min_area:        float — smallest triangle area
            max_area:        float — largest triangle area
            area_ratio:      float — max_area / min_area (condition number)
            edge_length_min: float — shortest edge
            edge_length_max: float — longest edge
            edge_length_median: float — median edge length
            non_manifold_edges: int — edges shared by ≠ 2 faces
    """
    # Area statistics
    median_area = float(np.median(areas)) if len(areas) > 0 else 0.0
    area_thresh = median_area * 1e-10 if median_area > 0 else 1e-30
    n_degenerate = int((areas < area_thresh).sum())

    valid_areas = areas[areas > area_thresh] if n_degenerate < len(areas) else areas
    min_area = float(valid_areas.min()) if len(valid_areas) > 0 else 0.0
    max_area = float(valid_areas.max()) if len(valid_areas) > 0 else 0.0
    area_ratio = max_area / min_area if min_area > 0 else float("inf")

    # Edge length statistics (fully vectorized, no Python loop).
    # For M triangles with vertex indices (a, b, c), the 3 edges are:
    #   e0 = v[b] - v[a],  e1 = v[c] - v[b],  e2 = v[a] - v[c]
    # Vectorized via array slicing — O(M) with no Python overhead.
    e0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e1 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    e2 = vertices[faces[:, 0]] - vertices[faces[:, 2]]
    edge_arr = np.concatenate([
        np.linalg.norm(e0, axis=1),
        np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1),
    ])

    # Non-manifold edges (shared by ≠ 2 faces).
    # Vectorized edge hashing: represent each edge as (min(a,b), max(a,b)),
    # count occurrences using numpy bincount on a Cantor pairing function.
    all_edges_a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    all_edges_b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    edge_lo = np.minimum(all_edges_a, all_edges_b)
    edge_hi = np.maximum(all_edges_a, all_edges_b)
    # Cantor pairing: unique integer per edge pair
    edge_keys = edge_lo.astype(np.int64) * (edge_hi.max() + 1) + edge_hi.astype(np.int64)
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    non_manifold = int((counts != 2).sum())

    return {
        "n_degenerate": n_degenerate,
        "min_area": round(min_area, 10),
        "max_area": round(max_area, 6),
        "area_ratio": round(area_ratio, 1) if area_ratio != float("inf") else None,
        "edge_length_min": round(float(edge_arr.min()), 6),
        "edge_length_max": round(float(edge_arr.max()), 4),
        "edge_length_median": round(float(np.median(edge_arr)), 4),
        "non_manifold_edges": non_manifold,
    }


def compute_centroid(vertices: np.ndarray, faces: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Area-weighted centroid of the mesh surface.

    Each triangle's centroid is the average of its three vertices,
    weighted by the triangle's area.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    tri_centroids = (v0 + v1 + v2) / 3.0

    total_area = areas.sum()
    if total_area == 0:
        return tri_centroids.mean(axis=0)

    weighted = tri_centroids * areas[:, np.newaxis]
    return weighted.sum(axis=0) / total_area
