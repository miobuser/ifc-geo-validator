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

    V = (1/6) * |sum( v0 . (v1 x v2) )| for each triangle.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    signed = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    return abs(signed.sum() / 6.0)


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
