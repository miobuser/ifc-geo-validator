"""Mesh extraction from IFC elements using IfcOpenShell geometry processing."""

import numpy as np
import ifcopenshell
import ifcopenshell.geom

# Global geometry settings
SETTINGS = ifcopenshell.geom.settings()
SETTINGS.set("use-world-coords", True)
SETTINGS.set("weld-vertices", True)


def extract_mesh(element) -> dict:
    """Extract triangulated mesh from an IFC element.

    Returns:
        dict with keys:
            vertices: np.array (N, 3) — vertex positions
            faces:    np.array (M, 3) — triangle vertex indices
            normals:  np.array (M, 3) — unit face normals
            areas:    np.array (M,)   — per-face areas
            is_watertight: bool — True if every edge is shared by exactly 2 faces
    """
    shape = ifcopenshell.geom.create_shape(SETTINGS, element)
    geometry = shape.geometry

    vertices = np.array(geometry.verts).reshape(-1, 3)
    faces = np.array(geometry.faces).reshape(-1, 3)

    # Compute face normals and areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)

    areas = 0.5 * np.linalg.norm(cross, axis=1)

    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid div-by-zero for degenerate triangles
    normals = cross / norms

    watertight = _check_watertight(faces)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "areas": areas,
        "is_watertight": watertight,
    }


def _check_watertight(faces: np.ndarray) -> bool:
    """Check if the mesh is watertight (every edge shared by exactly 2 faces)."""
    edge_count: dict[tuple[int, int], int] = {}
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    return all(c == 2 for c in edge_count.values())
