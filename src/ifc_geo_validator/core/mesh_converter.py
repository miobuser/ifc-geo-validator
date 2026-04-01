"""Mesh extraction from IFC elements using IfcOpenShell geometry processing.

Converts any IFC geometry representation (IfcFacetedBrep, IfcExtrudedAreaSolid,
IfcTriangulatedFaceSet, IfcBooleanClippingResult, etc.) to a uniform
triangulated mesh via IfcOpenShell/OpenCASCADE (OCCT).

Settings:
  use-world-coords=True  — All vertices in global coordinate system
  weld-vertices=True     — Merge duplicate vertices at shared positions

References:
  - Krijnen, T. & Beetz, J. (2020). An IFC schema extension and binary
    serialization format to efficiently integrate point cloud data into
    building models. Advanced Engineering Informatics, 33, 2017.
  - IfcOpenShell documentation: ifcopenshell.org
"""

import numpy as np
import ifcopenshell
import ifcopenshell.geom

# Global geometry settings
SETTINGS = ifcopenshell.geom.settings()
SETTINGS.set("use-world-coords", True)
SETTINGS.set("weld-vertices", True)


class MeshExtractionError(Exception):
    """Raised when geometry cannot be extracted from an IFC element."""
    pass


def extract_mesh(element) -> dict:
    """Extract triangulated mesh from an IFC element.

    Handles any IFC geometry representation (BRep, ExtrudedAreaSolid,
    TriangulatedFaceSet, CSG, etc.) via IfcOpenShell/OpenCASCADE.

    Returns:
        dict with keys:
            vertices: np.array (N, 3) — vertex positions
            faces:    np.array (M, 3) — triangle vertex indices
            normals:  np.array (M, 3) — unit face normals
            areas:    np.array (M,)   — per-face areas
            is_watertight: bool — True if every edge is shared by exactly 2 faces

    Raises:
        MeshExtractionError: If geometry cannot be extracted.
    """
    elem_name = getattr(element, "Name", None) or f"#{element.id()}"

    try:
        shape = ifcopenshell.geom.create_shape(SETTINGS, element)
    except Exception as e:
        raise MeshExtractionError(
            f"Failed to create geometry for '{elem_name}': {e}"
        ) from e

    geometry = shape.geometry

    verts_flat = np.array(geometry.verts)
    faces_flat = np.array(geometry.faces)

    if len(verts_flat) == 0 or len(faces_flat) == 0:
        raise MeshExtractionError(
            f"Empty geometry for '{elem_name}' (0 vertices or 0 faces)"
        )

    vertices = verts_flat.reshape(-1, 3)
    faces = faces_flat.reshape(-1, 3)

    # Compute face normals and areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)

    areas = 0.5 * np.linalg.norm(cross, axis=1)

    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid div-by-zero for degenerate triangles
    normals = cross / norms

    # Filter degenerate triangles: area < ε² where ε is derived from the
    # mesh's characteristic length (bbox diagonal). This is scale-invariant:
    # a triangle is degenerate if its area is negligible relative to the
    # model size, not relative to an absolute threshold.
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    area_threshold = (bbox_diag * 1e-8) ** 2 if bbox_diag > 0 else 1e-30
    valid_mask = areas > area_threshold
    n_degenerate = int((~valid_mask).sum())

    if n_degenerate > 0 and valid_mask.sum() >= 4:
        faces = faces[valid_mask]
        normals = normals[valid_mask]
        areas = areas[valid_mask]

    watertight = _check_watertight(faces)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "areas": areas,
        "is_watertight": watertight,
        "n_degenerate_filtered": n_degenerate,
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
