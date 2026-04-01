"""Level 1 Validation: Geometric properties (Volume, Area, BBox, Centroid)."""

import numpy as np

from ifc_geo_validator.core.geometry import (
    compute_volume,
    compute_total_area,
    compute_bbox,
    compute_centroid,
    compute_mesh_quality,
)


def validate_level1(mesh_data: dict) -> dict:
    """Run all Level 1 checks on extracted mesh data.

    Args:
        mesh_data: dict from mesh_converter.extract_mesh() with keys
                   vertices, faces, normals, areas, is_watertight.

    Returns:
        dict with computed geometric properties.
    """
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    areas = mesh_data["areas"]

    volume = compute_volume(vertices, faces)
    total_area = compute_total_area(areas)
    bbox = compute_bbox(vertices)
    centroid = compute_centroid(vertices, faces, areas)

    quality = compute_mesh_quality(vertices, faces, areas)

    return {
        "volume": volume,
        "total_area": total_area,
        "bbox": bbox,
        "centroid": centroid.tolist(),
        "is_watertight": mesh_data["is_watertight"],
        "num_triangles": len(faces),
        "num_vertices": len(vertices),
        "n_degenerate_filtered": mesh_data.get("n_degenerate_filtered", 0),
        "mesh_quality": quality,
    }
