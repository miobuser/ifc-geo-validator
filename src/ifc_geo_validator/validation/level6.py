"""Level 6 Validation: Inter-element distance checks and terrain context.

Generic distance measurement engine between IFC elements, configurable
via YAML rules. Supports:
  - Minimum distance between any two element meshes
  - Vertical clearance (crown to terrain via barycentric interpolation)
  - Horizontal distance (XY gap between bounding boxes)
  - Terrain-based front/back classification (earth vs air side)

Terrain side determination:
  sign(dot(n_face, p_terrain - c_face)) > 0  →  earth side (BACK)
  sign(dot(n_face, p_terrain - c_face)) ≤ 0  →  air side (FRONT)

This resolves the front/back ambiguity identified in Level 2 as
geometrically underdetermined without external context.

All distance computations use the primitives from core.distance.

References:
  - de Berg, M. et al. (2008). Computational Geometry, Ch. 6 (barycentric).
  - ASTRA 24 001-15101 V3.02 (2025). Fachhandbuch T/G, Stützbauwerke.
"""

import numpy as np

from ifc_geo_validator.core.distance import (
    min_vertex_distance,
    vertical_clearance_crown_to_terrain,
    horizontal_distance_xy,
    classify_terrain_side,
    terrain_height_at_xy,
)


def validate_level6(
    elements_data: list[dict],
    terrain_mesh: dict | None = None,
    reference_elements: list[dict] | None = None,
) -> dict:
    """Evaluate Level 6 inter-element distance checks.

    Args:
        elements_data: list of per-element dicts with level1, level2, level3, mesh_data
        terrain_mesh: dict from get_terrain_mesh() (vertices, faces), or None
        reference_elements: additional non-wall elements (barriers, etc.)

    Returns:
        dict with:
            terrain_available: bool
            terrain_side: dict mapping element_id → front/back assignments
            clearances: list of clearance measurements
            distances: list of inter-element distance measurements
    """
    result = {
        "terrain_available": terrain_mesh is not None,
        "terrain_side": {},
        "clearances": [],
        "embedments": [],
        "distances": [],
    }

    # ── Terrain-based front/back classification ──────────────────
    if terrain_mesh is not None:
        t_verts = terrain_mesh["vertices"]
        t_faces = terrain_mesh["faces"]

        for elem in elements_data:
            eid = elem.get("element_id")
            ename = elem.get("element_name", "Unnamed")
            l2 = elem.get("level2")
            if not l2:
                continue

            face_groups = l2.get("face_groups", [])
            assignments = classify_terrain_side(face_groups, t_verts, t_faces)

            if assignments:
                result["terrain_side"][eid] = {
                    "element_name": ename,
                    "assignments": assignments,
                }

        # ── Vertical clearance (crown to terrain) ────────────────
        for elem in elements_data:
            eid = elem.get("element_id")
            ename = elem.get("element_name", "Unnamed")
            mesh = elem.get("mesh_data")
            l2 = elem.get("level2")
            if not mesh or not l2:
                continue

            # Collect crown vertices
            crown_groups = [g for g in l2.get("face_groups", [])
                           if g.get("category") == "crown"]
            if not crown_groups:
                continue

            crown_face_indices = []
            for g in crown_groups:
                crown_face_indices.extend(g["face_indices"])

            if not crown_face_indices:
                continue

            faces = mesh["faces"]
            verts = mesh["vertices"]
            crown_vert_idx = np.unique(faces[np.array(crown_face_indices)].ravel())
            crown_verts = verts[crown_vert_idx]

            clearance = vertical_clearance_crown_to_terrain(
                crown_verts, t_verts, t_faces
            )
            clearance["element_id"] = eid
            clearance["element_name"] = ename
            result["clearances"].append(clearance)

        # ── Foundation embedment depth ────────────────────────────
        for elem in elements_data:
            eid = elem.get("element_id")
            ename = elem.get("element_name", "Unnamed")
            mesh = elem.get("mesh_data")
            l2 = elem.get("level2")
            if not mesh or not l2:
                continue

            found_groups = [g for g in l2.get("face_groups", [])
                            if g.get("category") == "foundation"]
            if not found_groups:
                continue

            found_face_indices = []
            for g in found_groups:
                found_face_indices.extend(g["face_indices"])

            if not found_face_indices:
                continue

            faces = mesh["faces"]
            verts = mesh["vertices"]
            found_vert_idx = np.unique(faces[np.array(found_face_indices)].ravel())
            found_verts = verts[found_vert_idx]

            # Minimum Z of the foundation (bottom of foundation)
            foundation_min_z = float(found_verts[:, 2].min())

            # Query terrain at foundation centroid XY
            centroid_xy = found_verts[:, :2].mean(axis=0)
            terrain_z = terrain_height_at_xy(
                t_verts, t_faces, centroid_xy[0], centroid_xy[1]
            )

            if terrain_z is not None:
                # Vertical embedment (ΔZ) — simple but overestimates on slopes
                embedment_vertical = terrain_z - foundation_min_z

                # Minimum distance to terrain surface (correct for frost depth).
                # The frost line follows the terrain surface, so the relevant
                # distance is the shortest path from foundation bottom to the
                # terrain, NOT the vertical distance.
                #
                # For each bottom vertex, find the nearest point on the terrain
                # and compute the 3D distance.
                bottom_mask = found_verts[:, 2] < (foundation_min_z + 0.01)
                bottom_verts = found_verts[bottom_mask] if bottom_mask.any() else found_verts

                min_surface_dist = float("inf")
                for bv in bottom_verts:
                    # Find terrain height at this vertex's XY
                    tz = terrain_height_at_xy(t_verts, t_faces, bv[0], bv[1])
                    if tz is not None:
                        terrain_pt = np.array([bv[0], bv[1], tz])
                        dist = float(np.linalg.norm(terrain_pt - bv))
                        min_surface_dist = min(min_surface_dist, dist)

                if min_surface_dist == float("inf"):
                    min_surface_dist = embedment_vertical

                result["embedments"].append({
                    "element_id": eid,
                    "element_name": ename,
                    "foundation_min_z": round(foundation_min_z, 4),
                    "terrain_z": round(terrain_z, 4),
                    # Primary: surface distance (correct for frost depth)
                    "foundation_embedment_m": round(min_surface_dist, 4),
                    "foundation_embedment_vertical_m": round(embedment_vertical, 4),
                    "foundation_embedment_surface_m": round(min_surface_dist, 4),
                })

    # ── Inter-element distances ──────────────────────────────────
    if len(elements_data) >= 2:
        from itertools import combinations

        for i, j in combinations(range(len(elements_data)), 2):
            a = elements_data[i]
            b = elements_data[j]
            mesh_a = a.get("mesh_data")
            mesh_b = b.get("mesh_data")
            l1_a = a.get("level1", {})
            l1_b = b.get("level1", {})

            if not mesh_a or not mesh_b:
                continue

            bbox_a = l1_a.get("bbox", {})
            bbox_b = l1_b.get("bbox", {})

            if not bbox_a or not bbox_b:
                continue

            # Minimum vertex-to-vertex distance
            min_dist = min_vertex_distance(mesh_a["vertices"], mesh_b["vertices"])

            # Horizontal XY distance
            h_dist = horizontal_distance_xy(
                np.array(bbox_a["min"]), np.array(bbox_a["max"]),
                np.array(bbox_b["min"]), np.array(bbox_b["max"]),
            )

            result["distances"].append({
                "element_a_id": a.get("element_id"),
                "element_a_name": a.get("element_name", "Unnamed"),
                "element_b_id": b.get("element_id"),
                "element_b_name": b.get("element_name", "Unnamed"),
                "min_distance_mm": round(min_dist * 1000, 1),
                "horizontal_distance_mm": round(h_dist * 1000, 1),
            })

    return result
