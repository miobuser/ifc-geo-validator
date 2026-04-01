"""
Create test model T14: Curved retaining wall with L-shaped cross-section (45° arc).

Specifications:
- 45° arc in plan view, 16 segments
- Inner radius R=12.0 m (back edge of foundation / stem)
- L-shaped cross-section (Winkelstützmauer):
    Foundation: R=12.0..13.5 (1.5m wide), z=0..0.5
    Stem:       R=12.0..12.3 (0.3m thick), z=0.5..3.0
    Total height: 3.0m
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces + L-shaped end caps via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Cross-section (radial slice):

    R=12.0  R=12.3            R=13.5
    z=3.0   +-----+
            |stem |
            |0.3m |
    z=0.5   |     +------------+
            |     | foundation |
    z=0     +-----+------------+
                    1.5m

Output: tests/test_models/T14_curved_l_profile.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
R_INNER = 12.0          # inner radius (back edge of foundation and stem) [m]
STEM_THICK = 0.3        # stem radial thickness [m]
FOUND_WIDTH = 1.5       # foundation radial width [m]
FOUND_HEIGHT = 0.5      # foundation vertical thickness [m]
TOTAL_HEIGHT = 3.0      # total wall height [m]

R_STEM_OUTER = R_INNER + STEM_THICK          # 12.3
R_FOUND_OUTER = R_INNER + FOUND_WIDTH         # 13.5

N_SEG = 16              # number of arc segments
ARC_DEG = 45.0          # arc angle [degrees]

# L-profile vertices (radial R, vertical z) — CCW when viewed from the right:
#   0: (R_INNER,      0.0)          — back-bottom of foundation
#   1: (R_FOUND_OUTER, 0.0)         — front-bottom of foundation
#   2: (R_FOUND_OUTER, FOUND_HEIGHT)— front-top of foundation
#   3: (R_STEM_OUTER,  FOUND_HEIGHT)— step: where stem meets foundation
#   4: (R_STEM_OUTER,  TOTAL_HEIGHT)— front-top of stem (crown outer)
#   5: (R_INNER,       TOTAL_HEIGHT)— back-top of stem (crown inner)

L_PROFILE_RZ = [
    (R_INNER,       0.0),
    (R_FOUND_OUTER, 0.0),
    (R_FOUND_OUTER, FOUND_HEIGHT),
    (R_STEM_OUTER,  FOUND_HEIGHT),
    (R_STEM_OUTER,  TOTAL_HEIGHT),
    (R_INNER,       TOTAL_HEIGHT),
]
N_VERTS = len(L_PROFILE_RZ)  # 6


def rz_to_xyz(r: float, z: float, angle_rad: float) -> tuple[float, float, float]:
    """Convert (R, z, angle) to 3D Cartesian coordinates."""
    return (r * math.cos(angle_rad), r * math.sin(angle_rad), z)


def main():
    # ---------- generate geometry vertices ----------
    n_pts = N_SEG + 1  # 17 angle positions for 16 segments

    # For each angle position, compute the 6 L-profile vertices in 3D
    # rings[j][k] = 3D point at angle position j, profile vertex k
    angles = [math.radians(ARC_DEG * i / (n_pts - 1)) for i in range(n_pts)]
    rings = []
    for angle in angles:
        ring = [rz_to_xyz(r, z, angle) for (r, z) in L_PROFILE_RZ]
        rings.append(ring)

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T14"
    )

    # Units (metres)
    length = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="LENGTHUNIT")
    area = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="AREAUNIT")
    volume = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="VOLUMEUNIT")
    ifcopenshell.api.run("unit.assign_unit", ifc, units=[length, area, volume])

    # Context
    ctx = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context", ifc,
        context_type="Model", context_identifier="Body", target_view="MODEL_VIEW",
        parent=ctx,
    )

    # Spatial hierarchy: Project > Site > Facility
    site = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände"
    )
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[site], relating_object=project
    )

    facility = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1"
    )
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[facility], relating_object=site
    )

    # ---------- build IfcFacetedBrep ----------
    pt_cache: dict[tuple[float, float, float], object] = {}

    def ifc_pt(coords: tuple[float, float, float]):
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key not in pt_cache:
            pt_cache[key] = ifc.createIfcCartesianPoint(key)
        return pt_cache[key]

    def make_face(pts_list):
        """Create an IfcFace from a list of coordinate tuples (forming a polygon)."""
        loop = ifc.createIfcPolyLoop([ifc_pt(p) for p in pts_list])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        return ifc.createIfcFace([bound])

    faces = []

    # Lateral faces: for each segment (pair of consecutive angle positions),
    # connect each edge of the L-profile as a quad.
    # The L-profile has 6 vertices → 6 edges (0→1, 1→2, 2→3, 3→4, 4→5, 5→0).
    for i in range(N_SEG):
        for k in range(N_VERTS):
            k_next = (k + 1) % N_VERTS
            # Quad: rings[i][k], rings[i+1][k], rings[i+1][k_next], rings[i][k_next]
            # Winding must be consistent (outward normals).
            # For edges along the outside of the L-profile (CCW profile, arc goes
            # in increasing angle), the outward normal points away from the center.
            # We use: current_ring first, then next_ring — CCW when viewed from outside.
            faces.append(make_face([
                rings[i][k],
                rings[i][k_next],
                rings[i + 1][k_next],
                rings[i + 1][k],
            ]))

    # End cap at angle = 0° (start) — L-shaped polygon with 6 vertices
    # Normal should point in the -angle direction (towards the start).
    # The profile vertices are CCW when viewed from the +angle side,
    # so for the start cap we reverse the winding.
    faces.append(make_face(list(reversed(rings[0]))))

    # End cap at angle = 45° (end) — L-shaped polygon with 6 vertices
    # Normal should point in the +angle direction (towards the end).
    faces.append(make_face(rings[-1]))

    closed_shell = ifc.createIfcClosedShell(faces)
    brep = ifc.createIfcFacetedBrep(closed_shell)

    # Shape representation
    shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [brep])
    prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

    # Placement
    origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    placement = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement = ifc.createIfcLocalPlacement(None, placement)

    # IfcWall
    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stützmauer T14 — Gekrümmte Winkelstützmauer (45° Bogen, L-Profil)",
        predefined_type="RETAININGWALL",
    )
    wall.ObjectPlacement = local_placement
    wall.Representation = prod_shape

    # Assign wall to facility
    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[wall], relating_structure=facility,
    )

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T14_curved_l_profile.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Quick sanity check: approximate volume
    # L-profile area = foundation_width * foundation_height + stem_thick * (total_height - foundation_height)
    a_profile = FOUND_WIDTH * FOUND_HEIGHT + STEM_THICK * (TOTAL_HEIGHT - FOUND_HEIGHT)
    # Arc length at centroid radius ≈ weighted average
    r_centroid_found = R_INNER + FOUND_WIDTH / 2.0
    r_centroid_stem = R_INNER + STEM_THICK / 2.0
    a_found = FOUND_WIDTH * FOUND_HEIGHT
    a_stem = STEM_THICK * (TOTAL_HEIGHT - FOUND_HEIGHT)
    r_centroid = (a_found * r_centroid_found + a_stem * r_centroid_stem) / (a_found + a_stem)
    arc_len = r_centroid * math.radians(ARC_DEG)
    v_approx = a_profile * arc_len
    print(f"  L-profile area: {a_profile:.4f} m²")
    print(f"  Approximate volume (Pappus): {v_approx:.4f} m³")


if __name__ == "__main__":
    main()
