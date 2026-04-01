"""
Create test model T14 (v2): Curved L-profile wall as 2 separate IFC elements.

Two elements along a 45-degree arc:
1. Curved wall stem (IfcWall, RETAININGWALL):
   - R_inner=12.0m, R_outer=12.3m (0.3m thick), z=0.5..3.0 (2.5m high)
2. Curved foundation slab (IfcWall, RETAININGWALL):
   - R_inner=12.0m, R_outer=13.5m (1.5m wide), z=0.0..0.5 (0.5m thick)

Both use IfcFacetedBrep with 16 segments, IFC4X3_ADD2.
Face winding: outward-pointing normals everywhere.

Output: tests/test_models/T14_curved_l_profile.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
R_INNER = 12.0          # shared inner radius [m]
ARC_DEG = 45.0          # arc angle [degrees]
N_SEG = 16              # number of arc segments

# Stem parameters
STEM_R_OUTER = 12.3     # outer radius of wall stem [m]
STEM_Z_BOT = 0.5        # bottom of stem [m]
STEM_Z_TOP = 3.0        # top of stem [m]

# Foundation parameters
FOUND_R_OUTER = 13.5    # outer radius of foundation [m]
FOUND_Z_BOT = 0.0       # bottom of foundation [m]
FOUND_Z_TOP = 0.5       # top of foundation [m]


def make_arc_points(radius: float, n_pts: int, z: float) -> list[tuple[float, float, float]]:
    """Generate points along an arc from 0 to ARC_DEG at given radius and z."""
    pts = []
    for i in range(n_pts):
        angle = math.radians(ARC_DEG * i / (n_pts - 1))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pts.append((round(x, 6), round(y, 6), round(z, 6)))
    return pts


def build_brep(ifc, body_ctx, r_inner, r_outer, z_bot, z_top, n_seg):
    """Build an IfcFacetedBrep for a curved wall segment.

    Face winding ensures outward-pointing normals:
    - Outer face: CCW viewed from outside (normal outward)
    - Inner face: CCW viewed from inside = CW from outside (normal inward)
    - Top face: CCW viewed from above (normal up)
    - Bottom face: CW viewed from above (normal down)
    """
    n_pts = n_seg + 1

    outer_bot = make_arc_points(r_outer, n_pts, z_bot)
    outer_top = make_arc_points(r_outer, n_pts, z_top)
    inner_bot = make_arc_points(r_inner, n_pts, z_bot)
    inner_top = make_arc_points(r_inner, n_pts, z_top)

    pt_cache: dict[tuple[float, float, float], object] = {}

    def ifc_pt(coords):
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key not in pt_cache:
            pt_cache[key] = ifc.createIfcCartesianPoint(key)
        return pt_cache[key]

    def make_face(pts_list):
        loop = ifc.createIfcPolyLoop([ifc_pt(p) for p in pts_list])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        return ifc.createIfcFace([bound])

    faces = []

    for i in range(n_seg):
        # Outer face: CCW from outside → normal points radially outward
        faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i]
        ]))

        # Inner face: CCW from inside → normal points radially inward
        faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1]
        ]))

        # Top face: CCW from above → normal points up
        faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1]
        ]))

        # Bottom face: CW from above → normal points down
        faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1]
        ]))

    # End cap at angle = 0 (start): looking at it from -Y direction (outside the arc start)
    # Normal should point toward angle=0 start, i.e. roughly in -Y direction
    faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0]
    ]))

    # End cap at angle = 45 (end): looking at it from beyond the arc end
    # Normal should point toward angle=45 end
    faces.append(make_face([
        outer_bot[-1], inner_bot[-1], inner_top[-1], outer_top[-1]
    ]))

    closed_shell = ifc.createIfcClosedShell(faces)
    brep = ifc.createIfcFacetedBrep(closed_shell)

    shape_rep = ifc.createIfcShapeRepresentation(body_ctx, "Body", "Brep", [brep])
    prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

    return prod_shape, len(faces), len(pt_cache)


def main():
    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject",
        name="Thesis Test T14 — Curved L-Profile",
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
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Testgelaende")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[facility], relating_object=site)

    # Shared placement at origin
    origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    placement = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement = ifc.createIfcLocalPlacement(None, placement)

    # ---------- Element 1: Wall stem ----------
    stem_shape, stem_faces, stem_pts = build_brep(
        ifc, body, R_INNER, STEM_R_OUTER, STEM_Z_BOT, STEM_Z_TOP, N_SEG,
    )

    stem_wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Mauersteg T14",
        predefined_type="RETAININGWALL",
    )
    stem_wall.ObjectPlacement = local_placement
    stem_wall.Representation = stem_shape

    # ---------- Element 2: Foundation slab ----------
    found_shape, found_faces, found_pts = build_brep(
        ifc, body, R_INNER, FOUND_R_OUTER, FOUND_Z_BOT, FOUND_Z_TOP, N_SEG,
    )

    # Need a separate local placement instance (same coords) for the foundation
    placement2 = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement2 = ifc.createIfcLocalPlacement(None, placement2)

    found_wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Fundament T14",
        predefined_type="RETAININGWALL",
    )
    found_wall.ObjectPlacement = local_placement2
    found_wall.Representation = found_shape

    # Assign both to facility
    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[stem_wall, found_wall], relating_structure=facility,
    )

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T14_curved_l_profile.ifc")
    ifc.write(out_path)

    print(f"Written: {out_path}")
    print(f"  Stem:       {stem_faces} faces, {stem_pts} unique points")
    print(f"  Foundation: {found_faces} faces, {found_pts} unique points")

    # Sanity check: approximate volumes
    # V = (arc_rad / 2) * (R_outer^2 - R_inner^2) * height
    arc_rad = math.radians(ARC_DEG)
    v_stem = (arc_rad / 2.0) * (STEM_R_OUTER**2 - R_INNER**2) * (STEM_Z_TOP - STEM_Z_BOT)
    v_found = (arc_rad / 2.0) * (FOUND_R_OUTER**2 - R_INNER**2) * (FOUND_Z_TOP - FOUND_Z_BOT)
    print(f"  Stem volume (approx):       {v_stem:.4f} m3")
    print(f"  Foundation volume (approx): {v_found:.4f} m3")

    # Verify element count
    walls = ifc.by_type("IfcWall")
    print(f"  IfcWall entities: {len(walls)}")
    for w in walls:
        print(f"    - {w.Name} (PredefinedType={w.PredefinedType})")


if __name__ == "__main__":
    main()
