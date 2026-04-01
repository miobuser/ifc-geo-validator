"""
Create test model T8: Curved retaining wall (90° arc).

Specifications:
- 90° arc in plan view (quarter circle), 20 segments
- Inner radius R=10.0 m, wall thickness 0.4 m (outer R=10.4 m)
- Height: 3.0 m, constant thickness (no inclination)
- Crown with 3% transverse slope (inner edge z=3.0, outer edge z=3.012)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T8_curved_wall.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
R_INNER = 10.0          # inner radius [m]
THICKNESS = 0.4         # wall thickness [m]
R_OUTER = R_INNER + THICKNESS
HEIGHT = 3.0            # wall height [m]
CROWN_SLOPE = 0.03      # 3 % transverse slope on the crown
N_SEG = 20              # number of arc segments
ARC_DEG = 90.0          # arc angle [degrees]

# Crown: inner edge at z=HEIGHT, outer edge at z=HEIGHT + THICKNESS * CROWN_SLOPE
Z_CROWN_INNER = HEIGHT
Z_CROWN_OUTER = HEIGHT + THICKNESS * CROWN_SLOPE  # 3.0 + 0.012 = 3.012


def make_arc_points(radius: float, n_pts: int, z: float) -> list[tuple[float, float, float]]:
    """Generate points along an arc from 0° to ARC_DEG at the given radius and z."""
    pts = []
    for i in range(n_pts):
        angle = math.radians(ARC_DEG * i / (n_pts - 1))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pts.append((x, y, z))
    return pts


def main():
    # ---------- generate geometry vertices ----------
    n_pts = N_SEG + 1  # 21 points for 20 segments

    # Bottom ring (z=0)
    outer_bot = make_arc_points(R_OUTER, n_pts, 0.0)
    inner_bot = make_arc_points(R_INNER, n_pts, 0.0)

    # Top ring (crown with slope)
    outer_top = make_arc_points(R_OUTER, n_pts, Z_CROWN_OUTER)
    inner_top = make_arc_points(R_INNER, n_pts, Z_CROWN_INNER)

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T8")

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
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[facility], relating_object=site)

    # ---------- build IfcFacetedBrep ----------
    # Helper: create an IfcCartesianPoint and cache
    pt_cache: dict[tuple[float, float, float], object] = {}

    def ifc_pt(coords: tuple[float, float, float]):
        # Round to avoid floating-point noise in the file
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

    for i in range(N_SEG):
        # Four corners of each segment (looking from outside):
        # outer_bot[i], outer_bot[i+1], outer_top[i+1], outer_top[i]
        # inner_bot[i], inner_bot[i+1], inner_top[i+1], inner_top[i]

        # --- Outer face (front) ---
        faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i]
        ]))

        # --- Inner face (back) ---
        faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1]
        ]))

        # --- Top face (crown) ---
        faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1]
        ]))

        # --- Bottom face (foundation) ---
        faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1]
        ]))

    # --- End cap at angle = 0° (start) ---
    faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0]
    ]))

    # --- End cap at angle = 90° (end) ---
    faces.append(make_face([
        outer_bot[-1], inner_bot[-1], inner_top[-1], outer_top[-1]
    ]))

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
        name="Stützmauer T8 — Gekrümmte Wand (90° Bogen)",
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
    out_path = os.path.join(out_dir, "T8_curved_wall.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Quick sanity check: approximate volume
    # V = (pi/4) * (R_outer^2 - R_inner^2) * H_avg
    h_avg = (Z_CROWN_INNER + Z_CROWN_OUTER) / 2.0
    v_approx = (math.pi / 4.0) * (R_OUTER**2 - R_INNER**2) * h_avg
    print(f"  Approximate volume: {v_approx:.4f} m³")
    print(f"  Crown z range: {Z_CROWN_INNER} .. {Z_CROWN_OUTER}")


if __name__ == "__main__":
    main()
