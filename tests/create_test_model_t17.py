"""
Create test model T17: Curved retaining wall with variable height and front inclination.

Specifications:
- 45° arc in plan view, 16 segments (17 angle positions)
- Inner radius R_inner=10.0 m (back face vertical)
- Outer radius at base: R_outer_base=10.5 m (constant along arc)
- Outer radius at crown: R_outer_top=10.35 m (constant, ~10:1 inclination)
- Variable height: 3.0 m at start (angle=0°), 5.0 m at end (angle=45°)
- Crown with 3% transverse slope
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T17_curved_variable.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
R_INNER = 10.0              # inner radius [m] (back face, vertical)
R_OUTER_BASE = 10.5         # outer radius at base [m]
R_OUTER_TOP = 10.35         # outer radius at crown [m] (front face inclined)
HEIGHT_START = 3.0           # wall height at angle=0° [m]
HEIGHT_END = 5.0             # wall height at angle=45° [m]
CROWN_SLOPE = 0.03           # 3% transverse slope on the crown
N_SEG = 16                   # number of arc segments
ARC_DEG = 45.0               # arc angle [degrees]


def main():
    n_pts = N_SEG + 1  # 17 points for 16 segments

    # ---------- generate geometry vertices ----------
    # For each angle position, compute the 4 corner points of the cross-section.
    inner_bot = []
    outer_bot = []
    inner_top = []
    outer_top = []

    for i in range(n_pts):
        t = i / N_SEG  # parameter 0..1
        angle = math.radians(ARC_DEG * t)
        height = HEIGHT_START + (HEIGHT_END - HEIGHT_START) * t

        # Crown z: inner edge at full height, outer edge with 3% slope
        z_crown_inner = height
        z_crown_outer = height + (R_OUTER_TOP - R_INNER) * CROWN_SLOPE

        # Bottom inner
        inner_bot.append((R_INNER * math.cos(angle), R_INNER * math.sin(angle), 0.0))
        # Bottom outer
        outer_bot.append((R_OUTER_BASE * math.cos(angle), R_OUTER_BASE * math.sin(angle), 0.0))
        # Top inner
        inner_top.append((R_INNER * math.cos(angle), R_INNER * math.sin(angle), z_crown_inner))
        # Top outer
        outer_top.append((R_OUTER_TOP * math.cos(angle), R_OUTER_TOP * math.sin(angle), z_crown_outer))

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T17",
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
        "root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände",
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1",
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[facility], relating_object=site)

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

    for i in range(N_SEG):
        # --- Outer face (front, inclined) ---
        faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i],
        ]))

        # --- Inner face (back, vertical) ---
        faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1],
        ]))

        # --- Top face (crown with 3% slope) ---
        faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1],
        ]))

        # --- Bottom face (foundation) ---
        faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1],
        ]))

    # --- End cap at angle=0° (start) ---
    faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0],
    ]))

    # --- End cap at angle=45° (end) ---
    faces.append(make_face([
        outer_bot[-1], inner_bot[-1], inner_top[-1], outer_top[-1],
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
        name="Stützmauer T17 \u2014 Gekrümmt mit variabler Höhe (3\u21925m, 45° Bogen, 10:1)",
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
    out_path = os.path.join(out_dir, "T17_curved_variable.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Summary of variable heights
    for i in [0, N_SEG // 2, N_SEG]:
        t = i / N_SEG
        h = HEIGHT_START + (HEIGHT_END - HEIGHT_START) * t
        a = ARC_DEG * t
        print(f"  Angle {a:5.1f}°: height={h:.2f}m, crown_z_inner={h:.3f}, "
              f"crown_z_outer={h + (R_OUTER_TOP - R_INNER) * CROWN_SLOPE:.3f}")

    # Approximate volume (trapezoidal average height * annular sector)
    h_avg = (HEIGHT_START + HEIGHT_END) / 2.0
    arc_rad = math.radians(ARC_DEG)
    # V ~ arc_angle/2 * (R_outer_avg^2 - R_inner^2) * h_avg
    r_outer_avg = (R_OUTER_BASE + R_OUTER_TOP) / 2.0
    v_approx = (arc_rad / 2.0) * (r_outer_avg**2 - R_INNER**2) * h_avg
    print(f"  Approximate volume: {v_approx:.4f} m³")


if __name__ == "__main__":
    main()
