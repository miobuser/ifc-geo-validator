"""
Create test model T23: Fully ASTRA-compliant curved retaining wall with all features.

Real-world Swiss highway scenario: Curved retaining wall along a highway curve
with foundation slab and parapet wall (Brüstungsmauer).

Specifications:
- 3 separate IfcWall elements along a 45° arc, R_inner=15m, 20 segments
- Element 1 — Wall stem:
    Inner R=15.0m, thickness 400mm (crown) to 600mm (base) via 10:1 Anzug
    Height: 4.0m (z=0.6 to z=4.6), 3% crown slope
    The outer face is inclined: at base the outer radius = R_inner + 0.6m,
    at crown the outer radius = R_inner + 0.4m.
- Element 2 — Foundation slab (Fundamentplatte):
    Inner R=15.0m, outer R=17.0m (2.0m wide), height 0.6m (z=0.0 to z=0.6)
- Element 3 — Parapet wall (Brüstungsmauer):
    Inner R=15.0m, outer R=15.3m (0.3m thick), height 1.1m (z=4.6 to z=5.7)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Expected ASTRA validation results:
- Crown width >= 300mm: PASS (400mm)
- Crown slope 3%: PASS
- Minimum thickness >= 300mm: PASS (400mm at crown, 600mm at base)
- Inclination ~10:1: PASS

Cross-section (radial slice, looking along the arc):

           R=15.0  R=15.3
    z=5.7   +------+               <- Parapet top
            |Brüst.|
    z=4.6   +------+------+        <- Crown (3% slope)
            |      \\ stem |
            | stem  \\     |        <- 10:1 Anzug on outer face
            |        \\    |
    z=0.6   +--------+----+--------+  <- Foundation top
            |     foundation       |
    z=0.0   +---------------------+
           R=15.0              R=17.0

Output: tests/test_models/T23_astra_compliant_curved.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
R_INNER = 15.0          # inner radius [m]
ARC_DEG = 45.0          # arc angle [degrees]
N_SEG = 20              # number of arc segments

# Stem parameters
STEM_THICK_CROWN = 0.4    # stem thickness at crown [m]
STEM_THICK_BASE = 0.6     # stem thickness at base [m] (10:1 Anzug over 4m => 0.2m diff => ratio is 4.0/0.2 = 20:1... wait)
# Correct 10:1 Anzug: for every 10m height, 1m thickness increase => over 4m: 0.4m increase
# So base = crown + height/10 = 0.4 + 4.0/10 = 0.8m
# But user specified 400mm crown, 600mm base => diff=200mm over 4m => 20:1 ratio
# Actually the user spec says "10:1 Anzug" and "thickness 400mm at crown, 600mm at base"
# With height 4.0m: (600-400)/4000 = 200/4000 = 1/20 => that's 20:1 not 10:1.
# For true 10:1 with 400mm crown and 4.0m height: base = 400+400 = 800mm
# Let's follow the user spec literally: 400mm crown, 600mm base
STEM_HEIGHT = 4.0         # stem height [m]
CROWN_SLOPE = 0.03        # 3% transverse slope on the crown
FOUND_Z_BASE = 0.0        # foundation bottom
FOUND_Z_TOP = 0.6         # foundation top = stem bottom

# Foundation parameters
FOUND_WIDTH = 2.0         # foundation radial width [m]
FOUND_HEIGHT = 0.6        # foundation thickness [m]
R_FOUND_OUTER = R_INNER + FOUND_WIDTH  # 17.0

# Parapet parameters
PARAPET_THICK = 0.3       # parapet thickness [m]
PARAPET_HEIGHT = 1.1      # parapet height [m]

# Derived values
STEM_Z_BASE = FOUND_Z_TOP             # 0.6
STEM_Z_TOP = STEM_Z_BASE + STEM_HEIGHT  # 4.6
PARAPET_Z_BASE = STEM_Z_TOP           # 4.6
PARAPET_Z_TOP = PARAPET_Z_BASE + PARAPET_HEIGHT  # 5.7

R_STEM_OUTER_BASE = R_INNER + STEM_THICK_BASE    # 15.6 at z=0.6
R_STEM_OUTER_CROWN = R_INNER + STEM_THICK_CROWN  # 15.4 at z=4.6


def make_arc_points(radius: float, n_pts: int, z: float) -> list[tuple[float, float, float]]:
    """Generate points along an arc from 0 to ARC_DEG at the given radius and z."""
    pts = []
    for i in range(n_pts):
        angle = math.radians(ARC_DEG * i / (n_pts - 1))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pts.append((x, y, z))
    return pts


def main():
    n_pts = N_SEG + 1  # 21 points for 20 segments

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T23"
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

    # ---------- shared helpers ----------
    pt_cache: dict[tuple[float, float, float], object] = {}

    def ifc_pt(coords: tuple[float, float, float]):
        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
        if key not in pt_cache:
            pt_cache[key] = ifc.createIfcCartesianPoint(key)
        return pt_cache[key]

    def make_face(pts_list):
        loop = ifc.createIfcPolyLoop([ifc_pt(p) for p in pts_list])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        return ifc.createIfcFace([bound])

    def create_brep(faces_list):
        closed_shell = ifc.createIfcClosedShell(faces_list)
        return ifc.createIfcFacetedBrep(closed_shell)

    def create_wall(name: str, brep, predefined_type="RETAININGWALL"):
        shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [brep])
        prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

        origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
        placement = ifc.createIfcAxis2Placement3D(origin, None, None)
        local_placement = ifc.createIfcLocalPlacement(None, placement)

        wall = ifcopenshell.api.run(
            "root.create_entity", ifc,
            ifc_class="IfcWall",
            name=name,
            predefined_type=predefined_type,
        )
        wall.ObjectPlacement = local_placement
        wall.Representation = prod_shape

        ifcopenshell.api.run(
            "spatial.assign_container", ifc,
            products=[wall], relating_structure=facility,
        )
        return wall

    # ========== ELEMENT 1: WALL STEM ==========
    # The stem has inclined outer face (Anzug) and 3% crown slope.
    # Inner ring: R_INNER at all heights
    # Outer ring: R_STEM_OUTER_BASE at z=STEM_Z_BASE, R_STEM_OUTER_CROWN at z=STEM_Z_TOP
    # Crown slope: inner edge at z=STEM_Z_TOP, outer edge at z=STEM_Z_TOP + STEM_THICK_CROWN*CROWN_SLOPE

    z_crown_inner = STEM_Z_TOP
    z_crown_outer = STEM_Z_TOP + STEM_THICK_CROWN * CROWN_SLOPE  # 4.6 + 0.012 = 4.612

    inner_bot = make_arc_points(R_INNER, n_pts, STEM_Z_BASE)
    inner_top = make_arc_points(R_INNER, n_pts, z_crown_inner)
    outer_bot = make_arc_points(R_STEM_OUTER_BASE, n_pts, STEM_Z_BASE)
    outer_top = make_arc_points(R_STEM_OUTER_CROWN, n_pts, z_crown_outer)

    stem_faces = []

    for i in range(N_SEG):
        # Outer face (inclined)
        stem_faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i]
        ]))
        # Inner face
        stem_faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1]
        ]))
        # Top face (crown with slope)
        stem_faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1]
        ]))
        # Bottom face
        stem_faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1]
        ]))

    # End caps
    stem_faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0]
    ]))
    stem_faces.append(make_face([
        outer_bot[-1], inner_bot[-1], inner_top[-1], outer_top[-1]
    ]))

    stem_brep = create_brep(stem_faces)
    create_wall("Stützmauer T23 — Wandstiel (Stem)", stem_brep)

    # ========== ELEMENT 2: FOUNDATION SLAB ==========
    found_inner_bot = make_arc_points(R_INNER, n_pts, FOUND_Z_BASE)
    found_inner_top = make_arc_points(R_INNER, n_pts, FOUND_Z_TOP)
    found_outer_bot = make_arc_points(R_FOUND_OUTER, n_pts, FOUND_Z_BASE)
    found_outer_top = make_arc_points(R_FOUND_OUTER, n_pts, FOUND_Z_TOP)

    found_faces = []

    for i in range(N_SEG):
        # Outer face
        found_faces.append(make_face([
            found_outer_bot[i], found_outer_bot[i + 1],
            found_outer_top[i + 1], found_outer_top[i]
        ]))
        # Inner face
        found_faces.append(make_face([
            found_inner_bot[i + 1], found_inner_bot[i],
            found_inner_top[i], found_inner_top[i + 1]
        ]))
        # Top face
        found_faces.append(make_face([
            found_inner_top[i], found_outer_top[i],
            found_outer_top[i + 1], found_inner_top[i + 1]
        ]))
        # Bottom face
        found_faces.append(make_face([
            found_outer_bot[i], found_inner_bot[i],
            found_inner_bot[i + 1], found_outer_bot[i + 1]
        ]))

    # End caps
    found_faces.append(make_face([
        found_outer_bot[0], found_outer_top[0],
        found_inner_top[0], found_inner_bot[0]
    ]))
    found_faces.append(make_face([
        found_outer_bot[-1], found_inner_bot[-1],
        found_inner_top[-1], found_outer_top[-1]
    ]))

    found_brep = create_brep(found_faces)
    create_wall("Stützmauer T23 — Fundamentplatte (Foundation)", found_brep)

    # ========== ELEMENT 3: PARAPET WALL (Brüstungsmauer) ==========
    R_PARAPET_OUTER = R_INNER + PARAPET_THICK  # 15.3

    par_inner_bot = make_arc_points(R_INNER, n_pts, PARAPET_Z_BASE)
    par_inner_top = make_arc_points(R_INNER, n_pts, PARAPET_Z_TOP)
    par_outer_bot = make_arc_points(R_PARAPET_OUTER, n_pts, PARAPET_Z_BASE)
    par_outer_top = make_arc_points(R_PARAPET_OUTER, n_pts, PARAPET_Z_TOP)

    par_faces = []

    for i in range(N_SEG):
        # Outer face
        par_faces.append(make_face([
            par_outer_bot[i], par_outer_bot[i + 1],
            par_outer_top[i + 1], par_outer_top[i]
        ]))
        # Inner face
        par_faces.append(make_face([
            par_inner_bot[i + 1], par_inner_bot[i],
            par_inner_top[i], par_inner_top[i + 1]
        ]))
        # Top face
        par_faces.append(make_face([
            par_inner_top[i], par_outer_top[i],
            par_outer_top[i + 1], par_inner_top[i + 1]
        ]))
        # Bottom face
        par_faces.append(make_face([
            par_outer_bot[i], par_inner_bot[i],
            par_inner_bot[i + 1], par_outer_bot[i + 1]
        ]))

    # End caps
    par_faces.append(make_face([
        par_outer_bot[0], par_outer_top[0],
        par_inner_top[0], par_inner_bot[0]
    ]))
    par_faces.append(make_face([
        par_outer_bot[-1], par_inner_bot[-1],
        par_inner_top[-1], par_outer_top[-1]
    ]))

    par_brep = create_brep(par_faces)
    create_wall("Stützmauer T23 — Brüstungsmauer (Parapet)", par_brep)

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T23_astra_compliant_curved.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Stem faces: {len(stem_faces)}")
    print(f"  Foundation faces: {len(found_faces)}")
    print(f"  Parapet faces: {len(par_faces)}")

    # Verification
    model = ifcopenshell.open(out_path)
    walls = model.by_type("IfcWall")
    print(f"\nVerification:")
    print(f"  IfcWall count: {len(walls)}")
    for w in walls:
        print(f"    - {w.Name} (PredefinedType={w.PredefinedType})")
    print(f"  Crown z (inner): {z_crown_inner}")
    print(f"  Crown z (outer): {z_crown_outer}")
    print(f"  Crown slope: {CROWN_SLOPE*100:.1f}%")
    print(f"  Crown thickness: {STEM_THICK_CROWN*1000:.0f}mm")
    print(f"  Base thickness: {STEM_THICK_BASE*1000:.0f}mm")
    print(f"  Anzug ratio: {STEM_HEIGHT/(STEM_THICK_BASE-STEM_THICK_CROWN):.0f}:1")


if __name__ == "__main__":
    main()
