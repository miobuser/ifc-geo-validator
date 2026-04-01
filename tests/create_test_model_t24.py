"""
Create test model T24: Highway retaining wall with terrain and road.

Real-world Swiss highway scenario: Straight retaining wall along a highway
with a foundation slab and sloped terrain behind the wall.

Specifications:
- Element 1 — Retaining wall:
    Straight, 12m long (x-direction), 0.35m thick (y-direction)
    Height: 3.5m, 10:1 Anzug on the back face (earth side, +y)
    Crown with 3% transverse slope
    IfcWall, PredefinedType=RETAININGWALL
- Element 2 — Foundation slab:
    12m x 2.0m x 0.5m, sitting under the wall (z=0 to z=0.5)
    Wall starts at z=0.5
    IfcSlab (or IfcWall for simplicity), PredefinedType=NOTDEFINED
- Element 3 — Terrain on IfcSite:
    8x8 grid of points covering x=-5 to x=17, y=-5 to y=15
    Terrain rises from z=0 on the road side (y<0) to z=5 behind wall (y>2)
    IfcTriangulatedFaceSet on IfcSite.Representation

Layout (plan view, y-axis pointing into hillside):
    y=-5  Road surface (flat, z=0)
    y=0   Wall front face
    y=0.35  Wall back face (at crown; thicker at base due to Anzug)
    y=2.0 Foundation outer edge
    y=5..15  Hillside terrain rising to z=5

Cross-section (looking along x-axis):

    Terrain z=5     _______________
                   /
                  /  Hillside
    z=4.0  +---+/
           | W |
           |   |    <- 10:1 Anzug
    z=0.5  | +-+--------+
           | |Foundation |
    z=0    +-+-----------+
    y=0   y=0.35       y=2.0

Tests: L5 (stem+foundation) + L6 (terrain) together.

Output: tests/test_models/T24_highway_with_terrain.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api

# ---------- wall parameters ----------
WALL_LENGTH = 12.0      # x-direction [m]
WALL_HEIGHT = 3.5       # stem height [m]
CROWN_THICK = 0.35      # thickness at crown [m]
CROWN_SLOPE = 0.03      # 3% transverse slope

# 10:1 Anzug: for 3.5m height => 0.35m extra thickness at base
BASE_THICK = CROWN_THICK + WALL_HEIGHT / 10.0  # 0.35 + 0.35 = 0.70m

# Foundation parameters
FOUND_LENGTH = 12.0     # same as wall [m]
FOUND_WIDTH = 2.0       # y-direction [m]
FOUND_HEIGHT = 0.5      # z-direction [m]

# Vertical positions
FOUND_Z_BASE = 0.0
FOUND_Z_TOP = FOUND_HEIGHT               # 0.5
STEM_Z_BASE = FOUND_Z_TOP                # 0.5
STEM_Z_TOP = STEM_Z_BASE + WALL_HEIGHT   # 4.0

# Crown slope: front edge (y=0) at z=STEM_Z_TOP, back edge higher
Z_CROWN_FRONT = STEM_Z_TOP
Z_CROWN_BACK = STEM_Z_TOP + CROWN_THICK * CROWN_SLOPE  # 4.0 + 0.0105 = 4.0105

# ---------- terrain parameters ----------
# 8x8 grid covering x=-5..17, y=-5..15
X_TERRAIN = [-5.0, -1.0, 3.0, 6.0, 9.0, 12.0, 14.0, 17.0]
Y_TERRAIN = [-5.0, -2.0, 0.0, 1.0, 2.0, 5.0, 10.0, 15.0]


def terrain_z(x: float, y: float) -> float:
    """
    Terrain height model:
    - y <= 0: road surface, flat at z=0
    - 0 < y <= 2: transition zone (retained earth at wall/foundation level)
    - y > 2: hillside rising linearly to z=5 at y=15
    """
    if y <= 0.0:
        return 0.0
    elif y <= 2.0:
        # Transition: ground level at foundation top
        return FOUND_Z_TOP * (y / 2.0)
    else:
        # Hillside: linear rise from z=0.5 at y=2 to z=5.0 at y=15
        t = (y - 2.0) / (15.0 - 2.0)
        return FOUND_Z_TOP + t * (5.0 - FOUND_Z_TOP)


def main():
    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T24"
    )

    # Units (metres)
    length_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="LENGTHUNIT")
    area_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="AREAUNIT")
    volume_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="VOLUMEUNIT")
    ifcopenshell.api.run("unit.assign_unit", ifc, units=[length_unit, area_unit, volume_unit])

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

    def create_placement():
        origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
        placement = ifc.createIfcAxis2Placement3D(origin, None, None)
        return ifc.createIfcLocalPlacement(None, placement)

    # ========== ELEMENT 1: RETAINING WALL STEM ==========
    # The wall runs along the x-axis from x=0 to x=12.
    # Front face at y=0 (road side), back face inclined (earth side).
    # At base (z=0.5): y_back = BASE_THICK = 0.70
    # At crown (z=4.0): y_back = CROWN_THICK = 0.35
    # Crown slope: front z=4.0, back z=4.0105

    # 8 vertices of the trapezoidal prism (stem with Anzug)
    #   Front face (y=0): vertical
    #   Back face (y=BASE_THICK at bottom, y=CROWN_THICK at top): inclined
    stem_verts = [
        # Bottom (z = STEM_Z_BASE = 0.5)
        (0.0,         0.0,        STEM_Z_BASE),    # 0: front-left-bottom
        (WALL_LENGTH, 0.0,        STEM_Z_BASE),    # 1: front-right-bottom
        (WALL_LENGTH, BASE_THICK, STEM_Z_BASE),    # 2: back-right-bottom
        (0.0,         BASE_THICK, STEM_Z_BASE),    # 3: back-left-bottom
        # Top/crown (z varies with crown slope)
        (0.0,         0.0,          Z_CROWN_FRONT),  # 4: front-left-top
        (WALL_LENGTH, 0.0,          Z_CROWN_FRONT),  # 5: front-right-top
        (WALL_LENGTH, CROWN_THICK,  Z_CROWN_BACK),   # 6: back-right-top
        (0.0,         CROWN_THICK,  Z_CROWN_BACK),   # 7: back-left-top
    ]

    v = stem_verts
    stem_faces = [
        # Front face (y=0, normal -Y) — vertical
        make_face([v[0], v[1], v[5], v[4]]),
        # Back face (inclined, normal +Y) — Anzug
        make_face([v[2], v[3], v[7], v[6]]),
        # Left face (x=0, normal -X) — trapezoid
        make_face([v[3], v[0], v[4], v[7]]),
        # Right face (x=12, normal +X) — trapezoid
        make_face([v[1], v[2], v[6], v[5]]),
        # Top face (crown with slope, normal ~+Z)
        make_face([v[4], v[5], v[6], v[7]]),
        # Bottom face (z=0.5, normal -Z)
        make_face([v[0], v[3], v[2], v[1]]),
    ]

    stem_shell = ifc.createIfcClosedShell(stem_faces)
    stem_brep = ifc.createIfcFacetedBrep(stem_shell)

    stem_shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [stem_brep])
    stem_prod_shape = ifc.createIfcProductDefinitionShape(None, None, [stem_shape_rep])

    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stützmauer T24 — Wandstiel (Stem)",
        predefined_type="RETAININGWALL",
    )
    wall.ObjectPlacement = create_placement()
    wall.Representation = stem_prod_shape

    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[wall], relating_structure=facility,
    )

    # ========== ELEMENT 2: FOUNDATION SLAB ==========
    # Box: x=0..12, y=0..2.0, z=0..0.5
    found_verts = [
        (0.0,         0.0,         FOUND_Z_BASE),   # 0
        (FOUND_LENGTH, 0.0,        FOUND_Z_BASE),   # 1
        (FOUND_LENGTH, FOUND_WIDTH, FOUND_Z_BASE),  # 2
        (0.0,         FOUND_WIDTH, FOUND_Z_BASE),   # 3
        (0.0,         0.0,         FOUND_Z_TOP),    # 4
        (FOUND_LENGTH, 0.0,        FOUND_Z_TOP),    # 5
        (FOUND_LENGTH, FOUND_WIDTH, FOUND_Z_TOP),   # 6
        (0.0,         FOUND_WIDTH, FOUND_Z_TOP),    # 7
    ]

    fv = found_verts
    found_faces = [
        make_face([fv[0], fv[1], fv[5], fv[4]]),  # Front
        make_face([fv[2], fv[3], fv[7], fv[6]]),  # Back
        make_face([fv[3], fv[0], fv[4], fv[7]]),  # Left
        make_face([fv[1], fv[2], fv[6], fv[5]]),  # Right
        make_face([fv[4], fv[5], fv[6], fv[7]]),  # Top
        make_face([fv[0], fv[3], fv[2], fv[1]]),  # Bottom
    ]

    found_shell = ifc.createIfcClosedShell(found_faces)
    found_brep = ifc.createIfcFacetedBrep(found_shell)

    found_shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [found_brep])
    found_prod_shape = ifc.createIfcProductDefinitionShape(None, None, [found_shape_rep])

    foundation = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stützmauer T24 — Fundamentplatte (Foundation)",
        predefined_type="RETAININGWALL",
    )
    foundation.ObjectPlacement = create_placement()
    foundation.Representation = found_prod_shape

    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[foundation], relating_structure=facility,
    )

    # ========== ELEMENT 3: TERRAIN ON IfcSite ==========
    terrain_vertices = []
    vertex_index = []
    idx = 1
    for iy, y in enumerate(Y_TERRAIN):
        row = []
        for ix, x in enumerate(X_TERRAIN):
            z = terrain_z(x, y)
            terrain_vertices.append((round(x, 4), round(y, 4), round(z, 4)))
            row.append(idx)
            idx += 1
        vertex_index.append(row)

    # Triangulate grid: 2 triangles per cell (1-based indices)
    triangles = []
    ny = len(Y_TERRAIN)
    nx = len(X_TERRAIN)
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = vertex_index[iy][ix]
            v10 = vertex_index[iy][ix + 1]
            v01 = vertex_index[iy + 1][ix]
            v11 = vertex_index[iy + 1][ix + 1]
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    point_list = ifc.createIfcCartesianPointList3D(terrain_vertices)
    face_set = ifc.createIfcTriangulatedFaceSet(point_list, None, None, triangles)

    terrain_shape_rep = ifc.createIfcShapeRepresentation(
        body, "Body", "Tessellation", [face_set]
    )
    terrain_prod_shape = ifc.createIfcProductDefinitionShape(
        None, None, [terrain_shape_rep]
    )

    # Placement for site
    site_origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    site_placement = ifc.createIfcAxis2Placement3D(site_origin, None, None)
    site_local_placement = ifc.createIfcLocalPlacement(None, site_placement)

    site.ObjectPlacement = site_local_placement
    site.Representation = terrain_prod_shape

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T24_highway_with_terrain.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Stem faces: {len(stem_faces)}")
    print(f"  Foundation faces: {len(found_faces)}")
    print(f"  Terrain vertices: {len(terrain_vertices)}")
    print(f"  Terrain triangles: {len(triangles)}")

    # Verification
    model = ifcopenshell.open(out_path)
    walls = model.by_type("IfcWall")
    sites = model.by_type("IfcSite")
    print(f"\nVerification:")
    print(f"  IfcWall count: {len(walls)}")
    for w in walls:
        print(f"    - {w.Name} (PredefinedType={w.PredefinedType})")
    print(f"  IfcSite count: {len(sites)}")
    print(f"  IfcSite has geometry: {sites[0].Representation is not None if sites else False}")
    print(f"\nWall dimensions:")
    print(f"  Length: {WALL_LENGTH}m")
    print(f"  Height: {WALL_HEIGHT}m")
    print(f"  Crown thickness: {CROWN_THICK*1000:.0f}mm")
    print(f"  Base thickness: {BASE_THICK*1000:.0f}mm")
    print(f"  Anzug: 10:1 ({WALL_HEIGHT}m / {BASE_THICK-CROWN_THICK:.2f}m)")
    print(f"  Crown slope: {CROWN_SLOPE*100:.1f}%")


if __name__ == "__main__":
    main()
