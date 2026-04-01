"""
Create test model T22: Retaining wall with terrain geometry on IfcSite.

Specifications:
- IfcWall (8x0.4x3m box, PredefinedType=RETAININGWALL)
  - Located at x=0..8, y=0..0.4, z=0..3
  - IfcFacetedBrep (6 faces)
  - Name: "Stützmauer T22"
- IfcSite with terrain mesh (IfcTriangulatedFaceSet)
  - 5x5 grid: x in {-2,0,4,8,10}, y in {-2,0,2,5,8}
  - z_terrain(y) = max(0, y) — 45° slope behind wall
  - Triangulated: 2 triangles per grid cell (4x4 cells = 32 triangles)
- IFC4X3_ADD2
- Hierarchy: IfcProject > IfcSite (with terrain) > IfcFacility > IfcWall

Output: tests/test_models/T22_with_terrain.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api


# ---------- wall parameters ----------
WALL_LENGTH = 8.0   # x-direction
WALL_THICK = 0.4    # y-direction (thickness)
WALL_HEIGHT = 3.0   # z-direction

# ---------- terrain parameters ----------
X_VALS = [-2.0, 0.0, 4.0, 8.0, 10.0]
Y_VALS = [-2.0, 0.0, 2.0, 5.0, 8.0]


def terrain_z(y: float) -> float:
    """Terrain height: flat at y<=0, 45° slope for y>0."""
    return max(0.0, y * 1.0)


def main():
    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T22"
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

    # ========== TERRAIN GEOMETRY ON IfcSite ==========
    # Build grid vertices
    terrain_vertices = []
    # Index map: vertex_index[iy][ix] -> 1-based index
    vertex_index = []
    idx = 1
    for iy, y in enumerate(Y_VALS):
        row = []
        for ix, x in enumerate(X_VALS):
            z = terrain_z(y)
            terrain_vertices.append((x, y, z))
            row.append(idx)
            idx += 1
        vertex_index.append(row)

    # Triangulate grid: 2 triangles per cell (1-based indices)
    triangles = []
    ny = len(Y_VALS)
    nx = len(X_VALS)
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            # Four corners of cell
            v00 = vertex_index[iy][ix]
            v10 = vertex_index[iy][ix + 1]
            v01 = vertex_index[iy + 1][ix]
            v11 = vertex_index[iy + 1][ix + 1]
            # Two triangles (CCW when viewed from above = +Z)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    # Create IfcTriangulatedFaceSet
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
    site_placement_3d = ifc.createIfcAxis2Placement3D(site_origin, None, None)
    site_local_placement = ifc.createIfcLocalPlacement(None, site_placement_3d)

    site.ObjectPlacement = site_local_placement
    site.Representation = terrain_prod_shape

    # ========== WALL GEOMETRY (IfcFacetedBrep, 6 faces) ==========
    # 8 vertices of the box
    #   x: 0..8, y: 0..0.4, z: 0..3
    box_verts = [
        (0.0, 0.0, 0.0),        # 0: front-left-bottom
        (WALL_LENGTH, 0.0, 0.0), # 1: front-right-bottom
        (WALL_LENGTH, WALL_THICK, 0.0),  # 2: back-right-bottom
        (0.0, WALL_THICK, 0.0),  # 3: back-left-bottom
        (0.0, 0.0, WALL_HEIGHT),        # 4: front-left-top
        (WALL_LENGTH, 0.0, WALL_HEIGHT), # 5: front-right-top
        (WALL_LENGTH, WALL_THICK, WALL_HEIGHT),  # 6: back-right-top
        (0.0, WALL_THICK, WALL_HEIGHT),  # 7: back-left-top
    ]

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

    v = box_verts
    faces = [
        # Front face (y=0, normal -Y)
        make_face([v[0], v[1], v[5], v[4]]),
        # Back face (y=0.4, normal +Y)
        make_face([v[2], v[3], v[7], v[6]]),
        # Left face (x=0, normal -X)
        make_face([v[3], v[0], v[4], v[7]]),
        # Right face (x=8, normal +X)
        make_face([v[1], v[2], v[6], v[5]]),
        # Top face (z=3, normal +Z) — crown
        make_face([v[4], v[5], v[6], v[7]]),
        # Bottom face (z=0, normal -Z)
        make_face([v[0], v[3], v[2], v[1]]),
    ]

    closed_shell = ifc.createIfcClosedShell(faces)
    brep = ifc.createIfcFacetedBrep(closed_shell)

    wall_shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [brep])
    wall_prod_shape = ifc.createIfcProductDefinitionShape(None, None, [wall_shape_rep])

    # Placement for wall
    wall_origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    wall_placement_3d = ifc.createIfcAxis2Placement3D(wall_origin, None, None)
    wall_local_placement = ifc.createIfcLocalPlacement(site_local_placement, wall_placement_3d)

    # IfcWall
    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stützmauer T22",
        predefined_type="RETAININGWALL",
    )
    wall.ObjectPlacement = wall_local_placement
    wall.Representation = wall_prod_shape

    # Assign wall to facility
    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[wall], relating_structure=facility,
    )

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T22_with_terrain.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Wall faces: {len(faces)}")
    print(f"  Terrain vertices: {len(terrain_vertices)}")
    print(f"  Terrain triangles: {len(triangles)}")

    # Verify
    model = ifcopenshell.open(out_path)
    walls = model.by_type("IfcWall")
    sites = model.by_type("IfcSite")
    print(f"\nVerification:")
    print(f"  IfcWall count: {len(walls)}")
    print(f"  IfcWall name: {walls[0].Name if walls else 'N/A'}")
    print(f"  IfcWall PredefinedType: {walls[0].PredefinedType if walls else 'N/A'}")
    print(f"  IfcSite count: {len(sites)}")
    print(f"  IfcSite has geometry: {sites[0].Representation is not None if sites else False}")

    # Check terrain face set
    if sites and sites[0].Representation:
        for rep in sites[0].Representation.Representations:
            for item in rep.Items:
                print(f"  Site geometry type: {item.is_a()}")
                if item.is_a("IfcTriangulatedFaceSet"):
                    coords = item.Coordinates.CoordList
                    faces_list = item.CoordIndex
                    print(f"  Terrain coord count: {len(coords)}")
                    print(f"  Terrain triangle count: {len(faces_list)}")


if __name__ == "__main__":
    main()
