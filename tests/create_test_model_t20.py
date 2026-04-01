"""
Create test model T20: Simple box wall using IfcTriangulatedFaceSet.

Specifications:
- Dimensions: 8.0m x 0.4m x 3.0m (same as T1)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcTriangulatedFaceSet (tessellated representation)
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T20_triangulated.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api


def main():
    # ---------- parameters ----------
    L = 8.0   # length along X
    W = 0.4   # width along Y
    H = 3.0   # height along Z

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T20")

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
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[facility], relating_object=site)

    # ---------- build IfcTriangulatedFaceSet ----------
    # 8 vertices of the box (1-based indexing for CoordIndex)
    #   1: (0,   0, 0)     5: (0,   0, H)
    #   2: (L,   0, 0)     6: (L,   0, H)
    #   3: (L,   W, 0)     7: (L,   W, H)
    #   4: (0,   W, 0)     8: (0,   W, H)
    coords = [
        [0.0, 0.0, 0.0],  # 1
        [L,   0.0, 0.0],  # 2
        [L,   W,   0.0],  # 3
        [0.0, W,   0.0],  # 4
        [0.0, 0.0, H],    # 5
        [L,   0.0, H],    # 6
        [L,   W,   H],    # 7
        [0.0, W,   H],    # 8
    ]

    point_list = ifc.createIfcCartesianPointList3D(coords)

    # Triangulated faces (1-based indices, consistent winding for outward normals)
    triangles = [
        [1, 3, 2], [1, 4, 3],  # bottom  (z=0, normal -Z)
        [5, 6, 7], [5, 7, 8],  # top     (z=H, normal +Z)
        [1, 2, 6], [1, 6, 5],  # front   (y=0, normal -Y)
        [3, 4, 8], [3, 8, 7],  # back    (y=W, normal +Y)
        [1, 5, 8], [1, 8, 4],  # left    (x=0, normal -X)
        [2, 3, 7], [2, 7, 6],  # right   (x=L, normal +X)
    ]

    face_set = ifc.createIfcTriangulatedFaceSet(point_list, None, None, triangles)

    # Shape representation — RepresentationType = "Tessellation"
    shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Tessellation", [face_set])
    prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

    # Placement
    origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    placement = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement = ifc.createIfcLocalPlacement(None, placement)

    # IfcWall
    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stützmauer T20 — IfcTriangulatedFaceSet",
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
    out_path = os.path.join(out_dir, "T20_triangulated.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Vertices: {len(coords)}")
    print(f"  Triangles: {len(triangles)}")
    print(f"  Expected volume: {L * W * H:.2f} m³")


if __name__ == "__main__":
    main()
