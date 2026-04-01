"""
Create test model T5 v2: T-shaped retaining wall as 2 separate IFC elements.

Replaces the old monolith T5 with a corrected version containing:
1. Main wall: 8.0m x 0.4m x 3.0m  (V=9.6 m³)
2. Spur:      2.0m x 1.5m x 2.5m  (V=7.5 m³)

Both are IfcWall with PredefinedType=RETAININGWALL, contained in one IfcFacility.
IFC4X3_ADD2 schema, IfcFacetedBrep geometry with absolute world coordinates.

Output: tests/test_models/T5_t_shaped.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api


def main():
    # ---------- box definitions (min_corner, max_corner) ----------
    boxes = [
        {
            "name": "Hauptwand T5",
            "min": (0.0, 0.0, 0.0),
            "max": (8.0, 0.4, 3.0),
        },
        {
            "name": "Sporn T5",
            "min": (3.0, 0.4, 0.0),
            "max": (5.0, 1.9, 2.5),
        },
    ]

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcProject",
        name="Thesis Test T5",
    )
    project.Description = "T5 — T-förmige Stützmauer (2 Elemente: Hauptwand + Sporn)"

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

    # ---------- shared placement at origin ----------
    origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    placement_3d = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    # ---------- helper: create box Brep ----------
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

    def make_box_brep(mn, mx):
        """Create an IfcFacetedBrep for an axis-aligned box from mn to mx."""
        x0, y0, z0 = mn
        x1, y1, z1 = mx

        # 8 vertices
        v = [
            (x0, y0, z0),  # 0: left-front-bottom
            (x1, y0, z0),  # 1: right-front-bottom
            (x1, y1, z0),  # 2: right-back-bottom
            (x0, y1, z0),  # 3: left-back-bottom
            (x0, y0, z1),  # 4: left-front-top
            (x1, y0, z1),  # 5: right-front-top
            (x1, y1, z1),  # 6: right-back-top
            (x0, y1, z1),  # 7: left-back-top
        ]

        faces = []

        # Bottom face (z=z0) -- outward normal pointing -Z
        faces.append(make_face([v[0], v[3], v[2], v[1]]))

        # Top face (z=z1) -- outward normal pointing +Z
        faces.append(make_face([v[4], v[5], v[6], v[7]]))

        # Front face (y=y0) -- outward normal pointing -Y
        faces.append(make_face([v[0], v[1], v[5], v[4]]))

        # Back face (y=y1) -- outward normal pointing +Y
        faces.append(make_face([v[2], v[3], v[7], v[6]]))

        # Left face (x=x0) -- outward normal pointing -X
        faces.append(make_face([v[0], v[4], v[7], v[3]]))

        # Right face (x=x1) -- outward normal pointing +X
        faces.append(make_face([v[1], v[2], v[6], v[5]]))

        closed_shell = ifc.createIfcClosedShell(faces)
        brep = ifc.createIfcFacetedBrep(closed_shell)
        return brep, len(faces)

    # ---------- create wall entities ----------
    walls = []
    total_faces = 0

    for box_def in boxes:
        brep, n_faces = make_box_brep(box_def["min"], box_def["max"])
        total_faces += n_faces

        shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [brep])
        prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

        wall = ifcopenshell.api.run(
            "root.create_entity", ifc,
            ifc_class="IfcWall",
            name=box_def["name"],
            predefined_type="RETAININGWALL",
        )
        wall.ObjectPlacement = local_placement
        wall.Representation = prod_shape

        walls.append(wall)

        # Compute volume for verification
        mn, mx = box_def["min"], box_def["max"]
        vol = (mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2])
        print(f"  Created: {box_def['name']} ({n_faces} faces, V={vol:.1f} m³)")

    # Assign all walls to facility
    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=walls, relating_structure=facility,
    )

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T5_t_shaped.ifc")
    ifc.write(out_path)

    print(f"\nWritten: {out_path}")
    print(f"  Total walls: {len(walls)}")
    print(f"  Total faces: {total_faces}")
    print(f"  Points cached: {len(pt_cache)}")


if __name__ == "__main__":
    main()
