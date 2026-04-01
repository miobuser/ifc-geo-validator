"""
Create test model T25: Non-compliant wall with multiple ASTRA rule failures.

Intentionally violates multiple ASTRA geometric validation rules to serve
as a negative test case for the validator.

Specifications:
- Single IfcWall element, straight
- Crown width: 200mm (FAIL: < 300mm minimum)
- No crown slope: 0% (FAIL: should be 3%)
- Wall thickness: 180mm uniform (FAIL: < 300mm minimum, also crown < 300mm)
- No inclination: vertical wall (INFO: no 10:1 Anzug)
- Height: 2.0m
- Length: 6.0m
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (6 faces, simple box)
- IFC version: IFC4X3_ADD2

Expected validation results:
- L1 Crown width >= 300mm:    FAIL (200mm — but note: crown width = thickness for
  a constant-thickness wall, so the validator sees 180mm)
  Actually: the crown width IS the thickness at the top = 180mm. FAIL.
- L2 Crown slope 3%:          FAIL (0%, flat crown)
- L3 Min thickness >= 300mm:  FAIL (180mm)
- L4 Inclination ~10:1:       INFO (vertical, no Anzug)

Note on crown width vs thickness: For this simple box, the crown (top face) has
width = wall thickness = 180mm in the y-direction. The 200mm mentioned in the
spec would require a different y-dimension, but since we want the wall thickness
to also be non-compliant at 180mm, we use 180mm for both. The "200mm" in the
original spec is overridden by the 180mm thickness to create a consistent model.

Actually, let's make the crown wider than the base to create an unusual scenario:
- Base thickness: 180mm (y=0 to y=0.18)
- Crown overhangs: crown from y=-0.01 to y=0.19 = 200mm wide
This way crown=200mm but min thickness=180mm — both fail independently.

Simplified approach: Just use a 200mm wide, 2m tall box. Both crown and thickness = 200mm.
This fails crown >= 300mm AND thickness >= 300mm.

Final decision: Use the simplest approach — a uniform 200mm box.
- Crown width: 200mm FAIL
- Thickness: 200mm FAIL (everywhere < 300mm)
- No slope: FAIL
- Vertical: INFO

Output: tests/test_models/T25_multi_failure.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
WALL_LENGTH = 6.0       # x-direction [m]
WALL_THICK = 0.200      # y-direction [m] — intentionally < 300mm
WALL_HEIGHT = 2.0       # z-direction [m]
# No crown slope (0%), no Anzug (vertical)


def main():
    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T25"
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

    # ---------- build IfcFacetedBrep ----------
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

    # 8 vertices of the box: x=0..6, y=0..0.2, z=0..2
    # Completely vertical, no slope, no Anzug — maximally non-compliant
    box_verts = [
        # Bottom (z=0)
        (0.0,         0.0,        0.0),           # 0: front-left-bottom
        (WALL_LENGTH, 0.0,        0.0),           # 1: front-right-bottom
        (WALL_LENGTH, WALL_THICK, 0.0),           # 2: back-right-bottom
        (0.0,         WALL_THICK, 0.0),           # 3: back-left-bottom
        # Top (z=2.0) — flat crown, no slope
        (0.0,         0.0,        WALL_HEIGHT),   # 4: front-left-top
        (WALL_LENGTH, 0.0,        WALL_HEIGHT),   # 5: front-right-top
        (WALL_LENGTH, WALL_THICK, WALL_HEIGHT),   # 6: back-right-top
        (0.0,         WALL_THICK, WALL_HEIGHT),   # 7: back-left-top
    ]

    v = box_verts
    faces = [
        # Front face (y=0, normal -Y)
        make_face([v[0], v[1], v[5], v[4]]),
        # Back face (y=0.2, normal +Y)
        make_face([v[2], v[3], v[7], v[6]]),
        # Left face (x=0, normal -X)
        make_face([v[3], v[0], v[4], v[7]]),
        # Right face (x=6, normal +X)
        make_face([v[1], v[2], v[6], v[5]]),
        # Top face (z=2.0, normal +Z) — flat crown, no slope
        make_face([v[4], v[5], v[6], v[7]]),
        # Bottom face (z=0, normal -Z)
        make_face([v[0], v[3], v[2], v[1]]),
    ]

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
        name="Stützmauer T25 — Nicht-konform (Multi-Failure)",
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
    out_path = os.path.join(out_dir, "T25_multi_failure.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Verification
    model = ifcopenshell.open(out_path)
    walls = model.by_type("IfcWall")
    print(f"\nVerification:")
    print(f"  IfcWall count: {len(walls)}")
    print(f"  IfcWall name: {walls[0].Name if walls else 'N/A'}")
    print(f"  IfcWall PredefinedType: {walls[0].PredefinedType if walls else 'N/A'}")

    print(f"\nIntended violations:")
    print(f"  Crown width:     {WALL_THICK*1000:.0f}mm (FAIL: < 300mm)")
    print(f"  Crown slope:     0% (FAIL: should be 3%)")
    print(f"  Min thickness:   {WALL_THICK*1000:.0f}mm (FAIL: < 300mm)")
    print(f"  Inclination:     vertical (INFO: no 10:1 Anzug)")
    print(f"  Wall dimensions: {WALL_LENGTH}m x {WALL_THICK}m x {WALL_HEIGHT}m")


if __name__ == "__main__":
    main()
