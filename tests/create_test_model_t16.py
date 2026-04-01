"""
Create test model T16: Retaining wall with a height step.

Specifications:
- Straight wall, 10.0m long (X axis), 0.4m thick (Y direction)
- LEFT half (x=0 to x=5): height 4.0m
- RIGHT half (x=5 to x=10): height 2.5m
- Transition at x=5 creates a vertical step face
- Foundation at z=0 (flat)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Cross-section (side view):
z=4.0  +----------+
       | LEFT     |
z=2.5  |  4.0m    +----------+
       |          | RIGHT    |
       |          |  2.5m    |
z=0    +----------+----------+
       x=0       x=5        x=10

Output: tests/test_models/T16_height_step.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
LENGTH = 10.0       # total wall length along X [m]
THICKNESS = 0.4     # wall thickness in Y direction [m]
H_LEFT = 4.0        # height of left half [m]
H_RIGHT = 2.5       # height of right half [m]
X_STEP = 5.0        # x-coordinate of the height step [m]

# Derived Y limits
Y0 = 0.0
Y1 = THICKNESS


def main():
    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T16")

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

    # === Face 1a: Foundation LEFT (bottom) — z=0, x=0 to X_STEP ===
    faces.append(make_face([
        (0.0, Y0, 0.0),
        (X_STEP, Y0, 0.0),
        (X_STEP, Y1, 0.0),
        (0.0, Y1, 0.0),
    ]))

    # === Face 1b: Foundation RIGHT (bottom) — z=0, x=X_STEP to LENGTH ===
    faces.append(make_face([
        (X_STEP, Y0, 0.0),
        (LENGTH, Y0, 0.0),
        (LENGTH, Y1, 0.0),
        (X_STEP, Y1, 0.0),
    ]))

    # === Face 2: Crown LEFT — z=H_LEFT, x=0 to X_STEP ===
    faces.append(make_face([
        (0.0, Y0, H_LEFT),
        (0.0, Y1, H_LEFT),
        (X_STEP, Y1, H_LEFT),
        (X_STEP, Y0, H_LEFT),
    ]))

    # === Face 3: Crown RIGHT — z=H_RIGHT, x=X_STEP to LENGTH ===
    faces.append(make_face([
        (X_STEP, Y0, H_RIGHT),
        (X_STEP, Y1, H_RIGHT),
        (LENGTH, Y1, H_RIGHT),
        (LENGTH, Y0, H_RIGHT),
    ]))

    # === Face 4: Step face (vertical, at x=X_STEP) — z=H_RIGHT to H_LEFT ===
    faces.append(make_face([
        (X_STEP, Y0, H_RIGHT),
        (X_STEP, Y0, H_LEFT),
        (X_STEP, Y1, H_LEFT),
        (X_STEP, Y1, H_RIGHT),
    ]))

    # === Face 5a: Front LEFT lower — y=0, x=0 to X_STEP, z=0 to H_RIGHT ===
    faces.append(make_face([
        (0.0, Y0, 0.0),
        (0.0, Y0, H_RIGHT),
        (X_STEP, Y0, H_RIGHT),
        (X_STEP, Y0, 0.0),
    ]))

    # === Face 5b: Front LEFT upper — y=0, x=0 to X_STEP, z=H_RIGHT to H_LEFT ===
    faces.append(make_face([
        (0.0, Y0, H_RIGHT),
        (0.0, Y0, H_LEFT),
        (X_STEP, Y0, H_LEFT),
        (X_STEP, Y0, H_RIGHT),
    ]))

    # === Face 6: Front RIGHT — y=0, x=X_STEP to LENGTH, z=0 to H_RIGHT ===
    faces.append(make_face([
        (X_STEP, Y0, 0.0),
        (X_STEP, Y0, H_RIGHT),
        (LENGTH, Y0, H_RIGHT),
        (LENGTH, Y0, 0.0),
    ]))

    # === Face 7a: Back LEFT lower — y=THICKNESS, x=0 to X_STEP, z=0 to H_RIGHT ===
    faces.append(make_face([
        (0.0, Y1, 0.0),
        (X_STEP, Y1, 0.0),
        (X_STEP, Y1, H_RIGHT),
        (0.0, Y1, H_RIGHT),
    ]))

    # === Face 7b: Back LEFT upper — y=THICKNESS, x=0 to X_STEP, z=H_RIGHT to H_LEFT ===
    faces.append(make_face([
        (0.0, Y1, H_RIGHT),
        (X_STEP, Y1, H_RIGHT),
        (X_STEP, Y1, H_LEFT),
        (0.0, Y1, H_LEFT),
    ]))

    # === Face 8: Back RIGHT — y=THICKNESS, x=X_STEP to LENGTH, z=0 to H_RIGHT ===
    faces.append(make_face([
        (X_STEP, Y1, 0.0),
        (LENGTH, Y1, 0.0),
        (LENGTH, Y1, H_RIGHT),
        (X_STEP, Y1, H_RIGHT),
    ]))

    # === Face 9a: End LEFT lower — x=0, y=0 to THICKNESS, z=0 to H_RIGHT ===
    faces.append(make_face([
        (0.0, Y0, 0.0),
        (0.0, Y1, 0.0),
        (0.0, Y1, H_RIGHT),
        (0.0, Y0, H_RIGHT),
    ]))

    # === Face 9b: End LEFT upper — x=0, y=0 to THICKNESS, z=H_RIGHT to H_LEFT ===
    faces.append(make_face([
        (0.0, Y0, H_RIGHT),
        (0.0, Y1, H_RIGHT),
        (0.0, Y1, H_LEFT),
        (0.0, Y0, H_LEFT),
    ]))

    # === Face 10: End RIGHT — x=LENGTH, y=0 to THICKNESS, z=0 to H_RIGHT ===
    faces.append(make_face([
        (LENGTH, Y0, 0.0),
        (LENGTH, Y0, H_RIGHT),
        (LENGTH, Y1, H_RIGHT),
        (LENGTH, Y1, 0.0),
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
        name="Stützmauer T16 \u2014 Höhenstufe (4.0m / 2.5m)",
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
    out_path = os.path.join(out_dir, "T16_height_step.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Volume sanity check
    v_left = X_STEP * THICKNESS * H_LEFT
    v_right = (LENGTH - X_STEP) * THICKNESS * H_RIGHT
    v_total = v_left + v_right
    print(f"  Expected volume: {v_total:.4f} m³ (left={v_left:.4f} + right={v_right:.4f})")


if __name__ == "__main__":
    main()
