"""
Create test model T9: Stepped retaining wall.

Specifications:
- Straight wall, 8.0m long (along X axis)
- Stepped cross-section: 300mm wide at crown, 600mm wide at base
- Step at z=1.5m: base 600mm (Y=0..0.6), upper 300mm (Y=0..0.3)
- Front face is flat/vertical (Y=0), step is on the back side
- Total height: 3.0m
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep
- IFC version: IFC4X3_ADD2

Cross-section (looking from the side, +Y to the right):
       Y=0    Y=0.3  Y=0.6
z=3.0  +------+
       |      | (upper: 300mm wide)
z=1.5  |      +------+
       |      |      | (lower: 600mm wide)
z=0    +------+------+

8 faces:
  1. Front face:     Y=0,   x=0..8, z=0..3.0
  2. Back upper:     Y=0.3, x=0..8, z=1.5..3.0
  3. Back lower:     Y=0.6, x=0..8, z=0..1.5
  4. Step (horiz.):  z=1.5, x=0..8, Y=0.3..0.6
  5. Crown (top):    z=3.0, x=0..8, Y=0..0.3
  6. Foundation:     z=0,   x=0..8, Y=0..0.6
  7. End left:       x=0,   L-shaped polygon
  8. End right:      x=8,   L-shaped polygon

Volume: (0.3*1.5 + 0.6*1.5) * 8 = 10.8 m³

Output: tests/test_models/T9_stepped_wall.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
LENGTH = 8.0            # wall length along X [m]
HEIGHT = 3.0            # total wall height [m]
STEP_Z = 1.5            # height of the step [m]
WIDTH_BASE = 0.6        # wall width at base (Y) [m]
WIDTH_CROWN = 0.3       # wall width at crown (Y) [m]


def main():
    # ---------- define the 8 corner vertices ----------
    # Naming: F=front(Y=0), B=back; L=left(x=0), R=right(x=8); 0/1/2 = z level
    #   z=0 level (bottom)
    #   z=1.5 level (step)
    #   z=3.0 level (top)

    # Front face vertices (Y=0)
    FL0 = (0.0, 0.0, 0.0)       # front-left-bottom
    FR0 = (LENGTH, 0.0, 0.0)    # front-right-bottom
    FL2 = (0.0, 0.0, HEIGHT)    # front-left-top
    FR2 = (LENGTH, 0.0, HEIGHT) # front-right-top

    # Back face vertices — base portion (Y=WIDTH_BASE=0.6)
    BL0 = (0.0, WIDTH_BASE, 0.0)       # back-left-bottom (wide)
    BR0 = (LENGTH, WIDTH_BASE, 0.0)    # back-right-bottom (wide)
    BL1 = (0.0, WIDTH_BASE, STEP_Z)    # back-left at step height (wide)
    BR1 = (LENGTH, WIDTH_BASE, STEP_Z) # back-right at step height (wide)

    # Back face vertices — upper portion (Y=WIDTH_CROWN=0.3)
    BL1n = (0.0, WIDTH_CROWN, STEP_Z)    # back-left at step height (narrow)
    BR1n = (LENGTH, WIDTH_CROWN, STEP_Z) # back-right at step height (narrow)
    BL2 = (0.0, WIDTH_CROWN, HEIGHT)     # back-left-top (narrow)
    BR2 = (LENGTH, WIDTH_CROWN, HEIGHT)  # back-right-top (narrow)

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T9"
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
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

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
        """Create an IfcFace from a list of coordinate tuples (polygon)."""
        loop = ifc.createIfcPolyLoop([ifc_pt(p) for p in pts_list])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        return ifc.createIfcFace([bound])

    faces = []

    # Face 1: Front face — Y=0, full rectangle z=0..3.0
    # Outward normal: -Y direction => vertices CCW when viewed from -Y
    faces.append(make_face([FL0, FR0, FR2, FL2]))

    # Face 2: Back upper — Y=0.3, z=1.5..3.0
    # Outward normal: +Y direction => vertices CCW when viewed from +Y
    faces.append(make_face([BL1n, BL2, BR2, BR1n]))

    # Face 3: Back lower — Y=0.6, z=0..1.5
    # Outward normal: +Y direction
    faces.append(make_face([BL0, BL1, BR1, BR0]))

    # Face 4: Step (horizontal) — z=1.5, Y=0.3..0.6
    # Outward normal: +Z direction (faces up)
    faces.append(make_face([BL1n, BR1n, BR1, BL1]))

    # Face 5: Crown (top) — z=3.0, Y=0..0.3
    # Outward normal: +Z direction
    faces.append(make_face([FL2, FR2, BR2, BL2]))

    # Face 6: Foundation (bottom) — z=0, Y=0..0.6
    # Outward normal: -Z direction
    faces.append(make_face([FL0, BL0, BR0, FR0]))

    # Face 7: End left — x=0, L-shaped polygon
    # Outward normal: -X direction => vertices CCW when viewed from -X
    # L-shape vertices going CCW from front-bottom:
    #   FL0(0,0,0) -> BL0(0,0.6,0) -> BL1(0,0.6,1.5) -> BL1n(0,0.3,1.5) -> BL2(0,0.3,3) -> FL2(0,0,3)
    faces.append(make_face([FL0, BL0, BL1, BL1n, BL2, FL2]))

    # Face 8: End right — x=8, L-shaped polygon
    # Outward normal: +X direction => vertices CCW when viewed from +X
    # L-shape vertices going CCW from front-bottom:
    #   FR0(8,0,0) -> FR2(8,0,3) -> BR2(8,0.3,3) -> BR1n(8,0.3,1.5) -> BR1(8,0.6,1.5) -> BR0(8,0.6,0)
    faces.append(make_face([FR0, FR2, BR2, BR1n, BR1, BR0]))

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
        name="Stützmauer T9 — Abgestufte Wand",
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
    out_path = os.path.join(out_dir, "T9_stepped_wall.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Sanity check: volume
    v_lower = WIDTH_BASE * STEP_Z * LENGTH   # 0.6 * 1.5 * 8 = 7.2
    v_upper = WIDTH_CROWN * (HEIGHT - STEP_Z) * LENGTH  # 0.3 * 1.5 * 8 = 3.6
    v_total = v_lower + v_upper
    print(f"  Volume: {v_lower:.2f} + {v_upper:.2f} = {v_total:.2f} m³")
    print(f"  Crown width: {WIDTH_CROWN * 1000:.0f} mm")


if __name__ == "__main__":
    main()
