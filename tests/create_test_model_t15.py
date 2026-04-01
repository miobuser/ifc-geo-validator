"""
Create test model T15: Retaining wall with variable height (terrain slope).

Specifications:
- Straight wall, 10.0 m long (along X axis), 0.4 m thick (Y axis)
- Variable height: 2.0 m at x=0, linearly increasing to 5.0 m at x=10
- Foundation at z=0 (flat), crown follows slope (~16.7°, 30% grade)
- 10 segments along the length (11 cross-section positions)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T15_variable_height.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
LENGTH = 10.0       # wall length along X [m]
THICKNESS = 0.4     # wall thickness along Y [m]
H_START = 2.0       # height at x=0 [m]
H_END = 5.0         # height at x=10 [m]
N_SEG = 10          # number of segments along the length


def main():
    # ---------- generate geometry vertices ----------
    n_pts = N_SEG + 1  # 11 positions

    # At each x-position compute 4 corners of the cross-section
    # bottom-front (y=0, z=0), bottom-back (y=T, z=0),
    # top-front (y=0, z=h), top-back (y=T, z=h)
    bot_front = []  # (x, 0, 0)
    bot_back = []   # (x, T, 0)
    top_front = []  # (x, 0, h)
    top_back = []   # (x, T, h)

    for i in range(n_pts):
        x = LENGTH * i / (n_pts - 1)
        h = H_START + (H_END - H_START) * (x / LENGTH)
        bot_front.append((x, 0.0, 0.0))
        bot_back.append((x, THICKNESS, 0.0))
        top_front.append((x, 0.0, h))
        top_back.append((x, THICKNESS, h))

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T15",
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
        "root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände",
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1",
    )
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[facility], relating_object=site,
    )

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
        # --- Front face (y=0, vertical) ---
        faces.append(make_face([
            bot_front[i], bot_front[i + 1], top_front[i + 1], top_front[i],
        ]))

        # --- Back face (y=T, vertical) ---
        faces.append(make_face([
            bot_back[i + 1], bot_back[i], top_back[i], top_back[i + 1],
        ]))

        # --- Crown (top, sloping surface) ---
        faces.append(make_face([
            top_front[i], top_front[i + 1], top_back[i + 1], top_back[i],
        ]))

        # --- Foundation (bottom, z=0, horizontal) ---
        faces.append(make_face([
            bot_front[i + 1], bot_front[i], bot_back[i], bot_back[i + 1],
        ]))

    # --- End cap at x=0 (start) — outward normal: -X ---
    faces.append(make_face([
        bot_front[0], top_front[0], top_back[0], bot_back[0],
    ]))

    # --- End cap at x=10 (end) — outward normal: +X ---
    faces.append(make_face([
        bot_front[-1], bot_back[-1], top_back[-1], top_front[-1],
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
        name="Stützmauer T15 \u2014 Variable Höhe (Hangverlauf 2m\u21925m)",
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
    out_path = os.path.join(out_dir, "T15_variable_height.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Quick sanity check: approximate volume
    # Trapezoidal cross-section: V = L * T * (H_start + H_end) / 2
    v_approx = LENGTH * THICKNESS * (H_START + H_END) / 2.0
    print(f"  Approximate volume: {v_approx:.4f} m³")
    print(f"  Height range: {H_START} m .. {H_END} m")
    print(f"  Crown slope: {(H_END - H_START) / LENGTH * 100:.1f}% grade")


if __name__ == "__main__":
    main()
