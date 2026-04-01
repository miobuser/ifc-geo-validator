"""
Create test model T18: Retaining wall with buttresses (counterforts).

Specifications:
- Straight wall, 12.0 m long (X axis), 0.3 m thick (Y), 3.0 m high (Z)
- 3 buttresses on the back side at x=3, x=6, x=9
  - Each: 0.5 m wide (X), 0.8 m deep (Y, from y=0.3 to y=1.1), full height
- Geometry: single IfcFacetedBrep body
- Entity: IfcWall with PredefinedType=RETAININGWALL
- IFC version: IFC4X3_ADD2

Cross-section (top view):
y=1.1       +--+       +--+       +--+
            |B |       |B |       |B |   (buttresses)
y=0.3  +----+  +-------+  +-------+  +----+
       |          main wall                |
y=0    +-----------------------------------+
       x=0  x=3      x=6      x=9       x=12

Output: tests/test_models/T18_buttressed.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
WALL_LENGTH = 12.0      # along X
WALL_THICK = 0.3        # Y direction (main wall)
HEIGHT = 3.0            # Z direction

# Buttress positions (center X) and dimensions
BUTTRESS_XS = [3.0, 6.0, 9.0]
BUTTRESS_WIDTH = 0.5    # X extent
BUTTRESS_DEPTH = 0.8    # Y extent (beyond main wall)
BUTTRESS_Y_START = WALL_THICK        # 0.3
BUTTRESS_Y_END = WALL_THICK + BUTTRESS_DEPTH  # 1.1


def main():
    # ---------- build cross-section outline ----------
    # CCW polygon in the XY plane, starting at (0, 0).
    # Walk along the bottom (y=0), then up the back with buttress indentations.
    outline = [
        (0.0, 0.0),
        (WALL_LENGTH, 0.0),          # (12, 0)
        (WALL_LENGTH, WALL_THICK),   # (12, 0.3)
    ]

    # Walk from x=12 back to x=0 along the back face (y=0.3),
    # inserting buttress notches in reverse order (x=9, 6, 3).
    for bx in reversed(BUTTRESS_XS):
        x_right = bx + BUTTRESS_WIDTH / 2.0   # 9.25, 6.25, 3.25
        x_left = bx - BUTTRESS_WIDTH / 2.0    # 8.75, 5.75, 2.75
        outline.append((x_right, WALL_THICK))
        outline.append((x_right, BUTTRESS_Y_END))
        outline.append((x_left, BUTTRESS_Y_END))
        outline.append((x_left, WALL_THICK))

    outline.append((0.0, WALL_THICK))  # back to the start edge

    # Close the polygon (first == last is implicit in IfcPolyLoop,
    # but we track the full list for face generation).
    # outline has 16 unique vertices.

    n_verts = len(outline)  # 16
    print(f"Outline vertices: {n_verts}")
    for i, v in enumerate(outline):
        print(f"  {i}: ({v[0]}, {v[1]})")

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject",
        name="Thesis Test T18",
    )

    # Units (metres)
    length_u = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="LENGTHUNIT")
    area_u = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="AREAUNIT")
    volume_u = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="VOLUMEUNIT")
    ifcopenshell.api.run("unit.assign_unit", ifc, units=[length_u, area_u, volume_u])

    # Context
    ctx = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context", ifc,
        context_type="Model", context_identifier="Body", target_view="MODEL_VIEW",
        parent=ctx,
    )

    # Spatial hierarchy
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
        """Create an IfcFace from a list of (x, y, z) tuples."""
        loop = ifc.createIfcPolyLoop([ifc_pt(p) for p in pts_list])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        return ifc.createIfcFace([bound])

    faces = []

    # --- Bottom face (z=0): the full outline polygon ---
    # Outward-facing normal should point down (-Z), so use CW winding
    # when viewed from below. Our outline is CCW when viewed from above,
    # which is CW when viewed from below => normal points down. Good.
    bottom_pts = [(v[0], v[1], 0.0) for v in outline]
    faces.append(make_face(bottom_pts))

    # --- Top face (z=HEIGHT): same shape, reversed winding for upward normal ---
    top_pts = [(v[0], v[1], HEIGHT) for v in reversed(outline)]
    faces.append(make_face(top_pts))

    # --- Vertical side faces ---
    # For each consecutive pair in the outline, create a vertical quad.
    for i in range(n_verts):
        j = (i + 1) % n_verts
        v0 = outline[i]
        v1 = outline[j]

        # Quad: bottom-left, bottom-right, top-right, top-left
        # Winding must have outward normal. Since outline is CCW,
        # the outward normal of a vertical face between v0→v1 points
        # to the right of the edge direction. Using this winding:
        # v0_bot, v1_bot, v1_top, v0_top => outward for CCW outline.
        faces.append(make_face([
            (v0[0], v0[1], 0.0),
            (v1[0], v1[1], 0.0),
            (v1[0], v1[1], HEIGHT),
            (v0[0], v0[1], HEIGHT),
        ]))

    print(f"Total faces: {len(faces)} (2 caps + {n_verts} sides)")

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
        name="Stützmauer T18 \u2014 Wand mit Strebepfeilern (3 Counterforts)",
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
    out_path = os.path.join(out_dir, "T18_buttressed.ifc")
    ifc.write(out_path)
    print(f"\nWritten: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Approximate volume
    wall_vol = WALL_LENGTH * WALL_THICK * HEIGHT
    buttress_vol = len(BUTTRESS_XS) * BUTTRESS_WIDTH * BUTTRESS_DEPTH * HEIGHT
    total_vol = wall_vol + buttress_vol
    print(f"  Approximate volume: {total_vol:.4f} m³ "
          f"(wall {wall_vol:.2f} + buttresses {buttress_vol:.2f})")


if __name__ == "__main__":
    main()
