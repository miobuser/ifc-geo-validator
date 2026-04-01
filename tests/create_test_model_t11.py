"""
Create test model T11: S-curved (serpentine) retaining wall.

Specifications:
- S-curve in plan view using a full sine wave: y = A * sin(2*pi*x/L)
- First half curves left, second half curves right (inflection at x=L/2)
- Amplitude A = 2.0 m, total length L = 16.0 m along x-axis
- Wall thickness: 0.4 m constant (offset ±0.2 m from centerline along normal)
- Height: 3.0 m, no crown slope, no inclination
- 24 segments (25 profile points)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T11_s_curved.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
N_SEG = 24              # number of segments
L = 16.0                # total length along x [m]
A = 2.0                 # amplitude of S-curve [m]
THICKNESS = 0.4         # wall thickness [m]
HEIGHT = 3.0            # wall height [m]
HALF_T = THICKNESS / 2  # 0.2 m offset from centerline


def generate_profile_points():
    """Generate inner and outer edge points for the S-curve wall.

    The centerline follows y = A * sin(2*pi*x/L), which creates a full
    sine wave (S-shape) with one inflection point at x = L/2.

    At each point, the local normal is computed perpendicular to the tangent,
    and inner/outer edges are offset by ±HALF_T along that normal.
    """
    n_pts = N_SEG + 1  # 25 points
    inner_pts = []
    outer_pts = []

    for i in range(n_pts):
        t = i / N_SEG
        x = t * L

        # Centerline
        y = A * math.sin(2 * math.pi * x / L)

        # Derivative dy/dx
        dy_dx = A * 2 * math.pi / L * math.cos(2 * math.pi * x / L)

        # Tangent direction (unit vector)
        mag = math.sqrt(1.0 + dy_dx ** 2)
        tx = 1.0 / mag
        ty = dy_dx / mag

        # Normal direction (90° CCW rotation of tangent)
        nx = -ty
        ny = tx

        # Inner and outer edges (offset from centerline)
        inner_x = x - HALF_T * nx
        inner_y = y - HALF_T * ny
        outer_x = x + HALF_T * nx
        outer_y = y + HALF_T * ny

        inner_pts.append((inner_x, inner_y))
        outer_pts.append((outer_x, outer_y))

    return inner_pts, outer_pts


def main():
    # ---------- generate geometry vertices ----------
    inner_2d, outer_2d = generate_profile_points()

    # 3D points: bottom (z=0) and top (z=HEIGHT)
    inner_bot = [(x, y, 0.0) for x, y in inner_2d]
    inner_top = [(x, y, HEIGHT) for x, y in inner_2d]
    outer_bot = [(x, y, 0.0) for x, y in outer_2d]
    outer_top = [(x, y, HEIGHT) for x, y in outer_2d]

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcProject",
        name="Thesis Test T11",
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
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=ctx,
    )

    # Spatial hierarchy: Project > Site > Facility
    site = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcSite",
        name="Testgelände",
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcFacility",
        name="Strasse A1",
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
        """Create an IfcFace from a list of coordinate tuples (forming a polygon)."""
        loop = ifc.createIfcPolyLoop([ifc_pt(p) for p in pts_list])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        return ifc.createIfcFace([bound])

    faces = []

    for i in range(N_SEG):
        # --- Outer face ---
        faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i],
        ]))

        # --- Inner face (reversed winding for outward normal) ---
        faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1],
        ]))

        # --- Top face ---
        faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1],
        ]))

        # --- Bottom face ---
        faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1],
        ]))

    # --- End cap at start (x=0) ---
    faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0],
    ]))

    # --- End cap at end (x=L) ---
    faces.append(make_face([
        outer_bot[-1], inner_bot[-1], inner_top[-1], outer_top[-1],
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
        name="Stützmauer T11 — S-förmig gekrümmte Wand (Serpentine)",
        predefined_type="RETAININGWALL",
    )
    wall.ObjectPlacement = local_placement
    wall.Representation = prod_shape

    # Assign wall to facility
    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[wall],
        relating_structure=facility,
    )

    # ---------- write file ----------
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T11_s_curved.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")
    print(f"  Segments: {N_SEG}")

    # Quick sanity: approximate arc length of the S-curve centerline
    total_arc = 0.0
    n_check = 1000
    for j in range(n_check):
        x0 = j / n_check * L
        x1 = (j + 1) / n_check * L
        y0 = A * math.sin(2 * math.pi * x0 / L)
        y1 = A * math.sin(2 * math.pi * x1 / L)
        total_arc += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    print(f"  Centerline arc length: {total_arc:.4f} m")
    print(f"  Approx volume: {total_arc * THICKNESS * HEIGHT:.4f} m³")


if __name__ == "__main__":
    main()
