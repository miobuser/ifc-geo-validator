"""
Create test model T13: Polygonal retaining wall (3 straight segments, Z-shape).

Specifications:
- 3 straight segments forming a zigzag in plan view:
    Segment 1: (0,0) -> (4,0)   — 4m along X
    Segment 2: (4,0) -> (6,2)   — ~2.83m diagonal at 45°
    Segment 3: (6,2) -> (10,2)  — 4m along X
- Wall thickness: 0.4m (offset ±0.2m along local normal)
- Height: 3.0m, no crown slope, no inclination
- Mitered corners at segment junctions
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T13_polygonal.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
THICKNESS = 0.4         # wall thickness [m]
HALF_T = THICKNESS / 2  # 0.2 m
HEIGHT = 3.0            # wall height [m]

# Centerline vertices (plan view, z=0)
CENTERLINE = [
    (0.0, 0.0),
    (4.0, 0.0),
    (6.0, 2.0),
    (10.0, 2.0),
]


def vec2_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vec2_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vec2_scale(v, s):
    return (v[0] * s, v[1] * s)


def vec2_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def vec2_normalize(v):
    ln = vec2_len(v)
    return (v[0] / ln, v[1] / ln)


def vec2_normal_ccw(v):
    """Rotate 90° counterclockwise: (x,y) -> (-y, x)."""
    return (-v[1], v[0])


def line_intersect_2d(p1, d1, p2, d2):
    """Find intersection of two 2D lines: p1 + t*d1 = p2 + s*d2.
    Returns the intersection point or None if parallel."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-12:
        return None  # parallel
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def compute_offset_polygon(centerline, offset):
    """Compute an offset polygon from a polyline centerline.

    For each segment, offset the line by `offset` along the CCW normal.
    At corners, intersect adjacent offset lines to get mitered points.
    Returns list of 2D points.
    """
    n_seg = len(centerline) - 1
    # For each segment, compute direction, normal, and offset line
    seg_dirs = []
    seg_normals = []
    seg_offset_starts = []
    seg_offset_ends = []

    for i in range(n_seg):
        d = vec2_sub(centerline[i + 1], centerline[i])
        d_norm = vec2_normalize(d)
        n = vec2_normal_ccw(d_norm)
        seg_dirs.append(d_norm)
        seg_normals.append(n)
        # Offset line: start and end shifted by offset * normal
        off = vec2_scale(n, offset)
        seg_offset_starts.append(vec2_add(centerline[i], off))
        seg_offset_ends.append(vec2_add(centerline[i + 1], off))

    # Build offset polygon points
    pts = []
    # First point: start of first offset segment
    pts.append(seg_offset_starts[0])

    # At each interior junction, intersect adjacent offset lines
    for i in range(n_seg - 1):
        p = line_intersect_2d(
            seg_offset_starts[i], seg_dirs[i],
            seg_offset_starts[i + 1], seg_dirs[i + 1],
        )
        if p is None:
            # Parallel segments — just use the end of current segment
            pts.append(seg_offset_ends[i])
        else:
            pts.append(p)

    # Last point: end of last offset segment
    pts.append(seg_offset_ends[-1])

    return pts


def main():
    # ---------- compute offset polygons ----------
    outer_2d = compute_offset_polygon(CENTERLINE, +HALF_T)   # left side (CCW normal)
    inner_2d = compute_offset_polygon(CENTERLINE, -HALF_T)   # right side

    n_pts = len(outer_2d)  # should be 4 (one per centerline vertex)
    assert n_pts == len(CENTERLINE), f"Expected {len(CENTERLINE)} points, got {n_pts}"

    # 3D points: bottom (z=0) and top (z=HEIGHT)
    outer_bot = [(p[0], p[1], 0.0) for p in outer_2d]
    outer_top = [(p[0], p[1], HEIGHT) for p in outer_2d]
    inner_bot = [(p[0], p[1], 0.0) for p in inner_2d]
    inner_top = [(p[0], p[1], HEIGHT) for p in inner_2d]

    # Print offset points for verification
    print("Outer polygon (left/+normal):")
    for i, p in enumerate(outer_2d):
        print(f"  [{i}] ({p[0]:.6f}, {p[1]:.6f})")
    print("Inner polygon (right/-normal):")
    for i, p in enumerate(inner_2d):
        print(f"  [{i}] ({p[0]:.6f}, {p[1]:.6f})")

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject",
        name="Thesis Test T13",
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
        "root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände",
    )
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[site], relating_object=project,
    )

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
    n_seg = len(CENTERLINE) - 1  # 3 segments

    for i in range(n_seg):
        # For segment i: vertices at index i and i+1

        # --- Outer face (front) ---
        faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i],
        ]))

        # --- Inner face (back) — reversed winding ---
        faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1],
        ]))

        # --- Top face (crown) ---
        faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1],
        ]))

        # --- Bottom face (foundation) ---
        faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1],
        ]))

    # --- End cap at start (i=0) ---
    faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0],
    ]))

    # --- End cap at end (i=-1) ---
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
        name="Stützmauer T13 \u2014 Polygonale Wand (3 Segmente, abgewinkelt)",
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
    out_path = os.path.join(out_dir, "T13_polygonal.ifc")
    ifc.write(out_path)
    print(f"\nWritten: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Quick volume estimate: sum of segment cross-section * length
    # Each segment: thickness * height * segment_length
    total_length = 0.0
    for i in range(n_seg):
        dx = CENTERLINE[i + 1][0] - CENTERLINE[i][0]
        dy = CENTERLINE[i + 1][1] - CENTERLINE[i][1]
        total_length += math.sqrt(dx ** 2 + dy ** 2)
    v_approx = THICKNESS * HEIGHT * total_length
    print(f"  Total centerline length: {total_length:.4f} m")
    print(f"  Approximate volume: {v_approx:.4f} m³")


if __name__ == "__main__":
    main()
