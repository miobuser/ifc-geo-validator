"""
Create test model T27: Complex long curved retaining wall (stress test).

Specifications:
- 50m long, S-curve with 2 inflection points
- Variable height: 2.5m (start) to 4.5m (end), linearly varying
- Variable thickness: 300mm at crown, 600mm at base (tapering)
- 3% crown slope toward the back
- 10:1 front face inclination
- Separate foundation element (1.5m wide, 0.5m thick)
- IfcFacetedBrep with fine tessellation (500+ triangles per element)
- IFC version: IFC4X3_ADD2

S-curve parametrization:
  x(s) = s
  y(s) = A * sin(2*pi*s/L) where A = 3.0m amplitude, L = 50m
  This gives an S-shape with 2 inflection points at s=L/4 and s=3L/4.

Output: tests/test_models/T27_complex_long.ifc
"""

import os
import math
import numpy as np
import ifcopenshell
import ifcopenshell.api


def s_curve_xy(s, L=50.0, A=3.0):
    """S-curve parametrization in plan view.
    Returns (x, y) for station s in [0, L].
    """
    x = s
    y = A * math.sin(2.0 * math.pi * s / L)
    return x, y


def s_curve_tangent(s, L=50.0, A=3.0):
    """Unit tangent of S-curve at station s."""
    dx = 1.0
    dy = A * 2.0 * math.pi / L * math.cos(2.0 * math.pi * s / L)
    mag = math.sqrt(dx * dx + dy * dy)
    return dx / mag, dy / mag


def s_curve_normal(s, L=50.0, A=3.0):
    """Unit normal (pointing left) of S-curve at station s."""
    tx, ty = s_curve_tangent(s, L, A)
    return -ty, tx


def build_wall_cross_section(s, L=50.0, crown_width_top=0.300, base_width=0.600,
                              height_start=2.5, height_end=4.5,
                              inclination_ratio=10.0, crown_slope_pct=3.0):
    """Build a wall cross-section at station s.

    Returns list of (offset_normal, z) points describing the cross-section
    polygon in the local normal-Z plane. Points are in order:
        back-bottom, back-top, crown front edge (top), front-top, front-bottom

    Convention:
        - offset_normal > 0 = front side (earth retention)
        - offset_normal < 0 = back side (air side)
        - Crown slopes toward back (negative normal direction)

    The cross-section is a trapezoid with:
        - Crown width = crown_width_top at the top
        - Base width = base_width at the bottom
        - Height varies linearly from height_start to height_end
        - Front face inclined at inclination_ratio:1 (horizontal:vertical)
        - Crown has crown_slope_pct % slope toward the back
    """
    t = s / L
    height = height_start + t * (height_end - height_start)

    # Inclination offset: front face leans 1/ratio per unit height
    incl_offset = height / inclination_ratio

    # Crown slope: 3% means the back edge is lower by 3% of crown width
    slope_drop = crown_slope_pct / 100.0 * crown_width_top

    # Cross-section points (normal_offset, z):
    # Centerline is the center of the crown.
    # Crown: from -crown_width_top/2 to +crown_width_top/2
    # At the base, front extends further due to tapering and inclination.
    # Back face is vertical.

    front_top = crown_width_top / 2.0
    front_bottom = crown_width_top / 2.0 + incl_offset + (base_width - crown_width_top)
    back_top = -crown_width_top / 2.0
    back_bottom = -crown_width_top / 2.0  # back is vertical

    # Crown slope: back edge drops by slope_drop
    z_crown_front = height
    z_crown_back = height - slope_drop

    # Build polygon in CW order when looking along the extrusion tangent.
    # This ensures outward-facing normals when connecting adjacent sections
    # via the quad rule: s0[j], s1[j], s1[j_next], s0[j_next].
    #
    # Order: back-bot -> back-top -> front-top -> front-bot
    # (CW looking from outside = correct outward normals)
    pts = [
        (back_bottom, 0.0),          # 0: back bottom
        (back_top, z_crown_back),    # 1: back top (lower due to slope)
        (front_top, z_crown_front),  # 2: front top
        (front_bottom, 0.0),         # 3: front bottom
    ]
    return pts, height


def triangulate_quad(v0, v1, v2, v3):
    """Split a quad into 2 triangles. Returns list of 2 triangles."""
    return [(v0, v1, v2), (v0, v2, v3)]


def build_faceted_brep_wall(n_stations=80, L=50.0, A=3.0,
                             crown_width_top=0.300, base_width=0.600,
                             height_start=2.5, height_end=4.5,
                             inclination_ratio=10.0, crown_slope_pct=3.0,
                             z_base=0.5):
    """Build wall mesh as list of triangular faces.

    Each face is a tuple of 3 3D points: ((x,y,z), (x,y,z), (x,y,z)).
    z_base is the Z offset (foundation top = wall bottom).

    Returns (triangles, stats) where stats is a dict with expected values.
    """
    stations = np.linspace(0.0, L, n_stations + 1)

    # Build cross-sections at each station
    sections = []
    for s in stations:
        cs_pts, height = build_wall_cross_section(
            s, L, crown_width_top, base_width, height_start, height_end,
            inclination_ratio, crown_slope_pct
        )
        # Transform cross-section to world coordinates
        cx, cy = s_curve_xy(s, L, A)
        nx, ny = s_curve_normal(s, L, A)
        tx, ty = s_curve_tangent(s, L, A)

        world_pts = []
        for (offset_n, z) in cs_pts:
            wx = cx + offset_n * nx
            wy = cy + offset_n * ny
            wz = z + z_base
            world_pts.append((wx, wy, wz))
        sections.append(world_pts)

    triangles = []

    # Connect adjacent cross-sections with quads, then triangulate
    for i in range(len(sections) - 1):
        s0 = sections[i]      # 4 points: back-bot, front-bot, front-top, back-top
        s1 = sections[i + 1]  # 4 points

        n_pts = len(s0)
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            # Quad: s0[j], s1[j], s1[j_next], s0[j_next]
            quads = triangulate_quad(s0[j], s1[j], s1[j_next], s0[j_next])
            triangles.extend(quads)

    # End caps (station 0 and station L)
    # Start cap: normal points opposite to tangent direction (inward at start)
    # Winding: CW when viewed from outside (looking along -tangent)
    s_start = sections[0]
    triangles.extend(triangulate_quad(s_start[3], s_start[2], s_start[1], s_start[0]))

    # End cap: normal points along tangent direction (outward at end)
    # Winding: CCW when viewed from outside (looking along tangent)
    s_end = sections[-1]
    triangles.extend(triangulate_quad(s_end[0], s_end[1], s_end[2], s_end[3]))

    # Further subdivide each quad-pair along the height for finer tessellation
    # The current approach gives: n_stations * 4 faces * 2 tri = n_stations*8 + 4 triangles
    # For 80 stations: 80*8 + 4 = 644 triangles. Good enough for 500+ target.

    stats = {
        "n_stations": n_stations,
        "n_triangles": len(triangles),
        "length_m": L,
        "height_start_m": height_start,
        "height_end_m": height_end,
        "crown_width_mm": crown_width_top * 1000,
        "base_width_mm": base_width * 1000,
    }
    return triangles, stats


def build_faceted_brep_foundation(n_stations=80, L=50.0, A=3.0,
                                   foundation_width=1.5, foundation_thickness=0.5,
                                   crown_width_top=0.300, base_width=0.600):
    """Build foundation mesh as list of triangular faces.

    Foundation is a slab under the wall, wider than the wall (extending on earth side).
    Foundation top is at z=foundation_thickness, bottom at z=0.
    The foundation is centered on the wall centerline but extends more toward the front.
    """
    stations = np.linspace(0.0, L, n_stations + 1)

    # Foundation cross-section: rectangle centered on wall center
    # but shifted toward front (earth side)
    # Wall crown center = 0, front at +crown_width_top/2, back at -crown_width_top/2
    # Foundation extends from back to (back + foundation_width)
    # We want it to extend beyond the wall on the earth side:
    # back edge aligns with wall back, front edge = back + foundation_width
    back_offset = -crown_width_top / 2.0 - 0.1  # 100mm beyond wall back
    front_offset = back_offset + foundation_width

    sections = []
    for s in stations:
        cx, cy = s_curve_xy(s, L, A)
        nx, ny = s_curve_normal(s, L, A)

        # 4 corners of foundation cross-section (CW when looking along tangent)
        pts = [
            # back-bottom
            (cx + back_offset * nx, cy + back_offset * ny, 0.0),
            # back-top
            (cx + back_offset * nx, cy + back_offset * ny, foundation_thickness),
            # front-top
            (cx + front_offset * nx, cy + front_offset * ny, foundation_thickness),
            # front-bottom
            (cx + front_offset * nx, cy + front_offset * ny, 0.0),
        ]
        sections.append(pts)

    triangles = []
    for i in range(len(sections) - 1):
        s0 = sections[i]
        s1 = sections[i + 1]
        n_pts = len(s0)
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            quads = triangulate_quad(s0[j], s1[j], s1[j_next], s0[j_next])
            triangles.extend(quads)

    # End caps
    s_start = sections[0]
    triangles.extend(triangulate_quad(s_start[3], s_start[2], s_start[1], s_start[0]))
    s_end = sections[-1]
    triangles.extend(triangulate_quad(s_end[0], s_end[1], s_end[2], s_end[3]))

    return triangles


def triangles_to_ifc_faceted_brep(ifc, triangles):
    """Convert a list of triangle tuples to an IfcFacetedBrep entity.

    Each triangle is ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3)).
    """
    # Collect unique vertices with welding
    vertex_map = {}
    vertex_list = []

    def get_vertex_idx(pt):
        key = (round(pt[0], 6), round(pt[1], 6), round(pt[2], 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertex_list)
            vertex_list.append(key)
        return vertex_map[key]

    # Build face data
    face_indices = []
    for tri in triangles:
        i0 = get_vertex_idx(tri[0])
        i1 = get_vertex_idx(tri[1])
        i2 = get_vertex_idx(tri[2])
        face_indices.append((i0, i1, i2))

    # Create IFC cartesian points
    ifc_points = []
    for v in vertex_list:
        ifc_points.append(ifc.createIfcCartesianPoint([float(v[0]), float(v[1]), float(v[2])]))

    # Create faces
    ifc_faces = []
    for (i0, i1, i2) in face_indices:
        # IfcFace > IfcFaceBound > IfcPolyLoop
        loop = ifc.createIfcPolyLoop([ifc_points[i0], ifc_points[i1], ifc_points[i2]])
        bound = ifc.createIfcFaceOuterBound(loop, True)
        face = ifc.createIfcFace([bound])
        ifc_faces.append(face)

    # Create closed shell and brep
    shell = ifc.createIfcClosedShell(ifc_faces)
    brep = ifc.createIfcFacetedBrep(shell)

    return brep


def main():
    # Parameters
    L = 50.0
    A = 3.0
    N_STATIONS = 80
    CROWN_WIDTH = 0.300
    BASE_WIDTH = 0.600
    HEIGHT_START = 2.5
    HEIGHT_END = 4.5
    INCLINATION_RATIO = 10.0
    CROWN_SLOPE_PCT = 3.0
    FOUNDATION_WIDTH = 1.5
    FOUNDATION_THICKNESS = 0.5

    # Build wall triangles
    wall_tris, stats = build_faceted_brep_wall(
        n_stations=N_STATIONS, L=L, A=A,
        crown_width_top=CROWN_WIDTH, base_width=BASE_WIDTH,
        height_start=HEIGHT_START, height_end=HEIGHT_END,
        inclination_ratio=INCLINATION_RATIO, crown_slope_pct=CROWN_SLOPE_PCT,
        z_base=FOUNDATION_THICKNESS,
    )

    # Build foundation triangles
    foundation_tris = build_faceted_brep_foundation(
        n_stations=N_STATIONS, L=L, A=A,
        foundation_width=FOUNDATION_WIDTH, foundation_thickness=FOUNDATION_THICKNESS,
        crown_width_top=CROWN_WIDTH, base_width=BASE_WIDTH,
    )

    print(f"Wall triangles: {len(wall_tris)}")
    print(f"Foundation triangles: {len(foundation_tris)}")

    # Create IFC file
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T27"
    )

    length_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="LENGTHUNIT")
    area_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="AREAUNIT")
    volume_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="VOLUMEUNIT")
    ifcopenshell.api.run("unit.assign_unit", ifc, units=[length_unit, area_unit, volume_unit])

    ctx = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context", ifc,
        context_type="Model", context_identifier="Body", target_view="MODEL_VIEW",
        parent=ctx,
    )

    site = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcSite", name="Testgelaende"
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1"
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[facility], relating_object=site)

    # ── Wall element ──
    wall_brep = triangles_to_ifc_faceted_brep(ifc, wall_tris)
    wall_shape = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [wall_brep])
    wall_prod_shape = ifc.createIfcProductDefinitionShape(None, None, [wall_shape])

    origin1 = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement1 = ifc.createIfcAxis2Placement3D(origin1, None, None)
    local_placement1 = ifc.createIfcLocalPlacement(None, placement1)

    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="T27 Wall - S-curved variable",
        predefined_type="RETAININGWALL",
    )
    wall.ObjectPlacement = local_placement1
    wall.Representation = wall_prod_shape

    ifcopenshell.api.run(
        "spatial.assign_container", ifc, products=[wall], relating_structure=facility,
    )

    # ── Foundation element ──
    foundation_brep = triangles_to_ifc_faceted_brep(ifc, foundation_tris)
    foundation_shape = ifc.createIfcShapeRepresentation(body, "Body", "Brep", [foundation_brep])
    foundation_prod_shape = ifc.createIfcProductDefinitionShape(None, None, [foundation_shape])

    origin2 = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement2 = ifc.createIfcAxis2Placement3D(origin2, None, None)
    local_placement2 = ifc.createIfcLocalPlacement(None, placement2)

    foundation = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="T27 Foundation",
        predefined_type="RETAININGWALL",
    )
    foundation.ObjectPlacement = local_placement2
    foundation.Representation = foundation_prod_shape

    ifcopenshell.api.run(
        "spatial.assign_container", ifc, products=[foundation], relating_structure=facility,
    )

    # Write file
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T27_complex_long.ifc")
    ifc.write(out_path)
    print(f"\nWritten: {out_path}")
    print(f"  Wall: {stats['n_triangles']} triangles, {L}m long")
    print(f"  Crown width: {CROWN_WIDTH*1000:.0f}mm, Base width: {BASE_WIDTH*1000:.0f}mm")
    print(f"  Height: {HEIGHT_START}m to {HEIGHT_END}m")
    print(f"  Inclination: {INCLINATION_RATIO:.0f}:1")
    print(f"  Crown slope: {CROWN_SLOPE_PCT:.0f}%")
    print(f"  Foundation: {FOUNDATION_WIDTH}m wide, {FOUNDATION_THICKNESS}m thick")


if __name__ == "__main__":
    main()
