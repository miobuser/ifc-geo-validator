"""
Create test model T26: Wall with curved front face (IfcExtrudedAreaSolid).

Specifications:
- Extruded along local X axis, 6.0m long
- Cross-section (polygon in YZ plane approximating curved front):
    - Crown: 300mm wide (top edge)
    - Base: 450mm wide (bottom edge)
    - Height: 3.0m
    - Front face: arc from (0.45, 0) to (0.3, 3.0) approximated by 8 segments
    - Back face: straight vertical at y=0
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcExtrudedAreaSolid with IfcArbitraryClosedProfileDef
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T26_extruded_curved.ifc
"""

import os
import math
import ifcopenshell
import ifcopenshell.api


def main():
    # ---------- parameters ----------
    DEPTH = 6.0         # extrusion length along X [m]
    BASE_WIDTH = 0.45   # bottom edge width [m]
    TOP_WIDTH = 0.30    # top edge / crown width [m]
    HEIGHT = 3.0        # wall height [m]
    N_ARC_SEGMENTS = 8  # number of intermediate points for arc approximation

    # ---------- build profile polygon (YZ plane) ----------
    # The profile is a closed polygon in 2D (y, z):
    #   - Back face: straight vertical from (0, 0) up to (0, 3.0)
    #   - Crown: horizontal from (0, 3.0) to (0.3, 3.0)
    #   - Front face: arc from (0.3, 3.0) down to (0.45, 0) with 8 intermediate points
    #   - Foundation: horizontal from (0.45, 0) back to (0, 0)
    #
    # Arc approximation: parameterize t from 0 to 1
    #   start = (0.3, 3.0)  (front-top)
    #   end   = (0.45, 0.0) (front-bottom)
    # Use a circular arc bulging outward (toward +Y)

    # Arc center and radius calculation:
    # We'll use a simple parametric arc. The arc goes from top-front to bottom-front,
    # bulging outward. We parameterize using a sine-based bulge.
    arc_start = (TOP_WIDTH, HEIGHT)   # (0.3, 3.0)
    arc_end = (BASE_WIDTH, 0.0)       # (0.45, 0.0)

    # Maximum bulge at midpoint: ~50mm outward from the straight line
    MAX_BULGE = 0.05  # 50mm

    arc_points = []
    for i in range(1, N_ARC_SEGMENTS + 1):
        t = i / (N_ARC_SEGMENTS + 1)
        # Linear interpolation along the line from start to end
        y_lin = arc_start[0] + t * (arc_end[0] - arc_start[0])
        z_lin = arc_start[1] + t * (arc_end[1] - arc_start[1])
        # Sinusoidal bulge outward (+Y direction)
        bulge = MAX_BULGE * math.sin(math.pi * t)
        arc_points.append((y_lin + bulge, z_lin))

    # Assemble profile polygon (counter-clockwise when viewed from +X)
    profile_points_2d = []
    # 1. Bottom-left (back face, foundation level)
    profile_points_2d.append((0.0, 0.0))
    # 2. Bottom-right (front face, foundation level)
    profile_points_2d.append((BASE_WIDTH, 0.0))
    # 3. Arc points from bottom-front up to top-front
    for pt in reversed(arc_points):
        profile_points_2d.append(pt)
    # 4. Top-right (front face, crown level)
    profile_points_2d.append((TOP_WIDTH, HEIGHT))
    # 5. Top-left (back face, crown level)
    profile_points_2d.append((0.0, HEIGHT))
    # Close back to (0, 0) — handled by repeating first point in polyline

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T26"
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
        "root.create_entity", ifc, ifc_class="IfcSite", name="Testgelaende"
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1"
    )
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[facility], relating_object=site
    )

    # ---------- build IfcExtrudedAreaSolid ----------

    # Profile polyline (2D points) — closed loop
    ifc_profile_pts = []
    for pt in profile_points_2d:
        ifc_profile_pts.append(ifc.createIfcCartesianPoint(pt))
    # Close the polyline by repeating the first point
    ifc_profile_pts.append(ifc.createIfcCartesianPoint(profile_points_2d[0]))

    polyline = ifc.createIfcPolyline(ifc_profile_pts)

    profile = ifc.createIfcArbitraryClosedProfileDef(
        "AREA", "Curved Front Profile", polyline
    )

    # Position of the extruded solid:
    # Profile in YZ plane, extruded along X.
    # IfcAxis2Placement3D(Location, Axis, RefDirection)
    #   Axis = local Z = world X (extrusion direction)
    #   RefDirection = local X = world Y
    #   (local Y = Axis x RefDirection = world Z)
    solid_origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    solid_axis = ifc.createIfcDirection((1.0, 0.0, 0.0))     # local Z = world X
    solid_ref_dir = ifc.createIfcDirection((0.0, 1.0, 0.0))  # local X = world Y
    solid_placement = ifc.createIfcAxis2Placement3D(
        solid_origin, solid_axis, solid_ref_dir
    )

    # Extrusion direction along local Z (= world X)
    extrude_dir = ifc.createIfcDirection((0.0, 0.0, 1.0))

    extruded_solid = ifc.createIfcExtrudedAreaSolid(
        profile, solid_placement, extrude_dir, DEPTH
    )

    # Shape representation
    shape_rep = ifc.createIfcShapeRepresentation(
        body, "Body", "SweptSolid", [extruded_solid]
    )
    prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

    # Placement
    origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    placement = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement = ifc.createIfcLocalPlacement(None, placement)

    # IfcWall
    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stuetzmauer T26 - Extrudiertes Profil mit gekruemmter Vorderseite",
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
    out_path = os.path.join(out_dir, "T26_extruded_curved.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")

    # Approximate volume: average cross-section area * depth
    # The polygon approximates a shape with base=0.45, top=0.30, height=3.0
    # plus a slight bulge. Approximate as trapezoid + bulge correction.
    trap_area = 0.5 * (BASE_WIDTH + TOP_WIDTH) * HEIGHT  # 1.125 m^2
    volume = trap_area * DEPTH
    print(f"  Profile: curved front, base={BASE_WIDTH}m, top={TOP_WIDTH}m, height={HEIGHT}m")
    print(f"  Approx cross-section area: {trap_area:.4f} m^2")
    print(f"  Extrusion depth: {DEPTH:.1f} m")
    print(f"  Approx expected volume: {volume:.4f} m^3 (+ bulge)")


if __name__ == "__main__":
    main()
