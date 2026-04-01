"""
Create test model T21: Wall with trapezoidal cross-section (IfcExtrudedAreaSolid).

Specifications:
- Extruded along X axis, 8.0m long
- Cross-section (trapezoid in YZ plane):
    - Bottom edge: y=0 to y=0.6 at z=0 (600mm wide)
    - Top edge: y=0.1 to y=0.4 at z=3.0 (300mm wide)
    - Front face: 10:1 inclination (Anzug)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcExtrudedAreaSolid with IfcArbitraryClosedProfileDef
- IFC version: IFC4X3_ADD2

Output: tests/test_models/T21_extruded_trapezoid.ifc
"""

import os
import ifcopenshell
import ifcopenshell.api


def main():
    # ---------- parameters ----------
    DEPTH = 8.0   # extrusion length along X [m]

    # Trapezoid cross-section vertices in the profile's 2D coordinate system
    # The profile plane will be oriented so that Y_profile -> Y_world, Z_profile -> Z_world
    # and extrusion goes along X_world.
    # Points: (y, z) in world terms
    profile_points_2d = [
        (0.0, 0.0),    # bottom-left (front-bottom)
        (0.6, 0.0),    # bottom-right (back-bottom)
        (0.4, 3.0),    # top-right (back-top)
        (0.1, 3.0),    # top-left (front-top)
    ]

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Thesis Test T21")

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
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Testgelände")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    facility = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcFacility", name="Strasse A1")
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[facility], relating_object=site)

    # ---------- build IfcExtrudedAreaSolid ----------

    # Profile polyline (2D points) — closed loop
    # IfcPolyline needs IfcCartesianPoint; for a closed profile, repeat the first point
    ifc_profile_pts = []
    for pt in profile_points_2d:
        ifc_profile_pts.append(ifc.createIfcCartesianPoint(pt))
    # Close the polyline by repeating the first point
    ifc_profile_pts.append(ifc.createIfcCartesianPoint(profile_points_2d[0]))

    polyline = ifc.createIfcPolyline(ifc_profile_pts)

    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", "Trapezprofil", polyline)

    # Position of the extruded solid:
    # We want the profile in the YZ plane, extruded along X.
    # IfcExtrudedAreaSolid extrudes along the ExtrudedDirection in the coordinate system
    # of the solid's Position (IfcAxis2Placement3D).
    #
    # Strategy: place the profile so its local X-axis = world Y, local Y-axis = world Z.
    # Then extrude along world X direction.
    #
    # IfcAxis2Placement3D(Location, Axis, RefDirection)
    #   Axis = local Z direction  -> we want extrusion along world X, so Axis = (1,0,0)
    #   RefDirection = local X direction -> world Y = (0,1,0)
    #   (local Y is derived as Axis x RefDirection = (1,0,0) x (0,1,0) = (0,0,1) = world Z)
    #
    # The profile is defined in local XY of this placement:
    #   profile X -> world Y, profile Y -> world Z  ✓

    solid_origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    solid_axis = ifc.createIfcDirection((1.0, 0.0, 0.0))       # local Z = world X (extrusion dir)
    solid_ref_dir = ifc.createIfcDirection((0.0, 1.0, 0.0))    # local X = world Y
    solid_placement = ifc.createIfcAxis2Placement3D(solid_origin, solid_axis, solid_ref_dir)

    # Extrusion direction is along the local Z of the placement (which is world X).
    # IfcExtrudedAreaSolid.ExtrudedDirection is relative to the Position's coordinate system,
    # so it should be (0,0,1) — i.e., along local Z.
    extrude_dir = ifc.createIfcDirection((0.0, 0.0, 1.0))

    extruded_solid = ifc.createIfcExtrudedAreaSolid(profile, solid_placement, extrude_dir, DEPTH)

    # Shape representation
    shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "SweptSolid", [extruded_solid])
    prod_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

    # Placement
    origin = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
    placement = ifc.createIfcAxis2Placement3D(origin, None, None)
    local_placement = ifc.createIfcLocalPlacement(None, placement)

    # IfcWall
    wall = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcWall",
        name="Stützmauer T21 — Extrudiertes Trapezprofil (10:1)",
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
    out_path = os.path.join(out_dir, "T21_extruded_trapezoid.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")

    # Trapezoid area = 0.5 * (b1 + b2) * h = 0.5 * (0.6 + 0.3) * 3.0 = 1.35 m²
    area = 0.5 * (0.6 + 0.3) * 3.0
    volume = area * DEPTH
    print(f"  Profile: trapezoid bottom=0.6m, top=0.3m, height=3.0m")
    print(f"  Cross-section area: {area:.4f} m²")
    print(f"  Extrusion depth: {DEPTH:.1f} m")
    print(f"  Expected volume: {volume:.4f} m³")


if __name__ == "__main__":
    main()
