"""
Create test model T27: Long curved retaining wall along a slope (realistic scenario).

This tests the limits of the face classifier with a geometry that combines:
- Long wall (30m arc length) with 90° curve
- Variable crown height along the slope (z varies from 3m to 6m)
- Foundation follows terrain (not flat)
- 300mm crown width, 400mm wall thickness
- 20 segments (realistic tessellation from BIM software)

This is the most complex single-element test model and verifies that:
1. Centerline extraction works for long curved walls with varying Z
2. Crown face filtering handles >0.5m Z-variation (old bug: hardcoded 0.5m)
3. Slice-based measurements remain accurate over long curves
4. Face classification groups all crown faces correctly (not splitting by Z)

Entity: IfcWall with PredefinedType=RETAININGWALL
Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
IFC version: IFC4X3_ADD2

Output: tests/test_models/T27_long_curved_slope.ifc
"""

import os
import math
import numpy as np
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
RADIUS = 20.0           # curve radius [m]
ARC_ANGLE = math.pi / 2 # 90° arc -> arc length ≈ 31.4m
CROWN_WIDTH = 0.3       # crown width [m]
THICKNESS = 0.4         # wall thickness [m]
H_START = 3.0           # wall height at arc start [m]
H_END = 6.0             # wall height at arc end [m]
N_SEG = 20              # number of segments
Z_BASE_START = 0.0      # foundation base Z at start
Z_BASE_END = 2.0        # foundation base Z at end (follows terrain slope)


def main():
    # ---------- generate geometry ----------
    n_pts = N_SEG + 1
    angles = np.linspace(0, ARC_ANGLE, n_pts)

    # Center of arc at origin
    # Inner radius = RADIUS, outer radius = RADIUS + THICKNESS
    r_inner = RADIUS
    r_outer = RADIUS + THICKNESS

    verts = []
    # For each cross-section along the arc:
    # 4 vertices: inner-bottom, outer-bottom, outer-top, inner-top
    for i, a in enumerate(angles):
        t = i / (n_pts - 1)  # parameter 0..1
        z_base = Z_BASE_START + t * (Z_BASE_END - Z_BASE_START)
        h = H_START + t * (H_END - H_START)
        z_top = z_base + h

        cos_a, sin_a = math.cos(a), math.sin(a)

        # Inner wall face (back side)
        xi = r_inner * cos_a
        yi = r_inner * sin_a
        # Outer wall face (front side)
        xo = r_outer * cos_a
        yo = r_outer * sin_a

        verts.append((xi, yi, z_base))   # inner bottom  [4*i + 0]
        verts.append((xo, yo, z_base))   # outer bottom  [4*i + 1]
        verts.append((xo, yo, z_top))    # outer top     [4*i + 2]
        verts.append((xi, yi, z_top))    # inner top     [4*i + 3]

    # Build faces (quads -> 2 triangles each)
    faces = []
    for i in range(N_SEG):
        base = 4 * i
        nxt = 4 * (i + 1)

        # Front face (outer): base+1, nxt+1, nxt+2, base+2
        faces.append((base+1, nxt+1, nxt+2, base+2))
        # Back face (inner): base, base+3, nxt+3, nxt
        faces.append((base, base+3, nxt+3, nxt))
        # Foundation (bottom): base, nxt, nxt+1, base+1
        faces.append((base, nxt, nxt+1, base+1))
        # Crown (top): base+3, base+2, nxt+2, nxt+3
        faces.append((base+3, base+2, nxt+2, nxt+3))

    # End caps (left = start, right = end)
    # Left end (i=0): verts 0,1,2,3
    faces.append((0, 1, 2, 3))
    # Right end (i=N_SEG): verts 4*N_SEG, 4*N_SEG+1, 4*N_SEG+2, 4*N_SEG+3
    e = 4 * N_SEG
    faces.append((e, e+3, e+2, e+1))

    # ---------- Create IFC ----------
    model = ifcopenshell.api.run("project.create_file", version="IFC4X3_ADD2")
    proj = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="T27 Test")
    ifcopenshell.api.run("unit.assign_unit", model)
    ctx = ifcopenshell.api.run("context.add_context", model, context_type="Model",
                                context_identifier="Body", target_view="MODEL_VIEW")
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    bldg = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Storey")
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=proj, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[bldg])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=bldg, products=[storey])

    wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall",
                                 name="Lange Stützmauer T27", predefined_type="RETAININGWALL")
    ifcopenshell.api.run("spatial.assign_container", model, relating_structure=storey, products=[wall])

    # Create IfcFacetedBrep
    # IFC default unit is millimeters, so convert from meters
    ifc_verts = [model.createIfcCartesianPoint((v[0]*1000, v[1]*1000, v[2]*1000)) for v in verts]

    ifc_faces = []
    for face in faces:
        loop_pts = [ifc_verts[j] for j in face]
        loop = model.createIfcPolyLoop(loop_pts)
        bound = model.createIfcFaceOuterBound(loop, True)
        ifc_face = model.createIfcFace([bound])
        ifc_faces.append(ifc_face)

    shell = model.createIfcClosedShell(ifc_faces)
    brep = model.createIfcFacetedBrep(shell)

    rep = model.createIfcShapeRepresentation(ctx, "Body", "Brep", [brep])
    prod_rep = model.createIfcProductDefinitionShape(None, None, [rep])
    wall.Representation = prod_rep

    placement = model.createIfcLocalPlacement(
        None, model.createIfcAxis2Placement3D(model.createIfcCartesianPoint((0.0, 0.0, 0.0)))  # origin in mm
    )
    wall.ObjectPlacement = placement

    # Write
    out_dir = os.path.join(os.path.dirname(__file__), "test_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T27_long_curved_slope.ifc")
    model.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Arc length: {RADIUS * ARC_ANGLE:.1f}m, {N_SEG} segments")
    print(f"  Height: {H_START}m -> {H_END}m (varies by {H_END - H_START}m)")
    print(f"  Base Z: {Z_BASE_START}m -> {Z_BASE_END}m (slope)")
    print(f"  Crown width: {CROWN_WIDTH*1000:.0f}mm, Thickness: {THICKNESS*1000:.0f}mm")
    print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")


if __name__ == "__main__":
    main()
