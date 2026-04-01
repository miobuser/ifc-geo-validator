"""
Create test model T28: Showcase retaining wall exercising ALL validation levels.

Swiss highway scenario: 60° curved retaining wall along a slope with terrain,
wall stem (10:1 Anzug, 3% crown slope), foundation slab, and terrain surface.
Exercises L1-L7 including terrain-based front/back classification.

Specifications:
- Wall stem: 60° arc, R=12m, height 3.5m, crown 350mm→base 700mm (10:1)
  3% crown slope toward earth side
- Foundation: same arc, 1.8m wide, 0.5m thick
- Terrain: sloped surface rising toward the wall (earth side)
- PredefinedType: RETAININGWALL
- Geometry: IfcFacetedBrep

Output: tests/test_models/T28_showcase.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# Parameters
R_INNER = 12.0
ARC_DEG = 60.0
N_SEG = 15

# Wall stem
STEM_HEIGHT = 3.5
STEM_THICK_CROWN = 0.35   # 350mm at crown
CROWN_SLOPE_PCT = 3.0      # 3% toward earth
INCLINATION_RATIO = 10.0   # 10:1

# Foundation
FOUND_WIDTH = 1.8          # 1.8m total
FOUND_HEIGHT = 0.5
FOUND_OVERHANG_IN = 0.3    # 0.3m beyond wall inner face
FOUND_OVERHANG_OUT = FOUND_WIDTH - STEM_THICK_CROWN - FOUND_OVERHANG_IN

# Heights
Z_FOUND_BOTTOM = 0.0
Z_FOUND_TOP = FOUND_HEIGHT
Z_WALL_BASE = Z_FOUND_TOP
Z_WALL_CROWN = Z_WALL_BASE + STEM_HEIGHT

# Wall base thickness (from inclination)
STEM_THICK_BASE = STEM_THICK_CROWN + STEM_HEIGHT / INCLINATION_RATIO

# Crown slope: 3% means Z drops by 0.03 * width across the crown
CROWN_DZ = CROWN_SLOPE_PCT / 100.0 * STEM_THICK_CROWN


def arc_point(radius, angle_deg):
    """Point on arc in XY plane."""
    a = math.radians(angle_deg)
    return (radius * math.cos(a), radius * math.sin(a))


def create_brep_wall(model, context, name, inner_r, outer_r_base, outer_r_crown,
                     z_bottom, z_top, crown_dz=0.0, n_seg=N_SEG, arc_deg=ARC_DEG):
    """Create a curved wall as IfcFacetedBrep with quad faces."""
    angles = [arc_deg * i / n_seg for i in range(n_seg + 1)]

    # Generate cross-section vertices per segment
    # Each segment has 4 vertices: inner_bottom, outer_bottom, outer_top, inner_top
    all_pts = []
    for angle in angles:
        # Crown slope: outer edge is lower by crown_dz
        ib = (*arc_point(inner_r, angle), z_bottom)
        ob = (*arc_point(outer_r_base, angle), z_bottom)
        ot = (*arc_point(outer_r_crown, angle), z_top - crown_dz)
        it_ = (*arc_point(inner_r, angle), z_top)
        all_pts.extend([ib, ob, ot, it_])

    # Create IFC points
    ifc_pts = [model.createIfcCartesianPoint(p) for p in all_pts]

    faces = []
    for i in range(n_seg):
        base = i * 4
        nxt = (i + 1) * 4
        # Inner face (vertical)
        faces.append([base + 0, base + 3, nxt + 3, nxt + 0])
        # Outer face (inclined)
        faces.append([base + 1, nxt + 1, nxt + 2, base + 2])
        # Top face (crown)
        faces.append([base + 3, base + 2, nxt + 2, nxt + 3])
        # Bottom face (foundation contact)
        faces.append([base + 0, nxt + 0, nxt + 1, base + 1])

    # End caps
    faces.append([0, 1, 2, 3])  # Start end
    faces.append([(n_seg) * 4, (n_seg) * 4 + 3, (n_seg) * 4 + 2, (n_seg) * 4 + 1])  # End end

    ifc_faces = []
    for f in faces:
        loop = model.createIfcPolyLoop([ifc_pts[vi] for vi in f])
        bound = model.createIfcFaceOuterBound(loop, True)
        ifc_faces.append(model.createIfcFace([bound]))

    shell = model.createIfcClosedShell(ifc_faces)
    brep = model.createIfcFacetedBrep(shell)

    rep = model.createIfcShapeRepresentation(
        context, "Body", "Brep", [brep]
    )
    product_rep = model.createIfcProductDefinitionShape(None, None, [rep])

    wall = ifcopenshell.api.run("root.create_entity", model,
                                 ifc_class="IfcWall", name=name,
                                 predefined_type="RETAININGWALL")
    wall.Representation = product_rep
    return wall


def create_terrain(model, context, n_seg=N_SEG, arc_deg=ARC_DEG):
    """Create a sloped terrain surface as IfcSite with geometry.

    Terrain rises toward the outer (earth) side of the wall.
    """
    angles = [arc_deg * i / n_seg for i in range(n_seg + 1)]

    # Terrain grid: inner edge (low, valley side) to outer edge (high, earth side)
    terrain_inner_r = R_INNER - 3.0   # 3m in front of wall (valley)
    terrain_outer_r = R_INNER + FOUND_WIDTH + 3.0  # 3m behind wall (earth)
    terrain_z_inner = -1.0  # Valley is 1m below foundation
    terrain_z_outer = Z_WALL_CROWN - 0.5  # Earth is 0.5m below crown

    pts = []
    for angle in angles:
        # Inner (valley) edge
        pi = (*arc_point(terrain_inner_r, angle), terrain_z_inner)
        # Outer (earth) edge
        po = (*arc_point(terrain_outer_r, angle), terrain_z_outer)
        pts.extend([pi, po])

    ifc_pts = [model.createIfcCartesianPoint(p) for p in pts]

    faces = []
    for i in range(n_seg):
        base = i * 2
        nxt = (i + 1) * 2
        # Quad: inner_i, outer_i, outer_i+1, inner_i+1
        faces.append([base, base + 1, nxt + 1, nxt])

    ifc_faces = []
    for f in faces:
        loop = model.createIfcPolyLoop([ifc_pts[vi] for vi in f])
        bound = model.createIfcFaceOuterBound(loop, True)
        ifc_faces.append(model.createIfcFace([bound]))

    shell = model.createIfcOpenShell(ifc_faces)
    face_surface = model.createIfcShellBasedSurfaceModel([shell])

    rep = model.createIfcShapeRepresentation(
        context, "Body", "SurfaceModel", [face_surface]
    )
    product_rep = model.createIfcProductDefinitionShape(None, None, [rep])
    return product_rep


def main():
    model = ifcopenshell.api.run("project.create_file", version="IFC4X3_ADD2")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject",
                                    name="T28 Showcase")
    ifcopenshell.api.run("unit.assign_unit", model, length={"is_metric": True, "raw": "METRE"})
    context = ifcopenshell.api.run("context.add_context", model,
                                    context_type="Model",
                                    context_identifier="Body",
                                    target_view="MODEL_VIEW")

    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite",
                                 name="Terrain")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding",
                                     name="Highway Structure")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey",
                                   name="Level 0")

    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])

    # Wall stem
    wall = create_brep_wall(
        model, context,
        name="Stützmauer T28 — Showcase Stem",
        inner_r=R_INNER,
        outer_r_base=R_INNER + STEM_THICK_BASE,
        outer_r_crown=R_INNER + STEM_THICK_CROWN,
        z_bottom=Z_WALL_BASE,
        z_top=Z_WALL_CROWN,
        crown_dz=CROWN_DZ,
    )
    ifcopenshell.api.run("spatial.assign_container", model,
                          relating_structure=storey, products=[wall])

    # Foundation slab
    foundation = create_brep_wall(
        model, context,
        name="Stützmauer T28 — Foundation",
        inner_r=R_INNER - FOUND_OVERHANG_IN,
        outer_r_base=R_INNER + STEM_THICK_CROWN + FOUND_OVERHANG_OUT,
        outer_r_crown=R_INNER + STEM_THICK_CROWN + FOUND_OVERHANG_OUT,
        z_bottom=Z_FOUND_BOTTOM,
        z_top=Z_FOUND_TOP,
        crown_dz=0,
    )
    ifcopenshell.api.run("spatial.assign_container", model,
                          relating_structure=storey, products=[foundation])

    # Terrain
    terrain_rep = create_terrain(model, context)
    site.Representation = terrain_rep

    out = os.path.join(os.path.dirname(__file__), "test_models", "T28_showcase.ifc")
    model.write(out)
    print(f"Written: {out}")
    print(f"Wall stem: R={R_INNER}m, {ARC_DEG}° arc, h={STEM_HEIGHT}m, "
          f"crown={STEM_THICK_CROWN*1000:.0f}mm, base={STEM_THICK_BASE*1000:.0f}mm, "
          f"slope={CROWN_SLOPE_PCT}%, 10:1")
    print(f"Foundation: {FOUND_WIDTH}m wide, {FOUND_HEIGHT}m thick")


if __name__ == "__main__":
    main()
