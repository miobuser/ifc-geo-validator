"""
Create test model T10: Complex curved retaining wall with inclination,
crown slope, and variable thickness.

Specifications:
- 60° arc in plan view, 16 segments
- Inner radius R_inner = 8.0 m (constant at all heights, vertical back face)
- Outer radius at base (z=0): R_outer_base = 8.6 m (thickness at base = 600 mm)
- Outer radius at crown (z~4.0): R_outer_top = 8.3 m (thickness at crown = 300 mm)
- Front face inclination: (8.6 - 8.3) / 4.0 = 0.3 / 4.0 = 1:13.3 (~4.29°)
- Crown slope 3%: inner edge z = 4.0, outer edge z = 4.0 + 0.3 * 0.03 = 4.009
- Height: 4.0 m (at inner edge)
- Entity: IfcWall with PredefinedType=RETAININGWALL
- Geometry: IfcFacetedBrep (quad faces via IfcPolyLoop)
- IFC version: IFC4X3_ADD2

Expected analysis results:
- is_curved: True
- Crown width: ~300 mm (local measurement at top)
- Front inclination: ~4.29° (inclined outer face)
- Crown slope: ~3%
- 6 face groups after merge (crown, foundation, front, back, 2 ends)

Output: tests/test_models/T10_complex_curved.ifc
"""

import math
import os
import ifcopenshell
import ifcopenshell.api

# ---------- parameters ----------
R_INNER = 8.0               # inner radius [m] — constant at all heights
R_OUTER_BASE = 8.6           # outer radius at base (z=0) [m]
R_OUTER_TOP = 8.3            # outer radius at crown [m]
HEIGHT = 4.0                 # wall height at inner edge [m]
CROWN_SLOPE = 0.03           # 3% transverse slope on the crown
N_SEG = 16                   # number of arc segments
ARC_DEG = 60.0               # arc angle [degrees]

# Derived
THICKNESS_BASE = R_OUTER_BASE - R_INNER    # 0.6 m
THICKNESS_TOP = R_OUTER_TOP - R_INNER      # 0.3 m

# Crown z-coordinates
Z_CROWN_INNER = HEIGHT                                    # 4.0 m
Z_CROWN_OUTER = HEIGHT + THICKNESS_TOP * CROWN_SLOPE      # 4.0 + 0.3 * 0.03 = 4.009 m


def make_arc_points(radius: float, n_pts: int, z: float) -> list[tuple[float, float, float]]:
    """Generate points along an arc from 0° to ARC_DEG at the given radius and z."""
    pts = []
    for i in range(n_pts):
        angle = math.radians(ARC_DEG * i / (n_pts - 1))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pts.append((x, y, z))
    return pts


def main():
    # ---------- generate geometry vertices ----------
    n_pts = N_SEG + 1  # 17 points for 16 segments

    # Bottom ring (z=0)
    inner_bot = make_arc_points(R_INNER, n_pts, 0.0)
    outer_bot = make_arc_points(R_OUTER_BASE, n_pts, 0.0)

    # Top ring (crown with slope)
    # Inner top: vertical back face, z = HEIGHT
    inner_top = make_arc_points(R_INNER, n_pts, Z_CROWN_INNER)
    # Outer top: inclined front face (smaller radius) + crown slope (slightly higher z)
    outer_top = make_arc_points(R_OUTER_TOP, n_pts, Z_CROWN_OUTER)

    # ---------- create IFC file ----------
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4X3")

    # Project
    project = ifcopenshell.api.run(
        "root.create_entity", ifc,
        ifc_class="IfcProject",
        name="Thesis Test T10",
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
        """Create/cache an IfcCartesianPoint with rounded coordinates."""
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
        # --- Outer face (front, inclined) ---
        # outer_bot uses R_OUTER_BASE, outer_top uses R_OUTER_TOP
        # so the front face tilts inward from bottom to top
        faces.append(make_face([
            outer_bot[i], outer_bot[i + 1], outer_top[i + 1], outer_top[i]
        ]))

        # --- Inner face (back, vertical) ---
        faces.append(make_face([
            inner_bot[i + 1], inner_bot[i], inner_top[i], inner_top[i + 1]
        ]))

        # --- Top face (crown, sloped from inner to outer) ---
        # inner_top at z=4.0, outer_top at z=4.009
        faces.append(make_face([
            inner_top[i], outer_top[i], outer_top[i + 1], inner_top[i + 1]
        ]))

        # --- Bottom face (foundation) ---
        faces.append(make_face([
            outer_bot[i], inner_bot[i], inner_bot[i + 1], outer_bot[i + 1]
        ]))

    # --- End cap at angle = 0° (start) ---
    faces.append(make_face([
        outer_bot[0], outer_top[0], inner_top[0], inner_bot[0]
    ]))

    # --- End cap at angle = 60° (end) ---
    faces.append(make_face([
        outer_bot[-1], inner_bot[-1], inner_top[-1], outer_top[-1]
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
        name="Stützmauer T10 — Komplexe gekrümmte Wand (60° Bogen, geneigt, Kronengefälle, variable Dicke)",
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
    out_path = os.path.join(out_dir, "T10_complex_curved.ifc")
    ifc.write(out_path)
    print(f"Written: {out_path}")
    print(f"  Points cached: {len(pt_cache)}")
    print(f"  Faces created: {len(faces)}")

    # Summary
    inclination_deg = math.degrees(math.atan2(R_OUTER_BASE - R_OUTER_TOP, HEIGHT))
    print(f"\n  Parameters:")
    print(f"    Arc: {ARC_DEG}° in {N_SEG} segments")
    print(f"    R_inner = {R_INNER} m (constant)")
    print(f"    R_outer_base = {R_OUTER_BASE} m, R_outer_top = {R_OUTER_TOP} m")
    print(f"    Thickness at base = {THICKNESS_BASE * 1000:.0f} mm")
    print(f"    Thickness at crown = {THICKNESS_TOP * 1000:.0f} mm")
    print(f"    Front inclination = {inclination_deg:.2f}°  (1:{HEIGHT / (R_OUTER_BASE - R_OUTER_TOP):.1f})")
    print(f"    Crown slope = {CROWN_SLOPE * 100:.1f}%")
    print(f"    Crown z range: {Z_CROWN_INNER} .. {Z_CROWN_OUTER}")

    # Approximate volume (sector of annular ring, trapezoidal cross-section)
    arc_rad = math.radians(ARC_DEG)
    # Average outer radius
    r_outer_avg = (R_OUTER_BASE + R_OUTER_TOP) / 2.0
    # Average height
    h_avg = (Z_CROWN_INNER + Z_CROWN_OUTER) / 2.0
    # Annular sector volume with average radii
    v_approx = (arc_rad / 2.0) * (r_outer_avg**2 - R_INNER**2) * h_avg
    print(f"    Approximate volume: {v_approx:.4f} m³")


if __name__ == "__main__":
    main()
