"""Single source of truth for the YAML rule-expression variable catalog.

Each entry is `(type, unit, description)` where:
  - type: "float" | "bool" | "int" — expected value type
  - unit: SI unit or "—" for dimensionless
  - description: human-readable label used by the app and editor UIs

app.py, ruleset_editor.py, and docs/variable_reference.md all consume
this catalog. Keeping a single definition avoids the drift that used
to occur when one surface added a variable and the other forgot.
"""

VARIABLE_CATALOG: dict[str, dict[str, tuple[str, str, str]]] = {
    "Geometrie (L1)": {
        "volume": ("float", "m³", "Volumen (Divergenztheorem, Gauss 1813)"),
        "total_area": ("float", "m²", "Gesamtoberfläche"),
        "mesh_is_watertight": ("bool", "—", "Mesh geschlossen (Euler-Charakteristik)"),
        "num_triangles": ("int", "—", "Anzahl Dreiecke"),
        "bbox_dim_max_m": ("float", "m", "Längste BBox-Achse"),
        "bbox_dim_min_m": ("float", "m", "Kürzeste BBox-Achse"),
        "bbox_height_m": ("float", "m", "Vertikale BBox-Ausdehnung"),
        "volume_fill_ratio": ("float", "—", "V/V_bbox (1.0 = massiv)"),
        "slenderness_ratio": ("float", "—", "Höhe/Dicke"),
    },
    "Krone (L3)": {
        "crown_width_mm": ("float", "mm", "Kronenbreite (Oberfläche, p10 robust)"),
        "crown_slope_percent": ("float", "%", "Kronenneigung"),
        "crown_width_cv": ("float", "—", "Profilkonsistenz (CV)"),
        "measurement_uncertainty_mm": ("float", "mm", "Messunsicherheit (Sagitta, Farin 2002)"),
        "crown_area_m2": ("float", "m²", "Kronenfläche"),
    },
    "Wand (L3)": {
        "min_wall_thickness_mm": ("float", "mm", "Wandstärke senkrecht (min)"),
        "avg_wall_thickness_mm": ("float", "mm", "Wandstärke senkrecht (avg)"),
        "wall_height_m": ("float", "m", "Wandhöhe (max)"),
        "wall_height_min_m": ("float", "m", "Wandhöhe (min entlang Achse)"),
        "wall_height_avg_m": ("float", "m", "Wandhöhe (avg entlang Achse)"),
        "wall_length_m": ("float", "m", "Wandlänge (Centerline)"),
        "front_inclination_ratio": ("float", "n:1", "Neigung Ansichtsfläche (Anzug)"),
        "front_plumbness_deg": ("float", "°", "Lotabweichung"),
        "is_curved": ("bool", "—", "Gekrümmte Wand erkannt"),
        "min_radius_m": ("float", "m", "Min. Krümmungsradius"),
        "taper_ratio": ("float", "n:1", "Anzug-Verhältnis"),
        "thickness_at_crown_mm": ("float", "mm", "Dicke am Mauerkopf"),
        "thickness_at_base_mm": ("float", "mm", "Dicke am Mauerfuss"),
    },
    "Neigung (L3)": {
        "cross_slope_avg_pct": ("float", "%", "Quergefälle (Durchschnitt)"),
        "cross_slope_max_pct": ("float", "%", "Quergefälle (Maximum)"),
        "long_slope_avg_pct": ("float", "%", "Längsgefälle (Durchschnitt)"),
        "long_slope_max_pct": ("float", "%", "Längsgefälle (Maximum)"),
    },
    "Fundament / Terrain (L3/L6)": {
        "foundation_width_mm": ("float", "mm", "Fundamentbreite"),
        "foundation_width_ratio": ("float", "—", "Fundamentbreite/Wandhöhe"),
        "foundation_embedment_m": ("float", "m", "Einbindetiefe (3D-Oberflächendistanz)"),
        "crown_height_above_terrain_m": ("float", "m", "Kronenhöhe über Terrain"),
        "foundation_depth_below_terrain_m": ("float", "m", "Fundament unter Terrain"),
    },
    "Flächen (L2)": {
        "front_area_m2": ("float", "m²", "Ansichtsfläche"),
        "back_area_m2": ("float", "m²", "Rückfläche"),
        "front_back_area_ratio": ("float", "—", "Front/Back Verhältnis"),
    },
    "Inter-Element (L5/L6)": {
        "min_distance_to_nearest_mm": ("float", "mm", "Min. Abstand zu Nachbar"),
    },
}


__all__ = ["VARIABLE_CATALOG"]
