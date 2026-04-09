# YAML Variable Reference

Complete reference of all variables available for YAML rule definitions.

## Usage in Rules

```yaml
- id: MY-RULE-001
  name: "Crown width check"
  check: "crown_width_mm >= 300"
  severity: ERROR
```

## Level 1 — Geometric Properties

| Variable | Type | Unit | Description | Method |
|----------|------|------|-------------|--------|
| `volume` | float | m³ | Mesh volume (divergence theorem, centered) | Gauss 1813 |
| `total_area` | float | m² | Total surface area (cross product) | — |
| `mesh_is_watertight` | bool | — | Every edge shared by exactly 2 faces | Euler char. |
| `num_triangles` | int | — | Number of triangles in mesh | — |
| `bbox_dim_max_m` | float | m | Longest BBox axis | — |
| `bbox_dim_mid_m` | float | m | Middle BBox axis | — |
| `bbox_dim_min_m` | float | m | Shortest BBox axis | — |
| `bbox_height_m` | float | m | Z-extent (vertical height) | — |
| `slenderness_ratio` | float | — | height / min_thickness (h/t) | — |
| `length_height_ratio` | float | — | max_dim / height | — |
| `volume_fill_ratio` | float | — | V_actual / V_bbox (1.0 = solid box) | — |

## Level 2 — Face Classification

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `crown_area_m2` | float | m² | Total crown face area |
| `foundation_area_m2` | float | m² | Total foundation face area |
| `front_area_m2` | float | m² | Total front face area |
| `back_area_m2` | float | m² | Total back face area |
| `front_back_area_ratio` | float | — | max(front,back) / min(front,back) |

## Level 3 — Measurements

### Crown (Mauerkrone)

| Variable | Type | Unit | Description | Method |
|----------|------|------|-------------|--------|
| `crown_width_mm` | float | mm | Width along tilted surface (not horizontal) | 1/cos(α) correction |
| `crown_slope_percent` | float | % | Area-weighted per-triangle slope | tan(α) × 100 |
| `crown_width_cv` | float | — | Coefficient of variation along wall | std/mean |
| `measurement_uncertainty_mm` | float | mm | Sagitta-based uncertainty (0 for straight) | δ≈L²κ/8, Farin 2002 |

### Wall (Wandgeometrie)

| Variable | Type | Unit | Description | Method |
|----------|------|------|-------------|--------|
| `min_wall_thickness_mm` | float | mm | Perpendicular to face (not horizontal) | cos(α) correction |
| `avg_wall_thickness_mm` | float | mm | Average perpendicular thickness | Median projection |
| `wall_height_m` | float | m | Maximum Z-extent of vertical faces | — |
| `wall_height_min_m` | float | m | Minimum height along centerline | Per-slice |
| `wall_height_max_m` | float | m | Maximum height along centerline | Per-slice |
| `wall_height_avg_m` | float | m | Average height along centerline | Per-slice |
| `wall_length_m` | float | m | Arc length of centerline | — |
| `is_curved` | bool | — | Curvature detected (3σ test) | Rissanen 1978 |
| `front_inclination_deg` | float | ° | Angle from vertical | arctan |
| `front_inclination_ratio` | float | n:1 | Rise/run ratio (inf = vertical) | — |

### Foundation (Fundament)

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `foundation_width_mm` | float | mm | Perpendicular extent |
| `foundation_width_ratio` | float | — | foundation_width / (wall_height × 1000) |

### Advanced Geometry

| Variable | Type | Unit | Description | Method |
|----------|------|------|-------------|--------|
| `taper_ratio` | float | n:1 | Thickness change ratio (linear regression) | — |
| `thickness_at_crown_mm` | float | mm | Thickness at top of wall | Taper profile |
| `thickness_at_base_mm` | float | mm | Thickness at bottom of wall | Taper profile |
| `thickness_min_mm` | float | mm | Minimum from taper profile | — |
| `thickness_max_mm` | float | mm | Maximum from taper profile | — |
| `front_plumbness_deg` | float | ° | Front face angle from vertical | arcsin(|n·z|) |
| `is_plumb` | bool | — | Deviation < 0.5° from vertical | — |
| `min_radius_m` | float | m | Minimum curvature radius | κ=|Δθ|/L, do Carmo 1976 |
| `max_curvature` | float | 1/m | Maximum curvature | — |

### Slope (Neigung)

| Variable | Type | Unit | Description | Method |
|----------|------|------|-------------|--------|
| `cross_slope_avg_pct` | float | % | Average cross-slope (Quergefälle) | Local frames |
| `cross_slope_max_pct` | float | % | Maximum local cross-slope | Local frames |
| `long_slope_avg_pct` | float | % | Average longitudinal slope | Local frames |
| `long_slope_max_pct` | float | % | Maximum longitudinal slope | Local frames |

## Level 5 — Inter-Element

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `foundation_extends_beyond_wall` | bool | — | Foundation wider than wall stem |
| `wall_foundation_gap_mm` | float | mm | Gap between stacked elements |
| `min_distance_to_nearest_mm` | float | mm | Shortest vertex distance to any other element |

## Level 6 — Terrain

| Variable | Type | Unit | Description | Method |
|----------|------|------|-------------|--------|
| `earth_side_determined` | bool | — | Terrain-based front/back resolved | Gradient |
| `crown_slope_towards_earth_side` | bool | — | Crown slopes toward terrain | — |
| `crown_height_above_terrain_m` | float | m | Crown Z minus terrain Z | Clearance |
| `wall_exposure_height_m` | float | m | Visible wall height above ground | — |
| `foundation_depth_below_terrain_m` | float | m | Min 3D distance to terrain surface | Surface dist |

## Level 7 — Distance

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `foundation_embedment_m` | float | m | Min surface distance to terrain (frost depth) |
| `foundation_embedment_vertical_m` | float | m | Vertical ΔZ (legacy) |
| `foundation_embedment_surface_m` | float | m | Min 3D surface distance (correct) |
