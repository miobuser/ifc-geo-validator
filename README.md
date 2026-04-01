# ifc-geo-validator

[![Tests](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml/badge.svg)](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml)

Geometric validation of IFC infrastructure models against configurable requirements.

## Overview

`ifc-geo-validator` analyses the geometry of IFC building elements and validates them against configurable YAML rulesets. It extracts triangulated meshes from IFC geometry via IfcOpenShell/OpenCASCADE, classifies faces by surface normal direction, computes geometric properties per face group, and checks them against normative requirements.

Designed for retaining walls per ASTRA FHB T/G, but extensible to any element type (tunnel components, bridge elements, structural members) via custom rulesets.

## Validation Levels

| Level | Description | Status |
|-------|-------------|--------|
| L1 | Geometric properties (volume, area, bbox, centroid, watertight, mesh quality) | Done |
| L2 | Face classification (crown, foundation, front, back, ends) with curved wall support | Done |
| L3 | Face-specific measurements (crown width, slope, thickness, inclination, height) | Done |
| L4 | YAML rule evaluation (L1–L7 rules, composite checks, bbox dimensions) | Done |
| L5 | Inter-element context (Contact Surface Normal Analysis: stacked/side-by-side) | Done |
| L6 | Terrain context + distance checks (IfcSite geometry, clearance, embedment) | Done |
| L7 | Distance calculations (YAML-configurable, evaluated via L4 engine) | Done |

## Installation

```bash
pip install ifc-geo-validator
```

Or from source:

```bash
git clone https://github.com/miobuser/ifc-geo-validator.git
cd ifc-geo-validator
pip install -e ".[dev,viz]"
```

## Usage

### CLI

```bash
# Validate an IFC file against ASTRA rules
ifc-geo-validator model.ifc --ruleset rulesets/astra_fhb_stuetzmauer.yaml

# Filter for retaining walls only
ifc-geo-validator model.ifc --filter-type IfcWall --filter-predefined RETAININGWALL

# Multiple entity types (wall + foundation for L5 inter-element checks)
ifc-geo-validator model.ifc --filter-type IfcWall,IfcSlab,IfcFooting

# Run specific levels
ifc-geo-validator model.ifc --levels 1,2,3

# Write validation results back into IFC as Pset_GeoValidation
ifc-geo-validator model.ifc --enrich model_validated.ifc

# Export failed checks as BCF 2.1 issues
ifc-geo-validator model.ifc --bcf issues.bcf

# Machine-readable summary for CI/CD (exits with code 1 on failure)
ifc-geo-validator model.ifc --summary
```

### Web App

```bash
pip install -e ".[web]"
streamlit run src/ifc_geo_validator/app.py
```

Upload an IFC file in the browser, configure filters, and view interactive validation results. Download options: JSON report, enriched IFC (with Pset_GeoValidation), and BCF 2.1 issue file.

### Browser Viewer

```bash
python viewer/server.py
```

Interactive 3D viewer with centerline rendering, face classification highlighting, measurement annotations, and screenshot export.

## Rulesets

Validation rules are defined in YAML and fully configurable. Three bundled rulesets:

| Ruleset | Standard | Scope | Key Checks |
|---------|----------|-------|------------|
| `astra_fhb_stuetzmauer.yaml` | ASTRA FHB T/G 24 001-15101 | Retaining walls | Crown ≥ 300mm, thickness ≥ 300mm, slope 3%, inclination 10:1 |
| `sia_262_stuetzmauer.yaml` | SIA 262 / SIA 267 | Retaining walls | Crown ≥ 200mm, thickness ≥ 200mm, height ≤ 3m |
| `astra_fhb_tunnel.yaml` | ASTRA FHB T/G 24 001-10xxx | Tunnel components | Volume, mesh validity, min thickness, bbox dimensions |

```yaml
level_3:
  - id: ASTRA-SM-L3-001
    name: "Kronenbreite Minimum"
    check: "crown_width_mm >= 300"
    severity: ERROR
    reference: "FHB T/G 24 001-15101, Kap. 3"
```

Custom rulesets can define rules for any computed variable (L1–L7), including bounding box dimensions (`bbox_dim_min_m`, `bbox_height_m`) for non-wall elements.

## Key Features

- **Curved wall support**: Centerline extraction with local coordinate frames for accurate measurements on arcs, S-curves, and polygonal walls
- **Multi-entity-type validation**: `--filter-type IfcWall,IfcSlab,IfcFooting` in a single run for inter-element checks
- **Contact Surface Normal Analysis** (L5): Mathematically founded pair classification using contact surface orientation
- **Terrain context** (L6): Automatic earth/air side determination from IfcSite geometry, crown-terrain clearance, foundation embedment
- **No heuristics**: All algorithms are mathematically derived (cos(π/4) thresholds, 3-σ significance tests, CDR, natural-gap clustering)
- **Robustness**: Degenerate triangle filtering, multi-body detection, scale-invariant tolerances (mm/m/km/LV95 coordinates)
- **Mesh quality diagnostics**: Degenerate count, aspect ratio, non-manifold edges, edge length statistics
- **Profile consistency** (L3): Coefficient of variation of crown width along curved walls
- **6 output formats**: CLI, JSON, enriched IFC (Pset_GeoValidation), BCF 2.1, Streamlit web app, browser viewer
- **26 test models** (T1–T27): simple boxes, inclined walls, L/T-profiles, curved arcs, polygonal walls, stepped profiles, variable height, buttresses, terrain context, ASTRA-compliant, multi-failure scenarios
- **348 automated tests** across 10 test suites (edge cases, scale invariance, multi-body detection)
- **Tested on real IFC models**: IFC4 bridge model (46 elements, 7 entity types), IFC4 tunnel model (102 elements, 2.5M triangles)

## Testing

```bash
pytest                                    # Run all 348 tests
python validate_all_models.py             # Batch validate all 26 models
python sensitivity_analysis.py            # Threshold sensitivity sweep
python generate_thesis_figures.py         # Generate 3D classification figures
```

## Context

Bachelor thesis at BFH (Berner Fachhochschule), Civil Engineering, FS 2026.

**Topic:** Geometrische Interpretation und Validierung von digitalen Bauwerksmodellen am Beispiel von Stützmauern nach ASTRA-Fachhandbuch

## License

MIT
