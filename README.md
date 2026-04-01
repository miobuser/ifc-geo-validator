# ifc-geo-validator

[![Tests](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml/badge.svg)](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml)

Geometric validation of IFC infrastructure models against configurable requirements.

## Overview

`ifc-geo-validator` analyses the geometry of IFC building elements (focused on retaining walls per ASTRA FHB T/G) and validates them against configurable YAML rulesets. It extracts triangulated meshes from IFC geometry via IfcOpenShell/OpenCASCADE, classifies faces by surface normal direction, computes geometric properties per face group, and checks them against normative requirements.

## Validation Levels

| Level | Description | Status |
|-------|-------------|--------|
| L1 | Geometric properties (volume, area, bbox, centroid, watertight) | Done |
| L2 | Face classification (crown, foundation, front, back, ends) with curved wall support | Done |
| L3 | Face-specific measurements (crown width, slope, thickness, inclination, height) | Done |
| L4 | YAML rule evaluation (L1-L7 rules, composite checks) | Done |
| L5 | Inter-element context (Contact Surface Normal Analysis: stacked/side-by-side) | Done |
| L6 | Terrain context + distance checks (IfcSite geometry, clearance, element distances) | Done |
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

## Rulesets

Validation rules are defined in YAML and fully configurable. Two bundled rulesets demonstrate different normative requirements:

| Ruleset | Standard | Crown min | Thickness min | Inclination |
|---------|----------|-----------|---------------|-------------|
| `astra_fhb_stuetzmauer.yaml` | ASTRA FHB T/G 24 001-15101 | 300 mm | 300 mm | 10:1 recommended |
| `sia_262_stuetzmauer.yaml` | SIA 262 / SIA 267 | 200 mm | 200 mm | — |

```yaml
level_3:
  - id: ASTRA-SM-L3-001
    name: "Kronenbreite Minimum"
    check: "crown_width_mm >= 300"
    severity: ERROR
    reference: "FHB T/G 24 001-15101, Kap. 3"
```

Custom rulesets can define rules for any computed variable (L1–L7).

## Key Features

- **Curved wall support**: Centerline extraction with local coordinate frames for accurate measurements on arcs, S-curves, and polygonal walls
- **Multi-element validation**: Separate IFC elements for wall stem, foundation, buttresses — each validated independently
- **Contact Surface Normal Analysis** (L5): Mathematically founded pair classification using the contact surface orientation (no heuristics)
- **Terrain context** (L6): Automatic earth/air side determination from IfcSite geometry, crown-terrain clearance measurement
- **No heuristics**: All 17 algorithms are mathematically derived (cos(pi/4) thresholds, 3-sigma significance tests, CDR, natural-gap clustering)
- **Foundation embedment depth** (L7): Automatic depth computation from terrain surface via barycentric interpolation
- **Profile consistency** (L3): Coefficient of variation of crown width along curved walls
- **26 test models** (T1–T27) covering: simple boxes, inclined walls, L/T-profiles, curved arcs (90/180/S-curve), polygonal walls, stepped profiles, variable height, buttresses, different IFC geometry types, terrain, and real-world scenarios (ASTRA-compliant, highway+terrain, multi-failure)
- **340 automated tests** across 10 test suites

## Testing

```bash
pytest                                    # Run all 340 tests
python validate_all_models.py             # Batch validate all 26 models
python sensitivity_analysis.py            # Threshold sensitivity sweep
python generate_thesis_figures.py         # Generate 3D classification figures
```

## Context

Bachelor thesis at BFH (Berner Fachhochschule), Civil Engineering, FS 2026.

**Topic:** Geometrische Interpretation und Validierung von digitalen Bauwerksmodellen am Beispiel von Stützmauern nach ASTRA-Fachhandbuch

## License

MIT
