# ifc-geo-validator

[![Tests](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml/badge.svg)](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml)

Geometric validation of IFC infrastructure models against configurable requirements.

## Overview

`ifc-geo-validator` analyses the geometry of IFC building elements (focused on retaining walls per ASTRA FHB T/G) and validates them against configurable YAML rulesets. It extracts triangulated meshes from IFC geometry via IfcOpenShell/OpenCASCADE, classifies faces by surface normal direction, computes geometric properties per face group, and checks them against normative requirements.

## Validation Levels

| Level | Description | Status |
|-------|-------------|--------|
| L1 | Geometric properties (volume, area, bbox, centroid) | Done |
| L2 | Face classification (crown, foundation, front, back, end faces) | Done |
| L3 | Face-specific values (crown width, inclination, wall thickness) | Done |
| L4 | Requirement comparison (YAML ruleset) | Done |
| L5 | Inter-component context (wall-foundation connection) | Optional |
| L6 | External context (terrain, alignment) | Optional |
| L7 | Distance calculations | Optional |

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
```

### Web App

```bash
pip install -e ".[web]"
streamlit run src/ifc_geo_validator/app.py
```

Upload an IFC file in the browser, configure filters, and view interactive validation results with downloadable JSON reports.

## Rulesets

Validation rules are defined in YAML:

```yaml
metadata:
  name: "ASTRA FHB T/G — Stützmauern"
  ifc_filter:
    entity: "IfcWall"
    predefined_type: "RETAININGWALL"

level_3:
  - id: ASTRA-SM-L3-001
    name: "Kronenbreite Minimum"
    check: "crown_width_mm >= 300"
    severity: ERROR
    reference: "FHB T/G 24 001-15101, Kap. 3"
```

## Testing

```bash
pytest
```

## Context

Bachelor thesis at BFH (Berner Fachhochschule), Civil Engineering, FS 2026.

**Topic:** Geometrische Interpretation und Validierung von digitalen Bauwerksmodellen am Beispiel von Stützmauern nach ASTRA-Fachhandbuch

## License

MIT
