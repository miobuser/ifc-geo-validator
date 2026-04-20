# ifc-geo-validator

[![Tests](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml/badge.svg)](https://github.com/miobuser/ifc-geo-validator/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Live App](https://img.shields.io/badge/app-streamlit.app-FF4B4B.svg)](https://ifc-geo-validator2.streamlit.app)

Geometric validation of IFC infrastructure models against configurable requirements.

**➡️ Try the live app:** https://ifc-geo-validator2.streamlit.app (click "Demo-Modell laden" for a sample run).

**New here?** Start with [docs/quickstart.md](docs/quickstart.md) — zero-to-first-report in 5 minutes.

## Overview

`ifc-geo-validator` analyses the geometry of IFC building elements and validates them against configurable YAML rulesets. It extracts triangulated meshes via IfcOpenShell/OpenCASCADE, classifies faces by surface normal direction, computes geometric properties, and checks them against normative requirements.

Designed for retaining walls per ASTRA FHB T/G, extensible to tunnel components, bridge elements, and structural members via custom rulesets. Smart filtering automatically skips wall-specific rules for non-wall elements.

## Validation Levels

| Level | Description |
|-------|-------------|
| L1 | Geometric properties (volume, area, bbox, watertight, mesh quality diagnostics) |
| L2 | Face classification (crown, foundation, front, back, ends) with confidence score |
| L3 | Face-specific measurements (crown width, slope, thickness, inclination, height) |
| L4 | YAML rule evaluation with actionable FAIL messages and smart filtering |
| L5 | Inter-element context (Contact Surface Normal Analysis, stacking, gaps) |
| L6 | Terrain context (gradient-based earth/air side, clearance, embedment, inter-element distances) |

## Installation

```bash
pip install -e ".[dev,viz,bcf,web]"
```

## Usage

### CLI

```bash
# Standard validation (auto-detects ASTRA ruleset)
ifc-geo-validator model.ifc

# Custom ruleset
ifc-geo-validator model.ifc -r rulesets/sia_262_stuetzmauer.yaml

# Multiple entity types
ifc-geo-validator model.ifc --filter-type IfcWall,IfcSlab,IfcFooting

# Output formats
ifc-geo-validator model.ifc -o report.json          # JSON report
ifc-geo-validator model.ifc --html report.html       # HTML report (self-contained)
ifc-geo-validator model.ifc --enrich validated.ifc   # IFC with Pset_GeoValidation
ifc-geo-validator model.ifc --bcf issues.bcf         # BCF 2.1 issues

# Visualizations
ifc-geo-validator model.ifc --heatmap cross          # Slope heatmap (Quergefälle)
ifc-geo-validator model.ifc --heatmap long           # Longitudinal slope
ifc-geo-validator model.ifc --cross-section 0.5      # Cross-section at 50%

# Discovery & batch
ifc-geo-validator model.ifc --scan                   # Scan entity types
ifc-geo-validator *.ifc                              # Batch: multiple files
ifc-geo-validator project_dir/                       # Batch: directory

# CI/CD integration
ifc-geo-validator model.ifc --summary                # Machine-readable (exit 1 on fail)
```

### Web App

```bash
streamlit run src/ifc_geo_validator/app.py
```

Multi-entity selector, built-in ruleset picker (ASTRA/SIA/Tunnel), slope analysis, mesh quality warnings, 3 download formats.

### Browser Viewer

```bash
python viewer/export_for_viewer.py model.ifc
python viewer/server.py
```

Interactive 3D viewer with face classification, slope heatmap toggle (Quergefälle/Längsgefälle), centerline rendering, and measurement annotations.

## Rulesets

| Ruleset | Standard | Rules | Key Checks |
|---------|----------|-------|------------|
| `astra_fhb_stuetzmauer.yaml` | ASTRA FHB T/G 24 001-15101 | 18 | Crown >= 300mm, slope 3%, thickness >= 300mm, 10:1 inclination, cross-slope |
| `sia_262_stuetzmauer.yaml` | SIA 262 / SIA 267 | 7 | Crown >= 200mm, thickness >= 200mm |
| `astra_fhb_tunnel.yaml` | ASTRA FHB T/G 24 001-10xxx | 9 | Volume, mesh, thickness >= 200mm, cross-slope <= 5%, longitudinal <= 5% |

Custom rulesets can define rules for any computed variable (see [YAML rule syntax](#yaml-rule-syntax)).

### YAML Rule Syntax

```yaml
level_3:
  - id: ASTRA-SM-L3-001
    name: "Kronenbreite Minimum"
    check: "crown_width_mm >= 300"
    severity: ERROR
    reference: "FHB T/G 24 001-15101, Kap. 3"
    quote: "Die Mauerkrone ist mindestens 300 mm breit auszubilden."
```

Available variables: `volume`, `total_area`, `mesh_is_watertight`, `crown_width_mm`, `crown_slope_percent`, `min_wall_thickness_mm`, `front_inclination_ratio`, `foundation_width_mm`, `wall_height_m`, `is_curved`, `crown_width_cv`, `cross_slope_max_pct`, `long_slope_max_pct`, `bbox_dim_min_m`, `bbox_height_m`, `foundation_embedment_m`, `earth_side_determined`, and more.

## Key Features

- **Slope heatmap**: per-triangle Quergefälle/Längsgefälle with local centerline frames (correct on curves)
- **Dual centerline**: geometric (from crown faces) or IfcAlignment (when available)
- **Confidence score**: 0-100% classification reliability with geometry validation gate
- **Smart rule filtering**: wall-specific rules auto-skipped for non-wall elements (slabs, foundations)
- **Actionable FAIL messages**: actual value, requirement, source quote from ASTRA FHB
- **Terrain gradient classification**: earth/air side via terrain slope direction (robust for retaining walls)
- **Cross-section visualization**: 2D profile at any position along the wall with dimension annotations
- **HTML report**: self-contained, printable, archivable — professional layout
- **Batch mode**: validate multiple files or entire directories
- **Multi-entity types**: `--filter-type IfcWall,IfcSlab,IfcFooting` in a single run
- **Curved wall support**: local coordinate frames, slice-based measurements, curvature detection
- **No heuristics**: all algorithms mathematically derived (cos(pi/4), 3-sigma, CDR, natural-gap)
- **Robustness**: degenerate triangle filtering, multi-body detection, scale-invariant (mm to LV95)
- **7 output formats**: CLI (colored), JSON, HTML, IFC (Pset_GeoValidation), BCF 2.1, Streamlit, 3D viewer
- **385 automated tests** across 16 test suites
- **27 test models** (T1-T28) including showcase with all levels active (16/18 PASS)
- **Tested on real IFC models**: bridge (46 elements), building (14), tunnel (2.5M triangles)

## Testing

```bash
pytest                                    # Run all 385 tests
python validate_all_models.py             # Batch validate all 27 models
python sensitivity_analysis.py            # Threshold sensitivity sweep
python generate_thesis_figures.py         # Generate 3D classification figures
```

## Context

Bachelor thesis at BFH (Berner Fachhochschule), Civil Engineering, FS 2026.

**Topic:** Geometrische Interpretation und Validierung von digitalen Bauwerksmodellen am Beispiel von Stützmauern nach ASTRA-Fachhandbuch

## License

MIT
