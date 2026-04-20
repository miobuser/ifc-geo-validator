"""Project configuration file support.

Allows saving validation settings as a .igv.yaml file in the project
directory, so users don't need to specify flags every time.

Example .igv.yaml:
    project: "A1 Bern-Zürich, Los 3"
    author: "M. Buser, B+S AG"
    filter_type: ["IfcWall", "IfcSlab", "IfcFooting"]
    ruleset: "astra_fhb_komplett.yaml"
    levels: [1, 2, 3, 4, 5, 6, 7]
    distances: true
    output:
      html: "reports/{filename}_report.html"
      csv: "reports/{filename}_measurements.csv"

Usage:
    # Create default config
    ifc-geo-validator --init

    # Use config (auto-detected in current directory)
    ifc-geo-validator model.ifc
"""

import os
import yaml
from pathlib import Path


CONFIG_FILENAME = ".igv.yaml"

# All reproducibility-relevant tolerances are surfaced here so a thesis
# reviewer can re-run any validation with different numerical settings
# without editing source. Defaults match the values discussed in the
# thesis; overrides are merged shallow-per-key in load_config.
DEFAULT_CONFIG = {
    "project": "",
    "author": "",
    "filter_type": ["IfcWall"],
    "levels": [1, 2, 3, 4, 5, 6, 7],
    "auto": True,
    "distances": False,
    "output": {
        "html": "",
        "csv": "",
    },
    # Face-classification tolerances (L2). Each value is an angle in
    # degrees or a dimensionless ratio; see core/face_classifier.py
    # DEFAULT_THRESHOLDS for the in-code derivation (Farin 2002,
    # IfcOpenShell tessellation analysis).
    "classifier": {
        "horizontal_deg": 45.0,    # π/4: geometric equator between horizontal and vertical
        "coplanar_deg": 5.0,       # ≈ 180°/(2·18): conservative vs 10° tessellation jitter
        "lateral_deg": 45.0,       # same equator for lateral face separation
    },
    # Inter-element analysis tolerances (L5).
    "pair_candidacy": {
        "max_gap_3d_m": 1.0,       # reject element pairs > 1 m apart in 3D
        "min_z_overlap_ratio": 0.5,  # required overlap of Z-extents for stacked pairs
    },
    # Robust-percentile choices for L3 width/thickness measurements.
    "robust_stats": {
        "width_percentile": 10,    # p10 lower quantile for crown/foundation width
        "thickness_low_percentile": 10,   # front-face lower decile
        "thickness_high_percentile": 90,  # back-face upper decile
    },
    # Anomaly-detection cutoffs (see core/anomaly_detection.py).
    "anomaly": {
        "front_back_ratio_flag": 2.0,
        "aspect_ratio_slender_flag": 50.0,
        "crown_narrower_than_thickness_factor": 0.5,
    },
}


def find_config(start_dir: str = ".") -> str | None:
    """Search for .igv.yaml in current and parent directories."""
    current = Path(start_dir).resolve()
    for _ in range(5):  # max 5 levels up
        config_path = current / CONFIG_FILENAME
        if config_path.exists():
            return str(config_path)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_config(path: str) -> dict:
    """Load and validate a project config file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Merge with defaults
    result = {**DEFAULT_CONFIG}
    for key in config:
        if key in result:
            result[key] = config[key]
        else:
            result[key] = config[key]  # allow custom keys

    return result


def create_default_config(directory: str = ".") -> str:
    """Create a default .igv.yaml in the given directory."""
    path = os.path.join(directory, CONFIG_FILENAME)
    content = """# ifc-geo-validator Projektkonfiguration
# Wird automatisch geladen wenn im Projektverzeichnis vorhanden.

# ── Projekt-Metadaten (für Prüfprotokoll) ─────────────────────────
project: ""
author: ""

# ── Element-Filter ────────────────────────────────────────────────
filter_type:
  - IfcWall
  - IfcSlab
  - IfcFooting

# Validierungslevel (1-6)
levels: [1, 2, 3, 4, 5, 6]

# Automatische Konfiguration (Entity-Typen, Ruleset, Terrain)
auto: true

# Paarweise Distanzen berechnen
distances: false

# ── Ausgabe-Dateien ───────────────────────────────────────────────
# {filename} wird durch den IFC-Dateinamen ersetzt
output:
  html: ""
  csv: ""

# ── Numerische Toleranzen (für Reproduzierbarkeit) ────────────────
# Alle Default-Werte entsprechen den in der Thesis dokumentierten
# Ableitungen. Anpassen nur mit fachlicher Begründung.

classifier:
  horizontal_deg: 45.0      # Äquator horizontal↔vertikal (π/4)
  coplanar_deg: 5.0         # koplanar-Schwelle (Farin 2002)
  lateral_deg: 45.0         # lateral-Äquator

pair_candidacy:
  max_gap_3d_m: 1.0         # Element-Paare bis zu diesem 3D-Abstand prüfen
  min_z_overlap_ratio: 0.5  # erforderliche Z-Überlappung für Stapel-Paare

robust_stats:
  width_percentile: 10                # p10 für Kronenbreite/Fundamentbreite
  thickness_low_percentile: 10        # Front-Fläche untere Dezile
  thickness_high_percentile: 90       # Back-Fläche obere Dezile

anomaly:
  front_back_ratio_flag: 2.0                    # ASTRA Anzug ≤ 1:10 → max ~1.6
  aspect_ratio_slender_flag: 50.0               # Shell-Artefakte abfangen
  crown_narrower_than_thickness_factor: 0.5     # physikalisch instabil
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
