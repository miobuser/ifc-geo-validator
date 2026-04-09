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

# Projekt-Metadaten (für Prüfprotokoll)
project: ""
author: ""

# Element-Filter
filter_type:
  - IfcWall
  - IfcSlab
  - IfcFooting

# Validierungslevel (1-7)
levels: [1, 2, 3, 4, 5, 6, 7]

# Automatische Konfiguration
auto: true

# Paarweise Distanzen berechnen
distances: false

# Ausgabe-Dateien (leer = nicht erzeugen)
# {filename} wird durch den IFC-Dateinamen ersetzt
output:
  html: ""
  csv: ""
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
