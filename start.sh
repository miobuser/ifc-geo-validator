#!/usr/bin/env bash
# IFC Geometry Validator — one-shot launcher for macOS / Linux.
# Windows users: double-click start.bat instead.

set -euo pipefail

echo "============================================================"
echo "  IFC Geometry Validator v2.0.0"
echo "  Geometrische Validierung von IFC-Infrastrukturmodellen"
echo "============================================================"
echo

# Python check
if ! command -v python3 >/dev/null 2>&1; then
    echo "FEHLER: Python 3 ist nicht installiert."
    echo "Installation: https://www.python.org  oder  brew install python@3.12"
    exit 1
fi

# First-run install
if ! python3 -c "import ifc_geo_validator" >/dev/null 2>&1; then
    echo "Erste Ausführung: Installiere Abhängigkeiten..."
    echo "Dies dauert 2-5 Minuten."
    echo
    python3 -m pip install -e ".[dev,bcf,web,viz]" -q
    echo "Installation abgeschlossen."
    echo
fi

echo "Starte Web-App..."
echo
echo "Die App öffnet sich im Browser unter:"
echo "  http://localhost:8501"
echo
echo "Zum Beenden: Ctrl+C in diesem Fenster"
echo

streamlit run streamlit_app.py --server.headless false
