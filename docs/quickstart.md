# Quickstart — ifc-geo-validator

Von Null auf erste Validierung in 5 Minuten.

## Variante A: Web-App (kein Setup)

Öffne https://ifc-geo-validator2.streamlit.app, klicke **Demo-Modell
laden** oder lade eine eigene IFC-Datei per Drag & Drop in die
Seitenleiste. Resultate erscheinen automatisch.

## Variante B: Lokal per CLI

```bash
git clone https://github.com/miobuser/ifc-geo-validator.git
cd ifc-geo-validator
pip install -e ".[web,bcf,viz]"
ifc-geo-validator tests/test_models/T28_showcase.ifc --auto
```

Erwartete Ausgabe (gekürzt):

```
IFC Geometry Validator v2.0.0
File: T28_showcase.ifc
CRS: EPSG:2056 / LN02
Auto-Config: Infrastrukturbauwerk mit Wänden
  Types:    IfcWall
  Ruleset:  astra_fhb_komplett.yaml
...
ASTRA-SM-L3-001  Kronenbreite >= 300mm           PASS  301.2
ASTRA-SM-L3-002  Kronenneigung 2-4%              PASS  2.8
...
Summary: 28 PASS, 0 FAIL, 2 SKIP
```

Exit-Code **0** = alle Regeln bestanden.
Exit-Code **1** = mindestens ein `ERROR`-Check hat `FAIL`.

## Variante C: Docker self-host

```bash
docker compose up -d
open http://localhost:8501
```

Siehe [deployment.md](deployment.md) für vollständige Docker- und
Reverse-Proxy-Setups.

## Weiterführend

- [variable_reference.md](variable_reference.md) — jeder verfügbare
  YAML-Prüfvariablen-Name mit Einheit und Beschreibung
- [references.md](references.md) — Bibliographie aller im Code
  zitierten wissenschaftlichen Quellen
- [deployment.md](deployment.md) — Docker, Ports, Volumes, Reverse-Proxy
- `../src/ifc_geo_validator/rules/rulesets/*.yaml` — vier Beispiel-Rulesets
  (ASTRA FHB T/G, SIA 262) zum Anpassen
