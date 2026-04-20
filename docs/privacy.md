# Datenschutz / Privacy

Dieses Dokument beschreibt, wie `ifc-geo-validator` mit den hochgeladenen
IFC-Dateien umgeht.

## Zusammenfassung

- **Keine persistente Speicherung.** Alle Verarbeitung erfolgt im
  Arbeitsspeicher der Server-Instanz.
- **Temporäre Dateien** werden ausschliesslich lokal auf dem
  Betriebssystem-Temp-Verzeichnis angelegt und nach Abschluss der
  Validierung (bzw. im `finally`-Pfad bei Fehler) **sofort gelöscht**.
- **Streamlit-Cache** mit TTL 10 Minuten. Nach Ablauf wird der
  Cache-Eintrag vom Runtime evakuiert.
- **Kein Download/Upload ausserhalb der Session.** Das Tool ruft keine
  externen Analytics-, Telemetry- oder Logging-Endpunkte auf. Die
  Streamlit-Nutzungsstatistik (`gatherUsageStats`) ist deaktiviert.
- **Drittanbieter-CDN:** Der 3D-Viewer lädt Three.js-Module von
  `esm.sh`. Nur Code fliesst vom CDN zum Client; IFC-Daten verlassen
  niemals den Server in Richtung Drittanbieter.
- **Berichte (JSON/HTML/BCF/CSV/Enriched-IFC):** werden im Browser
  erzeugt und direkt als Download ausgeliefert. Keine Zwischenablage
  auf Server-Speicher nach dem Download.

## Gesetzliche Grundlage

- **Schweizer DSG (Datenschutzgesetz, revidiert 2023)**: Art. 6
  (Datensparsamkeit) ist eingehalten, weil keine personenbezogenen
  Daten verarbeitet werden.
- **EU DSGVO / GDPR**: IFC-Dateien von Bauwerken enthalten in der
  Regel keine personenbezogenen Daten. Sollten Elemente mit
  Personennamen (z. B. Architekten im `IfcOwnerHistory`) enthalten
  sein, werden diese **nicht** ausgewertet, nicht persistiert und
  nicht an Dritte weitergegeben.

## Technische Kontrollen (für Security-Audits)

| Kontrolle | Implementierung |
|---|---|
| Tempfile-Garbage-Collection | `try/finally: os.unlink(...)` in `app.py::run_validation` und beiden Download-Pfaden |
| Cache-TTL | `@st.cache_data(ttl=600)` |
| Keine Pfad-Leaks | `report/json_report.py` speichert nur den Basename, nie den absoluten Tempfile-Pfad |
| Kein externer Traffic | Keine `requests.*` / `urllib` Calls im `src/ifc_geo_validator/`; nur `esm.sh` als Client-Side-CDN für Three.js |
| Upload-Grösse | `maxUploadSize = 500` MB (DoS-Mitigation) |

## Self-hosted Deployment

Bei eigenem Docker-Deployment (siehe [deployment.md](deployment.md))
gilt derselbe In-Memory-Ansatz — der Container schreibt nichts in
gemountete Volumes ausser wenn der Betreiber explizit ein
`--volume` hinzufügt.

## Vulnerability Disclosure

Für Sicherheitsmeldungen siehe [../SECURITY.md](../SECURITY.md).
