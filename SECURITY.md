# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅ Aktiv  |
| 1.x     | ❌ End-of-life — bitte auf 2.0 aktualisieren |

## Reporting a Vulnerability

Wenn du eine Sicherheitslücke findest, melde sie **nicht** über ein
öffentliches GitHub-Issue. Stattdessen:

1. **E-Mail** an `mio.buser@baghira.ch` mit Betreff-Präfix
   `[SECURITY] ifc-geo-validator`.
2. Füge eine reproduzierbare Schritt-für-Schritt-Anleitung bei (IFC-
   oder YAML-Ruleset-Muster als Anhang falls zutreffend).
3. Nenne dein Zeitfenster für eine verantwortungsvolle Offenlegung
   (Default: 90 Tage nach Erstmeldung).

## Scope

Im Scope der Vulnerability Disclosure:

- Remote Code Execution über Ruleset- oder IFC-Upload
- Cross-Site Scripting im 3D-Viewer
- Path-Traversal / File-Inclusion
- Denial-of-Service durch ressourcenlastige Payloads
- Leaks personenbezogener oder vertraulicher Daten

**Out of scope** (keine CVE):

- UI-Bugs ohne Sicherheitsimpact
- Rate-Limiting (wird auf Streamlit-Cloud-Level gelöst)
- Fehlkonfigurationen im Self-Hosted-Deployment (siehe `docs/deployment.md`)

## Track Record

Historische Sicherheits-Fixes sind im `CHANGELOG.md` unter der jeweiligen
Version dokumentiert. In v2.0.0 wurden u. a. behoben:

- RCE über `eval()`-Sandbox-Escape im Ruleset-Evaluator
- XSS über IFC-Element-Namen im 3D-Viewer
- CSV-Formula-Injection beim Export
- Temp-File-Leaks, die den Privacy-Claim verletzten
- Streamlit-Pin angehoben auf ≥1.37 (CVE-2024-42474)

## PGP

Kein PGP-Key bereitgestellt. Verschlüsselung nach Bedarf auf Anfrage.
