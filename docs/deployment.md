# Deployment

`ifc-geo-validator` kann auf drei Arten betrieben werden.

## 1. Streamlit Community Cloud (hosted)

Live unter https://ifc-geo-validator2.streamlit.app. Keine Installation,
keine Wartung. Uploads werden nicht dauerhaft gespeichert (siehe
[privacy.md](privacy.md) bzw. den Commit-Verlauf).

**Grenzen:** ein Container mit ~2 GB RAM, keine Batch-Jobs, öffentlich.
Für Kundenprojekte mit vertraulichen IFC-Daten → Option 2 oder 3.

## 2. Docker / Docker Compose (self-hosted)

```bash
git clone https://github.com/miobuser/ifc-geo-validator.git
cd ifc-geo-validator
docker compose up -d
```

Standardport: **8501**.

### Ports und Volumes

Der `docker-compose.yml` Standard:

```yaml
services:
  validator:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
```

Eigener Host-Port (z. B. wegen Konflikt mit einer anderen Streamlit-App):

```yaml
ports:
  - "9501:8501"    # Host 9501 → Container 8501
```

Persistente Custom-Rulesets durchreichen:

```yaml
volumes:
  - ./my_rulesets:/app/custom_rulesets:ro
```

### Healthcheck

Der Container liefert `GET /_stcore/health`. Bereits in `Dockerfile`
konfiguriert (30s-Intervall, 3 Retries). Für eigene Orchestrierung:

```bash
curl -f http://localhost:8501/_stcore/health
```

### Reverse-Proxy (nginx)

```nginx
location / {
    proxy_pass         http://localhost:8501;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host $host;
    proxy_read_timeout 86400;
}
```

Der letzte Wert ist wichtig: Streamlit hält WebSocket-Verbindungen
lange offen, sonst reisst die Session nach 60s Default ab.

### TLS

Hinter einem Reverse-Proxy mit certbot / acme.sh:

```nginx
listen 443 ssl http2;
ssl_certificate     /etc/letsencrypt/live/example.ch/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/example.ch/privkey.pem;
```

## 3. Python-Paket (Bibliothek / Skripte)

```bash
pip install -e ".[viz,bcf,web]"
ifc-geo-validator --help
```

Eintrittspunkte:
- `ifc-geo-validator` — CLI
- `streamlit run streamlit_app.py` — Web-UI
- `from ifc_geo_validator import get_version` — Python-API

### Python-Version

- **Erfordert:** Python 3.10, 3.11 oder 3.12
- **Getestet auf:** macOS arm64, Ubuntu 24.04, Windows 11
- **Nicht getestet:** Python 3.13+ (sollte laufen, ifcopenshell-Wheels
  checken)

## Umgebungsvariablen

| Variable | Zweck | Default |
|---|---|---|
| `STREAMLIT_SERVER_PORT` | Port überschreiben | 8501 |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | Max IFC-Größe (MB) | 1000 |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | Telemetrie deaktivieren | false |

Alle anderen Streamlit-Variablen funktionieren auch, siehe
https://docs.streamlit.io/develop/api-reference/configuration.

## System-Abhängigkeiten (Linux self-host)

Installiere auf Debian/Ubuntu vor `pip install`:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

**Hinweis:** Auf Ubuntu 24.04 (`noble`) heißt das Paket
`libglib2.0-0t64` statt `libglib2.0-0`. `packages.txt` im Repo nutzt
die `t64`-Variante für Streamlit Cloud (Debian trixie-Basis). Der
`Dockerfile` nutzt die bookworm-Variante ohne `t64`.

## Ressourcenbedarf (Faustregeln)

| Modell-Grösse | Elemente | Speicher | Laufzeit |
|---|---|---|---|
| Kleines Projekt | 1–10 Wände | 500 MB | <5 s |
| Typisches Projekt | 10–100 Wände | 1 GB | 10–60 s |
| Grosses Infra-Los | 500+ Wände | 2–4 GB | 5–10 min |

Für >500 Elemente empfiehlt sich Batch-Verarbeitung per CLI
(`ifc-geo-validator ./ifcs/ --auto --summary`) statt Web-UI.
