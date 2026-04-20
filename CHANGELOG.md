# Changelog

All notable changes to ifc-geo-validator are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/), versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Proper open-source `LICENSE` (MIT) and this `CHANGELOG.md`.
- `plotly` and `xsdata` declared as optional extras in `pyproject.toml`.

### Fixed
- `.coverage` removed from git history; `.gitignore` hardened.
- Swiss-engineering IT/FR translations: `coronamento`, `rastremazione`,
  `infissione`, `ispettore/ispettrice`, `écart d'aplomb`, `voile` instead
  of the earlier ad-hoc equivalents.

## [2.0.0] - 2026-04-10

### Added
- Three.js-based 3D viewer (`viz/mesh_viewer.py`) with toolbar, section
  planes (X/Y/Z), two-click distance measurement, collapsible panels,
  click-to-zoom, smooth cubic-eased camera transitions.
- Trilingual UI (DE / FR / IT): ~120 i18n keys spanning Streamlit
  surface and viewer.
- IFC coordinate-reference declaration: `get_coordinate_system()` reads
  `IfcProjectedCRS` + `IfcMapConversion` and surfaces the frame in
  every report (LV95 / LN02 for Swiss infrastructure).
- `docs/references.md` — 25+ primary citations grouped by topic.
- Full `.igv.yaml` schema for reproducibility: `classifier`,
  `pair_candidacy`, `robust_stats`, `anomaly` sections.
- `validation/level5.validate_level5(config=…)` and
  `anomaly_detection.detect_anomalies(config=…)` honour `.igv.yaml`
  overrides.
- Rich test coverage for recent contracts
  (`tests/test_market_readiness.py`, 11 tests).

### Changed
- Crown/foundation width: representative value switched from raw `min()`
  to p10 robust quantile (Tukey 1977 lower hinge) — rejects sliver
  artifacts at curved-wall endpoints. `width_min_mm`, `width_p10_mm`,
  `width_median_mm` all retained for traceability.
- Face adjacency builder emits O(k) star-graph pairs instead of O(k²)
  for non-manifold edges — memory-safe on CSG-authored inputs.
- `validate_level4` kwarg order: `level2_result` moved before
  `level5_context` / `level6_context` so it follows level numbering.
- All module-level hardcoded thresholds promoted to named constants
  with inline derivation (SIA 262, ASTRA FHB T/G citations, Farin 2002,
  Welzl 1991, etc.).
- Dockerfile aligned with Streamlit deployment (port 8501) and
  `wget`-based healthcheck.

### Fixed
- **Security (XSS)**: `esc()` HTML-escape helper now covers every
  `innerHTML` write in the viewer. IFC element names with `<script>`
  or quote-breakouts are neutralised.
- **Security (JSON payload)**: `</script>`, `<!--`, `-->` are escaped
  server-side so a malicious IFC name cannot terminate the inline
  `<script type="module">` tag.
- **Logic**: bounding-box horizontal distance switched from Manhattan
  to Euclidean (`math.hypot`) — matches Arvo & Kirk 1989 AABB convention
  and the sister function in `core.distance.horizontal_distance_xy`.
- **Logic**: outward-normal convention now verified via signed-volume
  sign check and, if negative, the mesh winding + normals are flipped
  once at extraction time — prevents silent crown↔foundation swap on
  CSG-authored IFCs with inverted winding.
- **Performance**: viewer hover raycaster throttled to one pick per
  animation frame (was 60/s on mousemove). Pixel-ratio capped at 2.
  WebGL resources disposed on iframe unload.
- **Robustness**: `IFCLoadError` / `MeshExtractionError` caught at
  Streamlit boundary with localised error messages instead of raw
  traceback.
- **Docs**: README "L7" phantom row removed (distance checks are part
  of L6). Dockerfile port mismatch (8080 vs 8501) resolved.

### Removed
- `viz/webifc_viewer.py` — orphaned module superseded by `mesh_viewer`.
- `_json_default` shadow wrapper in `report/json_report.py`.

## [1.x]

Prior thesis-draft iterations (see `git log` for detail). First public
release is 2.0.0.
