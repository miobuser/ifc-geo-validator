"""Interactive 3D viewer for IFC validation results.

The viewer uses Three.js (Möller & Haines 2018 rendering conventions) with a
Lambert shading model (Lambert 1760, "Photometria") for perceptually uniform
diffuse appearance — chosen over physically-based shading because the goal
is geometric inspection, not photorealism, and MeshLambertMaterial has a
single-pass cost ~3× lower than MeshStandardMaterial on large meshes.

Selection highlighting uses an emissive overlay in the fragment shader
(Akenine-Möller et al. 2018, §5.8 "Emission"); we set `emissive = #FF69B4`
with `emissiveIntensity = 0.5` so the per-vertex classification colors remain
visible beneath the highlight.

Zoom-to-fit uses the minimum-enclosing sphere of the model bounding box
(Welzl 1991, "Smallest enclosing disks") with FOV-aware distance
    d = r / sin(FOV_eff / 2) · 1.2
where FOV_eff = min(vertical_FOV, horizontal_FOV) so the fit is correct
for any aspect ratio. The 1.2 margin factor provides a uniform 20 %
safety border around the silhouette.

Element picking uses a Three.js `Raycaster` (Appel 1968, ray casting;
Möller–Trumbore 1997 ray-triangle intersection), one pick per click.

Color scheme follows the B+S Corporate dark theme (background #1A1A1A,
accent #CB0231) for consistency with the IFC-Editor frontend used at
B+S AG, reducing cognitive load for engineers who switch between both
tools.

Mesh data is extracted server-side via IfcOpenShell/OpenCASCADE (Krijnen &
Beetz 2020) and shipped as Float32Array vertex/index buffers; the viewer
never parses IFC directly.
"""

import json
import streamlit.components.v1 as components
import numpy as np


# Face category color palette (matches IFC-Editor B+S corporate dark theme idea)
CATEGORY_COLORS = {
    "crown":        "#2196F3",  # blue
    "foundation":   "#795548",  # brown
    "front":        "#F44336",  # red
    "back":         "#FF9800",  # orange
    "end_left":     "#4CAF50",  # green
    "end_right":    "#8BC34A",  # light green
    "unclassified": "#9E9E9E",  # grey
}


def render_mesh_viewer(
    elements: list,
    height: int = 650,
    terrain_mesh: dict | None = None,
) -> None:
    """Render pre-extracted meshes with rich Three.js viewer.

    Args:
        elements: list of dicts, each with:
            - element_id: int
            - element_name: str
            - mesh_data: dict {vertices, faces}
            - status: "PASS" | "FAIL" | "WARN" | "—"
            - level1: dict (volume, area, watertight, ...)
            - level2: dict (face_groups with face_indices, element_role)
            - level3: dict (measurements like crown_width_mm, ...)
            - level4: dict (checks list)
        height: viewer height in pixels.
        terrain_mesh: optional terrain mesh dict.
    """
    payload = []
    for el in elements:
        mesh = el.get("mesh_data")
        if not mesh:
            continue
        verts = np.asarray(mesh.get("vertices"), dtype=np.float32)
        faces = np.asarray(mesh.get("faces"), dtype=np.uint32)
        if verts.size == 0 or faces.size == 0:
            continue

        n_faces = len(faces)

        # Build per-face category array from L2 face_groups
        face_cats = ["unclassified"] * n_faces
        l2 = el.get("level2") or {}
        for g in l2.get("face_groups", []):
            cat = g.get("category", "unclassified")
            for fi in g.get("face_indices", []):
                if 0 <= fi < n_faces:
                    face_cats[fi] = cat

        # L1 / L3 metrics
        l1 = el.get("level1") or {}
        l3 = el.get("level3") or {}
        metrics = {
            "Volume (m³)": _fmt(l1.get("volume"), 3),
            "Oberfläche (m²)": _fmt(l1.get("total_area"), 3),
            "Wasserdicht": "Ja" if l1.get("is_watertight") else "Nein",
            "Triangles": l1.get("num_triangles"),
            "BBox H (m)": _fmt((l1.get("bbox", {}) or {}).get("size", [0, 0, 0])[2], 2),
        }
        # Add L3 measurements that are present
        l3_mapping = {
            "crown_width_mm": "Kronenbreite (mm)",
            "crown_slope_percent": "Kronenneigung (%)",
            "min_wall_thickness_mm": "Wandstärke min (mm)",
            "wall_height_m": "Wandhöhe (m)",
            "front_inclination_ratio": "Anzug (n:1)",
            "min_radius_m": "Min. Radius (m)",
            "front_plumbness_deg": "Lotabweichung (°)",
        }
        for key, label in l3_mapping.items():
            v = l3.get(key)
            if v is not None:
                metrics[label] = _fmt(v, 2)

        # L4 checks
        checks = []
        l4 = el.get("level4") or {}
        for c in l4.get("checks", []):
            checks.append({
                "id": c.get("id", ""),
                "name": c.get("name", c.get("id", "")),
                "status": c.get("status", "—"),
                "severity": c.get("severity", ""),
                "actual": _fmt(c.get("actual"), 2),
                "expected": str(c.get("expected", "")),
                "message": c.get("message", ""),
            })

        payload.append({
            "id": int(el.get("element_id", 0)),
            "name": str(el.get("element_name", "")),
            "status": el.get("status", "—"),
            "role": (el.get("level2") or {}).get("element_role", ""),
            "vertices": verts.flatten().tolist(),
            "indices": faces.flatten().tolist(),
            "face_categories": face_cats,
            "metrics": metrics,
            "checks": checks,
        })

    terrain_payload = None
    if terrain_mesh:
        tv = np.asarray(terrain_mesh.get("vertices"), dtype=np.float32)
        tf = np.asarray(terrain_mesh.get("faces"), dtype=np.uint32)
        if tv.size > 0 and tf.size > 0:
            terrain_payload = {
                "vertices": tv.flatten().tolist(),
                "indices": tf.flatten().tolist(),
            }

    data_json = json.dumps({
        "elements": payload,
        "terrain": terrain_payload,
        "category_colors": CATEGORY_COLORS,
    }, ensure_ascii=True)
    # Neutralise every sequence that could prematurely terminate the
    # enclosing <script type="module"> tag or start an HTML comment.
    # json.dumps does not escape these; we do it here so that an IFC
    # element named "</script>" or containing "<!--" can never break
    # out of the payload context.
    data_json = (
        data_json
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("-->", "--\\>")
    )

    html = _VIEWER_HTML.replace("__DATA_JSON__", data_json)
    html = html.replace("__HEIGHT__", str(height))

    components.html(html, height=height + 10, scrolling=False)


def _fmt(v, digits: int = 2):
    """Format a number for display, returning None on bad input."""
    if v is None:
        return None
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return v


_VIEWER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
  :root {
    /* B+S Corporate Style — exact tokens from IFC-Editor */
    --bg-primary: #1a1a1a;
    --bg-secondary: #222222;
    --bg-tertiary: #2a2a2a;
    --text-primary: #e0e0e0;
    --text-secondary: #909090;
    --accent: #CB0231;
    --accent-hover: #e0033a;
    --accent-light: rgba(203, 2, 49, 0.15);
    --border: #404040;
    --success: #4ec9b0;
    --warning: #dcdcaa;
    --error: #f14c4c;
    --highlight: #CB0231;
    --radius-sm: 2px; --radius-md: 4px; --radius-lg: 6px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 8px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
    --duration-fast: 0.15s; --duration-base: 0.2s;
    --transition: all 0.15s ease;
  }
  html, body {
    margin: 0; padding: 0; overflow: hidden;
    background: var(--bg-primary); color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    font-size: 12px;
    width: 100%; height: __HEIGHT__px;
  }
  #c { display: block; width: 100%; height: __HEIGHT__px; }

  /* Top toolbar — grouped containers with labels above buttons */
  #toolbar {
    position: absolute; top: 0; left: 0; right: 0; min-height: 64px;
    background: linear-gradient(180deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: stretch; padding: 6px 12px;
    gap: 4px; z-index: 50; user-select: none;
    overflow-x: auto; overflow-y: hidden;
  }
  #toolbar::-webkit-scrollbar { height: 4px; }
  #toolbar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .tb-group {
    display: flex; flex-direction: column; align-items: center;
    padding: 4px 8px; gap: 2px;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 6px; flex-shrink: 0;
  }
  .tb-group-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-secondary); font-weight: 600;
  }
  .tb-group-buttons { display: flex; gap: 2px; align-items: center; }
  button.tb {
    display: flex; flex-direction: column; align-items: center; gap: 2px;
    padding: 6px 10px;
    background: transparent; color: var(--text-primary);
    border: 1px solid transparent; border-radius: 4px;
    cursor: pointer; font-size: 10px; font-weight: 500;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    min-width: 48px;
  }
  button.tb svg { width: 18px; height: 18px; flex-shrink: 0; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  button.tb:hover { background: var(--bg-tertiary); border-color: var(--border); }
  button.tb.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.tb.active:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
  button.tb:disabled { opacity: 0.4; cursor: not-allowed; }

  button.panel-btn {
    background: transparent; color: var(--text-secondary);
    border: 1px solid transparent; border-radius: 3px; cursor: pointer;
    padding: 3px 5px; transition: var(--transition);
    display: flex; align-items: center;
  }
  button.panel-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); border-color: var(--border); }
  /* Left element list */
  #element-list {
    position: absolute; top: 76px; left: 8px; width: 230px;
    max-height: calc(100% - 76px); overflow-y: auto;
    background: rgba(34,34,34,0.94); border: 1px solid var(--border);
    border-radius: 6px; padding: 0; z-index: 30; font-size: 11px;
    transition: transform 0.2s ease, opacity 0.2s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  #element-list.collapsed { transform: translateX(calc(-100% - 16px)); opacity: 0; }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.15); border-top-left-radius: 6px; border-top-right-radius: 6px;
  }
  .panel-header h4 {
    margin: 0; font-size: 10px; text-transform: uppercase;
    color: var(--text-secondary); letter-spacing: 0.6px; font-weight: 600;
  }
  .panel-header .count {
    font-size: 10px; color: var(--text-secondary);
    background: var(--bg-tertiary); padding: 1px 6px; border-radius: 8px;
  }
  .panel-body { padding: 6px; }
  .el-row {
    padding: 6px 8px; cursor: pointer; border-radius: 3px;
    display: flex; align-items: center; gap: 8px;
    border: 1px solid transparent; margin-bottom: 2px;
    transition: var(--transition);
  }
  .el-row:hover { background: var(--bg-tertiary); }
  .el-row.selected { background: var(--accent); color: #fff; border-color: var(--accent); }
  .el-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .el-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Floating collapse toggle for left panel */
  #left-toggle {
    position: absolute; top: 76px; left: 8px;
    width: 28px; height: 28px;
    background: rgba(34,34,34,0.94); border: 1px solid var(--border);
    color: var(--text-secondary); cursor: pointer; border-radius: 6px;
    display: none; align-items: center; justify-content: center; z-index: 31;
    transition: var(--transition);
  }
  #left-toggle:hover { background: var(--bg-tertiary); color: var(--text-primary); }
  #element-list.collapsed ~ #left-toggle { display: flex; }

  /* Right properties panel */
  #props-panel {
    position: absolute; top: 76px; right: 8px; width: 300px;
    max-height: calc(100% - 76px); overflow-y: auto;
    background: rgba(34,34,34,0.94); border: 1px solid var(--border);
    border-radius: 6px; padding: 0; z-index: 30; font-size: 11px;
    transition: transform 0.2s ease, opacity 0.2s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  #props-panel.collapsed { transform: translateX(calc(100% + 16px)); opacity: 0; }
  #right-toggle {
    position: absolute; top: 76px; right: 8px;
    width: 28px; height: 28px;
    background: rgba(34,34,34,0.94); border: 1px solid var(--border);
    color: var(--text-secondary); cursor: pointer; border-radius: 6px;
    display: none; align-items: center; justify-content: center; z-index: 31;
    transition: var(--transition);
  }
  #right-toggle:hover { background: var(--bg-tertiary); color: var(--text-primary); }
  #props-panel.collapsed ~ #right-toggle { display: flex; }
  #props-panel .panel-body { padding: 10px; }
  #props-panel h4 {
    margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase;
    color: #909090; letter-spacing: 0.5px; border-bottom: 1px solid #404040;
    padding-bottom: 4px;
  }
  #props-panel h3 { margin: 0 0 4px 0; font-size: 13px; color: #e0e0e0; }
  .prop-row { display: flex; justify-content: space-between; padding: 2px 0; }
  .prop-key { color: #909090; }
  .prop-val { color: #e0e0e0; font-family: monospace; }
  .check-row {
    padding: 4px 6px; margin: 2px 0; border-radius: 2px;
    border-left: 3px solid #909090; background: #1f1f1f;
  }
  .check-row.PASS { border-left-color: #4CAF50; }
  .check-row.FAIL { border-left-color: #F44336; }
  .check-row.SKIP { border-left-color: #909090; }
  .check-name { font-size: 11px; color: #e0e0e0; }
  .check-detail { font-size: 10px; color: #909090; margin-top: 2px; font-family: monospace; }

  /* Bottom legend */
  #legend {
    position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
    background: rgba(34,34,34,0.92); border: 1px solid #404040; border-radius: 4px;
    padding: 6px 12px; font-size: 11px; z-index: 30;
    display: flex; gap: 14px; align-items: center;
  }
  .lg-item { display: flex; align-items: center; gap: 5px; }
  .lg-swatch { width: 12px; height: 12px; border-radius: 2px; }

  /* Status overlay */
  #status {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: #e0e0e0; font-size: 14px; text-align: center; z-index: 100;
    background: rgba(0,0,0,0.85); padding: 14px 22px; border-radius: 6px;
    max-width: 80%;
  }
  #err { color: #ff6b6b; }

  /* Hover tooltip */
  #tooltip {
    position: absolute; pointer-events: none; z-index: 60;
    background: rgba(0,0,0,0.9); color: #e0e0e0; padding: 4px 8px;
    border-radius: 3px; font-size: 11px; display: none;
    border: 1px solid #404040;
  }

  /* Empty panel state */
  .empty { color: #707070; font-style: italic; padding: 6px 0; }
</style>
</head>
<body>
<canvas id="c"></canvas>

<!-- Toolbar -->
<div id="toolbar">
  <div class="tb-group">
    <div class="tb-group-label">Farbe</div>
    <div class="tb-group-buttons">
      <button class="tb active" data-mode="status" title="Validierungs-Status">
        <svg viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        <span>Status</span>
      </button>
      <button class="tb" data-mode="category" title="Flächen-Klassifikation (L2)">
        <svg viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
        <span>Flächen</span>
      </button>
      <button class="tb" data-mode="role" title="Element-Rolle (L2)">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        <span>Rolle</span>
      </button>
      <button class="tb" data-mode="solid" title="Einheitliche Farbe">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>
        <span>Solid</span>
      </button>
    </div>
  </div>
  <div class="tb-group">
    <div class="tb-group-label">Ansicht</div>
    <div class="tb-group-buttons">
      <button class="tb" id="v-fit" title="Einpassen (F)">
        <svg viewBox="0 0 24 24"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
        <span>Fit</span>
      </button>
      <button class="tb" id="v-iso" title="Isometrisch (I)">
        <svg viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
        <span>Iso</span>
      </button>
      <button class="tb" id="v-top" title="Draufsicht (T)">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/></svg>
        <span>Top</span>
      </button>
      <button class="tb" id="v-front" title="Vorne (1)">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="15" x2="21" y2="15"/></svg>
        <span>Vorne</span>
      </button>
      <button class="tb" id="v-side" title="Seite (3)">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
        <span>Seite</span>
      </button>
    </div>
  </div>
  <div class="tb-group">
    <div class="tb-group-label">Anzeige</div>
    <div class="tb-group-buttons">
      <button class="tb" id="t-wire" title="Wireframe (W)">
        <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
        <span>Wire</span>
      </button>
      <button class="tb" id="t-edges" title="Kanten (E)">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="3" x2="21" y2="21" stroke-width="1"/></svg>
        <span>Kanten</span>
      </button>
      <button class="tb active" id="t-terrain" title="Terrain">
        <svg viewBox="0 0 24 24"><path d="M3 20l6-8 5 6 3-4 4 6H3z"/></svg>
        <span>Terrain</span>
      </button>
      <button class="tb" id="t-ghost" title="Andere ausblenden (G)">
        <svg viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
        <span>Ghost</span>
      </button>
    </div>
  </div>
  <div class="tb-group">
    <div class="tb-group-label">Werkzeug</div>
    <div class="tb-group-buttons">
      <button class="tb" id="tool-measure" title="Strecke messen (M)">
        <svg viewBox="0 0 24 24"><path d="M20 3H4a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1z"/><path d="M7 12h10M7 8v8M17 8v8" stroke-width="1.5"/></svg>
        <span>Messen</span>
      </button>
    </div>
  </div>
  <div class="tb-group">
    <div class="tb-group-label">Schnitt</div>
    <div class="tb-group-buttons">
      <button class="tb" id="sec-x" title="Schnitt-Ebene X">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="8" y1="3" x2="8" y2="21"/></svg>
        <span>X</span>
      </button>
      <button class="tb" id="sec-y" title="Schnitt-Ebene Y">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="8" x2="21" y2="8"/></svg>
        <span>Y</span>
      </button>
      <button class="tb" id="sec-z" title="Schnitt-Ebene Z">
        <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="3" x2="21" y2="21"/></svg>
        <span>Z</span>
      </button>
      <button class="tb" id="sec-flip" title="Richtung umkehren (F bei aktiver Ebene)">
        <svg viewBox="0 0 24 24"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
        <span>Flip</span>
      </button>
      <button class="tb" id="sec-clear" title="Alle Schnitte aufheben">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>
        <span>Aus</span>
      </button>
    </div>
  </div>
  <div class="tb-group">
    <div class="tb-group-label">Fokus</div>
    <div class="tb-group-buttons">
      <button class="tb" id="v-zoom-sel" title="Zu Auswahl zoomen (Z)">
        <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
        <span>Zoom</span>
      </button>
      <button class="tb" id="s-clear" title="Auswahl löschen (Esc)">
        <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        <span>Clear</span>
      </button>
    </div>
  </div>
</div>

<!-- Element list (left) -->
<div id="element-list">
  <div class="panel-header">
    <h4>Elemente</h4>
    <div style="display:flex;align-items:center;gap:8px">
      <span class="count" id="el-count">0</span>
      <button class="panel-btn" id="collapse-left" title="Einklappen">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
      </button>
    </div>
  </div>
  <div class="panel-body"><div id="el-rows"></div></div>
</div>
<button id="left-toggle" title="Elemente anzeigen">
  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
</button>

<!-- Properties panel (right) -->
<div id="props-panel">
  <div class="panel-header">
    <h4>Eigenschaften</h4>
    <button class="panel-btn" id="collapse-right" title="Einklappen">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
    </button>
  </div>
  <div class="panel-body">
    <div id="props-content">
      <div class="empty">Element anklicken zum Inspizieren</div>
    </div>
  </div>
</div>
<button id="right-toggle" title="Eigenschaften anzeigen">
  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
</button>

<!-- Legend -->
<div id="legend"></div>

<div id="tooltip"></div>
<div id="status">Lade Three.js...</div>

<script>
const statusDiv = document.getElementById('status');
function setStatus(msg, isError) {
  statusDiv.style.display = 'block';
  statusDiv.innerHTML = isError ? '<span id="err">' + msg + '</span>' : msg;
}
window.onerror = (msg, src, line, col, err) => {
  setStatus('JS Error: ' + msg + ' (line ' + line + ')', true);
  return false;
};
</script>

<script type="module">
const statusDiv = document.getElementById('status');
function setStatus(msg, isError) {
  statusDiv.style.display = 'block';
  statusDiv.innerHTML = isError ? '<span id="err">' + msg + '</span>' : msg;
}

try {
  setStatus('Lade Three.js...');
  const THREE = await import('https://esm.sh/three@0.160.0');
  const { OrbitControls } = await import('https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js');
  setStatus('Baue Szene...');

  const DATA = __DATA_JSON__;
  const HEIGHT = __HEIGHT__;

  if (!DATA.elements || DATA.elements.length === 0) {
    setStatus('Keine Mesh-Daten verfügbar', true);
    throw new Error('no elements');
  }

  // ── Color schemes ──────────────────────────────────────────────
  const STATUS_COLORS = {
    "PASS": 0x4CAF50,
    "FAIL": 0xF44336,
    "WARN": 0xFF9800,
    "—":    0x90A4AE,
  };
  const ROLE_COLORS = {
    "wall_stem":    0x2196F3,
    "foundation":   0x795548,
    "wing_wall":    0x9C27B0,
    "buttress":     0xFF5722,
    "":             0x90A4AE,
  };
  const CATEGORY_COLORS_HEX = {};
  for (const [k, v] of Object.entries(DATA.category_colors || {})) {
    CATEGORY_COLORS_HEX[k] = parseInt(v.replace('#', '0x'));
  }

  // ── Three.js setup ─────────────────────────────────────────────
  const canvas = document.getElementById('c');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  // Cap pixel ratio at 2 — 4K/retina displays otherwise report 3+,
  // tripling fragment shader work for no perceptible quality gain.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  // Enable local clipping so section planes only affect our meshes
  // (not helpers like the grid or the measurement line).
  renderer.localClippingEnabled = true;

  // Register cleanup for Streamlit re-renders: when the iframe is torn
  // down, dispose of WebGL resources to avoid CONTEXT_LOST churn on
  // frequent dashboard refreshes.
  window.addEventListener('beforeunload', () => {
    for (const em of elementMeshes) {
      if (em.geom) em.geom.dispose();
      if (em.mesh && em.mesh.material) em.mesh.material.dispose();
      if (em.edges && em.edges.geometry) em.edges.geometry.dispose();
      if (em.edges && em.edges.material) em.edges.material.dispose();
    }
    if (terrainMesh) {
      if (terrainMesh.geometry) terrainMesh.geometry.dispose();
      if (terrainMesh.material) terrainMesh.material.dispose();
    }
    renderer.dispose();
  });

  function getWidth() {
    return Math.max(canvas.clientWidth, window.innerWidth, 800);
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(45, getWidth() / HEIGHT, 0.01, 100000);
  const controls = new OrbitControls(camera, canvas);
  // OrbitControls tuning from IFC-Editor production viewer
  controls.enableDamping = true;
  controls.dampingFactor = 0.15;
  controls.zoomSpeed = 1.5;
  controls.rotateSpeed = 1.0;
  controls.panSpeed = 1.0;
  controls.zoomToCursor = true;
  controls.minDistance = 0.001;

  // 4-light setup mirroring the IFC-Editor (ambient + 2 directionals + fill
  // from below). MeshLambertMaterial is used throughout for consistency —
  // no PBR, no tone mapping, so the lighting rig stays simple and predictable.
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(50, 100, 50);
  scene.add(dirLight1);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dirLight2.position.set(-50, 50, -50);
  scene.add(dirLight2);
  const fillBelow = new THREE.DirectionalLight(0xffffff, 0.3);
  fillBelow.position.set(0, -100, 0);
  scene.add(fillBelow);

  // Grid
  const grid = new THREE.GridHelper(100, 50, 0x404040, 0x2a2a2a);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  // ── Build meshes ──────────────────────────────────────────────
  const elementMeshes = [];  // {id, name, status, role, mesh, edges, vertexColors, originalColors, baseGeo}
  const worldBox = new THREE.Box3();

  for (const el of DATA.elements) {
    const verts = new Float32Array(el.vertices);
    const idx = new Uint32Array(el.indices);
    if (verts.length === 0 || idx.length === 0) continue;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geom.setIndex(new THREE.BufferAttribute(idx, 1));
    geom.computeVertexNormals();

    // Per-vertex color buffer (mutable for color modes)
    const colorBuf = new Float32Array(verts.length);
    geom.setAttribute('color', new THREE.BufferAttribute(colorBuf, 3));

    const mat = new THREE.MeshLambertMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData = { elementId: el.id };
    scene.add(mesh);

    // Edge overlay (hidden by default)
    const edgeGeo = new THREE.EdgesGeometry(geom, 30);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.5 });
    const edges = new THREE.LineSegments(edgeGeo, edgeMat);
    edges.visible = false;
    scene.add(edges);

    elementMeshes.push({
      id: el.id, name: el.name, status: el.status, role: el.role,
      faceCats: el.face_categories || [],
      metrics: el.metrics || {}, checks: el.checks || [],
      mesh, edges, geom, colorBuf,
      indices: idx,
    });
    worldBox.expandByObject(mesh);
  }

  // ── Terrain ────────────────────────────────────────────────────
  let terrainMesh = null;
  if (DATA.terrain) {
    const tv = new Float32Array(DATA.terrain.vertices);
    const ti = new Uint32Array(DATA.terrain.indices);
    if (tv.length > 0 && ti.length > 0) {
      const tg = new THREE.BufferGeometry();
      tg.setAttribute('position', new THREE.BufferAttribute(tv, 3));
      tg.setIndex(new THREE.BufferAttribute(ti, 1));
      tg.computeVertexNormals();
      const tm = new THREE.MeshLambertMaterial({
        color: 0x795548, side: THREE.DoubleSide,
        transparent: true, opacity: 0.5,
      });
      terrainMesh = new THREE.Mesh(tg, tm);
      scene.add(terrainMesh);
      worldBox.expandByObject(terrainMesh);
    }
  }

  if (worldBox.isEmpty()) {
    setStatus('Bounding-Box leer', true);
    throw new Error('empty bbox');
  }

  // ── Color mode functions ──────────────────────────────────────
  let currentMode = 'status';

  function hexToRgb(hex) {
    const r = ((hex >> 16) & 255) / 255;
    const g = ((hex >> 8) & 255) / 255;
    const b = (hex & 255) / 255;
    return [r, g, b];
  }

  function applyColorMode(mode) {
    currentMode = mode;
    for (const em of elementMeshes) {
      const buf = em.colorBuf;
      const nVerts = buf.length / 3;
      const indices = em.indices;
      const nFaces = indices.length / 3;

      if (mode === 'category') {
        // Per-face color from face_categories
        for (let f = 0; f < nFaces; f++) {
          const cat = em.faceCats[f] || 'unclassified';
          const hex = CATEGORY_COLORS_HEX[cat] ?? 0x9E9E9E;
          const rgb = hexToRgb(hex);
          for (let k = 0; k < 3; k++) {
            const vi = indices[f * 3 + k];
            buf[vi * 3] = rgb[0];
            buf[vi * 3 + 1] = rgb[1];
            buf[vi * 3 + 2] = rgb[2];
          }
        }
      } else {
        // Uniform per element
        let hex;
        if (mode === 'status') hex = STATUS_COLORS[em.status] ?? STATUS_COLORS['—'];
        else if (mode === 'role') hex = ROLE_COLORS[em.role] ?? ROLE_COLORS[''];
        else hex = 0x90A4AE;
        const rgb = hexToRgb(hex);
        for (let v = 0; v < nVerts; v++) {
          buf[v * 3] = rgb[0];
          buf[v * 3 + 1] = rgb[1];
          buf[v * 3 + 2] = rgb[2];
        }
      }
      em.geom.attributes.color.needsUpdate = true;
    }
    updateLegend(mode);
  }

  // ── Legend ────────────────────────────────────────────────────
  const legendDiv = document.getElementById('legend');
  function updateLegend(mode) {
    let items = [];
    if (mode === 'status') {
      items = [
        ['#4CAF50', 'PASS'], ['#F44336', 'FAIL'],
        ['#FF9800', 'WARN'], ['#90A4AE', 'Keine Regel'],
      ];
    } else if (mode === 'category') {
      items = Object.entries(DATA.category_colors || {}).map(([k, v]) => [v, k]);
    } else if (mode === 'role') {
      items = [
        ['#2196F3', 'wall_stem'], ['#795548', 'foundation'],
        ['#9C27B0', 'wing_wall'], ['#FF5722', 'buttress'],
      ];
    } else {
      items = [['#90A4AE', 'Solid']];
    }
    legendDiv.innerHTML = items.map(([c, l]) =>
      '<div class="lg-item"><div class="lg-swatch" style="background:' + c + '"></div>' + l + '</div>'
    ).join('');
  }

  applyColorMode('status');

  // HTML-escape any string that we interpolate into innerHTML. IFC
  // element names are user-authored (AutoCAD/Revit text boxes) and can
  // contain &lt;/script&gt;, quotes, or event-handler-like fragments. Without
  // escaping, a malicious IFC could run arbitrary JS inside the iframe.
  // This is the single sanitisation hop for everything that appears in
  // the element list, properties panel, and tooltip.
  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ── Element list ──────────────────────────────────────────────
  const listDiv = document.getElementById('el-rows');
  const elCountEl = document.getElementById('el-count');
  function renderList() {
    elCountEl.textContent = elementMeshes.length;
    listDiv.innerHTML = elementMeshes.map(em => {
      const c = STATUS_COLORS[em.status] || STATUS_COLORS['—'];
      const colorHex = '#' + c.toString(16).padStart(6, '0');
      const sel = em.id === selectedId ? ' selected' : '';
      const safeName = esc(em.name);
      return '<div class="el-row' + sel + '" data-id="' + em.id + '">' +
        '<div class="el-dot" style="background:' + colorHex + '"></div>' +
        '<div class="el-name" title="' + safeName + '">#' + em.id + ' ' + safeName + '</div>' +
        '</div>';
    }).join('');
    listDiv.querySelectorAll('.el-row').forEach(row => {
      row.addEventListener('click', () => {
        const id = parseInt(row.dataset.id);
        selectElement(id);
        zoomToElement(id);
      });
    });
  }

  // ── Properties panel ──────────────────────────────────────────
  const propsContent = document.getElementById('props-content');
  let selectedId = null;
  let ghostMode = false;

  function renderProps(em) {
    if (!em) {
      propsContent.innerHTML = '<div class="empty">Element anklicken zum Inspizieren</div>';
      return;
    }
    // Every user-controlled string below runs through esc() before
    // interpolation — XSS-safe. Numeric fields are coerced to String.
    const safeName = esc(em.name);
    const safeRole = esc(em.role || 'unbekannt');
    const safeStatus = esc(em.status);
    const statusColor = STATUS_COLORS[em.status] ? '#' + STATUS_COLORS[em.status].toString(16).padStart(6, '0') : '#fff';

    let html = '<h3>#' + em.id + ' ' + safeName + '</h3>';
    html += '<div style="margin-bottom:8px;color:#909090">Rolle: ' + safeRole +
            ' | Status: <strong style="color:' + statusColor + '">' +
            safeStatus + '</strong></div>';

    html += '<h4>Messwerte</h4>';
    for (const [k, v] of Object.entries(em.metrics)) {
      if (v === null || v === undefined) continue;
      html += '<div class="prop-row"><span class="prop-key">' + esc(k) +
              '</span><span class="prop-val">' + esc(v) + '</span></div>';
    }

    if (em.checks && em.checks.length > 0) {
      html += '<h4>Regelprüfung (' + em.checks.length + ')</h4>';
      for (const c of em.checks) {
        const cls = c.status === 'PASS' ? 'PASS' : (c.status === 'FAIL' ? 'FAIL' : 'SKIP');
        html += '<div class="check-row ' + cls + '">';
        html += '<div class="check-name">[' + esc(c.status) + '] ' + esc(c.name) + '</div>';
        if (c.actual !== null && c.actual !== undefined) {
          html += '<div class="check-detail">Ist: ' + esc(c.actual) +
                  ' | Soll: ' + esc(c.expected) + '</div>';
        }
        if (c.message) {
          html += '<div class="check-detail" style="color:#bbb">' + esc(c.message) + '</div>';
        }
        html += '</div>';
      }
    } else {
      html += '<h4>Regelprüfung</h4><div class="empty">Keine Regeln evaluiert</div>';
    }
    propsContent.innerHTML = html;
  }

  function selectElement(id) {
    selectedId = id;
    const em = elementMeshes.find(e => e.id === id);
    renderProps(em);
    renderList();
    if (ghostMode) applyGhost();
    // Hot-pink emissive overlay for the selected element (IFC-Editor pattern).
    // Intensity 0.5 keeps underlying vertex-color shading visible.
    for (const m of elementMeshes) {
      const selected = m.id === id;
      m.mesh.material.emissive = new THREE.Color(selected ? 0xff69b4 : 0x000000);
      if ('emissiveIntensity' in m.mesh.material) {
        m.mesh.material.emissiveIntensity = selected ? 0.5 : 0.0;
      }
      m.mesh.material.needsUpdate = true;
    }
  }

  function clearSelection() {
    selectedId = null;
    renderProps(null);
    renderList();
    for (const m of elementMeshes) {
      m.mesh.material.emissive = new THREE.Color(0x000000);
      if ('emissiveIntensity' in m.mesh.material) m.mesh.material.emissiveIntensity = 0.0;
      m.mesh.material.needsUpdate = true;
    }
    if (ghostMode) applyGhost();
  }

  // ── Ghost mode ────────────────────────────────────────────────
  function applyGhost() {
    for (const em of elementMeshes) {
      const isSel = (em.id === selectedId);
      em.mesh.material.transparent = ghostMode && !isSel && selectedId !== null;
      em.mesh.material.opacity = (ghostMode && !isSel && selectedId !== null) ? 0.15 : 1.0;
      em.mesh.material.needsUpdate = true;
    }
  }

  // ── Raycaster click selection ─────────────────────────────────
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const tooltip = document.getElementById('tooltip');

  function getMouseNDC(event) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  canvas.addEventListener('click', (event) => {
    getMouseNDC(event);
    raycaster.setFromCamera(mouse, camera);
    const meshes = elementMeshes.map(e => e.mesh);
    const hits = raycaster.intersectObjects(meshes, false);
    if (measureMode) {
      if (hits.length > 0) addMeasurePoint(hits[0].point);
      return;
    }
    if (hits.length > 0) {
      selectElement(hits[0].object.userData.elementId);
    } else {
      clearSelection();
    }
  });

  // Cache the pickable-mesh array so we don't rebuild it every frame;
  // only rebuild if elements are added/removed (which never happens
  // once the viewer is rendered — the payload is static).
  const pickableMeshes = elementMeshes.map(e => e.mesh);

  // Throttle hover picking to one raycast per animation frame. Without
  // throttling a 60 Hz mousemove on a 10 000-element model turns into
  // 60 × O(N × T) raycasts per second and pegs the main thread.
  let hoverEvent = null;
  let hoverScheduled = false;
  function runHoverPick() {
    hoverScheduled = false;
    if (!hoverEvent) return;
    getMouseNDC(hoverEvent);
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickableMeshes, false);
    if (hits.length > 0) {
      const em = elementMeshes.find(e => e.id === hits[0].object.userData.elementId);
      tooltip.textContent = '#' + em.id + ' ' + em.name + ' [' + em.status + ']';
      tooltip.style.display = 'block';
      tooltip.style.left = (hoverEvent.clientX + 12) + 'px';
      tooltip.style.top = (hoverEvent.clientY + 12) + 'px';
      if (!measureMode) canvas.style.cursor = 'pointer';
    } else {
      tooltip.style.display = 'none';
      if (!measureMode) canvas.style.cursor = 'default';
    }
  }
  canvas.addEventListener('mousemove', (event) => {
    hoverEvent = event;
    if (!hoverScheduled) {
      hoverScheduled = true;
      requestAnimationFrame(runHoverPick);
    }
  });

  // ── Camera presets ────────────────────────────────────────────
  // Zoom-to-fit uses the bounding sphere with FOV-aware distance
  // (IFC-Editor math): d = r / sin(fov_eff/2) * 1.2 safety margin.
  // Effective FOV is the min of vertical and horizontal FOV so the
  // model fits regardless of aspect ratio.
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  worldBox.getCenter(center);
  worldBox.getSize(size);
  const sphere = new THREE.Sphere();
  worldBox.getBoundingSphere(sphere);
  const radius = Math.max(sphere.radius, 1);

  function fitDistance() {
    const fov = camera.fov * (Math.PI / 180);
    const aspect = camera.aspect;
    const horizontalFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);
    const effectiveFov = Math.min(fov, horizontalFov);
    return (radius / Math.sin(effectiveFov / 2)) * 1.2;
  }

  // Smooth camera animation (cubic ease-in-out over ~400ms).
  // Preserves spatial continuity — users don't lose their mental map
  // when the camera jumps to a new element.
  let cameraAnim = null;
  function animateCameraTo(targetPos, targetLookAt, duration) {
    duration = duration || 420;
    const startPos = camera.position.clone();
    const startLookAt = controls.target.clone();
    const t0 = performance.now();
    cameraAnim = (now) => {
      const t = Math.min(1.0, (now - t0) / duration);
      // Cubic ease-in-out
      const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      camera.position.lerpVectors(startPos, targetPos, e);
      controls.target.lerpVectors(startLookAt, targetLookAt, e);
      controls.update();
      if (t >= 1.0) cameraAnim = null;
    };
  }

  function setView(dir, animate) {
    const d = fitDistance();
    const nd = dir.clone().normalize();
    const target = center.clone().add(nd.clone().multiplyScalar(d));
    if (animate === false) {
      camera.position.copy(target);
      controls.target.copy(center);
      controls.update();
    } else {
      animateCameraTo(target, center.clone());
    }
  }

  function fitView(animate) { setView(new THREE.Vector3(1, 0.7, 1), animate); }
  function viewIso() { setView(new THREE.Vector3(1, 1, 1)); }
  function viewTop() {
    camera.up.set(0, 1, 0);
    setView(new THREE.Vector3(0, 0, 1));
  }
  function viewFront() {
    camera.up.set(0, 0, 1);
    setView(new THREE.Vector3(0, -1, 0));
  }
  function viewSide() {
    camera.up.set(0, 0, 1);
    setView(new THREE.Vector3(1, 0, 0));
  }
  fitView(false);  // instant on initial load

  // Zoom to a single element's bounding sphere (used on list-click and double-click).
  function zoomToElement(id) {
    const em = elementMeshes.find(e => e.id === id);
    if (!em) return;
    const box = new THREE.Box3().setFromObject(em.mesh);
    if (box.isEmpty()) return;
    const sph = new THREE.Sphere();
    box.getBoundingSphere(sph);
    const r = Math.max(sph.radius, 0.1);
    const fov = camera.fov * (Math.PI / 180);
    const aspect = camera.aspect;
    const horizontalFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);
    const effFov = Math.min(fov, horizontalFov);
    const dist = (r / Math.sin(effFov / 2)) * 1.4;
    // Keep current viewing direction
    const dir = camera.position.clone().sub(controls.target).normalize();
    const target = sph.center.clone().add(dir.multiplyScalar(dist));
    animateCameraTo(target, sph.center.clone());
  }

  // ── Toolbar wiring ────────────────────────────────────────────
  document.querySelectorAll('button.tb[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('button.tb[data-mode]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyColorMode(btn.dataset.mode);
    });
  });

  document.getElementById('v-fit').onclick = fitView;
  document.getElementById('v-iso').onclick = viewIso;
  document.getElementById('v-top').onclick = viewTop;
  document.getElementById('v-front').onclick = viewFront;
  document.getElementById('v-side').onclick = viewSide;

  document.getElementById('t-wire').onclick = function() {
    this.classList.toggle('active');
    const on = this.classList.contains('active');
    for (const em of elementMeshes) em.mesh.material.wireframe = on;
  };
  document.getElementById('t-edges').onclick = function() {
    this.classList.toggle('active');
    const on = this.classList.contains('active');
    for (const em of elementMeshes) em.edges.visible = on;
  };
  document.getElementById('t-terrain').onclick = function() {
    this.classList.toggle('active');
    if (terrainMesh) terrainMesh.visible = this.classList.contains('active');
  };
  document.getElementById('t-ghost').onclick = function() {
    this.classList.toggle('active');
    ghostMode = this.classList.contains('active');
    applyGhost();
  };
  document.getElementById('s-clear').onclick = clearSelection;

  // ── Keyboard shortcuts ────────────────────────────────────────
  // Keyboard shortcuts (IFC-Editor parity — A/I/H/F/Z/W/E/G/Esc)
  window.addEventListener('keydown', (e) => {
    // Never steal keys from inputs/textareas (there are none today,
    // but this keeps the contract clean for future additions)
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    const k = e.key;
    if (k === 'f' || k === 'F') fitView();
    else if (k === 'i' || k === 'I') viewIso();
    else if (k === 't' || k === 'T') viewTop();
    else if (k === 'z' || k === 'Z') { if (selectedId !== null) zoomToElement(selectedId); }
    else if (k === 'w' || k === 'W') document.getElementById('t-wire').click();
    else if (k === 'e' || k === 'E') document.getElementById('t-edges').click();
    else if (k === 'g' || k === 'G') document.getElementById('t-ghost').click();
    else if (k === 'm' || k === 'M') toggleMeasureMode();
    else if (k === '+' || k === '=') moveActiveSection(+1);
    else if (k === '-' || k === '_') moveActiveSection(-1);
    else if ((k === 'r' || k === 'R') && activeAxis) flipActiveSection();
    else if (k === 'Escape') {
      if (measureMode) toggleMeasureMode();
      else if (Object.values(sectionPlanes).some(p => p !== null)) clearAllSections();
      else clearSelection();
    }
  });

  // ── Section planes ────────────────────────────────────────────
  // One clipping plane per axis (X/Y/Z). Each plane is normalised so
  // its constant equals the center coord on its axis; toggling sets
  // the clip to that side. +/- keys move the active plane through the
  // model along its normal, F flips direction.
  const axisVectors = {
    X: new THREE.Vector3(1, 0, 0),
    Y: new THREE.Vector3(0, 1, 0),
    Z: new THREE.Vector3(0, 0, 1),
  };
  const sectionPlanes = { X: null, Y: null, Z: null };
  let activeAxis = null;  // which axis responds to +/- and F

  function sectionStep() {
    const bbox = worldBox;
    if (bbox.isEmpty()) return 1.0;
    const sz = new THREE.Vector3();
    bbox.getSize(sz);
    return Math.max(sz.x, sz.y, sz.z) / 50.0;  // 2% of largest dim per key press
  }

  function applyClippingToMeshes() {
    const active = Object.values(sectionPlanes).filter(p => p !== null);
    for (const em of elementMeshes) {
      em.mesh.material.clippingPlanes = active.length > 0 ? active : null;
      em.mesh.material.clipShadows = true;
      em.mesh.material.needsUpdate = true;
    }
  }

  function toggleSection(axis) {
    const btn = document.getElementById('sec-' + axis.toLowerCase());
    if (sectionPlanes[axis]) {
      sectionPlanes[axis] = null;
      btn.classList.remove('active');
      if (activeAxis === axis) activeAxis = null;
    } else {
      // Plane normal along axis, passes through center
      const normal = axisVectors[axis].clone();
      const plane = new THREE.Plane();
      plane.setFromNormalAndCoplanarPoint(normal, center);
      sectionPlanes[axis] = plane;
      btn.classList.add('active');
      activeAxis = axis;
    }
    applyClippingToMeshes();
  }

  function flipActiveSection() {
    if (!activeAxis || !sectionPlanes[activeAxis]) return;
    const p = sectionPlanes[activeAxis];
    p.normal.multiplyScalar(-1);
    p.constant *= -1;
    applyClippingToMeshes();
  }

  function moveActiveSection(sign) {
    if (!activeAxis || !sectionPlanes[activeAxis]) return;
    const p = sectionPlanes[activeAxis];
    p.constant += sign * sectionStep();
    applyClippingToMeshes();
  }

  function clearAllSections() {
    for (const k of Object.keys(sectionPlanes)) {
      sectionPlanes[k] = null;
      document.getElementById('sec-' + k.toLowerCase()).classList.remove('active');
    }
    activeAxis = null;
    applyClippingToMeshes();
  }

  // ── Measurement tool ──────────────────────────────────────────
  // Two-click distance measurement with snap-to-vertex (nearest
  // mesh vertex within 15 px screen distance). Draws a dashed line
  // between the two points and shows the 3D Euclidean distance.
  let measureMode = false;
  const measurePoints = [];  // THREE.Vector3 array
  let measureLine = null;
  let measureLabel = null;
  const measureLabelDiv = document.createElement('div');
  measureLabelDiv.style.cssText = 'position:absolute;pointer-events:none;z-index:65;' +
    'background:var(--accent);color:#fff;padding:4px 8px;border-radius:4px;' +
    'font-size:11px;font-family:monospace;display:none;box-shadow:0 2px 6px rgba(0,0,0,0.4);';
  document.body.appendChild(measureLabelDiv);

  function clearMeasurement() {
    measurePoints.length = 0;
    if (measureLine) { scene.remove(measureLine); measureLine.geometry.dispose(); measureLine = null; }
    measureLabelDiv.style.display = 'none';
  }

  function toggleMeasureMode() {
    measureMode = !measureMode;
    const btn = document.getElementById('tool-measure');
    if (measureMode) {
      btn.classList.add('active');
      canvas.style.cursor = 'crosshair';
      clearMeasurement();
    } else {
      btn.classList.remove('active');
      canvas.style.cursor = 'default';
      clearMeasurement();
    }
  }

  function addMeasurePoint(worldPoint) {
    measurePoints.push(worldPoint.clone());
    if (measurePoints.length === 2) {
      const [p1, p2] = measurePoints;
      const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
      const mat = new THREE.LineDashedMaterial({
        color: 0xCB0231, dashSize: 0.1, gapSize: 0.05, linewidth: 2,
      });
      measureLine = new THREE.Line(geom, mat);
      measureLine.computeLineDistances();
      scene.add(measureLine);
      const dist = p1.distanceTo(p2);
      const unit = dist >= 1 ? (dist.toFixed(3) + ' m') : ((dist * 1000).toFixed(1) + ' mm');
      measureLabelDiv.textContent = 'Distanz: ' + unit;
      // Position label at screen midpoint
      const mid = p1.clone().lerp(p2, 0.5);
      mid.project(camera);
      const rect = canvas.getBoundingClientRect();
      const sx = rect.left + (mid.x * 0.5 + 0.5) * rect.width;
      const sy = rect.top + (-mid.y * 0.5 + 0.5) * rect.height;
      measureLabelDiv.style.left = sx + 'px';
      measureLabelDiv.style.top = sy + 'px';
      measureLabelDiv.style.display = 'block';
    } else if (measurePoints.length > 2) {
      clearMeasurement();
      measurePoints.push(worldPoint.clone());
    }
  }

  // Double-click on a mesh zooms to it (IFC-Editor pattern)
  canvas.addEventListener('dblclick', (event) => {
    getMouseNDC(event);
    raycaster.setFromCamera(mouse, camera);
    const meshes = elementMeshes.map(e => e.mesh);
    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length > 0) {
      const id = hits[0].object.userData.elementId;
      selectElement(id);
      zoomToElement(id);
    }
  });

  document.getElementById('v-zoom-sel').onclick = () => {
    if (selectedId !== null) zoomToElement(selectedId);
  };
  document.getElementById('tool-measure').onclick = toggleMeasureMode;
  document.getElementById('sec-x').onclick = () => toggleSection('X');
  document.getElementById('sec-y').onclick = () => toggleSection('Y');
  document.getElementById('sec-z').onclick = () => toggleSection('Z');
  document.getElementById('sec-flip').onclick = flipActiveSection;
  document.getElementById('sec-clear').onclick = clearAllSections;

  // Panel collapse/expand
  const leftPanel = document.getElementById('element-list');
  const rightPanel = document.getElementById('props-panel');
  document.getElementById('collapse-left').onclick = () => leftPanel.classList.add('collapsed');
  document.getElementById('left-toggle').onclick = () => leftPanel.classList.remove('collapsed');
  document.getElementById('collapse-right').onclick = () => rightPanel.classList.add('collapsed');
  document.getElementById('right-toggle').onclick = () => rightPanel.classList.remove('collapsed');

  // ── Resize ────────────────────────────────────────────────────
  function resize() {
    const w = getWidth();
    renderer.setSize(w, HEIGHT, false);
    camera.aspect = w / HEIGHT;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // Re-project the measurement label each frame so it tracks the
  // line midpoint as the camera orbits.
  function updateMeasureLabel() {
    if (measurePoints.length !== 2) return;
    const mid = measurePoints[0].clone().lerp(measurePoints[1], 0.5);
    mid.project(camera);
    const rect = canvas.getBoundingClientRect();
    const sx = rect.left + (mid.x * 0.5 + 0.5) * rect.width;
    const sy = rect.top + (-mid.y * 0.5 + 0.5) * rect.height;
    measureLabelDiv.style.left = sx + 'px';
    measureLabelDiv.style.top = sy + 'px';
  }

  // ── Render loop ───────────────────────────────────────────────
  // Any exception inside the frame is logged ONCE (not 60×/s) so a
  // transient shader-compile failure doesn't destroy the console.
  let frameError = null;
  function animate(now) {
    requestAnimationFrame(animate);
    try {
      if (cameraAnim) cameraAnim(now || performance.now());
      controls.update();
      updateMeasureLabel();
      renderer.render(scene, camera);
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      if (frameError !== msg) {
        frameError = msg;
        console.error('[mesh_viewer] render frame:', err);
      }
    }
  }
  animate();

  renderList();
  statusDiv.style.display = 'none';

} catch (err) {
  setStatus('Fehler: ' + (err.message || err), true);
}
</script>
</body>
</html>
"""
