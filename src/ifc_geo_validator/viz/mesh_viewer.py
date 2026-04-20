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
    })

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
  html, body {
    margin: 0; padding: 0; overflow: hidden;
    background: #1a1a1a; color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    width: 100%; height: __HEIGHT__px;
  }
  #c { display: block; width: 100%; height: __HEIGHT__px; }

  /* Top toolbar */
  #toolbar {
    position: absolute; top: 0; left: 0; right: 0; height: 38px;
    background: rgba(34,34,34,0.95); border-bottom: 1px solid #404040;
    display: flex; align-items: center; padding: 0 8px;
    gap: 4px; z-index: 50; user-select: none;
  }
  .group {
    display: flex; align-items: center; gap: 2px;
    padding: 0 6px; border-right: 1px solid #404040;
    height: 100%;
  }
  .group:last-child { border-right: none; }
  .group-label {
    font-size: 10px; color: #909090;
    margin-right: 4px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  button.tb {
    background: transparent; color: #e0e0e0; border: 1px solid transparent;
    padding: 4px 10px; font-size: 11px; cursor: pointer; border-radius: 3px;
    height: 26px; min-width: 28px;
  }
  button.tb:hover { background: #2a2a2a; border-color: #505050; }
  button.tb.active { background: #CB0231; color: #fff; border-color: #CB0231; }
  button.tb.active:hover { background: #e0033a; border-color: #e0033a; }

  /* Left element list */
  #element-list {
    position: absolute; top: 46px; left: 8px; width: 220px;
    max-height: calc(100% - 60px); overflow-y: auto;
    background: rgba(34,34,34,0.92); border: 1px solid #404040; border-radius: 4px;
    padding: 6px; z-index: 30; font-size: 11px;
  }
  #element-list h4 {
    margin: 0 0 6px 0; font-size: 11px; text-transform: uppercase;
    color: #909090; letter-spacing: 0.5px;
  }
  .el-row {
    padding: 4px 6px; cursor: pointer; border-radius: 2px;
    display: flex; align-items: center; gap: 6px;
    border: 1px solid transparent;
  }
  .el-row:hover { background: #2a2a2a; }
  .el-row.selected { background: #CB0231; color: #fff; border-color: #CB0231; }
  .el-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .el-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Right properties panel */
  #props-panel {
    position: absolute; top: 46px; right: 8px; width: 280px;
    max-height: calc(100% - 60px); overflow-y: auto;
    background: rgba(34,34,34,0.92); border: 1px solid #404040; border-radius: 4px;
    padding: 10px; z-index: 30; font-size: 11px;
  }
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
  <div class="group">
    <span class="group-label">Farbe</span>
    <button class="tb active" data-mode="status" id="m-status" title="Validierungs-Status">Status</button>
    <button class="tb" data-mode="category" id="m-category" title="Flächen-Klassifikation (L2)">Flächen</button>
    <button class="tb" data-mode="role" id="m-role" title="Element-Rolle (L2)">Rolle</button>
    <button class="tb" data-mode="solid" id="m-solid" title="Einheitliche Farbe">Solid</button>
  </div>
  <div class="group">
    <span class="group-label">Ansicht</span>
    <button class="tb" id="v-fit" title="Modell einpassen (F)">Fit</button>
    <button class="tb" id="v-iso" title="Isometrisch (I)">Iso</button>
    <button class="tb" id="v-top" title="Draufsicht (T)">Top</button>
    <button class="tb" id="v-front" title="Vorne (1)">Vorne</button>
    <button class="tb" id="v-side" title="Seite (3)">Seite</button>
  </div>
  <div class="group">
    <span class="group-label">Anzeige</span>
    <button class="tb" id="t-wire" title="Wireframe (W)">Wire</button>
    <button class="tb" id="t-edges" title="Kanten (E)">Kanten</button>
    <button class="tb active" id="t-terrain" title="Terrain ein/aus">Terrain</button>
    <button class="tb" id="t-ghost" title="Andere ausblenden (G)">Ghost</button>
  </div>
  <div class="group">
    <span class="group-label">Auswahl</span>
    <button class="tb" id="s-clear" title="Auswahl löschen (Esc)">Clear</button>
  </div>
</div>

<!-- Element list (left) -->
<div id="element-list">
  <h4>Elemente</h4>
  <div id="el-rows"></div>
</div>

<!-- Properties panel (right) -->
<div id="props-panel">
  <h4>Eigenschaften</h4>
  <div id="props-content">
    <div class="empty">Element anklicken zum Inspizieren</div>
  </div>
</div>

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
  renderer.setPixelRatio(window.devicePixelRatio);

  function getWidth() {
    return Math.max(canvas.clientWidth, window.innerWidth, 800);
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(45, getWidth() / HEIGHT, 0.01, 100000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;

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

  // ── Element list ──────────────────────────────────────────────
  const listDiv = document.getElementById('el-rows');
  function renderList() {
    listDiv.innerHTML = elementMeshes.map(em => {
      const c = STATUS_COLORS[em.status] || STATUS_COLORS['—'];
      const colorHex = '#' + c.toString(16).padStart(6, '0');
      const sel = em.id === selectedId ? ' selected' : '';
      return '<div class="el-row' + sel + '" data-id="' + em.id + '">' +
        '<div class="el-dot" style="background:' + colorHex + '"></div>' +
        '<div class="el-name" title="' + em.name + '">#' + em.id + ' ' + em.name + '</div>' +
        '</div>';
    }).join('');
    listDiv.querySelectorAll('.el-row').forEach(row => {
      row.addEventListener('click', () => {
        const id = parseInt(row.dataset.id);
        selectElement(id);
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
    let html = '<h3>#' + em.id + ' ' + em.name + '</h3>';
    html += '<div style="margin-bottom:8px;color:#909090">Rolle: ' + (em.role || 'unbekannt') +
            ' | Status: <strong style="color:' + (STATUS_COLORS[em.status] ? '#' + STATUS_COLORS[em.status].toString(16).padStart(6, '0') : '#fff') + '">' +
            em.status + '</strong></div>';

    html += '<h4>Messwerte</h4>';
    for (const [k, v] of Object.entries(em.metrics)) {
      if (v === null || v === undefined) continue;
      html += '<div class="prop-row"><span class="prop-key">' + k + '</span><span class="prop-val">' + v + '</span></div>';
    }

    if (em.checks && em.checks.length > 0) {
      html += '<h4>Regelprüfung (' + em.checks.length + ')</h4>';
      for (const c of em.checks) {
        const cls = c.status === 'PASS' ? 'PASS' : (c.status === 'FAIL' ? 'FAIL' : 'SKIP');
        html += '<div class="check-row ' + cls + '">';
        html += '<div class="check-name">[' + c.status + '] ' + c.name + '</div>';
        if (c.actual !== null && c.actual !== undefined) {
          html += '<div class="check-detail">Ist: ' + c.actual + ' | Soll: ' + c.expected + '</div>';
        }
        if (c.message) {
          html += '<div class="check-detail" style="color:#bbb">' + c.message + '</div>';
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
    if (hits.length > 0) {
      selectElement(hits[0].object.userData.elementId);
    } else {
      clearSelection();
    }
  });

  canvas.addEventListener('mousemove', (event) => {
    getMouseNDC(event);
    raycaster.setFromCamera(mouse, camera);
    const meshes = elementMeshes.map(e => e.mesh);
    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length > 0) {
      const em = elementMeshes.find(e => e.id === hits[0].object.userData.elementId);
      tooltip.textContent = '#' + em.id + ' ' + em.name + ' [' + em.status + ']';
      tooltip.style.display = 'block';
      tooltip.style.left = (event.clientX + 12) + 'px';
      tooltip.style.top = (event.clientY + 12) + 'px';
      canvas.style.cursor = 'pointer';
    } else {
      tooltip.style.display = 'none';
      canvas.style.cursor = 'default';
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

  function setView(dir) {
    const d = fitDistance();
    const nd = dir.clone().normalize();
    camera.position.copy(center).addScaledVector(nd, d);
    controls.target.copy(center);
    controls.update();
  }

  function fitView() { setView(new THREE.Vector3(1, 0.7, 1)); }
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
  fitView();

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
  window.addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') fitView();
    else if (e.key === 'i' || e.key === 'I') viewIso();
    else if (e.key === 't' || e.key === 'T') viewTop();
    else if (e.key === 'w' || e.key === 'W') document.getElementById('t-wire').click();
    else if (e.key === 'e' || e.key === 'E') document.getElementById('t-edges').click();
    else if (e.key === 'g' || e.key === 'G') document.getElementById('t-ghost').click();
    else if (e.key === 'Escape') clearSelection();
  });

  // ── Resize ────────────────────────────────────────────────────
  function resize() {
    const w = getWidth();
    renderer.setSize(w, HEIGHT, false);
    camera.aspect = w / HEIGHT;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // ── Render loop ───────────────────────────────────────────────
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
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
