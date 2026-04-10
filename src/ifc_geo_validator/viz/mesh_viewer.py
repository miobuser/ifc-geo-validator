"""Fast Three.js mesh viewer for pre-extracted IFC element meshes.

Unlike webifc_viewer.py, this skips the WASM IFC parser entirely. Meshes are
already extracted server-side via IfcOpenShell/OCCT, so we just ship the
vertex/index buffers as JSON and render them directly with Three.js.

This is dramatically faster:
  - No WASM download (~3 MB)
  - No IFC re-parsing in browser
  - Only the small mesh JSON crosses the wire

Usage:
    from ifc_geo_validator.viz.mesh_viewer import render_mesh_viewer
    render_mesh_viewer(elements, height=550)
"""

import json
import streamlit.components.v1 as components
import numpy as np


def render_mesh_viewer(elements: list, height: int = 550, terrain_mesh: dict | None = None) -> None:
    """Render pre-extracted meshes with Three.js (no WASM, no IFC parsing).

    Args:
        elements: list of dicts, each with:
            - element_id: int
            - element_name: str
            - mesh_data: dict {vertices: ndarray (N,3), faces: ndarray (M,3)}
            - status: "PASS" | "FAIL" | "WARN" | "—"
        height: viewer height in pixels.
        terrain_mesh: optional terrain mesh dict with same structure.
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
        payload.append({
            "id": int(el.get("element_id", 0)),
            "name": str(el.get("element_name", "")),
            "status": el.get("status", "—"),
            "vertices": verts.flatten().tolist(),
            "indices": faces.flatten().tolist(),
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

    data_json = json.dumps({"elements": payload, "terrain": terrain_payload})

    html = _VIEWER_HTML.replace("__DATA_JSON__", data_json)
    html = html.replace("__HEIGHT__", str(height))

    components.html(html, height=height + 10, scrolling=False)


_VIEWER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; font-family: sans-serif; }
  #c { width: 100%; height: __HEIGHT__px; display: block; }
  #legend {
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,0,0,0.6); color: #e0e0e0;
    padding: 8px 12px; border-radius: 4px; font-size: 12px;
  }
  .swatch { display: inline-block; width: 12px; height: 12px;
            margin-right: 6px; vertical-align: middle; border-radius: 2px; }
  #info {
    position: absolute; bottom: 10px; left: 10px;
    background: rgba(0,0,0,0.6); color: #e0e0e0;
    padding: 6px 10px; border-radius: 4px; font-size: 11px;
  }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="legend">
  <div><span class="swatch" style="background:#4CAF50"></span>PASS</div>
  <div><span class="swatch" style="background:#F44336"></span>FAIL</div>
  <div><span class="swatch" style="background:#FF9800"></span>WARN</div>
  <div><span class="swatch" style="background:#795548"></span>Terrain</div>
</div>
<div id="info">Maus: Drehen | Scroll: Zoom | Rechtsklick: Pan</div>

<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const DATA = __DATA_JSON__;
const HEIGHT = __HEIGHT__;

const STATUS_COLOR = {
  "PASS": 0x4CAF50,
  "FAIL": 0xF44336,
  "WARN": 0xFF9800,
  "—":    0x90A4AE,
};

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

function resize() {
  const w = canvas.clientWidth;
  renderer.setSize(w, HEIGHT, false);
  if (camera) {
    camera.aspect = w / HEIGHT;
    camera.updateProjectionMatrix();
  }
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

let camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
dirLight.position.set(50, 100, 50);
scene.add(dirLight);
scene.add(new THREE.HemisphereLight(0x8888ff, 0x443322, 0.4));

const worldBox = new THREE.Box3();

// Build element meshes
for (const el of DATA.elements) {
  const verts = new Float32Array(el.vertices);
  const idx = new Uint32Array(el.indices);

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geom.setIndex(new THREE.BufferAttribute(idx, 1));
  geom.computeVertexNormals();

  const colorHex = STATUS_COLOR[el.status] ?? STATUS_COLOR["—"];
  const mat = new THREE.MeshLambertMaterial({
    color: colorHex,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.userData = { id: el.id, name: el.name, status: el.status };
  scene.add(mesh);

  // Edge overlay for clearer geometry
  const edges = new THREE.EdgesGeometry(geom, 30);
  const edgeMat = new THREE.LineBasicMaterial({
    color: 0x000000, transparent: true, opacity: 0.3,
  });
  scene.add(new THREE.LineSegments(edges, edgeMat));

  worldBox.expandByObject(mesh);
}

// Build terrain (semi-transparent brown)
if (DATA.terrain) {
  const tv = new Float32Array(DATA.terrain.vertices);
  const ti = new Uint32Array(DATA.terrain.indices);
  const tg = new THREE.BufferGeometry();
  tg.setAttribute('position', new THREE.BufferAttribute(tv, 3));
  tg.setIndex(new THREE.BufferAttribute(ti, 1));
  tg.computeVertexNormals();
  const tm = new THREE.MeshLambertMaterial({
    color: 0x795548,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5,
  });
  const tmesh = new THREE.Mesh(tg, tm);
  scene.add(tmesh);
  worldBox.expandByObject(tmesh);
}

// Center & frame the model
const center = new THREE.Vector3();
const size = new THREE.Vector3();
worldBox.getCenter(center);
worldBox.getSize(size);
const maxDim = Math.max(size.x, size.y, size.z, 1);
camera.position.set(center.x + maxDim, center.y + maxDim * 0.7, center.z + maxDim);
controls.target.copy(center);
controls.update();

resize();
window.addEventListener('resize', resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""
