"""Embedded web-ifc 3D viewer for Streamlit.

Renders IFC files natively in the browser using web-ifc (WASM).
The IFC data is passed as base64 to avoid server roundtrips.

Usage in Streamlit:
    from ifc_geo_validator.viz.webifc_viewer import render_ifc_viewer
    render_ifc_viewer(ifc_bytes, height=600)
"""

import base64
import streamlit.components.v1 as components


def render_ifc_viewer(
    ifc_bytes: bytes,
    height: int = 600,
    highlight_elements: dict = None,
    classification_data: dict = None,
) -> None:
    """Render an IFC file with web-ifc + Three.js and classification overlay.

    Args:
        ifc_bytes: raw IFC file content (bytes).
        height: viewer height in pixels.
        highlight_elements: optional dict {element_id: color_hex} for highlighting.
        classification_data: optional dict {expressID: {category, pass_fail}}
                            for color-coding elements by validation result.
    """
    ifc_b64 = base64.b64encode(ifc_bytes).decode("ascii")

    highlight_js = ""
    if highlight_elements:
        for eid, color in highlight_elements.items():
            highlight_js += f"highlightMap[{eid}] = '{color}';\n"

    # Pass classification data for overlay modes
    class_js = ""
    if classification_data:
        import json
        class_js = f"const classData = {json.dumps(classification_data)};"
    else:
        class_js = "const classData = {};"

    html = _VIEWER_HTML.replace("__IFC_DATA_B64__", ifc_b64)
    html = html.replace("__HIGHLIGHT_JS__", highlight_js)
    html = html.replace("__CLASS_DATA_JS__", class_js)
    html = html.replace("__HEIGHT__", str(height))

    components.html(html, height=height + 10, scrolling=False)


_VIEWER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; }
  canvas { width: 100%; height: __HEIGHT__px; display: block; }
  #loading {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: #e0e0e0; font-family: sans-serif; font-size: 14px; text-align: center;
  }
  .spinner {
    width: 30px; height: 30px; border: 3px solid #333;
    border-top-color: #2196F3; border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div>Modell wird geladen...</div>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import * as WebIFC from 'https://cdn.jsdelivr.net/npm/web-ifc@0.0.57/web-ifc-api.js';

const highlightMap = {};
__HIGHLIGHT_JS__
__CLASS_DATA_JS__

// Category colors for classification overlay
const CATEGORY_COLORS = {
  crown:        new THREE.Color(0x2196F3),  // blue
  foundation:   new THREE.Color(0x795548),  // brown
  front:        new THREE.Color(0xF44336),  // red
  back:         new THREE.Color(0xFF9800),  // orange
  end_left:     new THREE.Color(0x4CAF50),  // green
  end_right:    new THREE.Color(0x8BC34A),  // light green
  unclassified: new THREE.Color(0x9E9E9E),  // grey
};

// Pass/fail colors
const PASS_COLOR = new THREE.Color(0x4CAF50);
const FAIL_COLOR = new THREE.Color(0xF44336);
const SKIP_COLOR = new THREE.Color(0x9E9E9E);

async function init() {
  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / __HEIGHT__, 0.1, 10000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, __HEIGHT__);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(50, 100, 50);
  scene.add(dirLight);
  scene.add(new THREE.HemisphereLight(0x8888ff, 0x443322, 0.4));

  // Load IFC via web-ifc
  const ifcApi = new WebIFC.IfcAPI();
  ifcApi.SetWasmPath('https://cdn.jsdelivr.net/npm/web-ifc@0.0.57/');
  await ifcApi.Init();

  const b64 = '__IFC_DATA_B64__';
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const modelID = ifcApi.OpenModel(raw);

  // Extract and render geometry
  const meshes = ifcApi.LoadAllGeometry(modelID);
  const worldBox = new THREE.Box3();

  for (let i = 0; i < meshes.size(); i++) {
    const mesh = meshes.get(i);
    const geom = ifcApi.GetFlatMesh(modelID, mesh.expressID);

    for (let j = 0; j < geom.geometries.size(); j++) {
      const pg = geom.geometries.get(j);
      const gData = ifcApi.GetGeometry(modelID, pg.geometryExpressID);
      const vData = ifcApi.GetVertexArray(gData.GetVertexData(), gData.GetVertexDataSize());
      const iData = ifcApi.GetIndexArray(gData.GetIndexData(), gData.GetIndexDataSize());

      if (vData.length === 0 || iData.length === 0) continue;

      // Build Three.js geometry
      const bufGeo = new THREE.BufferGeometry();
      const positions = new Float32Array(vData.length / 2);
      const normals = new Float32Array(vData.length / 2);

      for (let k = 0; k < vData.length; k += 6) {
        const idx = k / 6;
        positions[idx * 3] = vData[k];
        positions[idx * 3 + 1] = vData[k + 1];
        positions[idx * 3 + 2] = vData[k + 2];
        normals[idx * 3] = vData[k + 3];
        normals[idx * 3 + 1] = vData[k + 4];
        normals[idx * 3 + 2] = vData[k + 5];
      }

      bufGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      bufGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      bufGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(iData), 1));

      // Apply placement matrix
      const mat4 = new THREE.Matrix4();
      mat4.fromArray(pg.flatTransformation);

      // Material
      const c = pg.color;
      let color = new THREE.Color(c.x, c.y, c.z);
      let opacity = c.w;

      // Highlight override
      if (highlightMap[mesh.expressID]) {
        color = new THREE.Color(highlightMap[mesh.expressID]);
        opacity = 1.0;
      }

      const material = new THREE.MeshLambertMaterial({
        color: color,
        transparent: opacity < 0.99,
        opacity: opacity,
        side: THREE.DoubleSide,
      });

      const threeeMesh = new THREE.Mesh(bufGeo, material);
      threeeMesh.applyMatrix4(mat4);
      scene.add(threeeMesh);
      worldBox.expandByObject(threeeMesh);
    }
  }

  // Camera positioning
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  worldBox.getCenter(center);
  worldBox.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  camera.position.set(center.x + maxDim, center.y + maxDim * 0.5, center.z + maxDim);
  controls.target.copy(center);

  // Hide loading
  document.getElementById('loading').style.display = 'none';

  // Render loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize
  window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, __HEIGHT__);
    camera.aspect = window.innerWidth / __HEIGHT__;
    camera.updateProjectionMatrix();
  });
}

init().catch(err => {
  document.getElementById('loading').innerHTML =
    '<span style="color:#ff6b6b">Fehler: ' + err.message + '</span>';
});
</script>
</body>
</html>
"""
