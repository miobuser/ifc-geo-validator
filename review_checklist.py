"""Review-Checklist: Manuelle Prüfung des ifc-geo-validator.

Führe dieses Script aus:
    python review_checklist.py

Es prüft alle kritischen Funktionen automatisiert und zeigt dir
wo du visuell nachprüfen musst.
"""
import sys
import os
import math
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")

from ifc_geo_validator.core.ifc_parser import load_model, get_elements, get_terrain_mesh
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.validation.level1 import validate_level1
from ifc_geo_validator.validation.level2 import validate_level2
from ifc_geo_validator.validation.level3 import validate_level3
from ifc_geo_validator.validation.level4 import validate_level4, load_ruleset
from ifc_geo_validator.validation.level5 import validate_level5
from ifc_geo_validator.validation.level6 import validate_level6

ASTRA_RS = "src/ifc_geo_validator/rules/rulesets/astra_fhb_stuetzmauer.yaml"
SIA_RS = "src/ifc_geo_validator/rules/rulesets/sia_262_stuetzmauer.yaml"

errors = []

def check(condition, msg):
    status = "OK" if condition else "FEHLER"
    if not condition:
        errors.append(msg)
    print(f"  [{status:6s}] {msg}")
    return condition


print("=" * 80)
print("IFC-GEO-VALIDATOR v1.0.0 — REVIEW CHECKLIST")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════
print("\n1. ANALYTISCHE REFERENZWERTE (bekannte Geometrien)")
print("-" * 80)
ref = [
    ("T1_simple_box",     9.6,   400, 400, 0.0,   False, 6),
    ("T2_inclined_wall", 12.0,   350, 350, 0.0,   False, 6),
    ("T3_crown_slope",    7.211, 300, 300, 3.0,   False, 6),
    ("T7_compliant",     10.811, 300, 300, 3.0,   False, 6),
    ("T8_curved_wall",   19.245, 399, 399, 3.0,   True,  6),
    ("T15_variable_height", 14.0, 400, 400, 30.0, False, 6),
    ("T20_triangulated",  9.6,  400, 400, 0.0,   False, 6),
]
for name, exp_v, exp_cw, exp_th, exp_sl, exp_curved, exp_grp in ref:
    model = load_model(f"tests/test_models/{name}.ifc")
    walls = get_elements(model, "IfcWall")
    mesh = extract_mesh(walls[0])
    l1 = validate_level1(mesh)
    l2 = validate_level2(mesh)
    l3 = validate_level3(mesh, l2)
    cl = l2.get("centerline")

    check(abs(l1["volume"] - exp_v) < 0.5,
          f"{name}: Volume {l1['volume']:.3f} (erwartet {exp_v})")
    check(abs(l3.get("crown_width_mm", 0) - exp_cw) < 10,
          f"{name}: Kronenbreite {l3.get('crown_width_mm',0):.0f}mm (erwartet {exp_cw}mm)")
    check(abs(l3.get("min_wall_thickness_mm", 0) - exp_th) < 10,
          f"{name}: Wandstaerke {l3.get('min_wall_thickness_mm',0):.0f}mm (erwartet {exp_th}mm)")
    check(abs(l3.get("crown_slope_percent", 0) - exp_sl) < 1.0,
          f"{name}: Kronenneigung {l3.get('crown_slope_percent',0):.2f}% (erwartet {exp_sl}%)")
    if cl:
        check(cl.is_curved == exp_curved,
              f"{name}: curved={cl.is_curved} (erwartet {exp_curved})")
    check(l2["num_groups"] == exp_grp,
          f"{name}: {l2['num_groups']} Gruppen (erwartet {exp_grp})")

# ═══════════════════════════════════════════════════════════════════
print("\n2. MULTI-ELEMENT MODELLE (separate IFC-Elemente)")
print("-" * 80)
multi = [
    ("T4_l_shaped", 2),
    ("T5_t_shaped", 2),
    ("T9_stepped_wall", 2),
    ("T14_curved_l_profile", 2),
    ("T18_buttressed", 4),
]
for name, exp_n in multi:
    model = load_model(f"tests/test_models/{name}.ifc")
    walls = get_elements(model, "IfcWall")
    names = [getattr(w, "Name", "?") for w in walls]
    check(len(walls) == exp_n,
          f"{name}: {len(walls)} Elemente (erwartet {exp_n}) — {names}")

# ═══════════════════════════════════════════════════════════════════
print("\n3. LEVEL 5 — CONTACT SURFACE NORMAL ANALYSIS")
print("-" * 80)
l5_checks = [
    ("T4_l_shaped", "stacked"),
    ("T5_t_shaped", "side_by_side"),
    ("T9_stepped_wall", "stacked"),
    ("T18_buttressed", "side_by_side"),
]
for name, exp_type in l5_checks:
    model = load_model(f"tests/test_models/{name}.ifc")
    walls = get_elements(model, "IfcWall")
    elems = []
    for w in walls:
        mesh = extract_mesh(w)
        l1 = validate_level1(mesh)
        elems.append({"element_id": w.id(), "element_name": getattr(w, "Name", "?"),
                      "level1": l1, "mesh_data": mesh})
    l5 = validate_level5(elems)
    if l5["pairs"]:
        pair_type = l5["pairs"][0]["pair_type"]
        check(pair_type == exp_type,
              f"{name}: {pair_type} (erwartet {exp_type})")
    else:
        check(False, f"{name}: Keine Paare erkannt!")

# ═══════════════════════════════════════════════════════════════════
print("\n4. LEVEL 6 — TERRAIN (T22)")
print("-" * 80)
model = load_model("tests/test_models/T22_with_terrain.ifc")
terrain = get_terrain_mesh(model)
check(terrain is not None, "Terrain aus IfcSite extrahiert")
if terrain:
    check(len(terrain["vertices"]) > 10, f"Terrain hat {len(terrain['vertices'])} Vertices")
    walls = get_elements(model, "IfcWall")
    mesh = extract_mesh(walls[0])
    l1 = validate_level1(mesh)
    l2 = validate_level2(mesh)
    l3 = validate_level3(mesh, l2)
    elems = [{"element_id": walls[0].id(), "element_name": "T22",
              "level1": l1, "level2": l2, "level3": l3, "mesh_data": mesh}]
    l6 = validate_level6(elems, terrain_mesh=terrain)
    check(l6["terrain_available"], "Terrain available = True")
    check(len(l6["clearances"]) > 0, "Freibord gemessen")
    if l6["clearances"]:
        cl = l6["clearances"][0]
        check(cl["min_m"] > 0, f"Freibord min = {cl['min_m']:.2f}m (positiv)")

# ═══════════════════════════════════════════════════════════════════
print("\n5. YAML-KONFIGURIERBARKEIT (250mm Wand: SIA vs ASTRA)")
print("-" * 80)
verts = np.array([[0, 0, 0], [6, 0, 0], [6, 0.25, 0], [0, 0.25, 0],
                  [0, 0, 2], [6, 0, 2], [6, 0.25, 2], [0, 0.25, 2]], dtype=float)
faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
cross = np.cross(v1 - v0, v2 - v0)
areas = 0.5 * np.linalg.norm(cross, axis=1)
norms = np.linalg.norm(cross, axis=1, keepdims=True); norms[norms == 0] = 1
normals = cross / norms
mesh_250 = {"vertices": verts, "faces": faces, "normals": normals, "areas": areas, "is_watertight": True}

l1 = validate_level1(mesh_250)
l2 = validate_level2(mesh_250)
l3 = validate_level3(mesh_250, l2)

astra = load_ruleset(ASTRA_RS)
sia = load_ruleset(SIA_RS)
r_astra = validate_level4(l1, l3, astra)
r_sia = validate_level4(l1, l3, sia)

astra_t = next((c for c in r_astra["checks"] if "L3-003" in c["rule_id"]), None)
sia_t = next((c for c in r_sia["checks"] if "L3-001" in c["rule_id"]), None)
check(astra_t and astra_t["status"] == "FAIL",
      f"250mm vs ASTRA (min 300mm): {astra_t['status'] if astra_t else '?'} (erwartet FAIL)")
check(sia_t and sia_t["status"] == "PASS",
      f"250mm vs SIA (min 200mm): {sia_t['status'] if sia_t else '?'} (erwartet PASS)")

# ═══════════════════════════════════════════════════════════════════
print("\n6. T7 NORMKONFORM — Alle ASTRA-Regeln bestanden?")
print("-" * 80)
model = load_model("tests/test_models/T7_compliant.ifc")
walls = get_elements(model, "IfcWall")
mesh = extract_mesh(walls[0])
l1 = validate_level1(mesh)
l2 = validate_level2(mesh)
l3 = validate_level3(mesh, l2)
l4 = validate_level4(l1, l3, load_ruleset(ASTRA_RS))
for c in l4["checks"]:
    if c["status"] == "FAIL":
        check(False, f"T7 {c['rule_id']} {c['name']}: FAIL (erwartet PASS)")
passed = l4["summary"]["passed"]
total = l4["summary"]["total"]
skipped = l4["summary"]["skipped"]
check(l4["summary"]["failed"] == 0,
      f"T7 normkonform: {passed} passed, {l4['summary']['failed']} failed, {skipped} skipped von {total}")
check(total == 17, f"ASTRA Ruleset hat {total} Regeln (erwartet 17)")

# ═══════════════════════════════════════════════════════════════════
print("\n7. REGELSET-VOLLSTÄNDIGKEIT")
print("-" * 80)
astra_full = load_ruleset(ASTRA_RS)
sia_full = load_ruleset(SIA_RS)
astra_rules = sum(len(astra_full.get(k, [])) for k in ["level_1", "level_3", "level_4", "level_5", "level_6", "level_7"])
sia_rules = sum(len(sia_full.get(k, [])) for k in ["level_1", "level_3", "level_4", "level_5", "level_6", "level_7"])
check(astra_rules == 17, f"ASTRA Regelset: {astra_rules} Regeln (erwartet 17)")
check(sia_rules == 8, f"SIA Regelset: {sia_rules} Regeln (erwartet 8)")
check("classification_thresholds" in astra_full, "ASTRA hat classification_thresholds")
check(astra_full["classification_thresholds"]["horizontal_deg"] == 45.0, "horizontal_deg = 45.0 (π/4)")
check(astra_full["classification_thresholds"]["coplanar_deg"] == 5.0, "coplanar_deg = 5.0 (180°/(2×18))")

# ═══════════════════════════════════════════════════════════════════
print("\n8. TESTABDECKUNG")
print("-" * 80)
import subprocess, re
result = subprocess.run([sys.executable, "-m", "pytest", "--co", "-q"], capture_output=True, text=True, cwd=".")
test_count = 0
for line in result.stdout.splitlines():
    m = re.search(r'(\d+)\s+test', line)
    if m:
        test_count = int(m.group(1))
check(test_count >= 340, f"{test_count} Tests gesammelt (erwartet >= 340)")

model_dir = "tests/test_models"
model_count = len([f for f in os.listdir(model_dir) if f.endswith(".ifc") and f.startswith("T")])
check(model_count == 26, f"{model_count} Testmodelle (erwartet 26)")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"ERGEBNIS: {len(errors)} Fehler gefunden" if errors else "ERGEBNIS: ALLE PRÜFUNGEN BESTANDEN")
if errors:
    print("\nFehler:")
    for e in errors:
        print(f"  - {e}")
print("=" * 80)

print("""
MANUELLE PRÜFUNG (visuell):

1. VISUALISIERUNGEN prüfen:
   Öffne die Dateien in viz_output/ und prüfe:
   - T1_simple_box_classified.png     → 6 Farben, Box korrekt?
   - T4_l_shaped_classified.png       → 2 Elemente, verschiedene Helligkeit?
   - T8_curved_wall_classified.png    → Bogen, alle Flächen korrekt?
   - T18_buttressed_classified.png    → 4 Elemente sichtbar?
   - T22_with_terrain_classified.png  → Mauer + Terrain sichtbar?

2. IFC in BIM-VIEWER öffnen:
   Öffne tests/test_models/T4_l_shaped.ifc in BIMcollab/Solibri/IFC.js:
   - Sind 2 separate Elemente sichtbar (Steg + Fundament)?
   - Stimmen die Dimensionen (6m x 0.3m x 2.5m Steg, 6m x 2m x 0.5m Fundament)?

3. STREAMLIT APP testen:
   streamlit run src/ifc_geo_validator/app.py
   - Lade T4_l_shaped.ifc hoch
   - Prüfe: 2 Elemente in der Übersicht?
   - Prüfe: L5 Sektion zeigt "stacked"?
   - Lade T22_with_terrain.ifc hoch
   - Prüfe: Terrain "Detected"?

4. CLI mit JSON-Export testen:
   python -m ifc_geo_validator.cli tests/test_models/T7_compliant.ifc -o /tmp/t7.json
   - Öffne /tmp/t7.json — valides JSON?
   - Enthält level1, level2, level3, level4?

5. ENRICHED IFC testen:
   python -m ifc_geo_validator.cli tests/test_models/T7_compliant.ifc --enrich /tmp/t7_enriched.ifc
   - Öffne /tmp/t7_enriched.ifc in BIM-Viewer
   - Prüfe: Pset_GeoValidation vorhanden?
   - Enthält CrownWidth_mm, WallHeight_m, IsCurved, RulesPassed?

6. BCF-Export testen:
   python -m ifc_geo_validator.cli tests/test_models/T6_non_compliant.ifc --bcf /tmp/t6.bcf
   - Öffne /tmp/t6.bcf in BIMcollab
   - Prüfe: BCF-Topics für fehlgeschlagene Regeln?
""")
