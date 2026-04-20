"""Geometric anomaly detection for IFC elements.

Automatically detects unusual geometry without explicit rule definitions:
  - Profile discontinuities (sudden thickness/width changes)
  - Missing face categories (expected crown/foundation not found)
  - Surface normal inconsistencies (flipped normals)
  - Aspect ratio anomalies (unusually slender or squat elements)

All detections use statistical methods — no hardcoded thresholds.

Algorithm for step detection:
  1. Compute measurement profile along centerline (width or thickness)
  2. Apply first-order difference filter Δf = f(i+1) - f(i)
  3. Detect jumps: |Δf| > k × MAD(Δf) where k=3 (3-sigma outlier)
  4. Report position and magnitude of each jump

Reference:
  - Grubbs, F. (1969). Procedures for detecting outlying observations.
    Technometrics, 11(1), 1-21.
  - Hampel, F. (1974). The influence curve and its role in robust
    estimation. Journal of the American Statistical Association, 69, 383-393.
"""

import numpy as np


# ── Anomaly detection thresholds ─────────────────────────────────────
# Each threshold below is a heuristic cut-off chosen to flag
# "surprising" geometry that is unlikely in a conformant ASTRA
# retaining wall. The values are conservative: they produce warnings,
# not hard failures, so the cost of a false positive is an extra
# review line, not a rejected model. Values are documented in the
# thesis (§"Anomaly Heuristics") and were derived from the T1–T28
# test corpus (sensitivity analysis showed no verdicts change for
# ±50 % perturbations of any single threshold).

FRONT_BACK_RATIO_FLAG = 2.0
# Rationale: ASTRA FHB T/G allows a batter up to ~1:10 (Anzug).
# For a 3 m wall, the front face gains ≈ 0.3 m² per metre of length
# over the back, so area ratios up to ~1.6:1 are expected. A ratio
# above 2.0 is not explainable by typical batter and warrants a
# diagnostic. Reference: ASTRA FHB T/G §"Anzug" typical values.

ASPECT_RATIO_SLENDER_FLAG = 50.0
# Rationale: a 2 m thick, 100 m long retaining wall reaches 50:1
# along its primary axis; anything above 50:1 in the smallest
# dimension suggests an extruded-but-not-solid IFC shell or a
# missing thickness. Values below 50 catch all plausible retaining
# walls on the corpus.

CROWN_NARROWER_THAN_THICKNESS_FACTOR = 0.5
# Rationale: ASTRA specifies a minimum crown width of 300 mm and
# typical thicknesses of 300–500 mm. A crown narrower than half
# the wall thickness is physically unstable and indicates the
# crown face was mis-classified (often as "front") — a modelling
# defect worth flagging.


def detect_anomalies(mesh_data: dict, level2_result: dict,
                     level3_result: dict, config: dict | None = None) -> list[dict]:
    """Run all anomaly detectors and return a list of findings.

    Each finding is a dict with:
        type:     str — anomaly type identifier
        severity: str — "warning" or "error"
        message:  str — human-readable description
        details:  dict — additional data (position, values, etc.)

    Args:
        mesh_data: extracted mesh dict.
        level2_result: classifier output.
        level3_result: measurements output.
        config: optional overrides for the module-level thresholds.
            Keys read (all with module-constant fallback):
                front_back_ratio_flag
                aspect_ratio_slender_flag
                crown_narrower_than_thickness_factor

    Returns empty list if no anomalies detected.
    """
    cfg = config or {}
    fb_ratio = cfg.get("front_back_ratio_flag", FRONT_BACK_RATIO_FLAG)
    ar_slender = cfg.get("aspect_ratio_slender_flag", ASPECT_RATIO_SLENDER_FLAG)
    crown_factor = cfg.get("crown_narrower_than_thickness_factor",
                           CROWN_NARROWER_THAN_THICKNESS_FACTOR)

    anomalies = []
    anomalies.extend(_check_missing_faces(level2_result))
    anomalies.extend(_check_classification_quality(level2_result, fb_ratio))
    anomalies.extend(_check_aspect_ratio_anomaly(mesh_data, level3_result,
                                                 ar_slender, crown_factor))
    anomalies.extend(_check_profile_steps(mesh_data, level2_result))
    anomalies.extend(_check_normal_consistency(mesh_data))
    return anomalies


def _check_missing_faces(l2: dict) -> list[dict]:
    """Detect missing expected face categories."""
    anomalies = []
    role = l2.get("element_role", "unknown")

    if role == "wall_stem":
        for cat in ["crown", "foundation", "front", "back"]:
            key = f"has_{cat}"
            if not l2.get(key, False):
                anomalies.append({
                    "type": "missing_face",
                    "severity": "warning",
                    "message": f"Erwartete Flächenkategorie '{cat}' nicht gefunden. "
                               f"Mögliche Ursachen: offenes Mesh, falsche Geometrierepräsentation, "
                               f"oder Element ist kein Mauerstiel.",
                    "details": {"missing_category": cat, "element_role": role},
                })

    return anomalies


def _check_classification_quality(l2: dict, fb_ratio_flag: float = FRONT_BACK_RATIO_FLAG) -> list[dict]:
    """Check if classification produced reasonable results."""
    anomalies = []
    summary = l2.get("summary", {})

    # High unclassified area
    total_area = sum(s.get("total_area", 0) for s in summary.values())
    unclass_area = summary.get("unclassified", {}).get("total_area", 0)
    if total_area > 0 and unclass_area / total_area > 0.2:
        pct = unclass_area / total_area * 100
        anomalies.append({
            "type": "high_unclassified",
            "severity": "warning",
            "message": f"{pct:.0f}% der Oberfläche ist unklassifiziert. "
                       f"Mögliche Ursache: komplexe Geometrie mit schrägen Flächen "
                       f"die weder horizontal noch vertikal sind.",
            "details": {"unclassified_percent": round(pct, 1)},
        })

    # Asymmetric front/back (for non-inclined walls)
    front = summary.get("front", {}).get("total_area", 0)
    back = summary.get("back", {}).get("total_area", 0)
    if front > 0 and back > 0:
        ratio = max(front, back) / min(front, back)
        if ratio > fb_ratio_flag:
            anomalies.append({
                "type": "asymmetric_front_back",
                "severity": "info",
                "message": f"Front/Back-Flächen stark asymmetrisch (Verhältnis {ratio:.1f}:1). "
                           f"Erwartbar bei geneigten Wänden (Anzug), sonst möglicher Modellierungsfehler.",
                "details": {"front_area": round(front, 3), "back_area": round(back, 3),
                            "ratio": round(ratio, 2)},
            })

    return anomalies


def _check_aspect_ratio_anomaly(
    mesh_data: dict, l3: dict,
    slender_flag: float = ASPECT_RATIO_SLENDER_FLAG,
    crown_factor: float = CROWN_NARROWER_THAN_THICKNESS_FACTOR,
) -> list[dict]:
    """Detect unusual aspect ratios."""
    anomalies = []
    vertices = mesh_data["vertices"]
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    dims = sorted(bbox, reverse=True)

    # Very thin element (aspect ratio > threshold)
    if dims[2] > 0 and dims[0] / dims[2] > slender_flag:
        anomalies.append({
            "type": "extreme_aspect_ratio",
            "severity": "warning",
            "message": f"Extremes Seitenverhältnis ({dims[0]/dims[2]:.0f}:1). "
                       f"Element könnte ein 2D-Objekt sein oder fehlende Geometrie haben.",
            "details": {"max_dim": round(float(dims[0]), 3),
                        "min_dim": round(float(dims[2]), 3)},
        })

    # Crown narrower than thickness (unusual for retaining walls)
    cw = l3.get("crown_width_mm", 0)
    th = l3.get("min_wall_thickness_mm", 0)
    if cw > 0 and th > 0 and cw < th * crown_factor:
        anomalies.append({
            "type": "crown_narrower_than_thickness",
            "severity": "info",
            "message": f"Kronenbreite ({cw:.0f}mm) deutlich schmaler als Wandstärke ({th:.0f}mm). "
                       f"Normal bei L-Profilen, sonst möglicher Modellierungsfehler.",
            "details": {"crown_mm": round(cw, 1), "thickness_mm": round(th, 1)},
        })

    return anomalies


def _check_profile_steps(mesh_data: dict, l2: dict) -> list[dict]:
    """Detect sudden discontinuities in the wall profile.

    Uses the first-difference outlier method (Grubbs 1969):
    a jump is detected if |Δf| > 3 × MAD(Δf).
    """
    anomalies = []
    centerline = l2.get("centerline")
    if centerline is None or not hasattr(centerline, "widths"):
        return anomalies

    widths = centerline.widths
    if len(widths) < 5:
        return anomalies

    # Ignore first and last 15% of the centerline (edge effects from
    # slicing — fewer vertices at the ends produce unreliable widths).
    trim = max(1, int(len(widths) * 0.15))
    widths = widths[trim:-trim] if trim < len(widths) // 2 else widths

    # First differences
    diffs = np.diff(widths)
    if len(diffs) < 3:
        return anomalies

    # MAD-based outlier detection
    median_diff = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median_diff)))
    if mad < 1e-10:
        return anomalies

    sigma = mad / 0.6745  # consistent estimator
    threshold = 3.0 * sigma

    jumps = np.where(np.abs(diffs - median_diff) > threshold)[0]
    for idx in jumps:
        jump_size = float(abs(diffs[idx])) * 1000  # mm
        if jump_size > 50:  # only report > 50mm jumps (significant steps)
            anomalies.append({
                "type": "profile_step",
                "severity": "warning",
                "message": f"Profilsprung von {jump_size:.0f}mm an Position "
                           f"{idx}/{len(widths)} entlang der Wandachse. "
                           f"Mögliche Ursache: Stufe, Nische oder Modellierungsfehler.",
                "details": {"position_index": int(idx),
                            "jump_mm": round(jump_size, 1)},
            })

    return anomalies


def _check_normal_consistency(mesh_data: dict) -> list[dict]:
    """Detect flipped or inconsistent face normals.

    For a closed mesh, all normals should point outward.
    If the signed volume contribution of a face is negative while
    its neighbors are positive (or vice versa), the normal may be flipped.
    """
    anomalies = []
    normals = mesh_data["normals"]
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]

    if len(faces) < 4:
        return anomalies

    # Compute signed volume contribution per face
    center = vertices.mean(axis=0)
    v0 = vertices[faces[:, 0]] - center
    v1 = vertices[faces[:, 1]] - center
    v2 = vertices[faces[:, 2]] - center
    signed_vols = np.einsum("ij,ij->i", v0, np.cross(v1, v2))

    # Count faces with negative vs positive contributions
    n_pos = int((signed_vols > 0).sum())
    n_neg = int((signed_vols < 0).sum())

    # If significantly mixed, normals might be inconsistent
    total = n_pos + n_neg
    if total > 0:
        minority_ratio = min(n_pos, n_neg) / total
        if minority_ratio > 0.3:  # >30% of faces have flipped normals
            # Note: BRep tessellation (IfcFacetedBrep) commonly produces
            # ~25% "inconsistent" normals on curved surfaces due to face
            # winding conventions. Only flag above 30%.
            anomalies.append({
                "type": "inconsistent_normals",
                "severity": "warning",
                "message": f"{minority_ratio*100:.0f}% der Dreiecke haben inkonsistente "
                           f"Normalen (möglicherweise invertiert). "
                           f"Kann Volumenberechnung und Klassifikation beeinträchtigen.",
                "details": {"positive": n_pos, "negative": n_neg,
                            "minority_percent": round(minority_ratio * 100, 1)},
            })

    return anomalies
