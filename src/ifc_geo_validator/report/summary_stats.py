"""Statistical summary for multi-element validation results.

Computes aggregate statistics across all validated elements:
  - Distribution of measurements (min/max/mean/median)
  - Outlier detection (elements deviating from the group)
  - Compliance rate per rule
  - Element clustering by role

Useful for project-level QA reports.

Reference:
  - Grubbs, F. (1969). Procedures for detecting outlying observations.
"""

import numpy as np


def compute_summary_stats(all_results: list[dict]) -> dict:
    """Compute aggregate statistics across validated elements.

    Args:
        all_results: list of per-element result dicts with level1-level4.

    Returns:
        dict with measurement distributions, outliers, compliance rates.
    """
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        return {"n_elements": 0}

    # Collect measurements
    measurements = {}
    for key in ["crown_width_mm", "crown_slope_percent", "min_wall_thickness_mm",
                 "wall_height_m", "foundation_width_mm"]:
        vals = []
        for r in valid:
            l3 = r.get("level3", {})
            v = l3.get(key)
            if v is not None and v != 0:
                vals.append(float(v))
        if vals:
            arr = np.array(vals)
            measurements[key] = {
                "n": len(vals),
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
                "mean": round(float(arr.mean()), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(arr.std()), 2),
            }

    # Outlier detection (modified Z-score via MAD)
    outliers = []
    for key, stats in measurements.items():
        if stats["n"] < 3:
            continue
        vals = [float(r.get("level3", {}).get(key, 0))
                for r in valid if r.get("level3", {}).get(key) is not None]
        arr = np.array(vals)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-10:
            continue
        z_scores = 0.6745 * (arr - median) / mad
        for i, z in enumerate(z_scores):
            if abs(z) > 3.0:  # 3-sigma outlier
                outliers.append({
                    "element": valid[i].get("element_name", "?"),
                    "property": key,
                    "value": round(float(arr[i]), 2),
                    "z_score": round(float(z), 2),
                    "group_median": round(float(median), 2),
                })

    # Compliance rate per rule
    rule_stats = {}
    for r in valid:
        l4 = r.get("level4", {})
        for chk in l4.get("checks", []):
            rid = chk["rule_id"]
            if rid not in rule_stats:
                rule_stats[rid] = {"name": chk["name"], "pass": 0, "fail": 0, "skip": 0}
            rule_stats[rid][chk["status"].lower()] = rule_stats[rid].get(chk["status"].lower(), 0) + 1

    # Role distribution
    roles = {}
    for r in valid:
        role = r.get("level2", {}).get("element_role", "unknown")
        roles[role] = roles.get(role, 0) + 1

    return {
        "n_elements": len(valid),
        "n_errors": len([r for r in all_results if "error" in r]),
        "measurements": measurements,
        "outliers": outliers,
        "rule_compliance": rule_stats,
        "roles": roles,
    }


def format_summary(stats: dict) -> str:
    """Format summary statistics as human-readable text."""
    lines = []
    lines.append(f"Elements: {stats['n_elements']} validated, {stats.get('n_errors', 0)} errors")
    lines.append("")

    # Roles
    if stats.get("roles"):
        lines.append("Element roles:")
        for role, count in sorted(stats["roles"].items(), key=lambda x: -x[1]):
            lines.append(f"  {role}: {count}")
        lines.append("")

    # Measurements
    if stats.get("measurements"):
        lines.append("Measurement distributions:")
        for key, s in stats["measurements"].items():
            lines.append(f"  {key}: {s['min']}–{s['max']} (median={s['median']}, n={s['n']})")
        lines.append("")

    # Outliers
    if stats.get("outliers"):
        lines.append(f"Outliers ({len(stats['outliers'])}):")
        for o in stats["outliers"]:
            lines.append(f"  {o['element']}: {o['property']}={o['value']} "
                         f"(z={o['z_score']:.1f}, group median={o['group_median']})")
        lines.append("")

    # Compliance
    if stats.get("rule_compliance"):
        lines.append("Rule compliance:")
        for rid, s in sorted(stats["rule_compliance"].items()):
            total = s["pass"] + s["fail"] + s["skip"]
            if s["fail"] > 0:
                lines.append(f"  {rid}: {s['pass']}/{total} PASS ({s['fail']} FAIL)")
        lines.append("")

    return "\n".join(lines)
