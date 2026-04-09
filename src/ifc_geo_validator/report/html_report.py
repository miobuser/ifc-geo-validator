"""Generate a self-contained HTML validation report.

Produces a professional-looking report with:
  - Summary table with color-coded PASS/FAIL per element
  - Per-element detail sections with metrics and rule checks
  - Inline CSS (no external dependencies)
  - Suitable for email, archiving, and printing

Usage:
    from ifc_geo_validator.report.html_report import generate_html_report
    html = generate_html_report(results, ruleset_name="ASTRA FHB T/G")
    with open("report.html", "w") as f:
        f.write(html)
"""

import math
from datetime import datetime


def generate_html_report(
    results: list[dict],
    ifc_filename: str = "",
    ruleset_name: str = "",
    l5_result: dict = None,
    l6_result: dict = None,
    project_name: str = "",
    author: str = "",
) -> str:
    """Generate a complete HTML report from validation results.

    Args:
        results: list of per-element result dicts (with level1-level4).
        ifc_filename: source IFC file name.
        ruleset_name: name of the ruleset used.
        l5_result: optional L5 inter-element results.
        l6_result: optional L6 terrain/distance results.

    Returns:
        Complete HTML string (self-contained, inline CSS).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    # Aggregate L4 stats
    total_p = total_f = total_s = 0
    for r in valid:
        l4 = r.get("level4", {})
        s = l4.get("summary", {})
        total_p += s.get("passed", 0)
        total_f += s.get("failed", 0)
        total_s += s.get("skipped", 0)

    overall = "PASS" if total_f == 0 else "FAIL"

    parts = [_html_head(ifc_filename, timestamp)]

    # Header with project metadata
    project_section = ""
    if project_name:
        project_section += f'<span>Projekt: <b>{_esc(project_name)}</b></span>'
    if author:
        project_section += f'<span>Prüfer: <b>{_esc(author)}</b></span>'

    parts.append(f"""
    <div class="header">
        <h1>Geometrisches Prüfprotokoll</h1>
        <div class="meta">
            {project_section}
            <span>Datei: <b>{_esc(ifc_filename)}</b></span>
            <span>Regelwerk: <b>{_esc(ruleset_name)}</b></span>
            <span>Datum: {timestamp}</span>
            <span>Elemente: {len(results)}</span>
        </div>
        <div class="overall {'pass' if overall == 'PASS' else 'fail'}">
            {overall} — {total_p} bestanden, {total_f} fehlgeschlagen, {total_s} übersprungen
        </div>
    </div>
    """)

    # Summary table
    parts.append('<h2>Übersicht</h2>')
    parts.append('<table class="summary"><thead><tr>')
    parts.append('<th>Element</th><th>ID</th><th>Vol (m³)</th><th>Fläche (m²)</th>')
    parts.append('<th>Wasserdicht</th><th>Confidence</th><th>Ergebnis</th></tr></thead><tbody>')

    for r in results:
        name = _esc(r.get("element_name", "?"))
        eid = r.get("element_id", "?")
        if "error" in r:
            parts.append(f'<tr class="error"><td>{name}</td><td>{eid}</td>'
                         f'<td colspan="5">ERROR: {_esc(r["error"])}</td></tr>')
            continue
        l1 = r.get("level1", {})
        l2 = r.get("level2", {})
        l4 = r.get("level4", {})
        vol = f'{l1.get("volume", 0):.2f}'
        area = f'{l1.get("total_area", 0):.2f}'
        wt = "Ja" if l1.get("is_watertight") else "Nein"
        conf = f'{l2.get("confidence", 0):.0%}'
        s = l4.get("summary", {})
        if s:
            l4_str = f'{s.get("passed", 0)}P / {s.get("failed", 0)}F / {s.get("skipped", 0)}S'
            css = "pass" if s.get("failed", 0) == 0 else "fail"
        else:
            l4_str = "—"
            css = ""
        parts.append(f'<tr><td>{name}</td><td>{eid}</td><td>{vol}</td>'
                     f'<td>{area}</td><td>{wt}</td><td>{conf}</td>'
                     f'<td class="{css}">{l4_str}</td></tr>')

    parts.append('</tbody></table>')

    # Per-element details
    for r in valid:
        name = _esc(r.get("element_name", "?"))
        eid = r.get("element_id", "?")
        l1 = r.get("level1", {})
        l2 = r.get("level2", {})
        l3 = r.get("level3", {})
        l4 = r.get("level4", {})

        parts.append(f'<div class="element"><h3>{name} <span class="eid">#{eid}</span></h3>')

        # Diagnostics
        diags = l2.get("diagnostics", [])
        if diags:
            for d in diags:
                parts.append(f'<div class="diag">{_esc(d)}</div>')

        # Metrics grid
        parts.append('<div class="metrics">')
        _add_metric(parts, "Volumen", f'{l1.get("volume", 0):.2f} m³')
        _add_metric(parts, "Oberfläche", f'{l1.get("total_area", 0):.2f} m²')
        _add_metric(parts, "Dreiecke", str(l1.get("num_triangles", 0)))
        _add_metric(parts, "Confidence", f'{l2.get("confidence", 0):.0%}')

        cw = l3.get("crown_width_mm")
        if cw is not None:
            _add_metric(parts, "Kronenbreite", f'{cw:.0f} mm')
        cs = l3.get("crown_slope_percent")
        if cs is not None:
            _add_metric(parts, "Kronenneigung", f'{cs:.2f} %')
        th = l3.get("min_wall_thickness_mm")
        if th is not None:
            _add_metric(parts, "Wandstärke", f'{th:.0f} mm')
        wh = l3.get("wall_height_m")
        if wh is not None:
            _add_metric(parts, "Wandhöhe", f'{wh:.2f} m')

        # Slope analysis
        slope = r.get("slope_analysis", {})
        if slope:
            _add_metric(parts, "Quergefälle", f'{slope.get("area_weighted_cross_pct", 0):.2f} %')
            _add_metric(parts, "Längsgefälle", f'{slope.get("area_weighted_long_pct", 0):.2f} %')

        # Curvature
        import math
        min_r = l3.get("min_radius_m")
        if min_r is not None and not math.isinf(min_r):
            _add_metric(parts, "Min. Radius", f'{min_r:.1f} m')

        # Measurement uncertainty
        unc = l3.get("measurement_uncertainty_mm", 0)
        if unc > 0:
            _add_metric(parts, "Messunsicherheit", f'±{unc:.1f} mm')

        # Min distance to nearest element
        min_d = l3.get("min_distance_to_nearest_mm")
        if min_d is not None:
            _add_metric(parts, "Min. Abstand", f'{min_d:.0f} mm')

        parts.append('</div>')

        # Clearance result
        clearance = r.get("clearance")
        if clearance:
            if clearance.get("clear"):
                parts.append('<div class="diag" style="border-color:#4CAF50;background:#E8F5E9">'
                             'Lichtraumprofil: KEIN VERSTOSS</div>')
            else:
                pen = clearance.get("max_penetration_mm", 0)
                n_v = clearance.get("n_violations", 0)
                parts.append(f'<div class="diag" style="border-color:#F44336;background:#FFEBEE">'
                             f'Lichtraumprofil: VERSTOSS — {n_v} Vertices, '
                             f'max. Eindringtiefe {pen:.0f} mm</div>')

        # Rule checks
        if l4 and l4.get("checks"):
            parts.append('<h4>Regelprüfung</h4>')
            parts.append('<table class="rules"><thead><tr>')
            parts.append('<th>Status</th><th>Regel</th><th>Name</th><th>Severity</th><th>Details</th>')
            parts.append('</tr></thead><tbody>')
            for chk in l4["checks"]:
                status = chk["status"]
                css = status.lower()
                msg = _esc(chk.get("message", ""))
                # Truncate long messages
                if len(msg) > 120:
                    msg = msg[:120] + "..."
                parts.append(
                    f'<tr class="{css}"><td class="status">{status}</td>'
                    f'<td>{_esc(chk["rule_id"])}</td>'
                    f'<td>{_esc(chk["name"])}</td>'
                    f'<td>{_esc(chk["severity"])}</td>'
                    f'<td class="msg">{msg}</td></tr>'
                )
            parts.append('</tbody></table>')

        parts.append('</div>')

    # Signature block
    if author:
        parts.append(f"""
        <div class="element" style="margin-top:24px;page-break-inside:avoid">
            <h3>Prüfvermerk</h3>
            <p style="margin:8px 0">Dieses Prüfprotokoll wurde automatisch erstellt mit
            <b>ifc-geo-validator v2.0.0</b> basierend auf dem Regelwerk
            <b>{_esc(ruleset_name)}</b>.</p>
            <div style="display:flex;gap:40px;margin-top:20px">
                <div style="flex:1">
                    <div style="border-bottom:1px solid #999;height:40px"></div>
                    <div style="font-size:11px;color:#666;margin-top:4px">
                        Ort, Datum
                    </div>
                </div>
                <div style="flex:1">
                    <div style="border-bottom:1px solid #999;height:40px"></div>
                    <div style="font-size:11px;color:#666;margin-top:4px">
                        {_esc(author)} (Prüfer/in)
                    </div>
                </div>
            </div>
        </div>
        """)

    # Footer
    parts.append(f"""
    <div class="footer">
        IFC Geometry Validator v2.0.0 — BSc Thesis BFH · Generiert: {timestamp}
    </div>
    </body></html>
    """)

    return "\n".join(parts)


def _esc(s):
    """Escape HTML special characters."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _add_metric(parts, label, value):
    parts.append(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>')


def _html_head(filename, timestamp):
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validierungsbericht — {_esc(filename)}</title>
<style>
:root {{ --pass: #2e7d32; --fail: #c62828; --skip: #757575; --bg: #fafafa; --border: #e0e0e0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: #212121; padding: 20px; max-width: 1100px; margin: 0 auto; font-size: 13px; }}
h1 {{ font-size: 20px; margin-bottom: 8px; }}
h2 {{ font-size: 16px; margin: 24px 0 8px; border-bottom: 2px solid var(--border); padding-bottom: 4px; }}
h3 {{ font-size: 14px; margin-bottom: 8px; }}
h4 {{ font-size: 12px; margin: 12px 0 6px; color: #555; }}
.header {{ background: white; padding: 16px 20px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 16px; }}
.meta {{ display: flex; gap: 20px; flex-wrap: wrap; color: #666; font-size: 12px; margin: 8px 0; }}
.overall {{ display: inline-block; padding: 6px 16px; border-radius: 4px; font-weight: 700; font-size: 14px; margin-top: 8px; }}
.overall.pass {{ background: #e8f5e9; color: var(--pass); border: 1px solid #a5d6a7; }}
.overall.fail {{ background: #ffebee; color: var(--fail); border: 1px solid #ef9a9a; }}
.eid {{ color: #999; font-weight: 400; font-size: 12px; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px; }}
th {{ background: #f5f5f5; text-align: left; padding: 6px 8px; border: 1px solid var(--border); font-weight: 600; }}
td {{ padding: 5px 8px; border: 1px solid var(--border); }}
tr:nth-child(even) {{ background: #fafafa; }}
.summary td.pass {{ color: var(--pass); font-weight: 600; }}
.summary td.fail {{ color: var(--fail); font-weight: 600; }}
.summary tr.error td {{ background: #fff3e0; color: #e65100; }}
.rules .status {{ font-weight: 700; text-align: center; width: 50px; }}
.rules tr.pass .status {{ color: var(--pass); }}
.rules tr.fail .status {{ color: var(--fail); }}
.rules tr.skip .status {{ color: var(--skip); }}
.rules tr.fail {{ background: #fff8f8; }}
.rules tr.skip {{ color: #999; }}
.rules .msg {{ font-size: 11px; color: #666; max-width: 350px; }}
.element {{ background: white; padding: 12px 16px; border-radius: 6px; border: 1px solid var(--border); margin: 12px 0; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }}
.metric {{ background: #f5f5f5; padding: 6px 12px; border-radius: 4px; min-width: 120px; }}
.metric .label {{ font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.3px; }}
.metric .value {{ font-size: 14px; font-weight: 600; }}
.diag {{ background: #fff8e1; border-left: 3px solid #ffa000; padding: 4px 10px; margin: 4px 0; font-size: 11px; color: #795548; }}
.footer {{ text-align: center; color: #999; font-size: 11px; margin-top: 32px; padding-top: 12px; border-top: 1px solid var(--border); }}
@media print {{ body {{ padding: 0; }} .element {{ break-inside: avoid; }} }}
</style>
</head>
<body>
"""
