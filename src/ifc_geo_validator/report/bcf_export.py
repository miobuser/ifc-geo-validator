"""Export validation failures as BCF 2.1 issues.

Each failed or skipped rule check becomes a BCF topic with metadata
(type, status, description, labels) and a comment detailing the
actual vs. expected value.  The resulting .bcf file can be opened in
any BCF-compatible BIM viewer (e.g., BIM Collab, Solibri, Desite).

Usage:
    from ifc_geo_validator.report.bcf_export import export_bcf
    export_bcf(elements_results, output_path, ifc_name="model.ifc")
"""

import uuid
from datetime import datetime

from bcf.v2 import bcfxml, model as mdl
from xsdata.models.datatype import XmlDateTime

AUTHOR = "ifc-geo-validator"

# Map severity → BCF topic_type
_SEVERITY_TYPE = {
    "ERROR": "Error",
    "WARNING": "Warning",
    "INFO": "Information",
}


def export_bcf(
    elements_results: list[dict],
    output_path: str,
    ifc_name: str = "",
) -> str:
    """Export failed rule checks as BCF topics.

    Args:
        elements_results: List of per-element result dicts (with level4).
        output_path: Path for the .bcf output file.
        ifc_name: Original IFC filename for reference.

    Returns:
        The output path.
    """
    bcf_file = bcfxml.BcfXml.create_new(AUTHOR)

    now = XmlDateTime.from_datetime(datetime.now())
    n_topics = 0

    for elem in elements_results:
        if "error" in elem:
            continue

        l4 = elem.get("level4")
        if not l4:
            continue

        elem_name = elem.get("element_name", "Unknown")
        elem_id = elem.get("element_id", "?")

        for chk in l4.get("checks", []):
            if chk["status"] == "PASS":
                continue

            title = f"[{chk['rule_id']}] {chk['name']}"
            severity = chk.get("severity", "INFO")
            topic_type = _SEVERITY_TYPE.get(severity, "Information")

            description = _build_description(chk, elem_name, elem_id, ifc_name)

            handler = bcf_file.add_topic(
                title=title,
                description=description,
                author=AUTHOR,
                topic_type=topic_type,
                topic_status="Open" if chk["status"] == "FAIL" else "Info",
            )

            # Set labels
            handler.topic.labels = [chk["rule_id"], severity]
            if ifc_name:
                handler.topic.labels.append(ifc_name)

            # Add detail comment
            comment_text = _build_comment(chk, elem_name)
            comment = mdl.Comment(
                comment=comment_text,
                date=now,
                author=AUTHOR,
                guid=str(uuid.uuid4()),
            )
            handler.markup.comment.append(comment)
            n_topics += 1

    bcf_file.save(output_path)
    bcf_file.close()
    return output_path


def _build_description(chk: dict, elem_name: str, elem_id, ifc_name: str) -> str:
    """Build topic description from check result."""
    parts = [
        f"Element: {elem_name} (#{elem_id})",
        f"Rule: {chk['rule_id']} — {chk['name']}",
        f"Status: {chk['status']}",
        f"Severity: {chk.get('severity', 'INFO')}",
    ]
    if chk.get("reference"):
        parts.append(f"Reference: {chk['reference']}")
    if chk.get("message"):
        parts.append(f"Message: {chk['message']}")
    if ifc_name:
        parts.append(f"File: {ifc_name}")
    return "\n".join(parts)


def _build_comment(chk: dict, elem_name: str) -> str:
    """Build comment text with actual value details."""
    parts = [f"{elem_name}: {chk['name']}"]

    actual = chk.get("actual_value")
    if actual is not None:
        if isinstance(actual, dict):
            for k, v in actual.items():
                parts.append(f"  {k}: {v}")
        elif isinstance(actual, float):
            parts.append(f"Actual: {actual:.2f}")
        else:
            parts.append(f"Actual: {actual}")

    parts.append(f"Check: {chk.get('check_expr', '')}")
    return "\n".join(parts)
