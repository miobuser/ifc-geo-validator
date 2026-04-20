"""IDS (Information Delivery Specification) exporter.

IDS is the buildingSMART International standard for specifying IFC
information requirements in a machine-readable, vendor-neutral way.
https://github.com/buildingSMART/IDS

An IDS file is XML with one or more ``<specification>`` elements,
each consisting of:

  - ``<applicability>`` — which IFC entities the spec applies to
  - ``<requirements>`` — what must hold for those entities

Unlike BCF (which carries findings per element), IDS carries the
rules themselves. Clients can hand an IDS file to a Revit/Solibri/
ifcopenshell-checker and get a verdict without touching our tool.

This module converts an ``ifc-geo-validator`` ruleset (YAML) into an
IDS 1.0 XML document so users who need the requirements in the
industry-standard form can export and share them.

Reference: buildingSMART IDS 1.0 schema (Dec 2023)
https://standards.buildingsmart.org/IDS/1.0/ids.xsd
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
from xml.dom import minidom


IDS_NS = "http://standards.buildingsmart.org/IDS"
XS_NS = "http://www.w3.org/2001/XMLSchema"


def export_ids(
    ruleset: dict,
    output_path: str,
    *,
    title: str = "",
    author: str = "",
    copyright: str = "",
    purpose: str = "",
) -> str:
    """Convert a parsed ruleset dict to an IDS 1.0 XML file.

    Args:
        ruleset: dict as returned by validation.level4.load_ruleset.
        output_path: target .ids (XML) path.
        title / author / copyright / purpose: info block metadata.

    Returns:
        The output path on success.
    """
    ET.register_namespace("", IDS_NS)
    ET.register_namespace("xs", XS_NS)

    ids = ET.Element(f"{{{IDS_NS}}}ids")

    # ── Info block (buildingSMART mandatory) ─────────────────
    info = ET.SubElement(ids, f"{{{IDS_NS}}}info")
    meta = ruleset.get("metadata", {})
    title_text = title or meta.get("name", "ifc-geo-validator ruleset")
    ET.SubElement(info, f"{{{IDS_NS}}}title").text = title_text
    if copyright:
        ET.SubElement(info, f"{{{IDS_NS}}}copyright").text = copyright
    if meta.get("version"):
        ET.SubElement(info, f"{{{IDS_NS}}}version").text = str(meta["version"])
    if purpose or meta.get("scope"):
        ET.SubElement(info, f"{{{IDS_NS}}}purpose").text = purpose or meta["scope"]
    ET.SubElement(info, f"{{{IDS_NS}}}date").text = datetime.now(
        timezone.utc).strftime("%Y-%m-%d")
    if author:
        ET.SubElement(info, f"{{{IDS_NS}}}author").text = author
    if meta.get("source"):
        ET.SubElement(info, f"{{{IDS_NS}}}description").text = meta["source"]

    # ── Specifications: one per rule ─────────────────────────
    specs = ET.SubElement(ids, f"{{{IDS_NS}}}specifications")

    ifc_filter = meta.get("ifc_filter", {}) or {}
    entity_name = ifc_filter.get("entity", "IfcWall")
    predefined = ifc_filter.get("predefined_type")
    ifc_version = meta.get("ifc_version", "IFC4X3_ADD2")

    for level_key in ("level_1", "level_3", "level_6"):
        for rule in ruleset.get(level_key, []):
            _add_specification(
                specs, rule,
                entity_name=entity_name,
                predefined_type=predefined,
                ifc_version=ifc_version,
            )

    xml_str = _prettyprint(ids)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return output_path


def _add_specification(parent, rule: dict, *,
                       entity_name: str,
                       predefined_type: str | None,
                       ifc_version: str) -> None:
    """Emit one <specification> element for a single rule."""
    spec = ET.SubElement(parent, f"{{{IDS_NS}}}specification")
    spec.set("name", str(rule.get("name", rule.get("id", "rule"))))
    spec.set("ifcVersion", ifc_version)
    if rule.get("description"):
        spec.set("description", str(rule["description"]))
    sev = _map_severity(rule.get("severity", "INFO"))
    spec.set("cardinality", "required" if sev == "required" else "optional")

    # Applicability: which elements this rule applies to
    applicability = ET.SubElement(spec, f"{{{IDS_NS}}}applicability")
    applicability.set("minOccurs", "0")
    applicability.set("maxOccurs", "unbounded")

    ent = ET.SubElement(applicability, f"{{{IDS_NS}}}entity")
    # <entity><name><simpleValue>IfcWall</simpleValue></name></entity>
    name_wrap = ET.SubElement(ent, f"{{{IDS_NS}}}name")
    _simple_name(name_wrap, entity_name)
    if predefined_type:
        _simple_predefined(ent, predefined_type)

    # Requirements: derived from the rule's check expression
    requirements = ET.SubElement(spec, f"{{{IDS_NS}}}requirements")
    _add_requirement_from_check(
        requirements,
        rule.get("check", ""),
        rule_id=rule.get("id", ""),
        description=rule.get("description", ""),
    )


def _map_severity(severity: str) -> str:
    """Map our ERROR/WARNING/INFO to IDS 'required'/'optional'.

    IDS's cardinality model is binary (required vs optional). We map:
      ERROR    → required
      WARNING  → optional
      INFO     → optional
    """
    return "required" if severity.upper() == "ERROR" else "optional"


def _add_requirement_from_check(parent, check_expr: str, *,
                                 rule_id: str, description: str) -> None:
    """Convert a rule check expression to an IDS property requirement.

    Strategy: parse ``<lhs> <op> <rhs>`` into a property test on
    ``Pset_GeoValidation``. The validator writes these properties
    into the enriched IFC, so a third-party IDS checker reading the
    enriched IFC can verify the same conditions.

    For expressions with boolean operators we emit a single human-
    readable string and mark the spec as a description — an IDS
    checker won't automate it, but the rule is still catalogued.
    """
    prop = ET.SubElement(parent, f"{{{IDS_NS}}}property")
    if description:
        prop.set("instructions", description)
    prop.set("dataType", "IFCREAL")
    prop.set("cardinality", "required")

    # Property set name: our enriched-IFC writer uses this constant.
    # Schema: <propertySet><simpleValue>...</simpleValue></propertySet>
    _simple_name(
        ET.SubElement(prop, f"{{{IDS_NS}}}propertySet"),
        "Pset_GeoValidation",
    )

    # Extract variable name (LHS of the comparison)
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(.+)",
                 check_expr)
    if m:
        var, op, rhs_text = m.group(1), m.group(2), m.group(3).strip()
        # IDS <property> uses <baseName><simpleValue>...</simpleValue></baseName>
        _simple_name(
            ET.SubElement(prop, f"{{{IDS_NS}}}baseName"),
            var,
        )
        # Value constraint via xs:restriction
        try:
            threshold = float(rhs_text.split()[0])
            value = ET.SubElement(prop, f"{{{IDS_NS}}}value")
            restriction = ET.SubElement(
                value, f"{{{XS_NS}}}restriction",
                {"base": "xs:double"},
            )
            facet = {
                ">=": "minInclusive", ">": "minExclusive",
                "<=": "maxInclusive", "<": "maxExclusive",
                "==": "enumeration", "!=": "enumeration",
            }.get(op, "minInclusive")
            ET.SubElement(
                restriction, f"{{{XS_NS}}}{facet}",
                {"value": _fmt_number(threshold)},
            )
        except (ValueError, IndexError):
            # Non-numeric RHS → skip the value constraint, IDS checker
            # will still validate property-presence.
            pass
    else:
        # Composite / boolean expression — catalogue it as description
        _simple_name(
            ET.SubElement(prop, f"{{{IDS_NS}}}baseName"),
            rule_id or "composite_rule",
        )


def _simple_name(parent, name: str) -> None:
    simple = ET.SubElement(parent, f"{{{IDS_NS}}}simpleValue")
    simple.text = name


def _simple_predefined(entity_elem, predefined: str) -> None:
    # IDS applicability uses <predefinedType><simpleValue>
    pd = ET.SubElement(entity_elem, f"{{{IDS_NS}}}predefinedType")
    _simple_name(pd, predefined)


def _fmt_number(v: float) -> str:
    """Compact decimal representation without scientific notation."""
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def _prettyprint(element) -> str:
    """Return a human-readable indented XML string."""
    rough = ET.tostring(element, encoding="utf-8", xml_declaration=True)
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
