"""Streamlit-based YAML Ruleset Editor.

Visual interface to create and edit validation rulesets.
Run with: streamlit run src/ifc_geo_validator/ruleset_editor.py

Features:
  - Browse available variables with descriptions
  - Add/edit/delete rules visually
  - Preview YAML output
  - Download ruleset file
  - Test rules against a sample model
"""

import streamlit as st
import yaml
from pathlib import Path

# ── Available variables with descriptions ──────────────────────────

from ifc_geo_validator.rules.variables import VARIABLE_CATALOG as VARIABLES

SEVERITY_OPTIONS = ["ERROR", "WARNING", "INFO"]
OPERATORS = [">=", "<=", ">", "<", "==", "!="]
ROLE_OPTIONS = ["wall_stem", "foundation", "parapet", "column", "slab"]

# ── App ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Ruleset Editor", page_icon="📐", layout="wide")

st.title("📐 Ruleset Editor")
st.caption("Erstelle und bearbeite YAML-Validierungsregeln visuell")

# Sidebar: metadata
st.sidebar.header("Ruleset-Metadaten")
rs_name = st.sidebar.text_input("Name", "Mein Ruleset")
rs_version = st.sidebar.text_input("Version", "1.0.0")
rs_source = st.sidebar.text_input("Quelle/Norm", "")
rs_scope = st.sidebar.text_input("Geltungsbereich", "")

# Main area: rules
if "rules" not in st.session_state:
    st.session_state.rules = []

# Variable browser
with st.expander("📖 Verfügbare Variablen", expanded=False):
    for category, vars_dict in VARIABLES.items():
        st.markdown(f"**{category}**")
        rows = []
        for var, (vtype, unit, desc) in vars_dict.items():
            rows.append({"Variable": f"`{var}`", "Typ": vtype, "Einheit": unit, "Beschreibung": desc})
        st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()

# Add new rule
st.subheader("Neue Regel hinzufügen")
col1, col2 = st.columns(2)

with col1:
    rule_name = st.text_input("Regelname", "Kronenbreite Minimum")
    rule_desc = st.text_input("Beschreibung", "Mauerkrone mindestens 300 mm breit")
    rule_severity = st.selectbox("Severity", SEVERITY_OPTIONS)
    rule_reference = st.text_input("Referenz", "FHB T/G 24 001-15101")

with col2:
    # Variable picker
    all_vars = {}
    for cat, vars_dict in VARIABLES.items():
        for var, info in vars_dict.items():
            all_vars[f"{var} ({info[2]})"] = var

    selected_var_label = st.selectbox("Variable", list(all_vars.keys()))
    selected_var = all_vars[selected_var_label]

    operator = st.selectbox("Operator", OPERATORS)
    threshold = st.number_input("Schwellwert", value=300.0, step=0.1)

    applies_to = st.multiselect("Gilt für (leer = alle)", ROLE_OPTIONS)

check_expr = f"{selected_var} {operator} {threshold}"
st.code(f'check: "{check_expr}"', language="yaml")

if st.button("Regel hinzufügen", type="primary"):
    rule_id = f"CUSTOM-{len(st.session_state.rules) + 1:03d}"
    rule = {
        "id": rule_id,
        "name": rule_name,
        "description": rule_desc,
        "check": check_expr,
        "severity": rule_severity,
        "reference": rule_reference,
    }
    if applies_to:
        rule["applies_to"] = applies_to
    st.session_state.rules.append(rule)
    st.success(f"Regel {rule_id} hinzugefügt!")

st.divider()

# Show current rules
if st.session_state.rules:
    st.subheader(f"Aktuelle Regeln ({len(st.session_state.rules)})")
    for i, rule in enumerate(st.session_state.rules):
        col_r, col_del = st.columns([10, 1])
        with col_r:
            sev_color = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}[rule["severity"]]
            st.markdown(f"{sev_color} **{rule['id']}** — {rule['name']}: `{rule['check']}` ({rule['severity']})")
        with col_del:
            if st.button("🗑", key=f"del_{i}"):
                st.session_state.rules.pop(i)
                st.rerun()

# Generate YAML
st.divider()
st.subheader("YAML-Ausgabe")

ruleset = {
    "metadata": {
        "name": rs_name,
        "version": rs_version,
        "source": rs_source,
        "scope": rs_scope,
    },
    "classification_thresholds": {
        "horizontal_deg": 45.0,
        "coplanar_deg": 5.0,
        "lateral_deg": 45.0,
    },
}

# Group rules by level
l1_rules = [r for r in st.session_state.rules if any(v in r["check"] for v in ["volume", "mesh_is", "bbox_", "num_tri"])]
l3_rules = [r for r in st.session_state.rules if r not in l1_rules]

if l1_rules:
    ruleset["level_1"] = l1_rules
if l3_rules:
    ruleset["level_3"] = l3_rules

yaml_output = yaml.dump(ruleset, default_flow_style=False, allow_unicode=True, sort_keys=False)
st.code(yaml_output, language="yaml")

st.download_button(
    "📥 YAML herunterladen",
    data=yaml_output,
    file_name=f"{rs_name.replace(' ', '_').lower()}.yaml",
    mime="text/yaml",
    use_container_width=True,
)
