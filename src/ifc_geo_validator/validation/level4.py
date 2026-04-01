"""Level 4 Validation: Requirement comparison against YAML ruleset.

Evaluates Level 1–7 results against configurable rules defined in YAML.
Each rule produces a CheckResult with PASS/FAIL/SKIP status.

The rule evaluation follows the four-stage process defined by Eastman et al.
(2009): (1) rule interpretation (YAML parsing), (2) model preparation
(L1–L3 computation), (3) rule execution (expression evaluation), and
(4) report generation (PASS/FAIL/SKIP).

References:
  - Eastman, C. et al. (2009). Automatic rule-based checking of building
    designs. Automation in Construction, 18(8), 1011-1033.
  - Solihin, W. & Eastman, C. (2015). Classification of rules for automated
    BIM rule checking development. Automation in Construction, 53, 69-82.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path


# ── Severity & Status ───────────────────────────────────────────────

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    """Result of a single rule check."""
    rule_id: str
    name: str
    status: str          # PASS / FAIL / SKIP
    severity: str        # ERROR / WARNING / INFO
    actual_value: object
    check_expr: str
    reference: str
    message: str


# ── Public API ──────────────────────────────────────────────────────

def load_ruleset(path: str) -> dict:
    """Load and parse a YAML ruleset file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_level4(
    level1_result: dict,
    level3_result: dict,
    ruleset: dict,
    level5_context: dict = None,
    level6_context: dict = None,
) -> dict:
    """Evaluate rules from the ruleset against computed values.

    Args:
        level1_result: dict from validate_level1().
        level3_result: dict from validate_level3().
        ruleset: parsed YAML ruleset dict.
        level5_context: optional dict with L5 inter-element variables.
        level6_context: optional dict with L6 distance/terrain variables.

    Returns:
        dict with:
            checks:   list[dict] — serialised CheckResult per rule
            summary:  {total, passed, failed, skipped, errors, warnings}
    """
    # Build the evaluation context from computed values
    context = _build_context(level1_result, level3_result)

    # Merge L5/L6 context variables if provided
    if level5_context:
        context.update(level5_context)
    if level6_context:
        context.update(level6_context)

    checks = []

    # Evaluate Level 1 rules
    for rule in ruleset.get("level_1", []):
        checks.append(_evaluate_rule(rule, context))

    # Evaluate Level 3 rules
    for rule in ruleset.get("level_3", []):
        checks.append(_evaluate_rule(rule, context))

    # Evaluate Level 5 rules (inter-element context)
    for rule in ruleset.get("level_5", []):
        checks.append(_evaluate_rule(rule, context))

    # Evaluate Level 6 rules (distance/terrain context)
    for rule in ruleset.get("level_6", []):
        checks.append(_evaluate_rule(rule, context))

    # Evaluate Level 7 rules (distance calculations)
    for rule in ruleset.get("level_7", []):
        checks.append(_evaluate_rule(rule, context))

    # Evaluate Level 4 composite rules (depend on all other results)
    for rule in ruleset.get("level_4", []):
        checks.append(_evaluate_composite(rule, checks))

    # Build summary
    summary = {
        "total": len(checks),
        "passed": sum(1 for c in checks if c["status"] == PASS),
        "failed": sum(1 for c in checks if c["status"] == FAIL),
        "skipped": sum(1 for c in checks if c["status"] == SKIP),
        "errors": sum(1 for c in checks if c["status"] == FAIL and c["severity"] == ERROR),
        "warnings": sum(1 for c in checks if c["status"] == FAIL and c["severity"] == WARNING),
    }

    return {
        "checks": checks,
        "summary": summary,
    }


# ── Context builder ─────────────────────────────────────────────────

def _build_context(level1_result: dict, level3_result: dict) -> dict:
    """Build a flat dict of all variables available for rule evaluation."""
    ctx = {}

    # Level 1
    ctx["volume"] = level1_result.get("volume", 0.0)
    ctx["total_area"] = level1_result.get("total_area", 0.0)
    ctx["mesh_is_watertight"] = level1_result.get("is_watertight", False)

    # Level 3
    ctx["crown_width_mm"] = level3_result.get("crown_width_mm")
    ctx["crown_slope_percent"] = level3_result.get("crown_slope_percent")
    ctx["min_wall_thickness_mm"] = level3_result.get("min_wall_thickness_mm")
    ctx["avg_wall_thickness_mm"] = level3_result.get("avg_wall_thickness_mm")
    ctx["front_inclination_deg"] = level3_result.get("front_inclination_deg")
    ctx["front_inclination_ratio"] = level3_result.get("front_inclination_ratio")

    # Level 3 — Curved wall metrics
    ctx["is_curved"] = level3_result.get("is_curved", False)
    ctx["wall_length_m"] = level3_result.get("wall_length_m")
    ctx["wall_height_m"] = level3_result.get("wall_height_m")

    # Level 3 — Profile consistency
    ctx["crown_width_cv"] = level3_result.get("crown_width_cv")

    # Level 3 — Foundation measurements
    ctx["foundation_width_mm"] = level3_result.get("foundation_width_mm")
    ctx["foundation_width_ratio"] = level3_result.get("foundation_width_ratio")

    return ctx


# ── Rule evaluator ──────────────────────────────────────────────────

def _evaluate_rule(rule: dict, context: dict) -> dict:
    """Evaluate a single rule against the context.

    The 'check' field is a simple Python expression evaluated against
    the context dict.  Only comparisons and boolean operators are allowed.
    """
    rule_id = rule["id"]
    name = rule["name"]
    severity = rule.get("severity", INFO)
    check_expr = rule.get("check", "")
    reference = rule.get("reference", "")

    # Check if all required variables are available
    try:
        result = _safe_eval(check_expr, context)
    except _MissingVariable as e:
        return _make_result(
            rule_id, name, SKIP, severity, None, check_expr, reference,
            f"Skipped: variable '{e.var_name}' not available"
        )
    except Exception as e:
        return _make_result(
            rule_id, name, SKIP, severity, None, check_expr, reference,
            f"Skipped: evaluation error: {e}"
        )

    if bool(result):
        return _make_result(
            rule_id, name, PASS, severity,
            _get_actual_value(check_expr, context),
            check_expr, reference, "Check passed"
        )
    else:
        return _make_result(
            rule_id, name, FAIL, severity,
            _get_actual_value(check_expr, context),
            check_expr, reference,
            f"Check failed: {check_expr}"
        )


def _evaluate_composite(rule: dict, previous_checks: list) -> dict:
    """Evaluate a composite rule that depends on other rules.

    All dependencies must PASS for the composite to PASS.
    """
    rule_id = rule["id"]
    name = rule["name"]
    severity = rule.get("severity", INFO)
    depends_on = rule.get("depends_on", [])
    reference = rule.get("reference", "")

    dep_results = []
    for dep_id in depends_on:
        dep = next((c for c in previous_checks if c["rule_id"] == dep_id), None)
        if dep is None:
            dep_results.append(SKIP)
        else:
            dep_results.append(dep["status"])

    if SKIP in dep_results:
        status = SKIP
        msg = f"Skipped: dependency not evaluated"
    elif all(s == PASS for s in dep_results):
        status = PASS
        msg = f"All dependencies passed: {', '.join(depends_on)}"
    else:
        failed = [d for d, s in zip(depends_on, dep_results) if s == FAIL]
        status = FAIL
        msg = f"Failed dependencies: {', '.join(failed)}"

    return _make_result(
        rule_id, name, status, severity,
        {d: s for d, s in zip(depends_on, dep_results)},
        f"depends_on: {depends_on}", reference, msg
    )


# ── Safe expression evaluator ──────────────────────────────────────

class _MissingVariable(Exception):
    def __init__(self, var_name):
        self.var_name = var_name


def _safe_eval(expr: str, context: dict) -> bool:
    """Evaluate a simple comparison expression safely.

    Supports: >=, <=, >, <, ==, !=, and, or, true, false
    Variables are resolved from context dict.
    Raises _MissingVariable only if the expression references a None variable.
    """
    # Replace 'true'/'false' literals
    expr_clean = expr.replace("true", "True").replace("false", "False")

    # Build namespace — skip None values so they raise NameError on access
    namespace = {}
    none_keys = set()
    for key, val in context.items():
        if val is None:
            none_keys.add(key)
        else:
            namespace[key] = val

    safe_builtins = {"True": True, "False": False, "abs": abs}
    try:
        return eval(expr_clean, {"__builtins__": safe_builtins}, namespace)
    except NameError as e:
        # Extract variable name from NameError message
        var = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise _MissingVariable(var)


def _get_actual_value(expr: str, context: dict):
    """Extract the left-hand-side variable value from a check expression."""
    # Simple heuristic: first token before a comparison operator
    for op in [">=", "<=", "!=", "==", ">", "<"]:
        if op in expr:
            var = expr.split(op)[0].strip()
            return context.get(var)
    return None


def _make_result(rule_id, name, status, severity, actual, check_expr, reference, message):
    return {
        "rule_id": rule_id,
        "name": name,
        "status": status,
        "severity": severity,
        "actual_value": actual,
        "check_expr": check_expr,
        "reference": reference,
        "message": message,
    }
