"""Threshold sensitivity analysis for face classification.

Varies the three classification thresholds (horizontal_deg, coplanar_deg,
lateral_deg) across their plausible range and records how face classification
changes for each test model. Generates CSV data and summary plots.

Usage:
    py -3 sensitivity_analysis.py

Output:
    viz_output/sensitivity_*.csv  — raw data
    viz_output/sensitivity_*.png  — plots
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import os
from pathlib import Path

import numpy as np

from ifc_geo_validator.core.ifc_parser import load_model, get_elements
from ifc_geo_validator.core.mesh_converter import extract_mesh
from ifc_geo_validator.core.face_classifier import (
    classify_faces, CROWN, FOUNDATION, FRONT, BACK, END_LEFT, END_RIGHT,
    UNCLASSIFIED, DEFAULT_THRESHOLDS,
)

MODELS_DIR = Path(__file__).parent / "tests" / "test_models"
OUTPUT_DIR = Path(__file__).parent / "viz_output"

# Reference: expected classification for each model at default thresholds
EXPECTED_CATEGORIES = {
    "T1_simple_box.ifc":      {CROWN: 1, FOUNDATION: 1, FRONT: 1, BACK: 1, END_LEFT: 1, END_RIGHT: 1},
    "T2_inclined_wall.ifc":   {CROWN: 1, FOUNDATION: 1, FRONT: 1, BACK: 1, END_LEFT: 1, END_RIGHT: 1},
    "T3_crown_slope.ifc":     {CROWN: 1, FOUNDATION: 1, FRONT: 1, BACK: 1, END_LEFT: 1, END_RIGHT: 1},
    "T4_l_shaped.ifc":        {CROWN: 2, FOUNDATION: 1, FRONT: 1, BACK: 2, END_LEFT: 1, END_RIGHT: 1},
    "T5_t_shaped.ifc":        {CROWN: 2, FOUNDATION: 1, FRONT: 2, BACK: 1, END_LEFT: 2, END_RIGHT: 2},
    "T6_non_compliant.ifc":   {CROWN: 1, FOUNDATION: 1, FRONT: 1, BACK: 1, END_LEFT: 1, END_RIGHT: 1},
    "T7_compliant.ifc":       {CROWN: 1, FOUNDATION: 1, FRONT: 1, BACK: 1, END_LEFT: 1, END_RIGHT: 1},
}


def load_all_meshes():
    """Load and cache all test model meshes."""
    meshes = {}
    for model_path in sorted(MODELS_DIR.glob("T*.ifc")):
        name = model_path.name
        model = load_model(str(model_path))
        walls = get_elements(model, "IfcWall")
        if walls:
            meshes[name] = extract_mesh(walls[0])
    return meshes


def count_categories(result):
    """Count groups per category from classify_faces result."""
    counts = {}
    for g in result["face_groups"]:
        cat = g["category"] if hasattr(g, "__getitem__") else g.category
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def check_correct(counts, expected):
    """Check if all expected categories are present with correct counts."""
    for cat, exp_count in expected.items():
        if counts.get(cat, 0) != exp_count:
            return False
    return True


def sweep_single_param(meshes, param_name, values, defaults):
    """Sweep one parameter while keeping others at defaults."""
    rows = []
    for val in values:
        thresholds = {**defaults, param_name: val}
        for name, mesh in meshes.items():
            result = classify_faces(mesh, thresholds)
            counts = count_categories(result)
            n_groups = result["num_groups"]
            n_unclassified = counts.get(UNCLASSIFIED, 0)
            correct = check_correct(counts, EXPECTED_CATEGORIES.get(name, {}))

            rows.append({
                "param": param_name,
                "value": val,
                "model": name.replace(".ifc", ""),
                "num_groups": n_groups,
                "crown": counts.get(CROWN, 0),
                "foundation": counts.get(FOUNDATION, 0),
                "front": counts.get(FRONT, 0),
                "back": counts.get(BACK, 0),
                "end_left": counts.get(END_LEFT, 0),
                "end_right": counts.get(END_RIGHT, 0),
                "unclassified": n_unclassified,
                "correct": correct,
            })
    return rows


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading test models...")
    meshes = load_all_meshes()
    print(f"  Loaded {len(meshes)} models.\n")

    defaults = dict(DEFAULT_THRESHOLDS)

    # Define sweep ranges — extended to find actual breakpoints
    # Key angles: T3 crown slope ≈ 1.72°, T2/T7 front incl ≈ 84.3° from horiz
    sweeps = {
        "horizontal_deg": np.unique(np.concatenate([
            np.arange(0.5, 5, 0.25),   # fine steps near T3 breakpoint (~1.72°)
            np.arange(5, 84, 2.5),     # coarse through middle
            np.arange(84, 90, 0.25),   # fine steps near T2 breakpoint (~84.3°)
        ])),
        "coplanar_deg": np.unique(np.concatenate([
            np.arange(0.05, 1, 0.05),  # very fine near zero
            np.arange(1, 30, 1),       # coarse to 29°
        ])),
        "lateral_deg": np.unique(np.concatenate([
            np.arange(1, 10, 0.5),     # fine at low end
            np.arange(10, 85, 2.5),    # coarse through middle
            np.arange(85, 90, 0.5),    # fine at high end
        ])),
    }

    all_rows = []

    for param_name, values in sweeps.items():
        print(f"Sweeping {param_name}: {values[0]:.1f}° to {values[-1]:.1f}° ({len(values)} steps)...")
        rows = sweep_single_param(meshes, param_name, values, defaults)
        all_rows.extend(rows)

        # Count how many models are correct at each value
        by_value = {}
        for r in rows:
            v = r["value"]
            if v not in by_value:
                by_value[v] = {"total": 0, "correct": 0}
            by_value[v]["total"] += 1
            if r["correct"]:
                by_value[v]["correct"] += 1

        # Find robust range (all models correct)
        robust_values = [v for v, c in by_value.items() if c["correct"] == c["total"]]
        if robust_values:
            print(f"  Robust range: {min(robust_values):.1f}° – {max(robust_values):.1f}°")
            print(f"  Default ({defaults[param_name]:.1f}°) is {'within' if defaults[param_name] in robust_values else 'OUTSIDE'} robust range")
        else:
            print(f"  WARNING: No value achieves 100% correctness!")

        # Print correctness table
        print(f"  {'Value':>7}  {'Correct':>7}  {'Total':>5}  {'%':>5}")
        for v in sorted(by_value.keys()):
            c = by_value[v]
            pct = 100 * c["correct"] / c["total"]
            marker = " ◄" if v == defaults[param_name] else ""
            print(f"  {v:>7.1f}  {c['correct']:>7}  {c['total']:>5}  {pct:>5.0f}%{marker}")
        print()

    # Write CSV
    csv_path = OUTPUT_DIR / "sensitivity_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Data saved: {csv_path}")

    # Generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_colors = {
            "T1_simple_box": "#1f77b4",
            "T2_inclined_wall": "#ff7f0e",
            "T3_crown_slope": "#2ca02c",
            "T4_l_shaped": "#d62728",
            "T5_t_shaped": "#9467bd",
            "T6_non_compliant": "#8c564b",
            "T7_compliant": "#e377c2",
        }

        for param_name, values in sweeps.items():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                            gridspec_kw={"height_ratios": [2, 1]})

            param_rows = [r for r in all_rows if r["param"] == param_name]
            models_in_data = sorted(set(r["model"] for r in param_rows))

            # ── Top: per-model correctness (binary) ──
            for i, model_name in enumerate(models_in_data):
                model_rows = [r for r in param_rows if r["model"] == model_name]
                xs = [r["value"] for r in model_rows]
                ys = [i + 0.8 if r["correct"] else i + 0.2 for r in model_rows]
                color = model_colors.get(model_name, "#333")
                ax1.fill_between(xs, i, ys, alpha=0.4, color=color, step="mid")
                ax1.step(xs, ys, where="mid", color=color, linewidth=1.2,
                         label=model_name)

            ax1.axvline(defaults[param_name], color="#F44336", linestyle="--",
                        linewidth=1, zorder=5)
            ax1.set_yticks([i + 0.5 for i in range(len(models_in_data))])
            ax1.set_yticklabels(models_in_data, fontsize=8)
            ax1.set_ylim(-0.2, len(models_in_data) + 0.2)
            ax1.set_title(f"Sensitivität: {param_name}", fontsize=13)
            ax1.set_ylabel("Korrekt klassifiziert", fontsize=11)
            ax1.grid(True, alpha=0.3, axis="x")

            # ── Bottom: aggregate correctness ──
            val_correct = {}
            for r in param_rows:
                v = r["value"]
                if v not in val_correct:
                    val_correct[v] = {"n": 0, "ok": 0}
                val_correct[v]["n"] += 1
                if r["correct"]:
                    val_correct[v]["ok"] += 1

            xs = sorted(val_correct.keys())
            ys = [100 * val_correct[x]["ok"] / val_correct[x]["n"] for x in xs]

            ax2.fill_between(xs, ys, alpha=0.3, color="#2196F3")
            ax2.plot(xs, ys, "-", color="#2196F3", linewidth=2)
            ax2.axvline(defaults[param_name], color="#F44336", linestyle="--", linewidth=1)
            ax2.axhline(100, color="#4CAF50", linestyle=":", alpha=0.5)

            # Shade robust range
            robust = [x for x, y in zip(xs, ys) if y == 100]
            if robust:
                ax2.axvspan(min(robust), max(robust), alpha=0.1, color="#4CAF50")
                ax2.text((min(robust) + max(robust)) / 2, 50,
                         f"Robust: {min(robust):.1f}°–{max(robust):.1f}°",
                         ha="center", fontsize=9, color="#388E3C", alpha=0.8)

            ax2.set_xlabel(f"{param_name} (°)", fontsize=11)
            ax2.set_ylabel("Korrekte Modelle (%)", fontsize=11)
            ax2.set_ylim(-5, 110)
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            png_path = OUTPUT_DIR / f"sensitivity_{param_name}.png"
            fig.savefig(str(png_path), dpi=150)
            plt.close(fig)
            print(f"Plot saved: {png_path}")

    except ImportError:
        print("matplotlib not available — skipping plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
