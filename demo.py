"""Quick demo: validates T28 showcase model with all features.

Usage:
    python demo.py

Generates:
    viz_output/demo_report.html    — HTML validation report
    viz_output/demo_report.json    — JSON report
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("viz_output", exist_ok=True)

print("=" * 60)
print("IFC Geometry Validator — Demo")
print("=" * 60)
print()

# Run CLI on T28 showcase model
sys.argv = [
    "ifc-geo-validator",
    "tests/test_models/T28_showcase.ifc",
    "--filter-type", "IfcWall",
    "--levels", "1,2,3,4,5,6",
    "--html", "viz_output/demo_report.html",
    "-o", "viz_output/demo_report.json",
    "-v",
]

from ifc_geo_validator.cli import main
main()

print()
print("=" * 60)
print("Demo complete!")
print(f"  HTML report: viz_output/demo_report.html")
print(f"  JSON report: viz_output/demo_report.json")
print("=" * 60)
