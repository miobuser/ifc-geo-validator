"""Entry point for Streamlit Cloud deployment.

This file adds the src directory to Python path and imports the main app.
Streamlit Cloud runs this file directly.
"""
import sys
import os

# Add src to path so ifc_geo_validator package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import and run the actual app
from ifc_geo_validator.app import *  # noqa: F401,F403
