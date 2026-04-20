"""IFC Geometry Validator — Geometric validation of IFC infrastructure models.

The canonical version string is derived from the installed package metadata
(pyproject.toml) so the Python-level `__version__` and `pip show` never drift.
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("ifc-geo-validator")
except Exception:
    __version__ = "2.0.0"  # fallback if metadata is unavailable (e.g. source checkout)
