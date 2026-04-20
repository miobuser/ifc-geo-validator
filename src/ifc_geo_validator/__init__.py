"""IFC Geometry Validator — Geometric validation of IFC infrastructure models.

The canonical version string is derived from the installed package metadata
(pyproject.toml) so the Python-level `__version__` and `pip show` never drift.
"""

from pathlib import Path as _Path


def get_version() -> str:
    """Return the package version.

    Resolution order:
      1. Installed-package metadata (importlib.metadata) — canonical when
         the package was installed via pip/uv.
      2. pyproject.toml at the repo root — used when running from a
         source checkout without `pip install -e .`.
      3. "unknown" fallback.

    Having a single accessor avoids the drift between hardcoded version
    literals in `__init__.py`, `cli.py`, `app.py`, and reports.
    """
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("ifc-geo-validator")
    except Exception:
        pass
    # Source-checkout fallback: parse pyproject.toml
    try:
        py = _Path(__file__).resolve().parents[2] / "pyproject.toml"
        if py.exists():
            for line in py.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("version"):
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "unknown"


__version__ = get_version()
