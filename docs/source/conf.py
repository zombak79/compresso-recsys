"""Sphinx configuration for the Compresso Recsys documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which pyproject still supports
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


sys.path.insert(0, os.path.abspath("../../src"))

project = "Compresso Recsys"
author = "Compresso contributors"


def _release() -> str:
    """Version to label the built documentation with.

    pyproject first, installed metadata second. The other order looks like a
    reasonable fallback and is useless: metadata is present whenever the package
    is installed at all, so a stale install would always win, and stamping the
    docs with a version the source has moved past is exactly the failure this
    avoids. Building against an editable install of an older release is the
    normal case, not an exotic one.
    """
    if tomllib is not None:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                declared = tomllib.load(handle).get("project", {}).get("version")
            if declared:
                return str(declared)
    try:
        return package_version("compresso-recsys")
    except PackageNotFoundError:
        return "0.0.0+unknown"


release = _release()
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
nbsphinx_execute = "never"
# Sphinx appends ".txt" to copied sources by default, which would offer the
# notebook as an unrunnable text file. Clearing the suffix makes the page's
# source link a real .ipynb, and the prolog says so above every notebook.
html_sourcelink_suffix = ""
nbsphinx_prolog = """
.. raw:: html

    <div class="admonition tip">
      <p class="admonition-title">Run this notebook</p>
      <p>
        Download it with the <em>View page source</em> link at the top right,
        or from the repository at
        <code>docs/source/{{ env.doc2path(env.docname, base=None)|basename }}</code>.
      </p>
    </div>
"""

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release} documentation"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}
