"""Sphinx configuration for the Compresso Recsys documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version


sys.path.insert(0, os.path.abspath("../../src"))

project = "Compresso Recsys"
author = "Compresso contributors"

try:
    release = package_version("compresso-recsys")
except PackageNotFoundError:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

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
