import os

project = "SAGE26"
copyright = "2026, Michael Bradley, Darren Croton"
author = "Michael Bradley, Darren Croton"
release = "2026"

extensions = [
    "myst_parser",
    "breathe",
    "sphinx_rtd_theme",
]

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# developer/*.md is contributor documentation, not user-facing -- keep it on
# GitHub but exclude it from the rendered Sphinx site so it does not warn
# about missing toctree entries or stale cross-references between files
# that have been deleted as part of the cleanup pass.
exclude_patterns = ["developer/*.md", "_build", "_doxygen"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
}
html_static_path = []

# -- Breathe (Doxygen bridge) ------------------------------------------------
_docs_dir = os.path.dirname(__file__)
breathe_projects = {"SAGE26": os.path.join(_docs_dir, "_doxygen", "xml")}
breathe_default_project = "SAGE26"
breathe_default_members = ("members", "undoc-members")

# Several physics modules define identical file-private static const doubles
# (e.g. FIRE_V_CRIT_KMS, KD11_METAL_HALO_MASS). The C language treats these
# as distinct -- file scope makes them private to each translation unit --
# but breathe lifts them into a single namespace and warns. Suppress.
suppress_warnings = ["duplicate_declaration.cpp"]
