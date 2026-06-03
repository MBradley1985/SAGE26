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
