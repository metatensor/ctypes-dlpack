import os
from datetime import datetime

import tomllib

import ctypes_dlpack


ROOT = os.path.abspath(os.path.join("..", ".."))


master_doc = "index"

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as file:
    project_dict = tomllib.load(file)["project"]

project = project_dict["name"]
author = ", ".join(author["name"] for author in project_dict["authors"])
copyright = f"{datetime.now().date().year}, {author}"
release = ctypes_dlpack.__version__


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dlpack": ("https://dmlc.github.io/dlpack/latest/", None),
}


html_title = "ctypes-dlpack"
html_theme = "furo"
