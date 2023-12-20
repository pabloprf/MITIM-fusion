# Configuration file for the Sphinx documentation builder.

import os
import sys
import sphinx_rtd_theme

# Add your project directory to the sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information
project = "MITIM"
copyright = "2018, Pablo RF"
author = "Pablo Rodriguez-Fernandez"
version = "1.0"  # The short X.Y version
release = "1.0.0"  # The full version, including alpha/beta/rc tags
html_logo = "mitim_logo.png"

# -- General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_panels",
    'sphinxcontrib.images',
]

# Add mappings for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# Templates path for custom pages or layouts
templates_path = ["_templates"]

# The master document
master_doc = "index"

# List of patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"  #'arduino'

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": None,
    "style_external_links": True,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#90EE90",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": -1,
    "includehidden": True,
    "titles_only": False,
}

# html_sidebars = {
#     '**': ['custom_sidebar.html', 'searchbox.html']
# }

html_static_path = [
    "_static"
]
html_css_files = [
    "css/custom.css",
]
html_js_files = [
    'js/open_links_in_new_tab.js',  # Replace with the actual path to your JS file
]

# Only copy the code, not the prompt or output, from code blocks
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Change the tooltip text
copybutton_tooltip_text = "Copy to clipboard"
