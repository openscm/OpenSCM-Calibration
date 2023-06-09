# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenSCM-Calibration"
# put the authors in their own variable, so they can be reused later
authors = ", ".join(["Zeb Nicholls"])
author = authors
# add a copyright year variable, we can extend this over time in future as
# needed
copyright_year = "2023"
copyright = "{}, {}".format(copyright_year, authors)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # create documentation automatically from source code
    # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    "sphinx.ext.autodoc",
    # automatic summary
    "sphinx.ext.autosummary",
    # TODO: think about using autosummary instead rather than needing to manually maintain the reference lists
    # https://stackoverflow.com/questions/48074094/use-sphinx-autosummary-recursively-to-generate-api-documentation
    # tell sphinx that we're using numpy style docstrings
    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    "sphinx.ext.napoleon",
    # add support for type hints too (so type hints are included next to
    # argument and return types in docs)
    # https://github.com/tox-dev/sphinx-autodoc-typehints
    # this must come after napoleon
    # in the list for things to work properly
    # https://github.com/tox-dev/sphinx-autodoc-typehints#compatibility-with-sphinxextnapoleon
    "sphinx_autodoc_typehints",
    # jupytext rendered notebook support
    "myst_nb",
    # links to other docs
    "sphinx.ext.intersphinx",
    # add source code to docs
    "sphinx.ext.viewcode",
    # add copy code button to code examples
    "sphinx_copybutton",
]

# general sphinx settings
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Don't include module names in object names (can also be left on,
# depends a bit how your project is structured and what you prefer)
add_module_names = False
# Other global settings which we've never used but are included by default
templates_path = ["_templates"]
exclude_patterns = []

# configure default settings for autodoc directives
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directives
autodoc_default_options = {
    # Automatically document members of classes/modules/exceptions
    "members": True,
    # Automatically document members of classes/modules/exceptions
    # that are inherited from base classes
    "inherited-members": True,
    # Include members even if they don't have docstrings
    "undoc-members": True,
    # Don't document private members
    "private-members": False,
    # Show the inheritance of classes
    "show-inheritance": True,
    # put members of classes in order as they appear in source
    "member-order": "bysource",
}

# autosummary
autosummary_generate = True

# napoleon extension settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
# We don't use google docstrings
napoleon_google_docstring = False
# We use numpy style docstrings
napoleon_numpy_docstring = True
# Don't use separate rtype for the return documentation
napoleon_use_rtype = False

# autodoc type hints settings
# https://github.com/tox-dev/sphinx-autodoc-typehints
# include full name of classes when expanding type hints?
typehints_fully_qualified = True
# Add rtype directive if needed
typehints_document_rtype = True
# Put the return type as part of the return documentation
typehints_use_rtype = False

# Left-align maths equations
mathjax3_config = {"chtml": {"displayAlign": "center"}}

# myst-nb
myst_enable_extensions = ["amsmath", "dollarmath"]
nb_execution_mode = "auto"
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = 120

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Pick your theme for html output, we typically use the read the docs theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# Ignore ipynb files
def setup(app):
    # Ignore .ipynb files (see https://github.com/executablebooks/MyST-NB/issues/363).
    app.registry.source_suffix.pop(".ipynb", None)


# Intersphinx mapping
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyam": ("https://pyam-iamc.readthedocs.io/en/latest", None),
    "scmdata": ("https://scmdata.readthedocs.io/en/latest", None),
    "xarray": ("http://xarray.pydata.org/en/stable", None),
    "pint": (
        "https://pint.readthedocs.io/en/latest",
        None,
    ),
}
