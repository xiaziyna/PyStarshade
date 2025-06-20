# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyStarshade'
copyright = '2024, Jamila Taaki'
author = 'Jamila Taaki'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# paths from conf.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pystarshade')))

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples')))


# Add the examples directory to the path
#sys.path.insert(0, os.path.abspath('../examples'))


#import sys
#from pathlib import Path

#sys.path.insert(0, str(Path('..', 'pystarshade').resolve()))

extensions = [    'sphinx.ext.autodoc',	     # To generate autodocs
                   'sphinx.ext.imgmath',
    'nbsphinx',
#    'sphinx.ext.mathjax',           # autodoc with maths
    'sphinx.ext.napoleon',           # For auto-doc configuration
]


napoleon_google_docstring = False   # Turn off googledoc strings
napoleon_numpy_docstring = True     # Turn on numpydoc strings
napoleon_use_ivar = True 	     # For maths symbology
imgmath_image_format = 'svg'
highlight_language = 'python'


#imgmath_latex_preamble = r"""
#\usepackage{amsmath}
#\usepackage{amssymb}
#"""


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_theme = 'alabaster'
html_static_path = ['_static']
