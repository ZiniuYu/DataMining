# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'DataMining'
copyright = '2021, Ziniu Yu'
author = 'Ziniu Yu'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.mathjax'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for LaTeX output ------------------------------------------------
rst_prolog = r'''
.. math::
    
    \newcommand{\bs}{\boldsymbol}
    \newcommand{\dp}{\displaystyle}
    \newcommand{\rm}{\mathrm}
    \newcommand{\cl}{\mathcal}
    \newcommand{\pd}{\partial}
    
    \newcommand{\cd}{\cdot}
    \newcommand{\cds}{\cdots}
    \newcommand{\dds}{\ddots}
    \newcommand{\lag}{\langle}
    \newcommand{\lv}{\lVert}
    \newcommand{\ol}{\overline}
    \newcommand{\od}{\odot}
    \newcommand{\ra}{\rightarrow}
    \newcommand{\rag}{\rangle}
    \newcommand{\rv}{\rVert}
    \newcommand{\seq}{\subseteq}
    \newcommand{\td}{\tilde}
    \newcommand{\vds}{\vdots}
    \newcommand{\wh}{\widehat}

    \newcommand{\0}{\boldsymbol{0}}
    \newcommand{\1}{\boldsymbol{1}}
    \newcommand{\a}{\boldsymbol{\mathrm{a}}}
    \newcommand{\b}{\boldsymbol{\mathrm{b}}}
    \newcommand{\c}{\boldsymbol{\mathrm{c}}}
    \newcommand{\d}{\boldsymbol{\mathrm{d}}}
    \newcommand{\e}{\boldsymbol{\mathrm{e}}}
    \newcommand{\f}{\boldsymbol{\mathrm{f}}}
    \newcommand{\g}{\boldsymbol{\mathrm{g}}}
    \newcommand{\h}{\boldsymbol{\mathrm{h}}}
    \newcommand{\i}{\boldsymbol{\mathrm{i}}}
    \newcommand{\j}{\boldsymbol{j}}
    \newcommand{\m}{\boldsymbol{\mathrm{m}}}
    \newcommand{\n}{\boldsymbol{\mathrm{n}}}
    \newcommand{\o}{\boldsymbol{\mathrm{o}}}
    \newcommand{\p}{\boldsymbol{\mathrm{p}}}
    \newcommand{\q}{\boldsymbol{\mathrm{q}}}
    \newcommand{\r}{\boldsymbol{\mathrm{r}}}
    \newcommand{\u}{\boldsymbol{\mathrm{u}}}
    \newcommand{\v}{\boldsymbol{\mathrm{v}}}
    \newcommand{\w}{\boldsymbol{\mathrm{w}}}
    \newcommand{\x}{\boldsymbol{\mathrm{x}}}
    \newcommand{\y}{\boldsymbol{\mathrm{y}}}
    \newcommand{\z}{\boldsymbol{\mathrm{z}}}

    \newcommand{\A}{\boldsymbol{\mathrm{A}}}
    \newcommand{\B}{\boldsymbol{\mathrm{B}}}
    \newcommand{\C}{\boldsymbol{\mathrm{C}}}
    \newcommand{\D}{\boldsymbol{\mathrm{D}}}
    \newcommand{\H}{\boldsymbol{\mathrm{H}}}
    \newcommand{\I}{\boldsymbol{\mathrm{I}}}
    \newcommand{\K}{\boldsymbol{\mathrm{K}}}
    \newcommand{\M}{\boldsymbol{\mathrm{M}}}
    \newcommand{\N}{\boldsymbol{\mathrm{N}}}
    \newcommand{\P}{\boldsymbol{\mathrm{P}}}
    \newcommand{\Q}{\boldsymbol{\mathrm{Q}}}
    \newcommand{\S}{\boldsymbol{\mathrm{S}}}
    \newcommand{\U}{\boldsymbol{\mathrm{U}}}
    \newcommand{\W}{\boldsymbol{\mathrm{W}}}
    \newcommand{\X}{\boldsymbol{\mathrm{X}}}
    \newcommand{\Y}{\boldsymbol{\mathrm{Y}}}
    \newcommand{\Z}{\boldsymbol{\mathrm{Z}}}

    \newcommand{\R}{\mathbb{R}}

    \newcommand{\cE}{\mathcal{E}}
    \newcommand{\cX}{\mathcal{X}}
    \newcommand{\cY}{\mathcal{Y}}

    \newcommand{\ld}{\lambda}
    \newcommand{\Ld}{\boldsymbol{\mathrm{\Lambda}}}
    \newcommand{\sg}{\sigma}
    \newcommand{\Sg}{\boldsymbol{\mathrm{\Sigma}}}
    \newcommand{\th}{\theta}
    \newcommand{\ve}{\varepsilon}

    \newcommand{\mmu}{\boldsymbol{\mu}}
    \newcommand{\ppi}{\boldsymbol{\pi}}
    \newcommand{\CC}{\mathcal{C}}
    \newcommand{\TT}{\mathcal{T}}


    \newcommand{\bb}{\begin{bmatrix}}
    \newcommand{\eb}{\end{bmatrix}}
    \newcommand{\bp}{\begin{pmatrix}}
    \newcommand{\ep}{\end{pmatrix}}
    \newcommand{\bv}{\begin{vmatrix}}
    \newcommand{\ev}{\end{vmatrix}}

    \newcommand{\im}{^{-1}}
    \newcommand{\pr}{^{\prime}}
    \newcommand{\ppr}{^{\prime\prime}}
'''