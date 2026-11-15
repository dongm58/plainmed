"""
plainmed/__init__.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Marks plainmed/ as a Python package so scripts, notebooks and tests can import it,
and hides one harmless warning: textstat 0.7.3 imports pkg_resources, which
setuptools marks as deprecated (setuptools is pinned to a version where it still works).

Input: none
Output: none
"""

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
