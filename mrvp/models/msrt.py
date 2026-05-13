"""Deprecated compatibility shim for old MSRT imports.

Use :mod:`mrvp.models.cmrt` and ``CounterfactualMotionResetTokenizer`` in new
code.  ``MSRT`` remains an alias so legacy checkpoints and scripts can load.
"""
from .cmrt import CMRT, MSRT, CounterfactualMotionResetTokenizer

__all__ = ["CounterfactualMotionResetTokenizer", "CMRT", "MSRT"]
