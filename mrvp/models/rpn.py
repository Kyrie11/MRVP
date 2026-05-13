"""Deprecated compatibility shim for old RPN imports.

Use :mod:`mrvp.models.rpfn` and ``RecoveryProgramFunnelNetwork`` in new code.
"""
from .rpfn import RPFN, RecoveryProfileNetwork, RecoveryProgramFunnelNetwork, ordering_loss

__all__ = ["RecoveryProgramFunnelNetwork", "RPFN", "RecoveryProfileNetwork", "ordering_loss"]
