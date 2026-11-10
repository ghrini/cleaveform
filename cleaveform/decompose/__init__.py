"""Modules 4 and 4b - deterministic chunk-mode decomposition and the cut graph."""

from .graph import CutGraph, Interface, build_cut_graph
from .partition import (
    CutPlane,
    DecompositionError,
    DecompositionResult,
    PartNode,
    decompose,
)

__all__ = [
    "CutGraph",
    "CutPlane",
    "DecompositionError",
    "DecompositionResult",
    "Interface",
    "PartNode",
    "build_cut_graph",
    "decompose",
]
