"""Module 0 - the geometry kernel.

Every geometry-touching module in Cleaveform depends on this package rather
than calling mesh libraries directly. Any geometry utility a second module
needs belongs here.
"""

from . import analysis, bbox, booleans, collision, repair, slicing, solids, units, voxel
from .booleans import BooleanError
from .units import UnitError

__all__ = [
    "analysis",
    "bbox",
    "booleans",
    "collision",
    "repair",
    "slicing",
    "solids",
    "units",
    "voxel",
    "BooleanError",
    "UnitError",
]
