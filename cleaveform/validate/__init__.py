"""Module 6 - chunk-mode validation and bounded recovery."""

from .assembly import AssemblyPlan, AssemblyStep, plan_assembly
from .constraints import (
    HARD,
    SOFT,
    WARNING,
    ConstraintReport,
    Finding,
    check_part,
    joint_face_on_bed_warning,
    stability_warning,
)

__all__ = [
    "HARD",
    "SOFT",
    "WARNING",
    "AssemblyPlan",
    "AssemblyStep",
    "ConstraintReport",
    "Finding",
    "check_part",
    "joint_face_on_bed_warning",
    "plan_assembly",
    "stability_warning",
]
