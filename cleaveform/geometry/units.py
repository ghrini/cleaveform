"""Unit and scale management.

STL files carry no units, so Cleaveform treats coordinates as millimetres by
default, always reports the detected bounding box, and refuses implausible
models unless the user supplies an explicit override (spec module 1).
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..config import N_AXES, Config

UNIT_SCALE_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
}


class UnitError(ValueError):
    """Raised when a model's size is implausible and no override was given."""


def scale_for_units(units: str) -> float:
    key = units.strip().lower()
    if key not in UNIT_SCALE_TO_MM:
        raise UnitError(
            f"unsupported unit {units!r}; expected one of {sorted(UNIT_SCALE_TO_MM)}"
        )
    return UNIT_SCALE_TO_MM[key]


def apply_scale(mesh: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    if factor <= 0:
        raise UnitError("scale factor must be positive")
    out = mesh.copy()
    out.apply_scale(factor)
    return out


def scale_to_target(mesh: trimesh.Trimesh, target_mm, config: Config) -> float:
    """Uniform factor that fits the mesh inside ``target_mm`` on every axis."""
    target = np.asarray(target_mm, dtype=float)
    if target.shape != (N_AXES,) or np.any(target <= 0):
        raise UnitError("target size must be three positive dimensions")
    extents = mesh.bounds[1] - mesh.bounds[0]
    safe = np.maximum(extents, config.length_epsilon_mm)
    return float(np.min(target / safe))


def check_plausible(mesh: trimesh.Trimesh, config: Config) -> None:
    """Raise :class:`UnitError` when the model is implausibly tiny or huge."""
    extents = mesh.bounds[1] - mesh.bounds[0]
    largest = float(np.max(extents))
    if largest < config.min_plausible_dim_mm:
        raise UnitError(
            f"model is only {largest:.4f} mm across, below the plausible minimum of "
            f"{config.min_plausible_dim_mm} mm. Pass --input-units, --target-size or "
            "--scale-factor if this is intentional."
        )
    if largest > config.max_plausible_dim_mm:
        raise UnitError(
            f"model is {largest:.1f} mm across, above the plausible maximum of "
            f"{config.max_plausible_dim_mm} mm. Pass --input-units, --target-size or "
            "--scale-factor if this is intentional."
        )


def describe_bbox(mesh: trimesh.Trimesh) -> str:
    extents = mesh.bounds[1] - mesh.bounds[0]
    return f"{extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm"
