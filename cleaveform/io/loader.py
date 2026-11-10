"""Module 1 - input handling.

Phase 1 accepts a single STL file. Loading applies the unit policy, runs a
repair pass, and reports the detected bounding box so the user can always see
what scale the tool believes the model is at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..config import Config
from ..geometry import repair, units

SUPPORTED_SUFFIXES = (".stl",)


class InputError(ValueError):
    """Raised when the input file cannot be used."""


@dataclass
class LoadedModel:
    mesh: trimesh.Trimesh
    source_path: Path
    raw_extents_mm: tuple[float, float, float]
    final_extents_mm: tuple[float, float, float]
    scale_applied: float
    repaired: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "source": str(self.source_path),
            "raw_bbox_mm": list(self.raw_extents_mm),
            "final_bbox_mm": list(self.final_extents_mm),
            "scale_applied": self.scale_applied,
            "repaired": self.repaired,
            "notes": list(self.notes),
        }


def load_model(
    path: str | Path,
    config: Config,
    input_units: str | None = None,
    target_size_mm: tuple[float, float, float] | None = None,
    scale_factor: float | None = None,
) -> LoadedModel:
    """Load an STL, apply unit overrides, and repair it to a watertight solid.

    Exactly one scale override may be supplied. With none, coordinates are read
    as millimetres and the model must fall inside the plausible size range.
    """
    source = Path(path)
    if not source.exists():
        raise InputError(f"input file not found: {source}")
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InputError(
            f"Phase 1 accepts {', '.join(SUPPORTED_SUFFIXES)} input; got {source.suffix!r}"
        )

    overrides = [input_units is not None, target_size_mm is not None, scale_factor is not None]
    if sum(overrides) > 1:
        raise InputError(
            "supply at most one of --input-units, --target-size, --scale-factor"
        )

    loaded = trimesh.load_mesh(str(source), process=True)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise InputError(f"no mesh geometry found in {source}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise InputError(f"unsupported geometry type in {source}")
    if len(loaded.faces) == 0:
        raise InputError(f"{source} contains no faces")

    raw_extents = tuple(float(v) for v in (loaded.bounds[1] - loaded.bounds[0]))
    notes: list[str] = []

    scale = 1.0
    if input_units is not None:
        scale = units.scale_for_units(input_units)
        notes.append(f"interpreted input coordinates as {input_units}")
    elif scale_factor is not None:
        scale = float(scale_factor)
        notes.append(f"applied explicit scale factor {scale}")
    elif target_size_mm is not None:
        scale = units.scale_to_target(loaded, target_size_mm, config)
        notes.append(f"scaled to fit target size {tuple(target_size_mm)} mm")
    else:
        notes.append("interpreted STL coordinates as millimetres (default policy)")

    mesh = units.apply_scale(loaded, scale) if scale != 1.0 else loaded

    if not any(overrides):
        units.check_plausible(mesh, config)

    mesh = repair.clean(mesh)
    was_watertight = bool(mesh.is_watertight and mesh.is_winding_consistent)
    mesh, ok = repair.ensure_watertight(mesh, config)
    if not ok:
        raise InputError(
            f"{source} could not be repaired to a watertight solid after "
            f"{config.repair_max_attempts} attempt(s); fix the mesh before splitting"
        )
    if not was_watertight:
        notes.append("input was repaired to a watertight solid")

    final_extents = tuple(float(v) for v in (mesh.bounds[1] - mesh.bounds[0]))
    return LoadedModel(
        mesh=mesh,
        source_path=source,
        raw_extents_mm=raw_extents,
        final_extents_mm=final_extents,
        scale_applied=float(scale),
        repaired=not was_watertight,
        notes=notes,
    )


def parse_target_size(text: str | None) -> tuple[float, float, float] | None:
    """Parse a ``X,Y,Z`` millimetre triple from the command line."""
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise InputError("--target-size expects three comma separated values: X,Y,Z")
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise InputError(f"--target-size values must be numbers: {text!r}") from exc
    if any(v <= 0 for v in values):
        raise InputError("--target-size values must be positive")
    return values  # type: ignore[return-value]
