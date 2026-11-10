"""Robust boolean operations.

``manifold3d`` is the required backend for Phase 1. Boolean failures are the
main technical risk in the pipeline, so every call funnels through here and
raises a single, recognisable exception type.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..config import Config

BOOLEAN_ENGINE = "manifold"


class BooleanError(RuntimeError):
    """Raised when the boolean backend cannot produce a valid result."""


def _check_engine() -> None:
    available = trimesh.boolean.engines_available
    if BOOLEAN_ENGINE not in available:
        raise BooleanError(
            f"boolean engine {BOOLEAN_ENGINE!r} is unavailable; install manifold3d"
        )


def union(meshes: list[trimesh.Trimesh], config: Config) -> trimesh.Trimesh:
    """Union a list of meshes. Order is preserved for determinism."""
    if not meshes:
        raise BooleanError("union requires at least one mesh")
    if len(meshes) == 1:
        return meshes[0].copy()
    _check_engine()
    try:
        result = trimesh.boolean.union(meshes, engine=BOOLEAN_ENGINE)
    except Exception as exc:  # pragma: no cover - backend specific
        raise BooleanError(f"union failed: {exc}") from exc
    return _validated(result, config, "union")


def difference(
    base: trimesh.Trimesh, cutters: list[trimesh.Trimesh], config: Config
) -> trimesh.Trimesh:
    """Subtract every cutter from ``base``."""
    if not cutters:
        return base.copy()
    _check_engine()
    try:
        result = trimesh.boolean.difference([base, *cutters], engine=BOOLEAN_ENGINE)
    except Exception as exc:  # pragma: no cover - backend specific
        raise BooleanError(f"difference failed: {exc}") from exc
    return _validated(result, config, "difference")


def intersection(
    meshes: list[trimesh.Trimesh], config: Config
) -> trimesh.Trimesh | None:
    """Intersect meshes, returning ``None`` when the result is empty."""
    if len(meshes) < 2:
        raise BooleanError("intersection requires at least two meshes")
    _check_engine()
    try:
        result = trimesh.boolean.intersection(meshes, engine=BOOLEAN_ENGINE)
    except Exception as exc:  # pragma: no cover - backend specific
        raise BooleanError(f"intersection failed: {exc}") from exc
    if result is None or len(result.faces) == 0:
        return None
    if abs(volume_of(result)) < config.volume_epsilon_mm3:
        return None
    return result


def volume_of(mesh: trimesh.Trimesh) -> float:
    """Mesh volume, tolerating the degenerate slivers booleans can produce.

    Two solids meeting exactly at a face can intersect in a zero-volume shell;
    computing its centre of mass divides by that zero. The volume itself is
    still well defined and is all this needs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(mesh.volume)


def intersection_volume(
    a: trimesh.Trimesh, b: trimesh.Trimesh, config: Config
) -> float:
    """Volume shared by two meshes; ``0.0`` when they do not overlap.

    A cheap AABB rejection runs first so the common non-overlapping case never
    reaches the boolean backend.
    """
    if not _bounds_overlap(a.bounds, b.bounds):
        return 0.0
    result = intersection([a, b], config)
    if result is None:
        return 0.0
    return abs(volume_of(result))


def _bounds_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


def _validated(
    result: trimesh.Trimesh | None, config: Config, op: str
) -> trimesh.Trimesh:
    if result is None or len(result.faces) == 0:
        raise BooleanError(f"{op} produced empty geometry")
    if abs(volume_of(result)) < config.volume_epsilon_mm3:
        raise BooleanError(f"{op} produced degenerate geometry")
    return result
