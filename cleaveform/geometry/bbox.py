"""Axis-aligned bounding box helpers and build-volume feasibility tests."""

from __future__ import annotations

import math

import numpy as np
import trimesh

from ..config import N_AXES, Config


def aabb(mesh: trimesh.Trimesh) -> np.ndarray:
    """``[[min_x, min_y, min_z], [max_x, max_y, max_z]]``."""
    return np.asarray(mesh.bounds, dtype=float)


def extents(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = aabb(mesh)
    return bounds[1] - bounds[0]


def axis_ratios(mesh: trimesh.Trimesh, config: Config) -> np.ndarray:
    """Per-axis ``dimension / ceiling_limit``. Values above 1 do not fit."""
    dims = extents(mesh)
    ceiling = np.asarray(config.design_ceiling_mm, dtype=float)
    return dims / ceiling


def fits_ceiling(mesh: trimesh.Trimesh, config: Config) -> bool:
    """True when the AABB is within the design ceiling on every axis."""
    return bool(np.all(axis_ratios(mesh, config) <= 1.0))


def worst_axis(mesh: trimesh.Trimesh, config: Config) -> int:
    """Axis with the greatest ``dimension / ceiling_limit`` ratio.

    Ties resolve to the lowest axis index so the choice is reproducible.
    """
    ratios = axis_ratios(mesh, config)
    return int(np.argmax(ratios))


def max_axis_ratio(mesh: trimesh.Trimesh, config: Config) -> float:
    return float(np.max(axis_ratios(mesh, config)))


def below_min_part_size(mesh: trimesh.Trimesh, config: Config) -> bool:
    """True when any AABB dimension is below the sliver threshold."""
    return bool(np.any(extents(mesh) < config.min_part_mm))


def estimated_part_count(mesh: trimesh.Trimesh, config: Config) -> int:
    """Lower bound on how many ceiling-sized parts this mesh must become.

    Used by the cost function to compare candidate cuts without recursing.
    """
    dims = extents(mesh)
    count = 1
    for axis in range(N_AXES):
        limit = config.ceiling_for_axis(axis)
        count *= max(int(math.ceil(dims[axis] / limit - 1e-9)), 1)
    return max(count, 1)


def aspect_ratio(mesh: trimesh.Trimesh, config: Config) -> float:
    """Longest AABB dimension over the shortest; 1.0 for a cube."""
    dims = np.sort(extents(mesh))
    if dims[0] < config.length_epsilon_mm:
        return math.inf
    return float(dims[-1] / dims[0])
