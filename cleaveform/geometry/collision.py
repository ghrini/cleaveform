"""Interference and insertion-sweep tests.

Phase 1 checks assembly feasibility by sampling the straight-line insertion
path of a part and measuring how much it overlaps the parts already placed.
An axis-aligned bounding box rejection runs before every boolean, so the common
non-overlapping sample costs almost nothing.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..config import Config
from . import booleans


def overlaps(a: trimesh.Trimesh, b: trimesh.Trimesh, config: Config) -> float:
    """Shared volume between two solids in mm^3."""
    return booleans.intersection_volume(a, b, config)


def sweep_is_clear(
    moving: list[trimesh.Trimesh],
    fixed: list[trimesh.Trimesh],
    direction,
    config: Config,
) -> tuple[bool, float]:
    """Test whether a group of parts can slide into place along ``direction``.

    The group is walked rigidly from a retracted position back to its seated
    pose. Returns ``(is_clear, worst_overlap_mm3)``. The seated pose itself is
    excluded because mating faces touch there by construction.
    """
    if not moving or not fixed:
        return True, 0.0
    axis = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < config.length_epsilon_mm:
        return True, 0.0
    axis = axis / norm

    lower = np.min([m.bounds[0] for m in moving], axis=0)
    upper = np.max([m.bounds[1] for m in moving], axis=0)
    travel = float(np.linalg.norm(upper - lower)) * config.sweep_distance_factor
    samples = max(config.sweep_samples, 1)

    worst = 0.0
    for index in range(samples):
        # Walk from fully retracted toward, but not reaching, the seated pose.
        fraction = (samples - index) / samples
        offset = -axis * (travel * fraction)
        total = 0.0
        for part in moving:
            shifted = part.copy()
            shifted.apply_translation(offset)
            for other in fixed:
                total += overlaps(shifted, other, config)
        worst = max(worst, total)
        if total > config.collision_volume_tolerance_mm3:
            return False, worst
    return True, worst
