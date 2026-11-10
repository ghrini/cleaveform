"""Module 10 - the decomposition cost function.

Implements the full six-term normalized cost from spec section 10:

    Cost = a*norm_part_count        + b*norm_support_volume
         + g*norm_visible_seam_area + d*norm_joint_count
         + e*norm_print_time        + z*norm_strength_penalty

Phase 1 also carried a ``norm_bbox_feasibility`` term, rewarding cuts that left
less oversize behind. It is gone: with a real part-count term in place it was
actively harmful, steering a 620 mm bar into four parts when three fit. Pieces
are kept inside the ceiling by the recursion itself, which cuts until they fit,
not by any term in the cost.

Every term is dimensionless and lands in [0, 1] before weighting, so the
weights mean what they look like they mean and no raw-magnitude term can
dominate. Lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import Polygon

from .config import N_AXES, Config
from .estimate import print_time_min
from .geometry import analysis, bbox, slicing


@dataclass(frozen=True)
class CutCandidate:
    """One proposed cut, with everything scoring needs already computed.

    Bundled rather than passed as loose arguments because the partitioner has
    already paid for the split and the cross-section, and recomputing either
    inside the cost function would double the most expensive work in the run.
    """

    parent: trimesh.Trimesh
    pieces: list[trimesh.Trimesh]
    axis: int
    polygons: list[Polygon]
    to_3d: np.ndarray

    @property
    def cut_face_area_mm2(self) -> float:
        return float(sum(p.area for p in self.polygons))


@dataclass(frozen=True)
class CostBreakdown:
    """Per-term cost detail, kept for diagnostics and the run report."""

    norm_part_count: float
    norm_visible_seam_area: float
    norm_support_volume: float
    norm_joint_count: float
    norm_print_time: float
    norm_strength_penalty: float
    seam_exposure: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "norm_part_count": self.norm_part_count,
            "norm_visible_seam_area": self.norm_visible_seam_area,
            "norm_support_volume": self.norm_support_volume,
            "norm_joint_count": self.norm_joint_count,
            "norm_print_time": self.norm_print_time,
            "norm_strength_penalty": self.norm_strength_penalty,
            "seam_exposure": self.seam_exposure,
            "total": self.total,
        }


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def baseline_print_time_min(mesh: trimesh.Trimesh, config: Config) -> float:
    """Print time for the whole object as a single piece.

    Hypothetical: the object is normally far too big for the bed. It exists
    only to give ``norm_print_time`` a fixed denominator, so that term compares
    every candidate against the same reference instead of against its own
    parent, which would shift under it at each level of the recursion.
    """
    return print_time_min(mesh, config)


def score_candidate(
    candidate: CutCandidate,
    baseline_time_min: float,
    config: Config,
) -> CostBreakdown:
    """Score one candidate cut. Lower is better."""
    parent = candidate.parent
    pieces = candidate.pieces

    estimated_parts = sum(bbox.estimated_part_count(p, config) for p in pieces)
    norm_part_count = _clamp01(estimated_parts / config.max_parts)

    exposure, norm_visible_seam_area = _seam(candidate, config)
    norm_support_volume = _support(parent, pieces, config)
    norm_joint_count = _joint_count(candidate, estimated_parts)
    norm_print_time = _print_time(pieces, baseline_time_min, config)
    norm_strength_penalty = _strength_penalty(candidate, config)

    total = (
        config.w_part_count * norm_part_count
        + config.w_visible_seam_area * norm_visible_seam_area
        + config.w_support_volume * norm_support_volume
        + config.w_joint_count * norm_joint_count
        + config.w_print_time * norm_print_time
        + config.w_strength_penalty * norm_strength_penalty
    )
    return CostBreakdown(
        norm_part_count=norm_part_count,
        norm_visible_seam_area=norm_visible_seam_area,
        norm_support_volume=norm_support_volume,
        norm_joint_count=norm_joint_count,
        norm_print_time=norm_print_time,
        norm_strength_penalty=norm_strength_penalty,
        seam_exposure=exposure,
        total=float(total),
    )


def _seam(candidate: CutCandidate, config: Config) -> tuple[float, float]:
    """Seam area weighted by how visible that seam actually is.

    Returns ``(exposure, normalized_visible_seam_area)``. The weighting is the
    whole point of this term: two cuts of identical area are not equally bad if
    one runs across an open front face and the other is buried in a recess.
    """
    parent_area = float(candidate.parent.area)
    if parent_area <= config.area_epsilon_mm2:
        return 1.0, 1.0

    points = slicing.seam_points(candidate.polygons, candidate.to_3d, config)
    exposure = analysis.seam_exposure(candidate.parent, points, config)
    weighted = candidate.cut_face_area_mm2 * exposure
    return exposure, _clamp01(weighted / parent_area)


def _support(
    parent: trimesh.Trimesh, pieces: list[trimesh.Trimesh], config: Config
) -> float:
    parent_volume = float(abs(parent.volume))
    if parent_volume <= config.volume_epsilon_mm3:
        return 1.0
    support = sum(analysis.estimate_support_volume(p, config) for p in pieces)
    return _clamp01(support / parent_volume)


def _joint_count(candidate: CutCandidate, estimated_parts: int) -> float:
    """Joints per part, spec section 10.

    A tree decomposition of P parts needs P-1 joints, so this term sits near 1
    for most cuts and does little of the ranking on its own. Where it earns its
    place is a cut whose cross-section breaks into several separate patches:
    each patch needs its own connector, and those extra joints show up here.
    """
    if estimated_parts <= 0:
        return 1.0
    extra_patches = max(len(candidate.polygons) - 1, 0)
    joints = max(estimated_parts - 1, 0) + extra_patches
    return _clamp01(joints / estimated_parts)


def _print_time(
    pieces: list[trimesh.Trimesh], baseline_time_min: float, config: Config
) -> float:
    """Print-time overhead the cut introduces, as a fraction of the baseline.

    Spec section 10 writes this as ``est_print_time / baseline``. Taken
    literally that ratio always exceeds 1 for a real split, because more pieces
    means more layers and more per-layer overhead, so it would sit pinned near
    the top of its range and resolve almost nothing. Subtracting the baseline
    keeps the dimensionless intent while measuring the part that actually
    varies between candidates: the overhead, where 0 is free and 1 is a
    doubling.
    """
    if baseline_time_min <= 0.0:
        return 0.0
    total = sum(print_time_min(p, config) for p in pieces)
    return _clamp01((total / baseline_time_min) - 1.0)


def _strength_penalty(candidate: CutCandidate, config: Config) -> float:
    """How weak the joint this cut would create, from 0 (solid) to 1.

    Compares the cut's cross-section against the bounding-box slice it sits in.
    Cutting an hourglass at the waist leaves little material to bond and scores
    near 1; cutting the same object through a full-width section scores near 0.
    This is the term that stops the optimizer picking the narrowest,
    cheapest-looking cut and handing back parts that snap at the joint.
    """
    extents = bbox.extents(candidate.parent)
    cross_section = 1.0
    for axis in range(N_AXES):
        if axis != candidate.axis:
            cross_section *= float(extents[axis])
    if cross_section <= config.area_epsilon_mm2:
        return 1.0
    fill_ratio = candidate.cut_face_area_mm2 / cross_section
    return _clamp01(1.0 - fill_ratio)
