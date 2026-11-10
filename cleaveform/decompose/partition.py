"""Modules 4 and 4b - chunk-mode decomposition.

This is the deterministic reference algorithm from the implementation directive
section 5. Given the same mesh and the same config it always produces the same
cuts, in the same order, with the same part identifiers.

The interface is deliberately narrow so a stronger optimizer (beam search, ILP,
or a learned policy) can replace :func:`decompose` later without any neighbouring
module changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..config import AXIS_NAMES, N_AXES, Config
from ..cost import CostBreakdown, CutCandidate, baseline_print_time_min, score_candidate
from ..geometry import bbox, repair, slicing

ROOT_PATH = "root"


class DecompositionError(RuntimeError):
    """Raised when no valid decomposition exists within the configured bounds."""

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


@dataclass(frozen=True)
class CutPlane:
    """A chosen cut, in world coordinates."""

    axis: int
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    fraction: float
    candidate_index: int
    face_area_mm2: float

    @property
    def axis_name(self) -> str:
        return AXIS_NAMES[self.axis]

    def to_dict(self) -> dict:
        return {
            "axis": self.axis_name,
            "origin_mm": list(self.origin),
            "normal": list(self.normal),
            "fraction": self.fraction,
            "candidate_index": self.candidate_index,
            "face_area_mm2": self.face_area_mm2,
        }


@dataclass
class PartNode:
    """One node of the binary partition tree."""

    path: str
    mesh: trimesh.Trimesh
    depth: int
    cut: CutPlane | None = None
    children: list["PartNode"] = field(default_factory=list)
    part_id: str | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> list["PartNode"]:
        if self.is_leaf:
            return [self]
        out: list[PartNode] = []
        for child in self.children:
            out.extend(child.leaves())
        return out

    def internal_nodes(self) -> list["PartNode"]:
        if self.is_leaf:
            return []
        out = [self]
        for child in self.children:
            out.extend(child.internal_nodes())
        return out


@dataclass
class Candidate:
    """A scored candidate cut, before selection."""

    index: int
    fraction: float
    axis: int
    origin: np.ndarray
    normal: np.ndarray
    pieces: list[trimesh.Trimesh]
    face_area_mm2: float
    cost: CostBreakdown


@dataclass
class DecompositionResult:
    root: PartNode
    parts: list[PartNode]
    warnings: list[str]
    cut_log: list[dict]

    @property
    def part_count(self) -> int:
        return len(self.parts)


def decompose(
    mesh: trimesh.Trimesh,
    config: Config,
    blacklist: set[tuple[str, int]] | None = None,
) -> DecompositionResult:
    """Partition ``mesh`` into pieces that each fit the design ceiling.

    ``blacklist`` holds ``(node_path, candidate_index)`` pairs that bounded
    recovery has already ruled out; those candidates are skipped so a retry
    explores a genuinely different decomposition.
    """
    blacklist = blacklist or set()
    warnings: list[str] = []
    cut_log: list[dict] = []
    state = {"leaf_count": 0}

    # Cut against the reduced ceiling so parts still fit once pegs are added.
    cut_config = config.with_overrides(
        design_ceiling_mm=config.decomposition_ceiling_mm
    )

    # One fixed denominator for the print-time term, computed on the whole
    # object before any cutting, so every candidate at every depth is measured
    # against the same reference.
    state["baseline_time_min"] = baseline_print_time_min(mesh, cut_config)

    root = PartNode(path=ROOT_PATH, mesh=mesh, depth=0)
    _partition_node(root, cut_config, blacklist, warnings, cut_log, state)

    parts = sorted(root.leaves(), key=lambda node: _path_sort_key(node.path))
    width = len(str(max(len(parts), 1)))
    for index, node in enumerate(parts, start=1):
        node.part_id = f"part_{index:0{max(width, 3)}d}"

    return DecompositionResult(
        root=root, parts=parts, warnings=warnings, cut_log=cut_log
    )


def _partition_node(
    node: PartNode,
    config: Config,
    blacklist: set[tuple[str, int]],
    warnings: list[str],
    cut_log: list[dict],
    state: dict,
) -> None:
    # Step 1 and 2: a part that already fits the ceiling is final.
    if bbox.fits_ceiling(node.mesh, config):
        state["leaf_count"] += 1
        _check_part_budget(node, config, state)
        return

    if node.depth >= config.max_cut_depth:
        raise DecompositionError(
            f"cut depth limit ({config.max_cut_depth}) reached at {node.path} while the "
            f"part still measures {_extent_text(node.mesh)}",
            {"path": node.path, "limit": "max_cut_depth", "value": config.max_cut_depth},
        )

    # Step 3: cut across the axis that overshoots its ceiling by the most.
    axis = bbox.worst_axis(node.mesh, config)
    candidates = _build_candidates(node, axis, config, warnings, state)

    allowed = [c for c in candidates if (node.path, c.index) not in blacklist]
    if not allowed:
        raise DecompositionError(
            f"no valid cut of {node.path} along {AXIS_NAMES[axis]} survived validation "
            f"(part measures {_extent_text(node.mesh)})",
            {
                "path": node.path,
                "axis": AXIS_NAMES[axis],
                "candidates_generated": len(candidates),
                "candidates_blacklisted": len(candidates) - len(allowed),
            },
        )

    # Steps 6 and 7: lowest cost wins; ties break to the lowest candidate index.
    best = min(allowed, key=lambda c: (c.cost.total, c.index))

    node.cut = CutPlane(
        axis=axis,
        origin=tuple(float(v) for v in best.origin),
        normal=tuple(float(v) for v in best.normal),
        fraction=best.fraction,
        candidate_index=best.index,
        face_area_mm2=best.face_area_mm2,
    )
    cut_log.append(
        {
            "path": node.path,
            "depth": node.depth,
            **node.cut.to_dict(),
            "cost": best.cost.to_dict(),
            "candidates_considered": len(allowed),
        }
    )

    # Step 8: recurse on each piece. Index 0 is the negative side of the normal.
    for child_index, piece in enumerate(best.pieces):
        child = PartNode(
            path=f"{node.path}.{child_index}", mesh=piece, depth=node.depth + 1
        )
        node.children.append(child)
    for child in node.children:
        _partition_node(child, config, blacklist, warnings, cut_log, state)


def _build_candidates(
    node: PartNode, axis: int, config: Config, warnings: list[str], state: dict
) -> list[Candidate]:
    """Steps 4 to 6: generate, filter and score the candidate cut planes."""
    bounds = bbox.aabb(node.mesh)
    low = float(bounds[0][axis])
    span = float(bounds[1][axis] - bounds[0][axis])
    normal = np.zeros(N_AXES)
    normal[axis] = 1.0

    candidates: list[Candidate] = []
    for index, fraction in enumerate(config.cut_fractions):
        origin = np.asarray(node.mesh.centroid, dtype=float).copy()
        origin[axis] = low + span * fraction

        negative, positive = slicing.split_by_plane(node.mesh, origin, normal, config)
        if negative is None or positive is None:
            continue
        pieces = [negative, positive]
        if any(bbox.below_min_part_size(p, config) for p in pieces):
            continue

        for side, piece in zip(("negative", "positive"), pieces):
            bodies = repair.connected_components(piece, config)
            if len(bodies) > 1:
                warnings.append(
                    f"cut at {fraction:.0%} of {AXIS_NAMES[axis]} on {node.path} leaves "
                    f"{len(bodies)} disconnected bodies on the {side} side; they stay "
                    "grouped as one part"
                )

        polygons, to_3d = slicing.cross_section_polygons(
            node.mesh, origin, normal, config
        )
        face_area = float(sum(p.area for p in polygons))
        cost = score_candidate(
            CutCandidate(
                parent=node.mesh,
                pieces=pieces,
                axis=axis,
                polygons=polygons,
                to_3d=to_3d,
            ),
            state["baseline_time_min"],
            config,
        )
        candidates.append(
            Candidate(
                index=index,
                fraction=float(fraction),
                axis=axis,
                origin=origin,
                normal=normal,
                pieces=pieces,
                face_area_mm2=face_area,
                cost=cost,
            )
        )
    return candidates


def _check_part_budget(node: PartNode, config: Config, state: dict) -> None:
    if state["leaf_count"] > config.max_parts:
        raise DecompositionError(
            f"part limit ({config.max_parts}) exceeded at {node.path}",
            {"path": node.path, "limit": "max_parts", "value": config.max_parts},
        )


def _extent_text(mesh: trimesh.Trimesh) -> str:
    extents = bbox.extents(mesh)
    return f"{extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm"


def _path_sort_key(path: str) -> tuple:
    """Sort tree paths depth-first in child order, not lexicographically."""
    head, _, rest = path.partition(".")
    if not rest:
        return (head,)
    return (head, *(int(token) for token in rest.split(".")))
