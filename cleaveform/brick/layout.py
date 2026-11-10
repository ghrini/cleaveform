"""The finished brick layout, and whether it will actually hold together.

Bricks connect vertically only. Two bricks side by side in a course have no
connection to each other at all; what binds them is a brick in the course above
overlapping both. Everything measured here follows from that one fact:

``bond_fraction``
    Fraction of bricks that sit on two or more distinct bricks below. A brick
    resting on exactly one brick ties nothing together, so this - not brick
    count, not seam count - is the number that says whether the courses are
    interlocked or merely stacked.
``seam_alignment_fraction``
    How often a seam in one course sits directly above a seam in the course
    below. Aligned seams are a split line through the model. Lower is better.
``weakest_link_cells``
    The fewest vertically-connected cell pairs across any horizontal plane. The
    direct analogue of chunk mode's minimum joint face area: it names the plane
    where the model will snap.

A model whose bricks form more than one connected component cannot be built as
a single object, so that is a hard failure rather than a warning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import Config
from ..validate.constraints import HARD, SOFT, WARNING, Finding
from .occupancy import OccupancyGrid
from .simplify import Changes, simplify_grid
from .tiling import BrickPlacement, tile_grid, tile_layer, TilingError
from .library import load_library

EMPTY = -1
# Findings list a sample rather than every offending brick; a few hundred ids
# in a diagnostic helps nobody.
MAX_REPORTED_BRICKS = 20


@dataclass
class BrickLayout:
    """Bricks placed on the grid, plus the structural verdict on them."""

    placements: list[BrickPlacement]
    grid: OccupancyGrid
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # Size in bricks of every connected piece, largest first. The first entry is
    # the main body; everything after it is a stray that nothing holds on.
    piece_sizes: list[int] = field(default_factory=list)
    # For each placement, the index of the piece it belongs to: 0 is the main
    # body, anything else is a stray. Lets a picture colour the strays.
    piece_of: list[int] = field(default_factory=list)
    # What simplification changed about the shape, if it ran (see simplify.py).
    # Empty when the model was tiled exactly as voxelized.
    changes: dict = field(default_factory=dict)

    @property
    def brick_count(self) -> int:
        return len(self.placements)

    @property
    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for placement in self.placements:
            counts[placement.type_id] = counts.get(placement.type_id, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def used_type_ids(self) -> set[str]:
        return {p.type_id for p in self.placements}

    @property
    def ok(self) -> bool:
        return not [f for f in self.findings if f.tier == HARD]

    @property
    def piece_count(self) -> int:
        return len(self.piece_sizes) or 1

    def change_summary(self) -> str:
        """One line on how the shape was altered, or an empty string if it was not."""
        c = self.changes
        parts = []
        if self.grid.added_thin:
            parts.append(f"kept {self.grid.added_thin} thin-feature cell(s)")
        if c and any(
            c.get(k) for k in ("dropped_cells", "thickened_cells", "stitched_cells")
        ):
            parts.append(
                Changes(
                    dropped=c.get("dropped_cells", 0),
                    thickened=c.get("thickened_cells", 0),
                    stitched=c.get("stitched_cells", 0),
                ).summary()
            )
        return ", ".join(parts)

    def piece_summary(self) -> str:
        """One line saying whether this builds as a single object.

        This is the headline result, so it is a sentence rather than a metric
        to dig out of the manifest: a model that builds as nine pieces and one
        that builds as one should not look alike in the output.
        """
        if self.piece_count <= 1:
            return "1 piece"
        strays = self.piece_sizes[1:]
        shown = ", ".join(str(n) for n in strays[:6])
        more = ", ..." if len(strays) > 6 else ""
        return (
            f"{self.piece_count} pieces: a main body of {self.piece_sizes[0]} bricks "
            f"and {len(strays)} stray group(s) of {shown}{more} brick(s)"
        )

    def to_dict(self) -> dict:
        return {
            "brick_count": self.brick_count,
            "type_count": len(self.counts_by_type),
            "counts_by_type": self.counts_by_type,
            "grid": self.grid.to_dict(),
            "metrics": dict(sorted(self.metrics.items())),
            "piece_count": self.piece_count,
            "piece_sizes": list(self.piece_sizes),
            "shape_changes": dict(self.changes),
            "placements": [p.to_dict() for p in self.placements],
        }


def build_layout(grid: OccupancyGrid, config: Config) -> BrickLayout:
    """Tile the grid, bridge whatever the tiler left split, then analyse.

    With ``brick_simplify`` on, a layout that still has stray bricks is given a
    few passes of minimal shape changes and tiled again; see ``simplify.py``.
    """
    layout = _tile_and_analyse(grid, config)
    if not config.brick_simplify or layout.piece_count <= 1:
        return layout
    return _simplified(layout, config)


def _tile_and_analyse(grid: OccupancyGrid, config: Config) -> BrickLayout:
    placements = tile_grid(grid, config)
    placements = repair_connectivity(placements, grid, config)
    return analyse(placements, grid, config)


def _badness(layout: BrickLayout) -> tuple[int, int]:
    """Smaller is better: bricks stranded, then pieces."""
    return (int(layout.metrics["stray_brick_count"]), layout.piece_count)


def _simplified(layout: BrickLayout, config: Config) -> BrickLayout:
    """Change the shape minimally and re-tile, keeping the best result found.

    Each pass uses the current strays as feedback. Some repairs take two steps -
    a flat rail is thickened, and only then does it touch the body in a way that
    can be stitched - so a pass that does not yet reduce the strays is not a
    reason to stop. The chain of passes is followed to its end and the best layout
    seen is kept, which is the original if no pass ever helped, so simplification
    can never leave a model worse attached than it found it.
    """
    best, best_total = layout, Changes()
    current, total = layout, Changes()
    for _ in range(config.brick_simplify_passes):
        if current.piece_count <= 1:
            break
        new_grid, step = simplify_grid(
            current.grid, current.placements, current.piece_of, config
        )
        if step.empty:
            break
        try:
            candidate = _tile_and_analyse(new_grid, config)
        except TilingError:
            break
        total += step
        total.passes += 1
        current = candidate
        if _badness(current) < _badness(best):
            best, best_total = current, replace(total)
    best.changes = best_total.to_dict()
    return add_change_finding(best)


def add_change_finding(layout: BrickLayout) -> BrickLayout:
    """Record in the findings that the shape was altered, so it is never silent."""
    summary = layout.change_summary()
    if summary:
        layout.findings.append(
            Finding(
                WARNING,
                "shape_simplified",
                f"the shape was changed so every brick could be held: {summary}. "
                "Compare against the original if that matters for this model",
                None,
                dict(layout.changes),
            )
        )
    return layout


def repair_connectivity(
    placements: list[BrickPlacement], grid: OccupancyGrid, config: Config
) -> list[BrickPlacement]:
    """Bridge disconnected groups by forcing a brick to span across a split.

    Course-offset heuristics get most of the way but cannot guarantee a
    connected result on an arbitrary shape, and a model in two pieces is a hard
    failure rather than a cosmetic one. This closes the gap directly: find two
    cells that are side by side but in different groups, and replace the bricks
    directly above them with one brick covering both, backfilling the remainder
    with singles. Each pass provably merges two groups, so the loop is bounded
    by the number of groups.
    """
    library = load_library(config)
    # A bridge is a two-stud brick laid across the split. A library without one
    # cannot repair anything, and saying so by returning the layout unchanged is
    # better than inventing a brick that does not exist.
    if _pair_brick(library) is None:
        return placements
    best = placements
    best_count = _component_count(placements, grid)

    for _ in range(config.brick_repair_passes):
        owner = _owner_grid(placements, grid.shape)
        components = _components(placements, _supports(placements, owner))
        if len(components) <= 1:
            return placements

        membership = {
            index: number
            for number, group in enumerate(components)
            for index in group
        }
        # Try the candidate bridges in turn and keep the first that really
        # reduces the split. Merging bricks frees their other cells, and
        # re-tiling those can leave a fragment that is worse connected than what
        # it replaced, so a bridge is judged by its RESULT. The loop used to take
        # only the first candidate found and give up entirely if that one did not
        # help, which abandoned every other bridge - on a pyramid it stopped with
        # a dozen single-brick strays that a neighbouring bridge would have fixed.
        improved = False
        for bridge in _bridges(owner, membership, grid):
            candidate = _apply_bridge(placements, owner, bridge, grid, library, config)
            count = _component_count(candidate, grid)
            if count < best_count:
                placements, best, best_count = candidate, candidate, count
                improved = True
                break
        if not improved:
            break
    return best


def _component_count(placements: list[BrickPlacement], grid: OccupancyGrid) -> int:
    owner = _owner_grid(placements, grid.shape)
    return len(_components(placements, _supports(placements, owner)))


def _bridges(
    owner: np.ndarray, membership: dict[int, int], grid: OccupancyGrid
):
    """Every pair of side-by-side cells in different groups, in a stable order.

    Each is ``(course, cell_a, cell_b)``. A bridge is made in the split's own
    course by merging the two bricks into one, which needs no room above or
    below - that matters at a narrowing feature like a cone tip or a vase neck,
    where neighbouring courses have nowhere to put a spanning brick.

    Only horizontal adjacencies can split a model in the first place: two cells
    stacked vertically are always owned by bricks that overlap, and overlapping
    bricks are joined by definition. So if none exist, the groups genuinely do
    not touch anywhere and no brick can join them.

    Scanned in index order so the choice is stable.
    """
    width, depth, courses = grid.shape
    for z in range(courses):
        for y in range(depth):
            for x in range(width):
                here = int(owner[x, y, z])
                if here == EMPTY:
                    continue
                for nx, ny in ((x + 1, y), (x, y + 1)):
                    if nx >= width or ny >= depth:
                        continue
                    other = int(owner[nx, ny, z])
                    if other == EMPTY or other == here:
                        continue
                    if membership.get(here) == membership.get(other):
                        continue
                    yield z, (x, y), (nx, ny)


def _find_bridge(
    owner: np.ndarray, membership: dict[int, int], grid: OccupancyGrid
) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    """The first bridge, or ``None`` if the groups do not touch anywhere."""
    return next(_bridges(owner, membership, grid), None)


def _pair_brick(library: list):
    """The library's two-stud brick, which is what spans a one-cell split."""
    return next((brick for brick in library if brick.stud_count == 2), None)


def _apply_bridge(
    placements: list[BrickPlacement],
    owner: np.ndarray,
    bridge: tuple[int, tuple[int, int], tuple[int, int]],
    grid: OccupancyGrid,
    library: list,
    config: Config,
) -> list[BrickPlacement]:
    """Merge the two bricks either side of a split into one that spans it.

    The cells the merged bricks no longer cover are re-tiled rather than filled
    with singles: a scatter of 1x1s is barely attached to anything and would
    trade one split for several.
    """
    course, (ax, ay), (bx, by) = bridge
    doomed = {int(owner[ax, ay, course]), int(owner[bx, by, course])} - {EMPTY}

    freed: list[tuple[int, int]] = []
    kept: list[BrickPlacement] = []
    for index, placement in enumerate(placements):
        if index in doomed:
            freed.extend(placement.footprint_cells)
        else:
            kept.append(placement)

    horizontal = ax != bx
    # The id is the LIBRARY's id for the pair brick in both orientations; a brick
    # that runs the other way is the same brick turned, flagged by ``rotated``.
    # This used to build the id from the footprint ("brick_1x2" for a bridge
    # across y), which names a brick that does not exist, so any model whose
    # repair crossed a split in that direction crashed later, in plate packing,
    # with a bare KeyError.
    kept.append(
        BrickPlacement(
            type_id=_pair_brick(library).id,
            x=min(ax, bx),
            y=min(ay, by),
            z=course,
            width_studs=2 if horizontal else 1,
            depth_studs=1 if horizontal else 2,
            rotated=not horizontal,
        )
    )

    leftover = np.zeros(grid.shape[:2], dtype=bool)
    for cell in set(freed) - {(ax, ay), (bx, by)}:
        leftover[cell] = True
    if leftover.any():
        below = None
        if course > 0:
            below = _owner_grid(kept, grid.shape)[:, :, course - 1]
        kept.extend(tile_layer(leftover, course, library, config, below))

    kept.sort(key=lambda p: (p.z, p.y, p.x))
    return kept


def analyse(
    placements: list[BrickPlacement], grid: OccupancyGrid, config: Config
) -> BrickLayout:
    """Build the layout and run every structural check on it."""
    layout = BrickLayout(placements=list(placements), grid=grid)
    if not placements:
        layout.findings.append(
            Finding(HARD, "empty_layout", "the model produced no bricks", None, {})
        )
        return layout

    owner = _owner_grid(placements, grid.shape)
    supports = _supports(placements, owner)

    components = _components(placements, supports)
    loose = [
        index
        for index, placement in enumerate(placements)
        if placement.z > 0 and not supports[index]
    ]
    bonded = [
        index
        for index, placement in enumerate(placements)
        if placement.z > 0 and len(supports[index]) >= 2
    ]
    upper = [index for index, p in enumerate(placements) if p.z > 0]

    layout.metrics = {
        "bond_fraction": (len(bonded) / len(upper)) if upper else 1.0,
        "seam_alignment_fraction": _seam_alignment(owner, grid),
        "weakest_link_cells": _weakest_link(grid),
        "loose_brick_count": len(loose),
        "component_count": len(components),
        "courses": grid.shape[2],
    }
    ordered_groups = sorted(components, key=len, reverse=True)
    ordered_sizes = [len(group) for group in ordered_groups]
    layout.piece_sizes = ordered_sizes
    layout.piece_of = [0] * len(placements)
    for number, group in enumerate(ordered_groups):
        for index in group:
            layout.piece_of[index] = number
    layout.metrics["largest_stray_bricks"] = (
        ordered_sizes[1] if len(ordered_sizes) > 1 else 0
    )
    layout.metrics["stray_brick_count"] = sum(ordered_sizes[1:])

    if len(components) > 1:
        ordered = sorted(components, key=len, reverse=True)
        sizes = [len(group) for group in ordered]
        detached = sum(sizes[1:])
        fraction = detached / len(placements)
        stray = [index for group in ordered[1:] for index in group]

        too_much = fraction > config.brick_detached_fraction_hard
        too_big = sizes[1] > config.brick_stray_hard_bricks
        if too_much or too_big:
            reason = (
                f"{fraction:.0%} of the bricks are detached"
                if too_much
                else f"one stray group is {sizes[1]} bricks, over the limit of "
                f"{config.brick_stray_hard_bricks}"
            )
            layout.findings.append(
                Finding(
                    HARD,
                    "disconnected_assembly",
                    f"the bricks form {len(components)} groups that never connect "
                    f"(sizes {sizes[:5]}); {reason}, so this cannot be built as one "
                    "object. Bricks join only top to bottom, so a feature that "
                    "narrows or runs sideways can leave a group nothing holds on"
                    + ("" if config.brick_simplify else
                       "; try --simplify, which changes the shape minimally to "
                       "attach such groups"),
                    None,
                    {"component_count": len(components), "component_sizes": sizes},
                )
            )
        else:
            # A few bricks around an edge with nothing above to grip is a
            # property of the shape, not a failed decomposition. The model is
            # still buildable; these are simply optional.
            layout.findings.append(
                Finding(
                    WARNING,
                    "detached_bricks",
                    f"{detached} brick(s) of {len(placements)} sit at an edge with "
                    "nothing above them to hold them on. The rest of the model is "
                    "one connected piece; glue these or leave them off",
                    None,
                    {
                        "detached": detached,
                        "fraction": fraction,
                        "bricks": [placements[i].to_dict() for i in stray[:MAX_REPORTED_BRICKS]],
                    },
                )
            )

    if loose:
        layout.findings.append(
            Finding(
                SOFT,
                "loose_bricks",
                f"{len(loose)} brick(s) rest on nothing and will fall off; they sit "
                "above empty space with no brick beneath them",
                None,
                {"count": len(loose), "cells": [placements[i].to_dict() for i in loose[:MAX_REPORTED_BRICKS]]},
            )
        )

    bond = layout.metrics["bond_fraction"]
    if upper and bond < config.brick_bond_fraction_warn:
        layout.findings.append(
            Finding(
                WARNING,
                "weak_bond",
                f"only {bond:.0%} of bricks span two or more bricks below "
                f"(target {config.brick_bond_fraction_warn:.0%}); courses are stacked "
                "more than interlocked, so the model may split along a course",
                None,
                {"bond_fraction": bond},
            )
        )

    return layout


def _owner_grid(placements: list[BrickPlacement], shape: tuple) -> np.ndarray:
    """Index of the brick occupying each cell, or ``EMPTY``."""
    owner = np.full(shape, EMPTY, dtype=np.int32)
    for index, placement in enumerate(placements):
        owner[
            placement.x : placement.x + placement.width_studs,
            placement.y : placement.y + placement.depth_studs,
            placement.z,
        ] = index
    return owner


def _supports(
    placements: list[BrickPlacement], owner: np.ndarray
) -> list[set[int]]:
    """For each brick, the distinct bricks directly beneath it."""
    supports: list[set[int]] = []
    for placement in placements:
        below: set[int] = set()
        if placement.z > 0:
            patch = owner[
                placement.x : placement.x + placement.width_studs,
                placement.y : placement.y + placement.depth_studs,
                placement.z - 1,
            ]
            below = {int(v) for v in np.unique(patch) if int(v) != EMPTY}
        supports.append(below)
    return supports


def _components(
    placements: list[BrickPlacement], supports: list[set[int]]
) -> list[list[int]]:
    """Connected groups of the vertical attachment graph."""
    neighbours: list[set[int]] = [set() for _ in placements]
    for index, below in enumerate(supports):
        for other in below:
            neighbours[index].add(other)
            neighbours[other].add(index)

    seen = [False] * len(placements)
    components: list[list[int]] = []
    for start in range(len(placements)):
        if seen[start]:
            continue
        group: list[int] = []
        queue = deque([start])
        seen[start] = True
        while queue:
            current = queue.popleft()
            group.append(current)
            for other in sorted(neighbours[current]):
                if not seen[other]:
                    seen[other] = True
                    queue.append(other)
        components.append(sorted(group))
    return components


def _seam_alignment(owner: np.ndarray, grid: OccupancyGrid) -> float:
    """Fraction of seams that sit directly above another seam.

    Only cell pairs solid in both courses count, so holes in the model do not
    skew the result.
    """
    width, depth, courses = grid.shape
    if courses < 2 or width < 2:
        return 0.0

    seams = 0
    aligned = 0
    for z in range(1, courses):
        upper = owner[:, :, z]
        lower = owner[:, :, z - 1]
        # Both axes. Checking only x would miss a seam running the full depth
        # of the model, which is exactly the kind that splits it in half.
        for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
            both = (upper[a] != EMPTY) & (upper[b] != EMPTY)
            both &= (lower[a] != EMPTY) & (lower[b] != EMPTY)
            if not both.any():
                continue
            seam_here = (upper[a] != upper[b]) & both
            seam_below = (lower[a] != lower[b]) & both
            seams += int(seam_here.sum())
            aligned += int((seam_here & seam_below).sum())
    return (aligned / seams) if seams else 0.0


def _weakest_link(grid: OccupancyGrid) -> int:
    """Fewest vertically-connected cell pairs across any horizontal plane."""
    courses = grid.shape[2]
    if courses < 2:
        return int(grid.filled[:, :, 0].sum()) if courses else 0
    counts = [
        int((grid.filled[:, :, z] & grid.filled[:, :, z + 1]).sum())
        for z in range(courses - 1)
    ]
    return min(counts) if counts else 0
