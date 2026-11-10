"""Changing a shape minimally so every brick of it can be held.

Some cells cannot be held by any tiling, however clever. A brick is held only by
bricks above or below it, so a cell with nothing over it and nothing under it
touches only its neighbours - and bricks never join sideways. The interior of a
one-brick-thick floor is the usual case: a vase's base, a tray's bottom. A stray
rim sliver, a mushroom's lip and a fence's rail are the same thing in other
clothes. No change to which brick goes where can attach them; the SHAPE has to
change.

The tiler already says which bricks could not be held, so that is the feedback
used here: each group of stray bricks is looked at and given the smallest change
that attaches it.

* a **sliver** (a cell or two) is dropped - holding it costs more than it is worth;
* a **flat group** (one course thick, like a floor) is thickened by a course, so
  the bricks of the course above can span it and tie it to what surrounds it;
* an **island** (several courses tall) that touches the main body sideways is
  **stitched**: its bricks and the body's sit side by side with a seam between
  them, and nothing above or below can span it because the cells there are empty,
  so the one or two cells missing over (or under) the seam are added. A fence rail
  one cell thick, with a post at each end, is the example: the rail's bricks meet
  at a seam that only a brick in the next course can cross;
An island that touches nothing is left alone. Joining it would mean inventing
material across empty space, and an empty gap is a real separation however narrow:
two separate objects in one file must stay two. Thin features that genuinely join
parts are the voxelizer's job (``brick_keep_thin``), which only bridges through
cells that hold some of the model.

Everything changed is counted and reported; the model is never altered silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from .occupancy import NEIGHBOURS, OccupancyGrid


@dataclass
class Changes:
    """What simplification did to the shape."""

    dropped: int = 0     # cells removed
    thickened: int = 0   # cells added to give a flat group a second course
    stitched: int = 0    # cells added over or under a seam so a brick can cross it
    passes: int = 0

    @property
    def added(self) -> int:
        return self.thickened + self.stitched

    @property
    def empty(self) -> bool:
        return not (self.dropped or self.added)

    def to_dict(self) -> dict:
        return {
            "dropped_cells": self.dropped,
            "thickened_cells": self.thickened,
            "stitched_cells": self.stitched,
            "passes": self.passes,
        }

    def __iadd__(self, other: "Changes") -> "Changes":
        self.dropped += other.dropped
        self.thickened += other.thickened
        self.stitched += other.stitched
        return self

    def summary(self) -> str:
        parts = []
        if self.dropped:
            parts.append(f"dropped {self.dropped} sliver cell(s)")
        if self.thickened:
            parts.append(f"thickened flat parts by {self.thickened} cell(s)")
        if self.stitched:
            parts.append(f"stitched seams with {self.stitched} cell(s)")
        return ", ".join(parts) or "no change"


def stray_cell_groups(placements, piece_of) -> list[set[tuple[int, int, int]]]:
    """Cells of every stray brick, grouped into face-connected clumps."""
    cells: set[tuple[int, int, int]] = set()
    for placement, piece in zip(placements, piece_of):
        if piece != 0:
            cells.update(placement.cells)
    groups = []
    seen: set = set()
    for start in sorted(cells):
        if start in seen:
            continue
        group = {start}
        stack = [start]
        seen.add(start)
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in NEIGHBOURS:
                n = (x + dx, y + dy, z + dz)
                if n in cells and n not in seen:
                    seen.add(n)
                    group.add(n)
                    stack.append(n)
        groups.append(group)
    return groups


def simplify_grid(
    grid: OccupancyGrid, placements, piece_of, config: Config
) -> tuple[OccupancyGrid, Changes]:
    """One pass: the smallest change that attaches each group of stray bricks."""
    changes = Changes()
    filled = grid.filled.copy()
    body = _body_cells(placements, piece_of)

    for group in stray_cell_groups(placements, piece_of):
        courses = {z for _, _, z in group}
        if len(group) <= config.brick_sliver_max_cells:
            for cell in group:
                filled[cell] = False
            changes.dropped += len(group)
        elif len(courses) == 1:
            filled, added = _thicken(filled, group)
            if added:
                changes.thickened += added
            else:
                for cell in group:
                    filled[cell] = False
                changes.dropped += len(group)
        else:
            stitch = _stitch(filled, group, body)
            if stitch:
                for cell in stitch:
                    filled[cell] = True
                changes.stitched += len(stitch)
                continue
    changes.passes = 1
    return (
        OccupancyGrid(
            filled=filled, pitch_mm=grid.pitch_mm, origin_mm=grid.origin_mm,
            added_thin=grid.added_thin,
        ),
        changes,
    )


def _body_cells(placements, piece_of) -> set[tuple[int, int, int]]:
    cells: set = set()
    for placement, piece in zip(placements, piece_of):
        if piece == 0:
            cells.update(placement.cells)
    return cells


def _thicken(filled: np.ndarray, group: set) -> tuple[np.ndarray, int]:
    """Give a flat group a second course, above it if there is room, else below.

    Above is preferred because a floor's upper face is its inside, so the added
    cells land in the cavity rather than showing on the outside. If neither side
    has room the model is one course tall and the grid grows by a course on top,
    the only way a single-course shape can have anything to hold it.
    """
    z = next(iter(group))[2]
    for step in (1, -1):
        target = z + step
        if 0 <= target < filled.shape[2]:
            cells = [(x, y, target) for x, y, _ in group]
            if all(not filled[c] for c in cells):
                for c in cells:
                    filled[c] = True
                return filled, len(cells)
    if z == filled.shape[2] - 1:
        grown = np.concatenate(
            [filled, np.zeros(filled.shape[:2] + (1,), dtype=bool)], axis=2
        )
        for x, y, _ in group:
            grown[x, y, z + 1] = True
        return grown, len(group)
    return filled, 0


def _stitch(
    filled: np.ndarray, island: set, body: set
) -> list[tuple[int, int, int]] | None:
    """Cells to add so a brick can cross the seam between an island and the body.

    Finds a pair of cells, one in the island and one in the body, that sit side by
    side in the same course. Bricks that meet there cannot be tied by anything in
    their own course, and a brick in the course above or below can span the seam
    only if both cells over (or under) it are filled; this adds whichever of the
    two are not. Prefers the side that needs fewer cells, and the upper side on a
    tie. ``None`` if the island does not touch the body or no seam has room.
    """
    shape = filled.shape
    best: tuple | None = None
    for x, y, z in sorted(island):
        for dx, dy, _ in NEIGHBOURS[:4]:
            other = (x + dx, y + dy, z)
            if other not in body:
                continue
            for step in (1, -1):
                target = z + step
                if not 0 <= target < shape[2]:
                    continue
                missing = [
                    c for c in ((x, y, target), (other[0], other[1], target)) if not filled[c]
                ]
                if not missing:
                    continue
                rank = (len(missing), -step, (x, y, z))
                if best is None or rank < best[0]:
                    best = (rank, missing)
    return None if best is None else sorted(best[1])
