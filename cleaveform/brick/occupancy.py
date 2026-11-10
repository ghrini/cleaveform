"""Turning a mesh into the occupancy grid brick mode tiles.

Two things here are easy to get wrong and invisible once wrong.

**Dilation.** ``trimesh``'s default voxelizer is a *surface* voxelization: it
marks every cell the surface passes through, then interior filling closes the
result. At an 8 mm pitch that inflates the model by up to a full cell in every
direction and turns a 3 mm wall into an 8 mm column, so a bricked model comes
out systematically fat. Sampling at half pitch and keeping only cells that are
at least ``brick_occupancy_threshold`` full recovers the intended shape.

**Origin.** ``trimesh`` reports the location of voxel ``[0,0,0]`` as that
cell's *centre*, not its minimum corner. Treating it as a corner offsets the
entire assembly by half a brick, which no test that merely counts bricks would
catch.
"""

from __future__ import annotations

import heapq

from dataclasses import dataclass

import numpy as np
import trimesh

from ..config import Config

SUBSAMPLE = 2  # cells per brick unit per axis when measuring fill fraction
# Finer, used when thin features are kept (see ``brick_keep_thin``).
SUBSAMPLE_THIN = 4


@dataclass(frozen=True)
class OccupancyGrid:
    """Which brick cells are solid, and where cell (0,0,0) sits in the model."""

    filled: np.ndarray  # bool, indexed [x, y, z]
    pitch_mm: float
    origin_mm: np.ndarray  # world position of the MIN CORNER of cell (0,0,0)
    # Cells added by ``brick_keep_thin`` to bridge thin features back, so the
    # report can say how much of the model the ordinary vote had thrown away.
    added_thin: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.filled.shape)  # type: ignore[return-value]

    @property
    def cell_count(self) -> int:
        return int(self.filled.sum())

    def cell_origin_mm(self, x: int, y: int, z: int) -> np.ndarray:
        """World position of the min corner of cell ``(x, y, z)``."""
        return self.origin_mm + np.array([x, y, z], dtype=float) * self.pitch_mm

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "pitch_mm": self.pitch_mm,
            "origin_mm": [float(v) for v in self.origin_mm],
            "cell_count": self.cell_count,
        }


def build_occupancy(mesh: trimesh.Trimesh, config: Config) -> OccupancyGrid:
    """Voxelize ``mesh`` onto the brick lattice by majority fill.

    The brick lattice is anchored on the model's own minimum corner rather than
    on whatever lattice ``trimesh`` happens to pick, so a model that is an exact
    multiple of the unit size produces exactly that many cells. Fill fractions
    are read off a half-pitch sample of the model by index lookup, which keeps
    this a couple of vectorized operations instead of a per-point containment
    test over the whole grid.

    With ``brick_keep_thin`` the same vote decides every cell as usual, and a
    second, finer sample is used only to find thin features worth bridging back.
    Sampling the vote itself more finely would change the model's ordinary
    outline, which is not what that option is for.
    """
    pitch = config.brick_unit_mm
    anchor = np.asarray(mesh.bounds[0], dtype=float)
    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    shape = np.maximum(np.ceil(extents / pitch - 1e-9).astype(int), 1)

    fraction = _sampled_fraction(mesh, config, anchor, shape, SUBSAMPLE)
    coarse = fraction >= config.brick_occupancy_threshold
    if config.brick_keep_thin:
        # Finer, because at the usual two samples per axis a rail a third of a
        # cell thick can fall between them and never be seen at all, and what is
        # never seen cannot be bridged back.
        thin = _sampled_fraction(mesh, config, anchor, shape, SUBSAMPLE_THIN)
        kept = preserve_connections(
            coarse, thin, config.brick_thin_min_fill, config.brick_thin_max_gap_cells
        )
        added = int(kept.sum() - coarse.sum())
        coarse = kept
    else:
        added = 0
    trimmed, origin = _trim(coarse, anchor, pitch)
    return OccupancyGrid(filled=trimmed, pitch_mm=pitch, origin_mm=origin, added_thin=added)


def _sampled_fraction(
    mesh: trimesh.Trimesh,
    config: Config,
    anchor: np.ndarray,
    shape: np.ndarray,
    subsample: int,
) -> np.ndarray:
    """Fraction of each brick cell that is solid, at ``subsample`` samples per axis."""
    pitch = config.brick_unit_mm
    fine_pitch = pitch / subsample
    voxels = mesh.voxelized(pitch=fine_pitch)
    if config.brick_fill_interior:
        voxels = voxels.fill()
    fine = np.asarray(voxels.matrix, dtype=bool)
    # trimesh reports the CENTRE of cell [0,0,0]; step back half a cell for the
    # corner the fine grid is really anchored on.
    fine_origin = np.asarray(voxels.translation, dtype=float) - fine_pitch / 2.0
    return _fill_fraction(fine, fine_origin, fine_pitch, anchor, shape, pitch, subsample)


def _fill_fraction(
    fine: np.ndarray,
    fine_origin: np.ndarray,
    fine_pitch: float,
    anchor: np.ndarray,
    shape: np.ndarray,
    pitch: float,
    subsample: int = SUBSAMPLE,
) -> np.ndarray:
    """Fraction of each brick cell that is solid, sampled ``subsample`` times per axis."""
    steps = (np.arange(subsample) + 0.5) * fine_pitch
    counts = np.zeros(tuple(shape), dtype=np.int32)

    cell_min = [
        anchor[axis] + np.arange(shape[axis]) * pitch for axis in range(len(shape))
    ]
    for dx in steps:
        ix = _lookup_index(cell_min[0] + dx, fine_origin[0], fine_pitch, fine.shape[0])
        for dy in steps:
            iy = _lookup_index(
                cell_min[1] + dy, fine_origin[1], fine_pitch, fine.shape[1]
            )
            for dz in steps:
                iz = _lookup_index(
                    cell_min[2] + dz, fine_origin[2], fine_pitch, fine.shape[2]
                )
                sample = fine[np.ix_(ix.clipped, iy.clipped, iz.clipped)]
                valid = (
                    ix.inside[:, None, None]
                    & iy.inside[None, :, None]
                    & iz.inside[None, None, :]
                )
                counts += (sample & valid).astype(np.int32)
    return counts / float(subsample**3)


class _Index:
    """Fine-grid indices for a row of sample points, plus an in-bounds mask."""

    __slots__ = ("clipped", "inside")

    def __init__(self, raw: np.ndarray, length: int) -> None:
        self.inside = (raw >= 0) & (raw < length)
        self.clipped = np.clip(raw, 0, max(length - 1, 0))


def _lookup_index(
    positions: np.ndarray, origin: float, fine_pitch: float, length: int
) -> _Index:
    return _Index(np.floor((positions - origin) / fine_pitch).astype(int), length)


def _trim(
    grid: np.ndarray, origin: np.ndarray, pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop empty border cells so the grid is exactly the model's extent."""
    if not grid.any():
        return grid, origin
    lower = []
    upper = []
    for axis in range(grid.ndim):
        occupied = np.any(grid, axis=tuple(a for a in range(grid.ndim) if a != axis))
        indices = np.flatnonzero(occupied)
        lower.append(int(indices[0]))
        upper.append(int(indices[-1]) + 1)
    sliced = grid[
        lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]
    ]
    return sliced, origin + np.asarray(lower, dtype=float) * pitch


# Face neighbours: bricks join through shared faces, so these are what "touching"
# means on the cell grid.
NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# Bridging adds cells in rounds because each round can unlock the next; a model
# needing more than this many is badly fragmented and is better left reported.
MAX_BRIDGE_ROUNDS = 8


def label_components(filled: np.ndarray) -> tuple[np.ndarray, int]:
    """Number the face-connected groups of filled cells, in scan order."""
    labels = np.zeros(filled.shape, dtype=np.int32)
    count = 0
    shape = filled.shape
    for start in zip(*np.nonzero(filled)):
        if labels[start]:
            continue
        count += 1
        labels[start] = count
        stack = [start]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in NEIGHBOURS:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (
                    0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]
                    and filled[nx, ny, nz] and not labels[nx, ny, nz]
                ):
                    labels[nx, ny, nz] = count
                    stack.append((nx, ny, nz))
    return labels, count


def preserve_connections(
    coarse: np.ndarray, fraction: np.ndarray, min_fill: float, max_gap: int
) -> np.ndarray:
    """Add empty cells that hold some of the model and join separate parts.

    The majority vote throws away a cell that is less than half full, which is
    right for a ragged edge and wrong for a rail or a bridge: a fence's rails are
    thinner than a 12 mm cell, so the vote deletes them and the posts they held
    come out as separate pieces. Cells are added only along the cheapest path of
    cells that each hold at least ``min_fill`` of material and that links two
    groups otherwise apart, no longer than ``max_gap`` cells. A path, not a single
    cell, because a rail across a two-cell gap has no one cell touching both
    posts. The cost of a cell is how little of it is material, so the strongest
    route wins and nothing is added anywhere that is not on a route.

    Each round joins every group to its nearest neighbour, then looks again,
    because one bridge can make the next one possible.
    """
    filled = coarse.copy()
    for _ in range(MAX_BRIDGE_ROUNDS):
        labels, count = label_components(filled)
        if count < 2:
            break
        candidates = ~filled & (fraction >= min_fill)
        chosen: dict[tuple, tuple] = {}
        for label in range(1, count + 1):
            found = _cheapest_route(labels, label, candidates, fraction, max_gap)
            if found is None:
                continue
            cost, path, other = found
            pair = tuple(sorted((label, other)))
            if pair not in chosen or (cost, path) < chosen[pair]:
                chosen[pair] = (cost, path)
        if not chosen:
            break
        for _, path in chosen.values():
            for cell in path:
                filled[cell] = True
    return filled


def _cheapest_route(
    labels: np.ndarray,
    label: int,
    candidates: np.ndarray,
    fraction: np.ndarray,
    max_gap: int,
):
    """Cheapest path of candidate cells from group ``label`` to any other group.

    Returns ``(cost, path, other_label)`` or ``None``. Deterministic: ties break
    on the cell coordinates.
    """
    shape = labels.shape

    def neighbours(cell):
        for dx, dy, dz in NEIGHBOURS:
            n = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if 0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and 0 <= n[2] < shape[2]:
                yield n

    heap: list = []
    best: dict = {}
    parent: dict = {}
    for cell in zip(*np.nonzero(labels == label)):
        for n in neighbours(cell):
            if candidates[n]:
                cost = 1.0 - float(fraction[n])
                if n not in best or cost < best[n][0]:
                    best[n] = (cost, 1)
                    parent[n] = None
                    heapq.heappush(heap, (cost, 1, n))
    while heap:
        cost, steps, cell = heapq.heappop(heap)
        if best.get(cell, (None,))[0] != cost:
            continue
        for n in neighbours(cell):
            owner = int(labels[n])
            if owner and owner != label:
                path = [cell]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return cost, tuple(sorted(path)), owner
        if steps >= max_gap:
            continue
        for n in neighbours(cell):
            if not candidates[n]:
                continue
            step_cost = cost + 1.0 - float(fraction[n])
            if n not in best or step_cost < best[n][0]:
                best[n] = (step_cost, steps + 1)
                parent[n] = cell
                heapq.heappush(heap, (step_cost, steps + 1, n))
    return None
