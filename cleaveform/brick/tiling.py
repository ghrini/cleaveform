"""Filling the occupancy grid with bricks, course by course.

Deterministic greedy: walk each course in a fixed order and take the largest
brick from the library that fits. The one refinement is stagger, and it is
structural rather than cosmetic. Bricks only connect vertically, so two bricks
side by side in a course are held together solely by a brick in the course above
spanning both of them. If every course breaks in the same place, nothing ties
across that line and the model splits along it.

Two approaches were tried and measured before the current one. Alternating the
scan direction on odd courses does nothing: an 8-wide run tiles ``4|4`` in both
directions, and any run that is a multiple of the longest brick is
mirror-symmetric. Preferring seam positions at a phase offset per course works
along one axis, but the x and y preferences cancel into ties that the tie-break
then resolves identically in every course, so a solid block still split cleanly
in half.

What actually works is to stop proxying the goal and optimise it directly: when
choosing a brick, first prefer one that covers cells belonging to *two or more
different bricks in the course below*. That is the definition of a bonded
course, so it is what gets maximised, with the phase offset kept only as a
tie-break and for the bottom course where there is nothing to bond to.

That still proxied the goal, and the benchmark showed where. "Covers two
different bricks below" is not the same as "joins two groups that are not yet
joined": two bricks already tied together by a lower course gain nothing from
being spanned again, while two bricks in separate towers are exactly what must
be tied. On a pyramid the tiler built a tower of rim bricks up each face that
stacked on one another and never touched the core, and a solid pyramid came out
as eighteen pieces. The bottom course made it worse, because it was tiled with
no objective at all, so a brick lying wholly in a rim with nothing above it was
isolated from the moment it was placed.

So the tiler now carries a map of which connected group every cell of the
course below belongs to, and chooses bricks by what they attach:

1. a brick that spans several *groups* (it merges them) beats one that spans one
   (it is merely supported) beats one that spans none (it rests on nothing);
2. where nothing is below to attach to, a brick that sits under cells of the
   course above (it will be attached from above) beats one that does not;
3. only then the original bond, size and stagger rules.

Merges made earlier in the same course count for later bricks in it, and a
whole course is scored the same way when its scan strategies are compared.

Cells with nothing to hold them get special handling. An overhang cell has
nothing under it, and on the bottom course a rim cell may have nothing over it;
a brick covering only such cells is loose or isolated whatever else is chosen.
So those cells are tiled first, and the brick chosen for each must also cover at
least one cell that is attached, reaching inward from the edge. The brick can
start anywhere that covers the cell, not only at it, because a brick only grows
towards +x and +y and an edge cell is often the far end of the brick that
should hold it.

One more thing had to change, and it was found by tracing a pyramid anchor by
anchor rather than by tuning. A pyramid's footprint is a diamond whose top is a
single cell, so the first brick anchored there can only run one way. It then
occupies the cell to the right of the next row's first cell, so that row can only
run the same way, and so on down the whole course: every course came out as
parallel stripes whose seams never crossed, and no choice of brick within a
stripe could tie one stripe to the next. The remedy is the one masonry uses,
laying alternate courses crosswise: each brick then spans every stripe it
crosses. So scan strategies include the transposed grid, and a course is free to
run the other way from the one below whenever that ties more groups together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from .library import BrickType, load_library
from .occupancy import OccupancyGrid

NO_BRICK = -1
# Rotation recorded in the spec's brick-layout record, in degrees.
ROTATED_DEGREES = 90
# Scan strategies tried per course, as (origin_x, origin_y, flip_x, flip_y).
# Origin shifts vary a rectangular footprint; direction flips vary an irregular
# one, where a shifted origin lands outside the shape and the scan slides back
# to the same first filled cell. Both are needed, so both are here.
BASE_STRATEGIES = (
    (0, 0, False, False),
    (1, 1, False, False),
    (0, 0, True, False),
    (0, 0, False, True),
    (0, 0, True, True),
    (1, 0, False, False),
    (0, 1, False, False),
)
# Each is also tried on the transposed grid, so a course can run crosswise to the
# one below. The plain ones come first: ties keep the first, so a shape that gains
# nothing from running the other way is tiled exactly as it always was.
SCAN_STRATEGIES = tuple(s + (False,) for s in BASE_STRATEGIES) + tuple(
    s + (True,) for s in BASE_STRATEGIES
)
# Spanning more than this many bricks below adds no useful bonding.
MAX_USEFUL_SPAN = 2
# Spanning more than this many separate groups at once adds nothing a pair of
# bricks could not do, and letting it grow would favour oversized bricks.
MAX_USEFUL_MERGE = 3
# Bond is compared as a share of the course's bricks, in thousandths. Comparing
# the raw sum rewards a tiling for having MORE bricks: a plain cube tiled with
# many small bricks scored higher than the same cube tiled with large ones, and
# one cube at the finest size ran over the brick limit because of it.
BOND_SCALE = 1000


@dataclass(frozen=True)
class BrickPlacement:
    """One brick, placed at a grid cell.

    ``width_studs``/``depth_studs`` are post-rotation, so they always describe
    the footprint as it sits on the grid.
    """

    type_id: str
    x: int
    y: int
    z: int
    width_studs: int
    depth_studs: int
    rotated: bool

    @property
    def cells(self) -> list[tuple[int, int, int]]:
        return [
            (self.x + i, self.y + j, self.z)
            for i in range(self.width_studs)
            for j in range(self.depth_studs)
        ]

    @property
    def footprint_cells(self) -> list[tuple[int, int]]:
        return [
            (self.x + i, self.y + j)
            for i in range(self.width_studs)
            for j in range(self.depth_studs)
        ]

    def to_dict(self) -> dict:
        """The spec's brick-layout record: id, position, orientation."""
        return {
            "brick_id": self.type_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "ori": ROTATED_DEGREES if self.rotated else 0,
        }


@dataclass
class CourseStats:
    """How well a tiled course attaches to the rest of the model.

    Compared lexicographically when scan strategies compete: no brick should be
    isolated, then none should rest on nothing, then as many groups as possible
    should be merged, then as many bricks as possible should span two below.
    """

    isolated: int = 0     # attached to nothing above or below: a stray from birth
    unsupported: int = 0  # nothing underneath (above the bottom course)
    merges: int = 0       # separate groups tied together by this course
    bonds: int = 0        # summed bond of each brick (0, 1 or 2 bricks spanned)
    bricks: int = 0

    def score(self) -> tuple[int, int, int, int, int]:
        """Higher is better. Bond is a share of the bricks, and fewer bricks win
        a tie, so using more pieces is never rewarded for its own sake."""
        share = self.bonds * BOND_SCALE // max(self.bricks, 1)
        return (-self.isolated, -self.unsupported, self.merges, share, -self.bricks)


class _Groups:
    """Which connected group each cell of the course below belongs to.

    Bricks join only top to bottom, so a group is everything linked by overlap
    between consecutive courses. Tracked while tiling, because by the time the
    layout is analysed a stray tower is already built.
    """

    def __init__(self) -> None:
        self._parent: list[int] = []

    def _new(self) -> int:
        self._parent.append(len(self._parent))
        return len(self._parent) - 1

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def advance(
        self, course: list["BrickPlacement"], below: np.ndarray | None, shape: tuple
    ) -> np.ndarray:
        """Group map for ``course``, after joining whatever its bricks span."""
        roots = []
        for placement in course:
            found: set[int] = set()
            if below is not None:
                patch = below[
                    placement.x : placement.x + placement.width_studs,
                    placement.y : placement.y + placement.depth_studs,
                ]
                found = {self.find(int(v)) for v in np.unique(patch) if int(v) != NO_BRICK}
            if found:
                root = min(found)
                for other in found:
                    self._parent[other] = root
            else:
                root = self._new()
            roots.append(root)
        out = np.full(shape, NO_BRICK, dtype=np.int32)
        for placement, root in zip(course, roots):
            out[
                placement.x : placement.x + placement.width_studs,
                placement.y : placement.y + placement.depth_studs,
            ] = self.find(root)
        return out


class TilingError(RuntimeError):
    """Raised when a grid cannot be tiled within the configured bounds."""

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


def tile_grid(grid: OccupancyGrid, config: Config) -> list[BrickPlacement]:
    """Fill every solid cell with bricks, bottom course first."""
    library = load_library(config)
    if not library:
        raise TilingError("brick_library is empty")

    placements: list[BrickPlacement] = []
    previous_offset: tuple | None = None
    groups = _Groups()
    below_groups: np.ndarray | None = None
    for z in range(grid.shape[2]):
        below = _owner_map(placements, grid.shape, z - 1) if z else None
        above = grid.filled[:, :, z + 1] if z + 1 < grid.shape[2] else None
        course, previous_offset = _best_course(
            grid.filled[:, :, z], z, library, config, below, previous_offset,
            below_groups, above,
        )
        below_groups = groups.advance(course, below_groups, grid.shape[:2])
        placements.extend(course)
        if len(placements) > config.max_bricks:
            raise TilingError(
                f"model needs more than max_bricks ({config.max_bricks}) at course "
                f"{z}; raise max_bricks or use a larger brick_unit_mm",
                {
                    "limit": "max_bricks",
                    "value": config.max_bricks,
                    "brick_unit_mm": config.brick_unit_mm,
                },
            )
    return placements


def _best_course(
    mask: np.ndarray,
    z: int,
    library: list[BrickType],
    config: Config,
    below: np.ndarray | None,
    previous_offset: tuple | None,
    groups: np.ndarray | None = None,
    above: np.ndarray | None = None,
) -> tuple[list[BrickPlacement], tuple]:
    """Tile a course at each scan offset and keep the best-attached result.

    Two things make this necessary rather than a fixed alternation. A fixed
    offset gets absorbed by an irregular footprint: if the shifted start cell
    falls outside the shape, the scan slides back to the same first filled cell
    and reproduces the course below. And where consecutive courses have the
    same footprint - a cylinder, say - the best-scoring offset is the same one
    every time, so every course comes out identical and nothing bonds. Hence
    the previous course's offset is excluded outright.

    The bottom course used to take the first offset unconditionally, there being
    nothing below to score against. It now competes like any other, scored on
    whether its bricks will be attached from above.

    Deterministic: offsets are tried in a fixed order and ties keep the first.
    """
    best: tuple[tuple, list[BrickPlacement], tuple] | None = None
    for strategy in SCAN_STRATEGIES:
        if strategy == previous_offset and below is not None:
            continue
        placements, stats = _tile_with(
            mask, z, library, config, below, strategy, groups, above
        )
        score = stats.score()
        if best is None or score > best[0]:
            best = (score, placements, strategy)
    assert best is not None
    return best[1], best[2]


def _tile_with(
    mask: np.ndarray,
    z: int,
    library: list[BrickType],
    config: Config,
    below: np.ndarray | None,
    strategy: tuple,
    groups: np.ndarray | None = None,
    above: np.ndarray | None = None,
) -> tuple[list[BrickPlacement], CourseStats]:
    """Tile one course under a scan strategy, in real coordinates."""
    ox, oy, flip_x, flip_y, transpose = strategy

    def prepare(array):
        if array is None:
            return None
        array = array.T if transpose else array
        return _mirror(array, flip_x, flip_y)

    work_shape = mask.shape[::-1] if transpose else mask.shape
    placed, stats = _tile_course(
        prepare(np.asarray(mask, dtype=bool)),
        z,
        library,
        config,
        prepare(below),
        (ox, oy),
        prepare(groups),
        prepare(above),
    )
    if not (flip_x or flip_y or transpose):
        return placed, stats
    placed = [_unmirror(p, work_shape, flip_x, flip_y) for p in placed]
    if transpose:
        placed = [_untranspose(p) for p in placed]
    return placed, stats


def _owner_map(
    placements: list[BrickPlacement], shape: tuple, z: int
) -> np.ndarray:
    """Which brick owns each cell of course ``z``, or ``NO_BRICK``."""
    owner = np.full(shape[:2], NO_BRICK, dtype=np.int32)
    for index, placement in enumerate(placements):
        if placement.z != z:
            continue
        owner[
            placement.x : placement.x + placement.width_studs,
            placement.y : placement.y + placement.depth_studs,
        ] = index
    return owner


def tile_layer(
    mask: np.ndarray,
    z: int,
    library: list[BrickType],
    config: Config,
    below: np.ndarray | None = None,
    origin: tuple[int, int] = (0, 0),
    groups: np.ndarray | None = None,
    above: np.ndarray | None = None,
) -> list[BrickPlacement]:
    """Tile one course. ``mask`` is a 2D bool array indexed ``[x, y]``.

    ``below`` is the owner map of the course underneath, used to prefer bricks
    that bond across its seams. ``groups`` says which connected group each cell
    below belongs to, and ``above`` which cells have something over them; with
    those the course is tiled to attach, not merely to interlock.
    """
    return _tile_course(mask, z, library, config, below, origin, groups, above)[0]


def _tile_course(
    mask: np.ndarray,
    z: int,
    library: list[BrickType],
    config: Config,
    below: np.ndarray | None,
    origin: tuple[int, int],
    groups: np.ndarray | None,
    above: np.ndarray | None,
) -> tuple[list[BrickPlacement], CourseStats]:
    remaining = np.asarray(mask, dtype=bool).copy()
    period = max(
        max(brick.width_studs, brick.depth_studs) for brick in library
    )
    candidates = _orientations(library)
    # A private copy: merges made by earlier bricks in this course are relabelled
    # here so later bricks see those groups as already joined.
    working = None if groups is None else groups.copy()

    # Each course prefers seams offset from the course below, along both axes.
    # The two phases are deliberately opposite: with the same phase on both, a
    # brick that satisfies x fails y and vice versa, every candidate ties, and
    # the tie-break picks the same brick in every course - which is exactly the
    # aligned-seam failure this is meant to prevent.
    phase_x = z % 2
    phase_y = (z + 1) % 2

    placements: list[BrickPlacement] = []
    stats = CourseStats()
    width, depth = remaining.shape

    def place(brick, rotated, x, y, spanned, touches_above, bond):
        remaining[x : x + brick.width_studs, y : y + brick.depth_studs] = False
        if working is not None and len(spanned) > 1:
            root = min(spanned)
            for other in spanned:
                working[working == other] = root
        if z > 0 and not spanned:
            stats.unsupported += 1
        if not spanned and not touches_above:
            stats.isolated += 1
        stats.merges += max(len(spanned) - 1, 0)
        stats.bonds += bond
        stats.bricks += 1
        placements.append(
            BrickPlacement(
                type_id=brick.id if not rotated else brick.rotated().id,
                x=x,
                y=y,
                z=z,
                width_studs=brick.width_studs,
                depth_studs=brick.depth_studs,
                rotated=rotated,
            )
        )

    order = _visit_order(width, depth, origin)

    # Cells with nothing to hold them go first, each in a brick that also covers
    # something attached. Left to the plain scan, a loose cell is covered by
    # whatever happens to be anchored there, which is usually a brick lying
    # wholly in the overhang.
    attached = _attached_cells(below, above, z, remaining.shape)
    if attached is not None:
        for ox, oy in order:
            if not remaining[ox, oy] or attached[ox, oy]:
                continue
            found = _choose_covering(
                remaining, ox, oy, candidates, attached, phase_x, phase_y, period,
                below, working, above,
            )
            if found is not None:
                place(*found)

    for x, y in order:
        if not remaining[x, y]:
            continue
        brick, rotated, spanned, touches_above, bond = _choose(
            remaining, x, y, candidates, phase_x, phase_y, period, below, working, above
        )
        place(brick, rotated, x, y, spanned, touches_above, bond)
    return placements, stats


def _attached_cells(
    below: np.ndarray | None, above: np.ndarray | None, z: int, shape: tuple
) -> np.ndarray | None:
    """Cells that something holds: a brick underneath, or on the bottom course,
    cells with something over them. ``None`` when there is nothing to attach to."""
    if z > 0:
        return None if below is None else below != NO_BRICK
    return None if above is None else np.asarray(above, dtype=bool)


def _choose_covering(
    remaining: np.ndarray,
    ox: int,
    oy: int,
    candidates: list[tuple[BrickType, bool]],
    attached: np.ndarray,
    phase_x: int,
    phase_y: int,
    period: int,
    below: np.ndarray | None,
    groups: np.ndarray | None,
    above: np.ndarray | None,
):
    """A brick covering cell ``(ox, oy)`` AND at least one attached cell.

    Returns ``(brick, rotated, x, y, spanned, touches_above, bond)`` or ``None``
    when no brick can reach an attached cell from here, in which case the plain
    scan covers the cell as before. The anchor ranges over every position that
    puts the brick over the cell, since a brick only grows towards +x and +y.
    """
    best = None
    for index, (brick, rotated) in enumerate(candidates):
        for x in range(ox - brick.width_studs + 1, ox + 1):
            for y in range(oy - brick.depth_studs + 1, oy + 1):
                if x < 0 or y < 0 or not _fits(remaining, x, y, brick):
                    continue
                footprint = attached[x : x + brick.width_studs, y : y + brick.depth_studs]
                if not footprint.any():
                    continue
                spanned = _groups_spanned(groups, x, y, brick)
                bond = _spanned_below(below, x, y, brick)
                key = (
                    -min(len(spanned), MAX_USEFUL_MERGE),
                    -bond,
                    -brick.stud_count,
                    _boundary_penalty(x + brick.width_studs, phase_x, period)
                    + _boundary_penalty(y + brick.depth_studs, phase_y, period),
                    index,
                    x,
                    y,
                )
                if best is None or key < best[0]:
                    best = (key, brick, rotated, x, y, spanned,
                            _touches(above, x, y, brick), bond)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5], best[6], best[7]


def _visit_order(width: int, depth: int, origin: tuple[int, int]) -> list[tuple[int, int]]:
    """Cell visit order for one course, starting ``origin`` cells in.

    Greedy tiling always starts a brick at the first free cell it reaches, so
    the visit order decides where brick boundaries land. The skipped border is
    filled afterwards with whatever still fits.
    """
    ox, oy = origin
    ox = ox if width > ox else 0
    oy = oy if depth > oy else 0
    inner = [(x, y) for y in range(oy, depth) for x in range(ox, width)]
    if not ox and not oy:
        return inner
    border = [(x, y) for y in range(depth) for x in range(width) if x < ox or y < oy]
    return inner + border


def _mirror(array: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    if flip_x:
        array = array[::-1, :]
    if flip_y:
        array = array[:, ::-1]
    return array


def _unmirror(
    placement: BrickPlacement, shape: tuple, flip_x: bool, flip_y: bool
) -> BrickPlacement:
    """Map a placement made on a mirrored grid back to real coordinates.

    Bricks always grow in +x/+y from their anchor cell, so a course cannot be
    tiled by simply walking the grid backwards - the anchor would sit at the far
    end of the brick and nothing but a 1x1 would ever fit. Mirroring the grid,
    tiling it normally and mapping the result back gives a genuinely different
    packing with the anchors still in the right place.
    """
    x, y = placement.x, placement.y
    if flip_x:
        x = shape[0] - (placement.x + placement.width_studs)
    if flip_y:
        y = shape[1] - (placement.y + placement.depth_studs)
    return BrickPlacement(
        type_id=placement.type_id,
        x=x,
        y=y,
        z=placement.z,
        width_studs=placement.width_studs,
        depth_studs=placement.depth_studs,
        rotated=placement.rotated,
    )


def _untranspose(placement: BrickPlacement) -> BrickPlacement:
    """Map a placement made on a transposed grid back to real coordinates.

    Swapping the axes turns a brick that ran along x into one that runs along y,
    so a non-square brick's ``rotated`` flag flips: ``type_id`` always names the
    library's own orientation, and the flag says whether it is turned.
    """
    square = placement.width_studs == placement.depth_studs
    return BrickPlacement(
        type_id=placement.type_id,
        x=placement.y,
        y=placement.x,
        z=placement.z,
        width_studs=placement.depth_studs,
        depth_studs=placement.width_studs,
        rotated=placement.rotated if square else not placement.rotated,
    )


def _orientations(library: list[BrickType]) -> list[tuple[BrickType, bool]]:
    """Each brick in both orientations, square ones only once."""
    out: list[tuple[BrickType, bool]] = []
    for brick in library:
        out.append((brick, False))
        if not brick.is_square:
            out.append((brick.rotated(), True))
    return out


def _choose(
    remaining: np.ndarray,
    x: int,
    y: int,
    candidates: list[tuple[BrickType, bool]],
    phase_x: int,
    phase_y: int,
    period: int,
    below: np.ndarray | None,
    groups: np.ndarray | None = None,
    above: np.ndarray | None = None,
) -> tuple[BrickType, bool, set[int], bool, int]:
    """Pick the brick to place at ``(x, y)``.

    Preference order: attach to the model (merge groups, then at least rest on
    something, then at least sit under something), then land a seam on this
    course's preferred boundary, cover the most studs, and take the squarest
    footprint. The first is what keeps the model in one piece; the rest are the
    plain greedy rule, and what staggers the courses.

    Returns the brick, whether it is rotated, the set of groups it spans, whether
    it sits under anything in the course above, and how many bricks below it
    spans (for the bond count).
    """
    best: tuple[tuple, BrickType, bool, set[int], bool, int] | None = None
    for index, (brick, rotated) in enumerate(candidates):
        if not _fits(remaining, x, y, brick):
            continue
        spanned = _groups_spanned(groups, x, y, brick)
        touches_above = _touches(above, x, y, brick)
        bond = _spanned_below(below, x, y, brick)
        key = (
            # Join groups first: spanning two separate towers ties the model
            # together, spanning one only rests on it, spanning none rests on
            # nothing.
            -min(len(spanned), MAX_USEFUL_MERGE),
            # With nothing to rest on, at least be held from above.
            0 if spanned else (0 if touches_above else 1),
            # Then bond: a brick spanning two bricks below ties them together,
            # a brick sitting on one ties nothing.
            -bond,
            -brick.stud_count,
            _boundary_penalty(x + brick.width_studs, phase_x, period)
            + _boundary_penalty(y + brick.depth_studs, phase_y, period),
            max(brick.width_studs, brick.depth_studs)
            / min(brick.width_studs, brick.depth_studs),
            index,
        )
        if best is None or key < best[0]:
            best = (key, brick, rotated, spanned, touches_above, bond)

    if best is None:
        # A 1x1 always fits a filled cell, so the library is missing one.
        raise TilingError(
            f"no brick in the library fits cell ({x}, {y}); brick_library must "
            "contain a 1x1 so every cell can be filled"
        )
    return best[1], best[2], best[3], best[4], best[5]


def _groups_spanned(
    groups: np.ndarray | None, x: int, y: int, brick: BrickType
) -> set[int]:
    """The separate connected groups a footprint would sit on."""
    if groups is None:
        return set()
    patch = groups[x : x + brick.width_studs, y : y + brick.depth_studs]
    return {int(v) for v in np.unique(patch) if int(v) != NO_BRICK}


def _touches(above: np.ndarray | None, x: int, y: int, brick: BrickType) -> bool:
    """Whether any cell of the footprint has something over it in the next course."""
    if above is None:
        return False
    return bool(above[x : x + brick.width_studs, y : y + brick.depth_studs].any())


def _spanned_below(
    below: np.ndarray | None, x: int, y: int, brick: BrickType
) -> int:
    """How many distinct bricks in the course below this footprint covers.

    Capped, because spanning three bricks is not meaningfully better than two
    and letting it grow without bound would pull the choice toward oversized
    bricks for their own sake.
    """
    if below is None:
        return 0
    patch = below[x : x + brick.width_studs, y : y + brick.depth_studs]
    distinct = {int(v) for v in np.unique(patch) if int(v) != NO_BRICK}
    return min(len(distinct), MAX_USEFUL_SPAN)


def _fits(remaining: np.ndarray, x: int, y: int, brick: BrickType) -> bool:
    width, depth = remaining.shape
    if x + brick.width_studs > width or y + brick.depth_studs > depth:
        return False
    patch = remaining[x : x + brick.width_studs, y : y + brick.depth_studs]
    return bool(patch.all())


def _boundary_penalty(end: int, phase: int, period: int) -> int:
    """0 when a brick's far edge lands on this course's preferred seam line.

    Even courses prefer seams at multiples of the longest brick; odd courses
    prefer the positions half way between. That is what makes consecutive
    courses break in different places.
    """
    if period < 2:
        return 0
    half = period // 2
    return 0 if end % period == (phase * half) % period else 1
