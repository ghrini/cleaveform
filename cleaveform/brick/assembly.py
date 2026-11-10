"""Placing brick meshes into world space, and packing them onto build plates.

A brick mesh is built once per type at the origin. Everything here is rigid
transforms of those meshes - no booleans - which is what keeps a five-hundred
brick model as cheap to export as a six-brick one.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..config import Config
from .layout import BrickLayout
from .occupancy import OccupancyGrid
from .tiling import BrickPlacement

QUARTER_TURN = np.pi / 2.0


def world_transform(
    placement: BrickPlacement, grid: OccupancyGrid, config: Config
) -> np.ndarray:
    """Transform putting a canonical brick mesh at its place in the model.

    A rotated brick turns a quarter turn about Z, which swings it into negative
    x, so it is shifted back by its own depth to sit on its grid cell again.
    """
    gap = config.brick_fit_clearance_mm
    transform = np.eye(4)

    if placement.rotated:
        transform = trimesh.transformations.rotation_matrix(
            QUARTER_TURN, [0.0, 0.0, 1.0]
        )
        # Post-rotation footprint depth equals the canonical width, so the
        # shift back is the canonical depth: the placement's own width.
        transform[:3, 3] = np.array(
            [placement.width_studs * config.brick_unit_mm - gap, 0.0, 0.0]
        )

    corner = grid.cell_origin_mm(placement.x, placement.y, placement.z)
    shift = np.eye(4)
    shift[:3, 3] = corner + np.array([gap / 2.0, gap / 2.0, 0.0])
    return shift @ transform


def assemble(
    layout: BrickLayout, meshes: dict[str, trimesh.Trimesh], config: Config
) -> trimesh.Trimesh:
    """The whole model as one mesh, for preview and verification only.

    Not an export format: the bricks are separate objects and only touch, so
    this is a scene flattened into one mesh rather than a solid.
    """
    parts: list[trimesh.Trimesh] = []
    for placement in layout.placements:
        brick = meshes.get(placement.type_id)
        if brick is None:
            continue
        copy = brick.copy()
        copy.apply_transform(world_transform(placement, layout.grid, config))
        parts.append(copy)
    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def _plate_grid(
    counts: dict[str, int], meshes: dict[str, trimesh.Trimesh], config: Config
) -> tuple[int, int, float, float]:
    """The row-major grid ``pack_plates`` lays bricks out in.

    Returns ``(columns, rows, cell_width, cell_depth)``. Split out so a caller
    that only needs the NUMBER of plates (a benchmark scoring hundreds of
    layouts) can get it without building every plate mesh, while both callers
    stay in step because there is only one place the grid is worked out.
    """
    cell_width = max(
        float(meshes[t].bounds[1][0] - meshes[t].bounds[0][0]) for t in counts
    ) + config.plate_margin_mm
    cell_depth = max(
        float(meshes[t].bounds[1][1] - meshes[t].bounds[0][1]) for t in counts
    ) + config.plate_margin_mm
    columns = max(int((config.plate_size_mm[0] - config.plate_margin_mm) // cell_width), 1)
    rows = max(int((config.plate_size_mm[1] - config.plate_margin_mm) // cell_depth), 1)
    return columns, rows, cell_width, cell_depth


def plate_count(
    counts: dict[str, int], meshes: dict[str, trimesh.Trimesh], config: Config
) -> int:
    """How many plates a bill of bricks needs."""
    total = sum(counts.values())
    if total == 0:
        return 0
    columns, rows, _, _ = _plate_grid(counts, meshes, config)
    return -(-total // (columns * rows))


def pack_plates(
    counts: dict[str, int], meshes: dict[str, trimesh.Trimesh], config: Config
) -> list[trimesh.Trimesh]:
    """Lay every brick copy out on build-plate-sized sheets.

    Handing over six STLs and "print 512 of them" is not a usable deliverable;
    arranged plates are. Bricks are placed in a row-major grid sized to the
    largest brick, which wastes some area but keeps the arrangement obvious and
    reproducible.
    """
    if not counts:
        return []
    columns, rows, cell_width, cell_depth = _plate_grid(counts, meshes, config)
    per_plate = columns * rows

    queue: list[str] = []
    for type_id in sorted(counts):
        queue.extend([type_id] * counts[type_id])

    plates: list[trimesh.Trimesh] = []
    for start in range(0, len(queue), per_plate):
        batch = queue[start : start + per_plate]
        parts: list[trimesh.Trimesh] = []
        for index, type_id in enumerate(batch):
            column, row = index % columns, index // columns
            copy = meshes[type_id].copy()
            copy.apply_translation(
                (
                    config.plate_margin_mm + column * cell_width,
                    config.plate_margin_mm + row * cell_depth,
                    0.0,
                )
            )
            parts.append(copy)
        plates.append(trimesh.util.concatenate(parts))
    return plates
