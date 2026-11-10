"""Regenerate the test fixture STLs.

Fixtures are generated rather than hand-modelled so they are reproducible and
reviewable. Run this from the repository root:

    python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent

# Sized against the 280 x 300 x 300 mm design ceiling.
SMALL_CUBE_MM = 60.0
OVERSIZE_BAR_MM = (620.0, 80.0, 80.0)
OVERSIZE_BLOCK_MM = (400.0, 400.0, 120.0)
TALL_CYLINDER = {"radius": 60.0, "height": 520.0, "sections": 64}
THIN_PLATE_MM = (400.0, 200.0, 3.0)
# Thicker than min_part_mm, so it decomposes, but far too thin to host a peg:
# this is the fixture that must trip the hard connector-thickness constraint.
NARROW_PLATE_MM = (400.0, 200.0, 7.0)


def _write(mesh: trimesh.Trimesh, name: str) -> Path:
    path = HERE / name
    path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    return path


def build() -> list[Path]:
    written: list[Path] = []

    cube = trimesh.creation.box(extents=(SMALL_CUBE_MM,) * 3)
    written.append(_write(cube, "small_cube.stl"))

    bar = trimesh.creation.box(extents=OVERSIZE_BAR_MM)
    written.append(_write(bar, "oversize_bar.stl"))

    block = trimesh.creation.box(extents=OVERSIZE_BLOCK_MM)
    written.append(_write(block, "oversize_block.stl"))

    cylinder = trimesh.creation.cylinder(**TALL_CYLINDER)
    written.append(_write(cylinder, "tall_cylinder.stl"))

    plate = trimesh.creation.box(extents=THIN_PLATE_MM)
    written.append(_write(plate, "thin_plate.stl"))

    narrow = trimesh.creation.box(extents=NARROW_PLATE_MM)
    written.append(_write(narrow, "narrow_plate.stl"))

    tiny = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    written.append(_write(tiny, "implausibly_tiny.stl"))

    return written


if __name__ == "__main__":
    for path in build():
        mesh = trimesh.load_mesh(str(path))
        extents = mesh.bounds[1] - mesh.bounds[0]
        print(f"{path.name:<24} {extents[0]:8.2f} x {extents[1]:7.2f} x {extents[2]:7.2f} mm")
