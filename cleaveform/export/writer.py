"""Module 7 - STL export.

Phase 1 requires one binary STL per part. Export asserts watertightness so a
non-manifold part can never reach the slicer unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from ..config import Config

PARTS_DIRNAME = "parts"
STL_SUFFIX = ".stl"


class ExportError(RuntimeError):
    """Raised when a part cannot be written."""


def write_parts(
    meshes: dict[str, trimesh.Trimesh],
    out_dir: str | Path,
    config: Config,
    subdirectory: str = PARTS_DIRNAME,
) -> dict[str, Path]:
    """Write one STL per part into ``out_dir/parts``.

    Returns a mapping of part id to the written path. Parts are written in
    sorted id order so a directory listing is reproducible.
    """
    parts_dir = Path(out_dir) / subdirectory
    parts_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for part_id in sorted(meshes):
        mesh = meshes[part_id]
        if not mesh.is_watertight:
            raise ExportError(
                f"refusing to export {part_id}: mesh is not watertight"
            )
        if abs(float(mesh.volume)) < config.volume_epsilon_mm3:
            raise ExportError(f"refusing to export {part_id}: mesh has no volume")
        path = parts_dir / f"{part_id}{STL_SUFFIX}"
        path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
        written[part_id] = path
    return written
