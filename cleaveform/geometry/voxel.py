"""Voxel and mesh interconversion.

Chunk mode does not voxelize, so nothing in the Phase 1 pipeline calls into
this module. It exists because the geometry kernel owns every geometry utility
a second module may need (spec module 0), and brick mode in Phase 5 will use it.
"""

from __future__ import annotations

import numpy as np
import trimesh


def voxelize(mesh: trimesh.Trimesh, pitch_mm: float) -> dict:
    """Return the voxel-grid contract from spec section 8."""
    if pitch_mm <= 0:
        raise ValueError("voxel pitch must be positive")
    grid = mesh.voxelized(pitch=pitch_mm).fill()
    return {
        "grid": np.asarray(grid.matrix, dtype=bool),
        "pitch_mm": float(pitch_mm),
        "origin_mm": np.asarray(grid.translation, dtype=float),
    }


def to_mesh(voxel_grid: dict) -> trimesh.Trimesh:
    """Rebuild a blocky mesh from the voxel-grid contract."""
    encoding = trimesh.voxel.VoxelGrid(voxel_grid["grid"])
    mesh = encoding.marching_cubes
    mesh.apply_scale(voxel_grid["pitch_mm"])
    mesh.apply_translation(voxel_grid["origin_mm"])
    return mesh
