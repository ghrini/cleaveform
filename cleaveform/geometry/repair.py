"""Mesh repair and validity helpers.

Repair is intentionally deterministic: the same input mesh always produces the
same repaired output, which the Phase 1 determinism acceptance test depends on.
Only operations with stable output ordering are used.

``trimesh`` handles the common cases. ``open3d`` is a second-chance fallback for
input meshes with non-manifold edges that hole filling alone will not fix; it is
an optional dependency and the pipeline degrades gracefully without it.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..config import Config


def is_empty(mesh: trimesh.Trimesh | None, config: Config) -> bool:
    """True when the mesh has no usable geometry."""
    if mesh is None:
        return True
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return True
    return abs(float(mesh.volume)) < config.volume_epsilon_mm3


def clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove degenerate and duplicated geometry without altering shape."""
    out = mesh.copy()
    out.remove_unreferenced_vertices()
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    return out


def repair(mesh: trimesh.Trimesh, config: Config) -> trimesh.Trimesh:
    """Attempt to make ``mesh`` watertight with consistent outward normals.

    Performs at most ``config.repair_max_attempts`` hole-filling passes. Returns
    the best mesh produced; the caller checks ``is_watertight`` to decide whether
    the repair succeeded.
    """
    out = clean(mesh)
    for _ in range(max(config.repair_max_attempts, 0)):
        if out.is_watertight and out.is_winding_consistent:
            break
        trimesh.repair.fill_holes(out)
        trimesh.repair.fix_winding(out)
        trimesh.repair.fix_inversion(out)
        trimesh.repair.fix_normals(out)
        out = clean(out)
    if out.is_watertight and out.volume < 0:
        out.invert()
    return out


def ensure_watertight(mesh: trimesh.Trimesh, config: Config) -> tuple[trimesh.Trimesh, bool]:
    """Return ``(mesh, ok)`` where ``ok`` reports watertightness after repair."""
    if mesh.is_watertight and mesh.is_winding_consistent:
        return mesh, True
    fixed = repair(mesh, config)
    if fixed.is_watertight:
        return fixed, True
    salvaged = _open3d_repair(fixed, config)
    if salvaged is not None and salvaged.is_watertight:
        return salvaged, True
    return fixed, False


def _open3d_repair(mesh: trimesh.Trimesh, config: Config) -> trimesh.Trimesh | None:
    """Second-chance repair for non-manifold input, via open3d.

    Returns ``None`` when open3d is unavailable or cannot help. Every operation
    used here is order-stable, so determinism is preserved.
    """
    try:
        import open3d as o3d
    except ImportError:
        return None

    try:
        geometry = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        )
        geometry.remove_duplicated_vertices()
        geometry.remove_duplicated_triangles()
        geometry.remove_degenerate_triangles()
        geometry.remove_non_manifold_edges()
        geometry.remove_unreferenced_vertices()
        geometry.orient_triangles()
        rebuilt = trimesh.Trimesh(
            vertices=np.asarray(geometry.vertices),
            faces=np.asarray(geometry.triangles),
            process=False,
        )
    except Exception:
        return None

    return repair(rebuilt, config)


def connected_components(mesh: trimesh.Trimesh, config: Config) -> list[trimesh.Trimesh]:
    """Split into connected bodies, ordered deterministically by volume then position."""
    try:
        bodies = list(mesh.split(only_watertight=False))
    except Exception:
        bodies = []
    if not bodies:
        return [mesh]
    bodies = [b for b in bodies if not is_empty(b, config)]
    if not bodies:
        return [mesh]
    bodies.sort(key=lambda b: (-float(abs(b.volume)), tuple(np.round(b.bounds[0], 6))))
    return bodies


def as_watertight_or_none(
    mesh: trimesh.Trimesh, config: Config
) -> trimesh.Trimesh | None:
    """Repair and return the mesh, or ``None`` when it cannot be made valid."""
    if is_empty(mesh, config):
        return None
    fixed, ok = ensure_watertight(mesh, config)
    if not ok or is_empty(fixed, config):
        return None
    return fixed
