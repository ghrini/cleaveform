"""Plane splitting and cross-section extraction.

Splitting is done with boolean intersection against an oversized half-space
solid rather than a bare mesh slice, because the manifold backend gives
watertight results on geometry where naive capping fails.

Cross-sections are projected into an explicitly constructed plane frame so the
resulting 2D coordinates never depend on library-internal basis choices. That
is what makes peg placement reproducible run to run.
"""

from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from ..config import N_AXES, Config
from . import booleans, repair


def unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("zero-length vector has no direction")
    return v / norm


def plane_frame(normal) -> np.ndarray:
    """Right-handed orthonormal frame ``(u, v, n)`` as a 3x3 matrix of rows.

    The in-plane axes are derived from the normal alone, deterministically.
    """
    n = unit(normal)
    reference = np.zeros(N_AXES)
    reference[int(np.argmin(np.abs(n)))] = 1.0
    u = unit(np.cross(reference, n))
    v = np.cross(n, u)
    return np.vstack([u, v, n])


def plane_transforms(origin, normal) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(to_2d, to_3d)`` 4x4 transforms for the given plane."""
    rotation = plane_frame(normal)
    point = np.asarray(origin, dtype=float)
    to_2d = np.eye(4)
    to_2d[:3, :3] = rotation
    to_2d[:3, 3] = -rotation @ point
    return to_2d, np.linalg.inv(to_2d)


def half_space(
    mesh: trimesh.Trimesh, origin, normal, config: Config
) -> trimesh.Trimesh:
    """A solid box covering everything on the positive side of the plane."""
    n = unit(normal)
    point = np.asarray(origin, dtype=float)
    bounds = mesh.bounds
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    size = diagonal + config.peg_edge_margin_mm
    box = trimesh.creation.box(extents=(size * 2, size * 2, size))
    transform = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), n)
    transform[:3, 3] = point + n * (size / 2.0)
    box.apply_transform(transform)
    return box


def split_by_plane(
    mesh: trimesh.Trimesh, origin, normal, config: Config
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Split into ``(negative_side, positive_side)`` relative to ``normal``.

    Either side is ``None`` when it is empty or cannot be repaired to a
    watertight solid within the configured repair budget.
    """
    n = unit(normal)
    positive_cutter = half_space(mesh, origin, n, config)
    negative_cutter = half_space(mesh, origin, -n, config)

    def carve(cutter: trimesh.Trimesh) -> trimesh.Trimesh | None:
        try:
            piece = booleans.intersection([mesh, cutter], config)
        except booleans.BooleanError:
            return None
        if piece is None:
            return None
        return repair.as_watertight_or_none(piece, config)

    return carve(negative_cutter), carve(positive_cutter)


def cross_section_polygons(
    mesh: trimesh.Trimesh, origin, normal, config: Config, frame_origin=None
) -> tuple[list[Polygon], np.ndarray]:
    """Polygons of the mesh cross-section, plus the plane's ``to_3d`` transform.

    Polygons are returned in the plane frame from :func:`plane_transforms`,
    sorted by descending area so downstream selection is stable.

    ``frame_origin`` lets a section taken slightly off a plane be expressed in
    that plane's own frame, so two probes either side of a cut share one 2D
    coordinate system and their polygons can be intersected directly.
    """
    to_2d, to_3d = plane_transforms(
        origin if frame_origin is None else frame_origin, normal
    )
    try:
        section = mesh.section(plane_origin=np.asarray(origin, dtype=float),
                               plane_normal=unit(normal))
    except Exception:
        return [], to_3d
    if section is None:
        return [], to_3d
    try:
        planar, _ = section.to_2D(to_2D=to_2d, check=False)
    except Exception:
        return [], to_3d
    polygons = [p for p in planar.polygons_full if p.area > config.area_epsilon_mm2]
    polygons.sort(key=lambda p: (-p.area, round(p.bounds[0], 6), round(p.bounds[1], 6)))
    return polygons, to_3d


def cross_section_area(
    mesh: trimesh.Trimesh, origin, normal, config: Config
) -> float:
    polygons, _ = cross_section_polygons(mesh, origin, normal, config)
    return float(sum(p.area for p in polygons))


def point_2d_to_3d(point_2d, to_3d: np.ndarray) -> np.ndarray:
    """Lift a plane-frame 2D point back into world coordinates."""
    homogeneous = np.array([point_2d[0], point_2d[1], 0.0, 1.0], dtype=float)
    return (to_3d @ homogeneous)[:3]


def seam_points(polygons, to_3d: np.ndarray, config: Config) -> np.ndarray:
    """Points spread evenly along the seam a cut would leave on the surface.

    The visible seam is the boundary of the cross-section, not its area: that
    boundary is exactly where the cutting plane breaks through the model's outer
    skin. Points are placed by arc length so a long edge gets proportionally
    more of them, and the half-step offset keeps them off the polygon's start
    vertex.
    """
    rings = [p.exterior for p in polygons if p.exterior.length > config.length_epsilon_mm]
    if not rings:
        return np.empty((0, N_AXES))
    total_length = sum(r.length for r in rings)

    points: list[np.ndarray] = []
    for ring in rings:
        share = ring.length / total_length
        count = max(int(round(config.seam_sample_points * share)), 1)
        for step in range(count):
            distance = ring.length * (step + 0.5) / count
            located = ring.interpolate(distance)
            points.append(point_2d_to_3d((located.x, located.y), to_3d))
    return np.asarray(points, dtype=float)
