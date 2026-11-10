"""Overhang, support, orientation and thickness analysis.

All figures produced here are heuristic estimates feeding the cost function and
the metadata report. They are explicitly not slicer output (spec section 12).
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from ..config import AXIS_NAMES, N_AXES, Config

UP = np.array([0.0, 0.0, 1.0])


def orientation_candidates() -> list[tuple[str, np.ndarray]]:
    """The six axis-aligned print orientations, in a fixed order.

    Each entry is ``(label, 4x4 rotation)`` where the label names the model axis
    that ends up pointing at the ceiling.
    """
    candidates: list[tuple[str, np.ndarray]] = []
    for axis in range(N_AXES):
        for sign in (1.0, -1.0):
            source = np.zeros(N_AXES)
            source[axis] = sign
            transform = trimesh.geometry.align_vectors(source, UP)
            label = f"{'+' if sign > 0 else '-'}{AXIS_NAMES[axis].upper()} up"
            candidates.append((label, transform))
    return candidates


def overhang_face_mask(mesh: trimesh.Trimesh, config: Config) -> np.ndarray:
    """Boolean mask of faces that overhang beyond the configured angle.

    A face needs support when the angle between its plane and the build plate is
    smaller than ``overhang_angle_deg``, which is equivalent to its downward
    normal component exceeding the cosine of that angle.
    """
    downward = -mesh.face_normals @ UP
    return downward > math.cos(math.radians(config.overhang_angle_deg))


def overhang_area(mesh: trimesh.Trimesh, config: Config) -> float:
    mask = overhang_face_mask(mesh, config)
    if not np.any(mask):
        return 0.0
    return float(np.sum(mesh.area_faces[mask]))


def estimate_support_volume(mesh: trimesh.Trimesh, config: Config) -> float:
    """Volume of support material the part would need, in mm^3.

    Each overhanging face contributes its projected area swept down to the build
    plate, scaled by the support infill factor.
    """
    mask = overhang_face_mask(mesh, config)
    if not np.any(mask):
        return 0.0
    downward = (-mesh.face_normals @ UP)[mask]
    projected = mesh.area_faces[mask] * downward
    heights = mesh.triangles_center[mask][:, 2] - float(mesh.bounds[0][2])
    heights = np.clip(heights, 0.0, None)
    return float(np.sum(projected * heights) * config.support_infill_factor)


def best_orientation(
    mesh: trimesh.Trimesh, config: Config
) -> tuple[str, np.ndarray, trimesh.Trimesh, float]:
    """Pick the axis-aligned orientation needing the least support.

    Only orientations that still fit the design ceiling are considered, so
    following the recommendation can never break the build-volume guarantee.
    Returns ``(label, transform, oriented_mesh, support_volume_mm3)``; ties
    resolve to the first candidate in the fixed order.
    """
    ceiling = np.asarray(config.design_ceiling_mm, dtype=float)
    best: tuple[str, np.ndarray, trimesh.Trimesh, float] | None = None
    fallback: tuple[str, np.ndarray, trimesh.Trimesh, float] | None = None

    for label, transform in orientation_candidates():
        rotated = mesh.copy()
        rotated.apply_transform(transform)
        support = estimate_support_volume(rotated, config)
        candidate = (label, transform, rotated, support)
        if fallback is None:
            fallback = candidate
        if np.any((rotated.bounds[1] - rotated.bounds[0]) > ceiling):
            continue
        if best is None or support < best[3] - config.volume_epsilon_mm3:
            best = candidate

    assert fallback is not None
    return best if best is not None else fallback


def requires_support(mesh: trimesh.Trimesh, config: Config) -> bool:
    return estimate_support_volume(mesh, config) > config.volume_epsilon_mm3


def min_local_thickness(mesh: trimesh.Trimesh, config: Config) -> float:
    """Coarse minimum wall thickness, sampled by inward ray casting.

    Returns ``inf`` when no sample produces a hit. This is advisory only: thin
    walls are flagged, never silently thickened (spec section 11).
    """
    face_count = len(mesh.faces)
    if face_count == 0:
        return math.inf
    step = max(face_count // config.thin_wall_sample_faces, 1)
    indices = np.arange(0, face_count, step)
    origins = mesh.triangles_center[indices]
    normals = mesh.face_normals[indices]
    starts = origins - normals * config.thin_wall_ray_offset_mm
    try:
        locations, ray_indices, _ = mesh.ray.intersects_location(
            ray_origins=starts, ray_directions=-normals, multiple_hits=False
        )
    except Exception:
        return math.inf
    if len(locations) == 0:
        return math.inf
    distances = np.linalg.norm(locations - starts[ray_indices], axis=1)
    distances = distances[distances > config.length_epsilon_mm]
    if len(distances) == 0:
        return math.inf
    return float(np.min(distances))


def solid_depth_along(
    mesh: trimesh.Trimesh, point, direction, config: Config
) -> float:
    """Depth of material from ``point`` travelling along ``direction``.

    Used to confirm a socket has enough material behind it. Returns ``0.0`` when
    the ray leaves the solid immediately.
    """
    origin = np.asarray(point, dtype=float)
    unit_direction = np.asarray(direction, dtype=float)
    unit_direction = unit_direction / float(np.linalg.norm(unit_direction))
    start = origin + unit_direction * config.thin_wall_ray_offset_mm
    try:
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=start.reshape(1, N_AXES),
            ray_directions=unit_direction.reshape(1, N_AXES),
            multiple_hits=True,
        )
    except Exception:
        return 0.0
    if len(locations) == 0:
        return 0.0
    distances = np.linalg.norm(locations - start, axis=1)
    return float(np.min(distances) + config.thin_wall_ray_offset_mm)


# Golden ratio, for the Fibonacci sphere used to spread visibility rays evenly
# over all directions without any random sampling.
GOLDEN_RATIO = (1.0 + 5.0**0.5) / 2.0


def sphere_directions(count: int) -> np.ndarray:
    """``count`` unit vectors spread evenly over the sphere, deterministically.

    A Fibonacci lattice, so repeated runs cast rays in exactly the same
    directions. Nothing here draws from a random generator.
    """
    index = np.arange(count, dtype=float)
    theta = 2.0 * np.pi * index / GOLDEN_RATIO
    z = 1.0 - (2.0 * index + 1.0) / count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


def seam_exposure(
    mesh: trimesh.Trimesh, seam_points: np.ndarray, config: Config
) -> float:
    """How visible a seam is, from 0 (fully buried) to 1 (fully exposed).

    Rays are fired from each seam point in every direction; the fraction that
    escape without striking the model again is its exposure. A seam running
    across an open outer face escapes in most directions and scores near 1. A
    seam tucked into a slot or a deep concave pocket has most directions
    blocked by the model's own walls and scores low, which is what makes a cost
    term built on this prefer hiding seams where they will not be seen.
    """
    if len(seam_points) == 0:
        return 1.0
    directions = sphere_directions(config.seam_visibility_rays)

    origins = np.repeat(seam_points, len(directions), axis=0)
    repeated = np.tile(directions, (len(seam_points), 1))
    # Step off the surface so a ray does not immediately hit its own origin.
    origins = origins + repeated * config.seam_ray_offset_mm

    try:
        blocked = mesh.ray.intersects_any(
            ray_origins=origins, ray_directions=repeated
        )
    except Exception:
        return 1.0
    return float(np.count_nonzero(~np.asarray(blocked)) / len(origins))
