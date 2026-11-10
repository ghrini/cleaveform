"""Module 12 - print time and material estimation.

These are coarse heuristics, not slicing. The user's slicer is authoritative;
every value produced here is labelled as an estimate wherever it is reported.
The model is calibrated once against real Bambu Studio slices by adjusting the
speed and infill constants in :mod:`cleaveform.config`.
"""

from __future__ import annotations

import math

import trimesh

from .config import Config
from .geometry import analysis

MM3_PER_CM3 = 1000.0
SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0
MM_PER_M = 1000.0


def material_volume_mm3(mesh: trimesh.Trimesh, config: Config) -> float:
    """Extruded plastic volume for the part body, excluding support.

    Two contributions, not one. Infill fills the interior at the configured
    density, but the perimeters and the top and bottom layers print solid
    whatever that density is, and they scale with surface area rather than
    volume. Calibration against real slices showed the volume-only form
    underestimating high-surface-area parts by around a quarter while
    overestimating bulky ones, which is exactly the shape of a missing skin.
    """
    volume = float(abs(mesh.volume))
    shell = float(mesh.area) * config.shell_thickness_mm
    # A thin part can be shell all the way through; it cannot be more than solid.
    return min(volume * config.infill_factor + shell, volume)


def weight_g(mesh: trimesh.Trimesh, config: Config) -> float:
    """Estimated part weight.

    Spec section 12 writes this as ``volume_cm3 x density x infill``. That omits
    the solid skin, so it understates every part. This uses the same extruded
    volume the time model uses, which is the physical amount of plastic.

    Unlike the time model, this has not been checked against real slices: the
    calibration run collected times, not filament weights. Treat it as better
    than the spec formula but still unverified.
    """
    return material_volume_mm3(mesh, config) / MM3_PER_CM3 * config.material_density_g_cm3


def filament_length_m(mesh: trimesh.Trimesh, config: Config) -> float:
    """Estimated filament consumed, in metres."""
    radius = config.filament_diameter_mm / 2.0
    cross_section = math.pi * radius * radius
    if cross_section <= 0:
        return 0.0
    return material_volume_mm3(mesh, config) / cross_section / MM_PER_M


def print_time_min(
    mesh: trimesh.Trimesh, config: Config, support_volume_mm3: float | None = None
) -> float:
    """Estimated print time in minutes.

    Extruded volume, including support, is converted to a bead length at the
    nozzle cross-section and divided by the nominal speed, plus a fixed
    per-layer overhead for travel and layer changes.
    """
    if support_volume_mm3 is None:
        support_volume_mm3 = analysis.estimate_support_volume(mesh, config)
    bead_area = config.nozzle_width_mm * config.layer_height_mm
    if bead_area <= 0 or config.nominal_speed_mm_s <= 0:
        return 0.0
    total_volume = material_volume_mm3(mesh, config) + float(support_volume_mm3)
    bead_length_mm = total_volume / bead_area
    extrusion_s = bead_length_mm / config.nominal_speed_mm_s

    height_mm = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    layers = max(int(math.ceil(height_mm / config.layer_height_mm)), 1)
    overhead_s = layers * config.layer_overhead_s

    return (extrusion_s + overhead_s) / SECONDS_PER_MINUTE


def summarize(mesh: trimesh.Trimesh, config: Config) -> dict[str, float]:
    """All estimates for one part, ready for the metadata report."""
    support = analysis.estimate_support_volume(mesh, config)
    return {
        "est_volume_mm3": float(abs(mesh.volume)),
        "est_surface_area_mm2": float(mesh.area),
        "est_support_volume_mm3": support,
        "est_weight_g": weight_g(mesh, config),
        "est_filament_m": filament_length_m(mesh, config),
        "est_print_time_min": print_time_min(mesh, config, support),
    }


def batch_print_time_min(
    meshes: dict[str, trimesh.Trimesh],
    counts: dict[str, int],
    plate_count: int,
    config: Config,
) -> float:
    """Print time for many copies of a few shapes, spread over plates.

    Charging every copy its own per-layer overhead would be badly wrong here.
    A brick is only a centimetre tall but that is still fifty layers, and at a
    second and a half each, five hundred bricks would book ten hours of layer
    changes that never happen: bricks printed together on one plate share their
    layers. Extrusion scales with the number of copies, overhead scales with the
    number of plates.
    """
    bead_area = config.nozzle_width_mm * config.layer_height_mm
    if bead_area <= 0 or config.nominal_speed_mm_s <= 0:
        return 0.0

    extrusion_s = 0.0
    tallest = 0.0
    for type_id, count in counts.items():
        mesh = meshes.get(type_id)
        if mesh is None or count <= 0:
            continue
        volume = material_volume_mm3(mesh, config) * count
        extrusion_s += volume / bead_area / config.nominal_speed_mm_s
        tallest = max(tallest, float(mesh.bounds[1][2] - mesh.bounds[0][2]))

    layers = max(int(math.ceil(tallest / config.layer_height_mm)), 1)
    overhead_s = layers * config.layer_overhead_s * max(plate_count, 1)
    return (extrusion_s + overhead_s) / SECONDS_PER_MINUTE


def batch_summary(
    meshes: dict[str, trimesh.Trimesh],
    counts: dict[str, int],
    plate_count: int,
    config: Config,
) -> dict:
    """Per-shape and total estimates for a brick build."""
    per_type = {}
    total_weight = 0.0
    total_filament = 0.0
    for type_id, count in sorted(counts.items()):
        mesh = meshes.get(type_id)
        if mesh is None:
            continue
        each_weight = weight_g(mesh, config)
        each_filament = filament_length_m(mesh, config)
        total_weight += each_weight * count
        total_filament += each_filament * count
        per_type[type_id] = {
            "count": count,
            "each_weight_g": each_weight,
            "total_weight_g": each_weight * count,
            "each_volume_mm3": float(abs(mesh.volume)),
        }
    return {
        "per_type": per_type,
        "plate_count": plate_count,
        "total_weight_g": total_weight,
        "total_filament_m": total_filament,
        "total_print_time_min": batch_print_time_min(
            meshes, counts, plate_count, config
        ),
    }
