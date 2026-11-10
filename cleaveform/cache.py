"""Cross-cutting cache (spec section 7).

Keys are the content hash of a stage's input combined with the config fields
that actually affect that stage. Stages chain: each one's key folds in the key
of the stage above it, so changing a parameter invalidates its own stage and
everything downstream, and nothing above.

Meshes are stored as raw ``float64`` vertex and ``int64`` face arrays. That is
not incidental. PLY and STL both round-trip through ``float32``, which shifts
vertices by around 1e-6 mm - invisible on a print, but enough to make a warm
run produce different bytes from a cold one and break the determinism
guarantee. Exact arrays keep a cache hit indistinguishable from a recompute.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .config import Config

# Stage names. ``voxel`` and ``brick`` belong to brick mode; ``chunk``,
# ``joint`` and ``validation`` to chunk mode. ``mesh`` is shared.
STAGE_MESH = "mesh"
STAGE_VOXEL = "voxel"
STAGE_CHUNK = "chunk"
STAGE_JOINT = "joint"
STAGE_VALIDATION = "validation"
STAGE_BRICK = "brick"

STAGES = (
    STAGE_MESH,
    STAGE_VOXEL,
    STAGE_CHUNK,
    STAGE_JOINT,
    STAGE_VALIDATION,
    STAGE_BRICK,
)

CACHE_FORMAT_VERSION = 2
MESH_NAMES_KEY = "__mesh_names__"
KEY_SHARD_CHARS = 2

# Which config fields affect which stage. A field in the wrong group is a
# correctness bug, not a performance one: too narrow and a stale result is
# served after a parameter change. ``test_cache.py`` asserts every field is
# accounted for, so adding a config field forces a decision here.
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    STAGE_MESH: (
        "repair_max_attempts",
        "min_plausible_dim_mm",
        "max_plausible_dim_mm",
        "volume_epsilon_mm3",
        "length_epsilon_mm",
    ),
    STAGE_VOXEL: (
        "brick_unit_mm",
        "brick_occupancy_threshold",
        "brick_fill_interior",
        "brick_keep_thin",
        "brick_thin_min_fill",
        "brick_thin_max_gap_cells",
    ),
    STAGE_BRICK: (
        # Tiling.
        "brick_unit_mm",
        "brick_library",
        "max_bricks",
        "brick_repair_passes",
        "brick_bond_fraction_warn",
        "brick_detached_fraction_hard",
        "brick_stray_hard_bricks",
        "brick_simplify",
        "brick_simplify_passes",
        "brick_sliver_max_cells",
        # Brick solid geometry.
        "brick_stud_radius_fraction",
        "brick_stud_height_fraction",
        "brick_fit_clearance_mm",
        "brick_roof_mm",
        "brick_tube_wall_mm",
        "brick_clutch_interference_mm",
        "brick_wall_mm",
        "brick_tube_clearance_mm",
        "brick_rib_relief_mm",
        "brick_fit_bias_mm",
        "peg_cylinder_segments",
        "min_wall_mm",
        "min_feature_mm",
        "area_epsilon_mm2",
        "volume_epsilon_mm3",
        "length_epsilon_mm",
    ),
    STAGE_CHUNK: (
        "decomposition_mode",
        "design_ceiling_mm",
        "min_part_mm",
        "max_parts",
        "max_cut_depth",
        "cut_fractions",
        "repair_max_attempts",
        # The cost function decides which cut wins, so every weight and every
        # input the cost reads belongs to this stage.
        "cost_preset",
        "w_part_count",
        "w_support_volume",
        "w_visible_seam_area",
        "w_joint_count",
        "w_print_time",
        "w_strength_penalty",
        "seam_sample_points",
        "seam_visibility_rays",
        "seam_ray_offset_mm",
        "overhang_angle_deg",
        "support_infill_factor",
        # Print time feeds the cost, so the estimation constants do too.
        "infill_factor",
        "layer_height_mm",
        "nominal_speed_mm_s",
        "nozzle_width_mm",
        "layer_overhead_s",
        "shell_thickness_mm",
        # The cutting ceiling is the design ceiling minus the peg depth.
        "peg_depth_mm",
        "area_epsilon_mm2",
        "volume_epsilon_mm3",
        "length_epsilon_mm",
    ),
    STAGE_JOINT: (
        "joint_type",
        "joint_clearance_mm",
        "peg_radius_mm",
        "peg_depth_mm",
        "min_pegs_per_face",
        "max_pegs_per_face",
        "peg_area_per_peg_mm2",
        "peg_edge_margin_mm",
        "socket_depth_margin_mm",
        "peg_grid_pitch_mm",
        "peg_cylinder_segments",
        "min_joint_face_area_mm2",
        "cut_face_probe_eps_mm",
        "min_wall_mm",
        "nozzle_width_mm",
        "thin_wall_ray_offset_mm",
        "area_epsilon_mm2",
        "volume_epsilon_mm3",
        "length_epsilon_mm",
    ),
    STAGE_VALIDATION: (
        "design_ceiling_mm",
        "min_feature_mm",
        "min_wall_mm",
        "sweep_samples",
        "sweep_distance_factor",
        "collision_volume_tolerance_mm3",
        "support_volume_soft_fraction",
        "print_time_soft_minutes",
        "seam_area_soft_fraction",
        "warp_aspect_ratio_warn",
        "tall_thin_aspect_warn",
        "auto_repair_thin_walls",
        "thin_wall_sample_faces",
        "thin_wall_ray_offset_mm",
        "stability_contact_area_fraction_warn",
        "overhang_angle_deg",
        "support_infill_factor",
        "infill_factor",
        "layer_height_mm",
        "nominal_speed_mm_s",
        "nozzle_width_mm",
        "layer_overhead_s",
        "shell_thickness_mm",
        "material_density_g_cm3",
        "filament_diameter_mm",
        "joint_type",
        "area_epsilon_mm2",
        "volume_epsilon_mm3",
        "length_epsilon_mm",
    ),
}

# Config fields that deliberately affect no cached stage, listed so the
# coverage test can tell "considered and excluded" from "forgotten".
UNCACHED_FIELDS = frozenset(
    {
        # Cache plumbing itself.
        "cache_enabled",
        "cache_dir",
        # Locked mode flags with a single legal value, or output-only settings
        # that never change computed geometry.
        "generation_backend",
        "export_format",
        "build_volume_single_nozzle_mm",
        "material_name",
        # Recovery and calibration operate outside the cached stages.
        "recovery_clearance_range_mm",
        "recovery_max_attempts",
        "experimental",
        "calibration_cube_mm",
        "calibration_peg_offset_fraction",
        "calibration_clearances_mm",
        "brick_calibration_interferences_mm",
        "brick_calibration_footprint",
        "brick_calibration_brim_mm",
        "brick_gauge_offsets_mm",
        "brick_calibration_batch",
        "brick_calibration_marker_radius_mm",
        "brick_calibration_marker_pitch_mm",
        # Plate packing is rigid transforms done at export time, so it never
        # changes a cached stage result.
        "plate_size_mm",
        "plate_margin_mm",
    }
)


class CacheError(RuntimeError):
    """Raised when a cache entry exists but cannot be read back."""


@dataclass
class CacheStats:
    hits: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)
    writes: dict[str, int] = field(default_factory=dict)

    def record(self, bucket: dict[str, int], stage: str) -> None:
        bucket[stage] = bucket.get(stage, 0) + 1

    @property
    def hit_count(self) -> int:
        return sum(self.hits.values())

    @property
    def miss_count(self) -> int:
        return sum(self.misses.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": dict(sorted(self.hits.items())),
            "misses": dict(sorted(self.misses.items())),
            "writes": dict(sorted(self.writes.items())),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
        }


class Cache:
    """A content-addressed store on disk, one directory per stage.

    A disabled cache is a working no-op: every lookup misses and every write is
    dropped, so callers need no branch of their own.
    """

    def __init__(self, config: Config) -> None:
        self.enabled = bool(config.cache_enabled)
        self.root = Path(config.cache_dir)
        self.stats = CacheStats()

    def stage_key(
        self, stage: str, config: Config, upstream: str = "", *extra: Any
    ) -> str:
        """Key for ``stage``: upstream key, its own config fields, and extras."""
        if stage not in STAGE_FIELDS:
            raise CacheError(f"unknown cache stage {stage!r}")
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "stage": stage,
            "upstream": upstream,
            "config": {name: _plain(getattr(config, name)) for name in STAGE_FIELDS[stage]},
            "extra": [_plain(item) for item in extra],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, stage: str, key: str) -> bytes | None:
        path = self._path(stage, key)
        if not self.enabled or not path.exists():
            if self.enabled:
                self.stats.record(self.stats.misses, stage)
            return None
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise CacheError(f"cache entry {path} could not be read: {exc}") from exc
        self.stats.record(self.stats.hits, stage)
        return blob

    def put(self, stage: str, key: str, blob: bytes) -> None:
        if not self.enabled:
            return
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and move into place, so a crash mid-write
        # cannot leave a truncated entry that later reads as a valid hit.
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(blob)
        temporary.replace(path)
        self.stats.record(self.stats.writes, stage)

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _path(self, stage: str, key: str) -> Path:
        return self.root / stage / key[:KEY_SHARD_CHARS] / f"{key}.bin"


def hash_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def hash_file(path: str | Path) -> str:
    return hash_bytes(Path(path).read_bytes())


def hash_mesh(mesh: trimesh.Trimesh) -> str:
    """Content hash of a mesh's exact vertex and face arrays."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def hash_meshes(meshes: dict[str, trimesh.Trimesh]) -> str:
    digest = hashlib.sha256()
    for name in sorted(meshes):
        digest.update(name.encode("utf-8"))
        digest.update(hash_mesh(meshes[name]).encode("ascii"))
    return digest.hexdigest()


def dump_mesh(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    return {
        "vertices": np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        "faces": np.ascontiguousarray(mesh.faces, dtype=np.int64),
    }


def load_mesh(arrays: dict[str, Any], prefix: str = "") -> trimesh.Trimesh:
    """Rebuild a mesh without reprocessing, so it is bit-identical to the original."""
    return trimesh.Trimesh(
        vertices=np.asarray(arrays[f"{prefix}vertices"], dtype=np.float64),
        faces=np.asarray(arrays[f"{prefix}faces"], dtype=np.int64),
        process=False,
    )


def pack(payload: dict[str, Any], meshes: dict[str, trimesh.Trimesh]) -> bytes:
    """Serialise a JSON-able payload plus named meshes into one blob.

    Mesh names go in the payload and the arrays are numbered, so a name
    containing dots or underscores - node paths look like ``root.0.1`` - can
    never collide with the array naming scheme.
    """
    names = sorted(meshes)
    arrays: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        for suffix, array in dump_mesh(meshes[name]).items():
            arrays[f"m{index}_{suffix}"] = array
    body = dict(payload)
    body[MESH_NAMES_KEY] = names
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        payload=np.frombuffer(
            json.dumps(body, sort_keys=True).encode("utf-8"), dtype=np.uint8
        ),
        **arrays,
    )
    return buffer.getvalue()


def unpack(blob: bytes) -> tuple[dict[str, Any], dict[str, trimesh.Trimesh]]:
    """Inverse of :func:`pack`."""
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            payload = json.loads(bytes(data["payload"]).decode("utf-8"))
            names = payload.pop(MESH_NAMES_KEY, [])
            meshes = {
                name: trimesh.Trimesh(
                    vertices=np.asarray(data[f"m{index}_vertices"], dtype=np.float64),
                    faces=np.asarray(data[f"m{index}_faces"], dtype=np.int64),
                    process=False,
                )
                for index, name in enumerate(names)
            }
    except Exception as exc:
        raise CacheError(f"cache entry is unreadable: {exc}") from exc
    return payload, meshes


def _plain(value: Any) -> Any:
    """Reduce a config value to something with a stable JSON encoding."""
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def all_config_field_names() -> set[str]:
    return {f.name for f in fields(Config)}
