"""Module 7 - the run manifest.

``metadata.json`` records what was produced, how it was configured, and every
estimate, clearly labelled as an estimate rather than slicer output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..config import Config
from ..estimate import summarize
from ..geometry import analysis, bbox

METADATA_FILENAME = "metadata.json"

ESTIMATE_NOTE = (
    "All est_* values are heuristics from cleaveform, not slicer output. "
    "Bambu Studio is authoritative."
)


def part_metadata(
    part_id: str,
    mesh: trimesh.Trimesh,
    oriented: trimesh.Trimesh,
    orientation_label: str,
    orientation_transform: np.ndarray,
    config: Config,
    joint_ids: list[str],
) -> dict[str, Any]:
    """Per-part record matching the Module 7 contract.

    Estimates are computed on the mesh in its recommended print orientation,
    because that is the orientation the figures are meant to describe. The
    bounding box is reported in the exported model frame.
    """
    estimates = summarize(oriented, config)
    bounds = bbox.aabb(mesh)
    extents = bbox.extents(mesh)
    return {
        "id": part_id,
        "material": config.material_name,
        "bbox_mm": {
            "min": [float(v) for v in bounds[0]],
            "max": [float(v) for v in bounds[1]],
            "extents": [float(v) for v in extents],
        },
        "orientation": {
            "recommended": orientation_label,
            "transform": [[float(v) for v in row] for row in orientation_transform],
            "oriented_extents_mm": [
                float(v) for v in (oriented.bounds[1] - oriented.bounds[0])
            ],
            "note": "apply this rotation in the slicer; the STL is exported in the model frame",
        },
        "supports_required": bool(
            estimates["est_support_volume_mm3"] > config.volume_epsilon_mm3
        ),
        "recommended_layer_height_mm": config.layer_height_mm,
        "joints": list(joint_ids),
        **estimates,
    }


def build_manifest(
    config: Config,
    input_summary: dict[str, Any],
    parts: dict[str, dict[str, Any]],
    joints: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    constraints: dict[str, Any],
    decomposition: dict[str, Any],
    recovery: dict[str, Any],
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = {
        "part_count": len(parts),
        "joint_count": len(joints),
        "est_total_weight_g": sum(p["est_weight_g"] for p in parts.values()),
        "est_total_filament_m": sum(p["est_filament_m"] for p in parts.values()),
        "est_total_print_time_min": sum(
            p["est_print_time_min"] for p in parts.values()
        ),
    }
    return {
        "tool": "cleaveform",
        "phase": 1,
        "note": ESTIMATE_NOTE,
        "input": input_summary,
        "config": config.to_dict(),
        "decomposition": decomposition,
        "recovery": recovery,
        "cache": cache or {},
        "totals": totals,
        "parts": [parts[k] for k in sorted(parts)],
        "joints": joints,
        "assembly": steps,
        "constraints": constraints,
    }


def write_manifest(manifest: dict[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / METADATA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def build_brick_manifest(
    config: Config,
    input_summary: dict[str, Any],
    layout: dict[str, Any],
    estimates: dict[str, Any],
    constraints: dict[str, Any],
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run manifest for a brick build.

    Separate from :func:`build_manifest` because the totals mean different
    things: ``part_count`` there is the number of files, which for bricks would
    report six when five hundred bricks were produced, and the weight total
    would be one of each rather than the whole bill of materials.
    """
    return {
        "tool": "cleaveform",
        "mode": "brick",
        "note": ESTIMATE_NOTE,
        "input": input_summary,
        "config": config.to_dict(),
        "cache": cache or {},
        "totals": {
            "brick_count": layout["brick_count"],
            "brick_type_count": layout["type_count"],
            "piece_count": layout.get("piece_count", 1),
            "largest_stray_bricks": layout["metrics"].get("largest_stray_bricks", 0),
            "plate_count": estimates.get("plate_count", 0),
            "est_total_weight_g": estimates.get("total_weight_g", 0.0),
            "est_total_filament_m": estimates.get("total_filament_m", 0.0),
            "est_total_print_time_min": estimates.get("total_print_time_min", 0.0),
        },
        "layout": layout,
        "estimates": estimates,
        "constraints": constraints,
    }
