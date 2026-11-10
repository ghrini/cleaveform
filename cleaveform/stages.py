"""Cached wrappers around the expensive pipeline stages.

Each wrapper computes its stage's key, returns a stored result when one is
present, and otherwise runs the real work and stores it. Keys chain, so a
parameter change invalidates its own stage and everything after it.

The stage boundaries are chosen around the boolean operations, which dominate a
run. In particular joints are their own stage: changing ``joint_clearance_mm``
after a calibration print re-cuts nothing, it only regenerates the connectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from shapely import wkb

from . import cache as cache_module
from .cache import (
    STAGE_BRICK,
    STAGE_CHUNK,
    STAGE_JOINT,
    STAGE_MESH,
    STAGE_VALIDATION,
    STAGE_VOXEL,
    Cache,
    hash_file,
    hash_mesh,
    hash_meshes,
    pack,
    unpack,
)
from .config import Config
from .decompose.graph import CutGraph, Interface, build_cut_graph
from .decompose.partition import (
    CutPlane,
    DecompositionResult,
    PartNode,
    decompose,
)
from .geometry import analysis
from .io import LoadedModel, load_model
from .joints import JointPlacement, JointResult, generate_joints
from .validate import (
    Finding,
    check_part,
    joint_face_on_bed_warning,
    plan_assembly,
    stability_warning,
)


# ----------------------------------------------------------------------
# Stage 1: load, scale and repair the input mesh
# ----------------------------------------------------------------------
def cached_load_model(
    path,
    config: Config,
    cache: Cache,
    input_units: str | None = None,
    target_size_mm: tuple[float, float, float] | None = None,
    scale_factor: float | None = None,
) -> tuple[LoadedModel, str]:
    """Load and repair the input, or return the stored result."""
    key = cache.stage_key(
        STAGE_MESH,
        config,
        hash_file(path),
        input_units,
        list(target_size_mm) if target_size_mm else None,
        scale_factor,
    )
    blob = cache.get(STAGE_MESH, key)
    if blob is not None:
        payload, meshes = unpack(blob)
        model = LoadedModel(
            mesh=meshes["mesh"],
            source_path=Path(payload["source_path"]),
            raw_extents_mm=tuple(payload["raw_extents_mm"]),
            final_extents_mm=tuple(payload["final_extents_mm"]),
            scale_applied=payload["scale_applied"],
            repaired=payload["repaired"],
            notes=list(payload["notes"]),
        )
        return model, key

    model = load_model(
        path,
        config,
        input_units=input_units,
        target_size_mm=target_size_mm,
        scale_factor=scale_factor,
    )
    cache.put(
        STAGE_MESH,
        key,
        pack(
            {
                "source_path": str(model.source_path),
                "raw_extents_mm": list(model.raw_extents_mm),
                "final_extents_mm": list(model.final_extents_mm),
                "scale_applied": model.scale_applied,
                "repaired": model.repaired,
                "notes": list(model.notes),
            },
            {"mesh": model.mesh},
        ),
    )
    return model, key


# ----------------------------------------------------------------------
# Stage 2: decomposition
# ----------------------------------------------------------------------
def cached_decompose(
    mesh: trimesh.Trimesh,
    config: Config,
    cache: Cache,
    upstream_key: str,
    blacklist: set[tuple[str, int]],
) -> tuple[DecompositionResult, str]:
    """Partition the mesh, or return the stored partition.

    The blacklist is part of the key: bounded recovery ruling out a cut is a
    genuinely different decomposition, not the same one recomputed.
    """
    key = cache.stage_key(
        STAGE_CHUNK, config, upstream_key, sorted(blacklist)
    )
    blob = cache.get(STAGE_CHUNK, key)
    if blob is not None:
        payload, meshes = unpack(blob)
        root = _load_tree(payload["tree"], meshes)
        parts = [node for node in _walk(root) if node.is_leaf]
        parts.sort(key=lambda node: payload["part_order"].index(node.path))
        return (
            DecompositionResult(
                root=root,
                parts=parts,
                warnings=list(payload["warnings"]),
                cut_log=list(payload["cut_log"]),
            ),
            key,
        )

    result = decompose(mesh, config, blacklist)
    cache.put(
        STAGE_CHUNK,
        key,
        pack(
            {
                "tree": _dump_tree(result.root),
                "part_order": [node.path for node in result.parts],
                "warnings": list(result.warnings),
                "cut_log": list(result.cut_log),
            },
            {node.path: node.mesh for node in _walk(result.root)},
        ),
    )
    return result, key


def _walk(node: PartNode) -> list[PartNode]:
    out = [node]
    for child in node.children:
        out.extend(_walk(child))
    return out


def _dump_tree(node: PartNode) -> dict[str, Any]:
    return {
        "path": node.path,
        "depth": node.depth,
        "part_id": node.part_id,
        "cut": None
        if node.cut is None
        else {
            "axis": node.cut.axis,
            "origin": list(node.cut.origin),
            "normal": list(node.cut.normal),
            "fraction": node.cut.fraction,
            "candidate_index": node.cut.candidate_index,
            "face_area_mm2": node.cut.face_area_mm2,
        },
        "children": [_dump_tree(child) for child in node.children],
    }


def _load_tree(data: dict[str, Any], meshes: dict[str, trimesh.Trimesh]) -> PartNode:
    cut = data["cut"]
    node = PartNode(
        path=data["path"],
        mesh=meshes[data["path"]],
        depth=data["depth"],
        cut=None
        if cut is None
        else CutPlane(
            axis=cut["axis"],
            origin=tuple(cut["origin"]),
            normal=tuple(cut["normal"]),
            fraction=cut["fraction"],
            candidate_index=cut["candidate_index"],
            face_area_mm2=cut["face_area_mm2"],
        ),
        part_id=data["part_id"],
    )
    node.children = [_load_tree(child, meshes) for child in data["children"]]
    return node


# ----------------------------------------------------------------------
# Stage 3: cut graph and connectors
# ----------------------------------------------------------------------
def cached_joints(
    decomposition: DecompositionResult,
    config: Config,
    cache: Cache,
    upstream_key: str,
) -> tuple[CutGraph, JointResult, str]:
    """Build the cut graph and fit connectors, or return the stored result."""
    key = cache.stage_key(STAGE_JOINT, config, upstream_key)
    blob = cache.get(STAGE_JOINT, key)
    parts = {node.part_id: node for node in decomposition.parts if node.part_id}

    if blob is not None:
        payload, meshes = unpack(blob)
        graph = CutGraph(
            parts=parts,
            interfaces=[_load_interface(item) for item in payload["interfaces"]],
            warnings=list(payload["graph_warnings"]),
        )
        joints = JointResult(
            meshes=meshes,
            placements=[_load_placement(item) for item in payload["placements"]],
            warnings=list(payload["joint_warnings"]),
        )
        return graph, joints, key

    graph = build_cut_graph(decomposition, config)
    joints = generate_joints(graph, config)
    cache.put(
        STAGE_JOINT,
        key,
        pack(
            {
                "interfaces": [_dump_interface(i) for i in graph.interfaces],
                "graph_warnings": list(graph.warnings),
                "placements": [_dump_placement(p) for p in joints.placements],
                "joint_warnings": list(joints.warnings),
            },
            joints.meshes,
        ),
    )
    return graph, joints, key


def _dump_interface(interface: Interface) -> dict[str, Any]:
    return {
        "id": interface.id,
        "male_id": interface.male_id,
        "female_id": interface.female_id,
        "axis": interface.axis,
        "origin": [float(v) for v in interface.origin],
        "normal": [float(v) for v in interface.normal],
        # WKB keeps the polygon's float64 coordinates exactly; WKT would round.
        "polygon_wkb": wkb.dumps(interface.polygon).hex(),
        "to_3d": [[float(v) for v in row] for row in interface.to_3d],
        "area_mm2": interface.area_mm2,
        "node_path": interface.node_path,
        "candidate_index": interface.candidate_index,
    }


def _load_interface(data: dict[str, Any]) -> Interface:
    return Interface(
        id=data["id"],
        male_id=data["male_id"],
        female_id=data["female_id"],
        axis=data["axis"],
        origin=np.asarray(data["origin"], dtype=float),
        normal=np.asarray(data["normal"], dtype=float),
        polygon=wkb.loads(bytes.fromhex(data["polygon_wkb"])),
        to_3d=np.asarray(data["to_3d"], dtype=float),
        area_mm2=data["area_mm2"],
        node_path=data["node_path"],
        candidate_index=data["candidate_index"],
    )


def _dump_placement(placement: JointPlacement) -> dict[str, Any]:
    return placement.to_dict()


def _load_placement(data: dict[str, Any]) -> JointPlacement:
    return JointPlacement(
        interface_id=data["id"],
        male_id=data["male_part"],
        female_id=data["female_part"],
        axis_name=data["axis"],
        normal=np.asarray(data["normal"], dtype=float),
        centers_mm=[np.asarray(c, dtype=float) for c in data["centers_mm"]],
        peg_radius_mm=data["peg_radius_mm"],
        socket_radius_mm=data["socket_radius_mm"],
        peg_depth_mm=data["peg_depth_mm"],
        clearance_mm=data["clearance_mm"],
        face_area_mm2=data["face_area_mm2"],
    )


# ----------------------------------------------------------------------
# Stage 4: orientation, constraint checks and the assembly plan
# ----------------------------------------------------------------------
def cached_validation(
    decomposition: DecompositionResult,
    graph: CutGraph,
    meshes: dict[str, trimesh.Trimesh],
    config: Config,
    cache: Cache,
    upstream_key: str,
) -> tuple[dict[str, tuple], list[Finding], list[dict], str | None, str]:
    """Orientations, manufacturability findings and the assembly plan.

    These are cached together because they share inputs and, profiled on a real
    model, together account for about a third of a run - the constraint checks
    alone ray-cast every part looking for thin walls.

    Orientations store only a label and a transform. The oriented mesh is
    rebuilt by applying that transform, a deterministic matrix multiply, rather
    than storing a second copy of every part.
    """
    key = cache.stage_key(STAGE_VALIDATION, config, upstream_key, hash_meshes(meshes))
    blob = cache.get(STAGE_VALIDATION, key)
    if blob is not None:
        payload, _ = unpack(blob)
        orientations = _rebuild_orientations(payload["orientations"], meshes)
        findings = [
            Finding(
                tier=item["tier"],
                code=item["code"],
                message=item["message"],
                part_id=item["part_id"],
                detail=item["detail"],
            )
            for item in payload["findings"]
        ]
        return orientations, findings, payload["steps"], payload["base_part"], key

    orientations = {
        part_id: analysis.best_orientation(meshes[part_id], config)
        for part_id in sorted(meshes)
    }

    joint_area: dict[str, float] = {}
    for interface in graph.interfaces:
        joint_area[interface.male_id] = (
            joint_area.get(interface.male_id, 0.0) + interface.area_mm2
        )
        joint_area[interface.female_id] = (
            joint_area.get(interface.female_id, 0.0) + interface.area_mm2
        )

    findings: list[Finding] = []
    for part_id in sorted(meshes):
        _, transform, oriented, _ = orientations[part_id]
        area = joint_area.get(part_id, 0.0)
        findings.extend(check_part(part_id, meshes[part_id], oriented, config, area))
        for warning in (
            stability_warning(part_id, meshes[part_id], area, config),
            joint_face_on_bed_warning(
                part_id,
                [i.normal for i in graph.interfaces_for(part_id)],
                transform,
                config,
            ),
        ):
            if warning is not None:
                findings.append(warning)

    plan = plan_assembly(decomposition, graph, meshes, config)
    findings.extend(plan.findings)
    steps = [step.to_dict() for step in plan.steps]

    cache.put(
        STAGE_VALIDATION,
        key,
        pack(
            {
                "orientations": {
                    part_id: {
                        "label": label,
                        "transform": [[float(v) for v in row] for row in transform],
                        "support": float(support),
                    }
                    for part_id, (label, transform, _, support) in orientations.items()
                },
                "findings": [f.to_dict() for f in findings],
                "steps": steps,
                "base_part": plan.base_part,
            },
            {},
        ),
    )
    return orientations, findings, steps, plan.base_part, key


def _rebuild_orientations(
    stored: dict[str, Any], meshes: dict[str, trimesh.Trimesh]
) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for part_id, item in stored.items():
        transform = np.asarray(item["transform"], dtype=float)
        oriented = meshes[part_id].copy()
        oriented.apply_transform(transform)
        out[part_id] = (item["label"], transform, oriented, item["support"])
    return out


# ----------------------------------------------------------------------
# Brick mode: occupancy grid, then the brick layout
# ----------------------------------------------------------------------
def cached_occupancy(mesh: trimesh.Trimesh, config: Config, cache: Cache, upstream_key: str):
    """Voxelize onto the brick lattice, or return the stored grid.

    Fills the ``voxel`` cache layer, which the spec reserved and chunk mode
    never had a use for.
    """
    from .brick.occupancy import OccupancyGrid, build_occupancy

    key = cache.stage_key(STAGE_VOXEL, config, upstream_key)
    blob = cache.get(STAGE_VOXEL, key)
    if blob is not None:
        payload, _ = unpack(blob)
        return grid_from_payload(payload), key

    grid = build_occupancy(mesh, config)
    cache.put(
        STAGE_VOXEL,
        key,
        pack(grid_to_payload(grid), {}),
    )
    return grid, key


def grid_to_payload(grid) -> dict:
    """A JSON-able form of an occupancy grid.

    Bit-packed: a bool array stores one byte per cell, and a 35x35x35 grid is
    mostly empty space either way.
    """
    return {
        "packed": np.packbits(grid.filled.ravel()).tolist(),
        "shape": list(grid.shape),
        "pitch_mm": grid.pitch_mm,
        "origin_mm": [float(v) for v in grid.origin_mm],
        "added_thin": int(grid.added_thin),
    }


def grid_from_payload(payload: dict):
    from .brick.occupancy import OccupancyGrid

    return OccupancyGrid(
        filled=np.unpackbits(
            np.asarray(payload["packed"], dtype=np.uint8),
            count=int(np.prod(payload["shape"])),
        )
        .astype(bool)
        .reshape(payload["shape"]),
        pitch_mm=payload["pitch_mm"],
        origin_mm=np.asarray(payload["origin_mm"], dtype=float),
        added_thin=int(payload.get("added_thin", 0)),
    )


def cached_brick_layout(grid, config: Config, cache: Cache, upstream_key: str):
    """Tile, repair and analyse the grid, or return the stored layout."""
    from .brick.layout import BrickLayout, add_change_finding, analyse, build_layout
    from .brick.tiling import BrickPlacement

    key = cache.stage_key(STAGE_BRICK, config, upstream_key)
    blob = cache.get(STAGE_BRICK, key)
    if blob is not None:
        payload, _ = unpack(blob)
        placements = [
            BrickPlacement(
                type_id=item["type_id"],
                x=item["x"],
                y=item["y"],
                z=item["z"],
                width_studs=item["width_studs"],
                depth_studs=item["depth_studs"],
                rotated=item["rotated"],
            )
            for item in payload["placements"]
        ]
        # The stored grid, not the one passed in: simplification may have changed
        # the shape, and the bricks only make sense against the grid they tile.
        stored = grid_from_payload(payload["grid"]) if "grid" in payload else grid
        layout = analyse(placements, stored, config)
        layout.changes = dict(payload.get("changes", {}))
        return add_change_finding(layout), key

    layout = build_layout(grid, config)
    cache.put(
        STAGE_BRICK,
        key,
        pack(
            {
                "placements": [
                    {
                        "type_id": p.type_id,
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                        "width_studs": p.width_studs,
                        "depth_studs": p.depth_studs,
                        "rotated": p.rotated,
                    }
                    for p in layout.placements
                ],
                "grid": grid_to_payload(layout.grid),
                "changes": dict(layout.changes),
            },
            {},
        ),
    )
    return layout, key
