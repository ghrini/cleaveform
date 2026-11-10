"""End-to-end Phase 1 run: STL in, printable parts and a guide out.

Also home to bounded recovery. When a cut produces a face that cannot carry a
connector, or a part that cannot be inserted, the offending cut is ruled out and
the decomposition is retried. Recovery is bounded by ``recovery_max_attempts``
and, critically, never changes the joint clearance in a production run:
the clearance schedule for a non-experimental run contains exactly one value,
the calibrated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from .cache import Cache
from .config import Config
from .decompose import DecompositionError
from .decompose.graph import CutGraph
from .export import (
    build_manifest,
    part_metadata,
    write_diagnostic,
    write_guide,
    write_manifest,
    write_parts,
)
from .geometry import BooleanError, analysis
from .io import LoadedModel
from .joints import JointError
from .stages import (
    cached_brick_layout,
    cached_decompose,
    cached_joints,
    cached_load_model,
    cached_occupancy,
    cached_validation,
)
from .validate import HARD, ConstraintReport, Finding, check_part

BRICK_DIRNAME = "bricks"
PLATE_DIRNAME = "plates"


class PipelineError(RuntimeError):
    """A run failed in a way bounded recovery could not resolve."""

    def __init__(self, message: str, detail: dict | None = None, notes=None):
        super().__init__(message)
        self.detail = detail or {}
        self.notes = list(notes or [])


@dataclass
class Attempt:
    """One pass through decompose, joints and validation."""

    index: int
    clearance_mm: float
    blacklisted: list[tuple[str, int]]
    outcome: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "attempt": self.index,
            "clearance_mm": self.clearance_mm,
            "ruled_out_cuts": [
                {"node": path, "candidate": index} for path, index in self.blacklisted
            ],
            "outcome": self.outcome,
            "reason": self.reason,
        }


@dataclass
class RunResult:
    ok: bool
    out_dir: Path
    model: LoadedModel
    manifest: dict[str, Any] | None = None
    part_paths: dict[str, Path] = field(default_factory=dict)
    guide_path: Path | None = None
    metadata_path: Path | None = None
    diagnostic_path: Path | None = None
    attempts: list[Attempt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cache_stats: dict = field(default_factory=dict)


def clearance_schedule(config: Config) -> list[float]:
    """Clearance values recovery is allowed to try, in order.

    A production run gets exactly one: the calibrated value. Only an explicitly
    experimental run may explore within ``recovery_clearance_range_mm``.
    """
    if not config.experimental:
        return [config.joint_clearance_mm]
    span = config.recovery_clearance_range_mm
    values = [
        config.joint_clearance_mm,
        config.joint_clearance_mm - span,
        config.joint_clearance_mm + span,
    ]
    return [v for v in values if v >= 0.0]


def assert_production_clearance(config: Config, clearance_used: float) -> None:
    """Guard the Phase 1 recovery-safety property at the export boundary."""
    if config.experimental:
        return
    if abs(clearance_used - config.joint_clearance_mm) > config.length_epsilon_mm:
        raise PipelineError(
            "refusing to export: a production run must use the calibrated clearance "
            f"{config.joint_clearance_mm:.3f} mm, but {clearance_used:.3f} mm was used",
            {"constraint": "recovery_clearance_protection"},
        )


def run(
    input_path: str | Path,
    out_dir: str | Path,
    config: Config,
    input_units: str | None = None,
    target_size_mm: tuple[float, float, float] | None = None,
    scale_factor: float | None = None,
    on_progress=None,
) -> RunResult:
    """Execute the full Phase 1 pipeline.

    On success the output directory holds ``parts/*.stl``, ``metadata.json`` and
    ``assembly_guide.txt``. On failure it holds ``diagnostic.txt`` instead, and
    :class:`PipelineError` is raised.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log = on_progress or (lambda message: None)

    cache = Cache(config)
    model, mesh_key = cached_load_model(
        input_path,
        config,
        cache,
        input_units=input_units,
        target_size_mm=target_size_mm,
        scale_factor=scale_factor,
    )
    log(f"loaded {model.source_path.name}, bbox {_extent_text(model.final_extents_mm)}")

    if config.decomposition_mode == "brick":
        return _run_brick(model, mesh_key, out_path, config, cache, log)

    attempts: list[Attempt] = []
    blacklist: set[tuple[str, int]] = set()
    attempt_index = 0
    last_error: PipelineError | None = None
    last_retry: _Retryable | None = None

    for clearance in clearance_schedule(config):
        attempt_config = (
            config
            if clearance == config.joint_clearance_mm
            else config.with_overrides(joint_clearance_mm=clearance)
        )
        blacklist = set()

        while attempt_index < config.recovery_max_attempts:
            attempt_index += 1
            attempt = Attempt(
                index=attempt_index,
                clearance_mm=clearance,
                blacklisted=sorted(blacklist),
                outcome="pending",
            )
            attempts.append(attempt)
            try:
                result = _attempt(
                    model.mesh, attempt_config, log, set(blacklist), cache, mesh_key
                )
            except _Retryable as retry:
                attempt.outcome = "retry"
                attempt.reason = str(retry)
                last_retry = retry
                if retry.cut is None:
                    last_error = PipelineError(str(retry), retry.detail, model.notes)
                    break
                if retry.cut in blacklist:
                    last_error = PipelineError(
                        f"{retry} (this cut was already ruled out; no alternative remains)",
                        retry.detail,
                        model.notes,
                    )
                    break
                blacklist.add(retry.cut)
                log(f"recovery: ruled out cut {retry.cut[0]} candidate {retry.cut[1]}")
                continue
            except DecompositionError as exc:
                attempt.outcome = "failed"
                attempt.reason = str(exc)
                if last_retry is not None and blacklist:
                    # Every alternative cut was ruled out by the same underlying
                    # problem; lead with that rather than with the symptom.
                    last_error = PipelineError(
                        f"{last_retry} - and every alternative cut was ruled out for "
                        f"the same reason ({exc})",
                        {**last_retry.detail, **exc.detail},
                        model.notes,
                    )
                else:
                    last_error = PipelineError(str(exc), exc.detail, model.notes)
                break

            attempt.outcome = "accepted"
            assert_production_clearance(config, clearance)
            run_result = _finalize(
                model=model,
                graph=result.graph,
                meshes=result.meshes,
                orientations=result.orientations,
                joints=result.joint_dicts,
                report=result.report,
                plan_steps=result.steps,
                base_part=result.base_part,
                decomposition=result.decomposition,
                cache_stats=result.cache_stats,
                config=attempt_config,
                production_config=config,
                out_path=out_path,
                attempts=attempts,
                clearance=clearance,
                warnings=result.warnings + model.notes,
                log=log,
            )
            return run_result

        if last_error is not None and attempt_index >= config.recovery_max_attempts:
            break

    message = (
        str(last_error)
        if last_error is not None
        else f"no valid decomposition found within {config.recovery_max_attempts} attempts"
    )
    detail = last_error.detail if last_error is not None else {}
    seen: set[str] = set()
    context: list[str] = []
    for note in model.notes + [a.reason for a in attempts if a.reason]:
        if note not in seen:
            seen.add(note)
            context.append(note)
    diagnostic = write_diagnostic(
        out_path, message, {**detail, "attempts": len(attempts)}, context
    )
    raise PipelineError(message, {**detail, "diagnostic": str(diagnostic)}, model.notes)


class _Retryable(RuntimeError):
    """An in-run failure that bounded recovery may be able to route around."""

    def __init__(self, message: str, cut: tuple[str, int] | None, detail: dict):
        super().__init__(message)
        self.cut = cut
        self.detail = detail


@dataclass
class _AttemptResult:
    graph: CutGraph
    meshes: dict[str, trimesh.Trimesh]
    orientations: dict[str, tuple]
    joint_dicts: list[dict]
    report: ConstraintReport
    steps: list[dict]
    base_part: str | None
    decomposition: dict
    warnings: list[str]
    cache_stats: dict


def _attempt(
    mesh: trimesh.Trimesh,
    config: Config,
    log,
    blacklist: set[tuple[str, int]],
    cache: Cache,
    mesh_key: str,
) -> _AttemptResult:
    """One decompose, joint and validate pass. Raises :class:`_Retryable`."""
    decomposition, chunk_key = cached_decompose(
        mesh, config, cache, mesh_key, blacklist
    )
    log(f"decomposed into {decomposition.part_count} part(s)")

    try:
        graph, joint_result, joint_key = cached_joints(
            decomposition, config, cache, chunk_key
        )
        log(f"found {len(graph.interfaces)} cut face(s) needing connectors")
    except JointError as exc:
        cut = (exc.node_path, exc.candidate_index) if exc.node_path else None
        raise _Retryable(str(exc), cut, exc.detail) from exc
    except BooleanError as exc:
        raise _Retryable(f"boolean backend failure: {exc}", None, {}) from exc

    meshes = joint_result.meshes

    orientations, findings, steps, base_part, _ = cached_validation(
        decomposition, graph, meshes, config, cache, joint_key
    )
    report = ConstraintReport(list(findings))

    hard = report.hard_failures
    if hard:
        _raise_for_hard_failure(hard, graph)

    log(f"validated {len(meshes)} part(s), {len(steps)} assembly step(s)")
    return _AttemptResult(
        graph=graph,
        meshes=meshes,
        orientations=orientations,
        joint_dicts=[p.to_dict() for p in joint_result.placements],
        report=report,
        steps=steps,
        base_part=base_part,
        decomposition={
            "part_count": decomposition.part_count,
            "cuts": decomposition.cut_log,
            "warnings": decomposition.warnings,
        },
        warnings=decomposition.warnings + graph.warnings + joint_result.warnings,
        cache_stats=cache.stats.to_dict(),
    )


def _raise_for_hard_failure(hard: list[Finding], graph: CutGraph) -> None:
    """Translate a hard validation failure into a retry against a specific cut."""
    first = hard[0]
    cut: tuple[str, int] | None = None
    if first.part_id:
        interfaces = sorted(graph.interfaces_for(first.part_id), key=lambda i: i.id)
        if interfaces:
            cut = (interfaces[-1].node_path, interfaces[-1].candidate_index)
    raise _Retryable(
        f"[{first.tier}] {first.code}: {first.message}",
        cut,
        {**first.detail, "code": first.code, "tier": HARD},
    )


def _finalize(
    model: LoadedModel,
    graph: CutGraph,
    meshes: dict[str, trimesh.Trimesh],
    orientations: dict[str, tuple],
    joints: list[dict],
    report: ConstraintReport,
    plan_steps: list[dict],
    base_part: str | None,
    decomposition: dict,
    cache_stats: dict,
    config: Config,
    production_config: Config,
    out_path: Path,
    attempts: list[Attempt],
    clearance: float,
    warnings: list[str],
    log,
) -> RunResult:
    """Write every output file for an accepted run."""
    joints_by_part: dict[str, list[str]] = {}
    for joint in joints:
        joints_by_part.setdefault(joint["male_part"], []).append(joint["id"])
        joints_by_part.setdefault(joint["female_part"], []).append(joint["id"])

    parts_meta = {}
    for part_id in sorted(meshes):
        label, transform, oriented, _ = orientations[part_id]
        parts_meta[part_id] = part_metadata(
            part_id,
            meshes[part_id],
            oriented,
            label,
            transform,
            config,
            sorted(joints_by_part.get(part_id, [])),
        )

    part_paths = write_parts(meshes, out_path, config)
    log(f"wrote {len(part_paths)} STL file(s)")

    constraints = report.to_dict()
    recovery = {
        "attempts": [a.to_dict() for a in attempts],
        "clearance_used_mm": clearance,
        "clearance_is_calibrated": not production_config.experimental,
        "experimental": production_config.experimental,
    }
    manifest = build_manifest(
        config=config,
        input_summary=model.summary(),
        parts=parts_meta,
        joints=joints,
        steps=plan_steps,
        constraints=constraints,
        decomposition=decomposition,
        recovery=recovery,
        cache=cache_stats,
    )
    metadata_path = write_manifest(manifest, out_path)
    guide_path = write_guide(
        out_path,
        config,
        model.summary(),
        parts_meta,
        joints,
        plan_steps,
        base_part,
        constraints,
        warnings + graph.warnings,
    )
    log(f"wrote {metadata_path.name} and {guide_path.name}")

    return RunResult(
        ok=True,
        out_dir=out_path,
        model=model,
        manifest=manifest,
        part_paths=part_paths,
        guide_path=guide_path,
        metadata_path=metadata_path,
        attempts=attempts,
        warnings=warnings + graph.warnings,
        cache_stats=cache_stats,
    )


def _extent_text(extents) -> str:
    return f"{extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm"


def _run_brick(
    model: LoadedModel,
    mesh_key: str,
    out_path: Path,
    config: Config,
    cache: Cache,
    log,
) -> RunResult:
    """Brick-mode run: voxelize, tile, export shapes and plates.

    Deliberately not routed through the chunk path. Bricks are repeated copies
    of a handful of shapes, so there is nothing per-part to decompose, joint or
    sequence: doing so would build five hundred unique parts, run five hundred
    boolean joint operations and write five hundred files to describe six
    distinct objects.
    """
    from .brick.assembly import pack_plates
    from .brick.guide import write_bill_of_materials, write_brick_guide
    from .brick.library import build_all, load_library
    from .brick.tiling import TilingError
    from .estimate import batch_summary
    from .export.metadata import build_brick_manifest

    try:
        grid, voxel_key = cached_occupancy(model.mesh, config, cache, mesh_key)
        log(
            f"voxelized to {grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]} cells "
            f"at {config.brick_unit_mm:.1f} mm"
        )
        layout, _ = cached_brick_layout(grid, config, cache, voxel_key)
    except TilingError as exc:
        diagnostic = write_diagnostic(out_path, str(exc), exc.detail, model.notes)
        raise PipelineError(str(exc), {**exc.detail, "diagnostic": str(diagnostic)}) from exc

    log(
        f"placed {layout.brick_count} brick(s) from "
        f"{len(layout.counts_by_type)} shape(s)"
    )
    log(f"builds as {layout.piece_summary()}")

    report = ConstraintReport(list(layout.findings))
    counts = layout.counts_by_type
    meshes = build_all(config, used=set(counts))

    # Manufacturability is checked per SHAPE, not per brick: every copy of a
    # shape is the same object, so checking each placement would run the same
    # test hundreds of times for the same answer.
    footprints = {b.id: b for b in load_library(config)}
    for type_id in sorted(meshes):
        report.extend(check_part(type_id, meshes[type_id], meshes[type_id], config))

    hard = report.hard_failures
    if hard:
        message = f"[{hard[0].tier}] {hard[0].code}: {hard[0].message}"
        diagnostic = write_diagnostic(out_path, message, hard[0].detail, model.notes)
        raise PipelineError(message, {"diagnostic": str(diagnostic)})

    plates = pack_plates(counts, meshes, config)
    estimates = batch_summary(meshes, counts, len(plates), config)
    for type_id, detail in estimates["per_type"].items():
        brick = footprints.get(type_id)
        detail["footprint"] = brick.footprint if brick else "?"

    part_paths = write_parts(meshes, out_path, config, subdirectory=BRICK_DIRNAME)
    plate_paths = _write_plates(plates, out_path, config)
    log(f"wrote {len(part_paths)} shape(s) and {len(plate_paths)} plate(s)")

    manifest = build_brick_manifest(
        config=config,
        input_summary=model.summary(),
        layout=layout.to_dict(),
        estimates=estimates,
        constraints=report.to_dict(),
        cache=cache.stats.to_dict(),
    )
    metadata_path = write_manifest(manifest, out_path)
    guide_path = write_brick_guide(
        out_path, layout, config, model.summary(), estimates
    )
    write_bill_of_materials(out_path, layout, estimates)
    log(f"wrote {metadata_path.name} and {guide_path.name}")

    return RunResult(
        ok=True,
        out_dir=out_path,
        model=model,
        manifest=manifest,
        part_paths={**part_paths, **plate_paths},
        guide_path=guide_path,
        metadata_path=metadata_path,
        warnings=list(model.notes) + _layout_advisories(layout),
        cache_stats=cache.stats.to_dict(),
    )


def _layout_advisories(layout) -> list[str]:
    """Non-blocking layout findings, as lines the CLI can print.

    These used to live only in the guide's notes, so a run that left a stray
    group of bricks looked identical on the command line to one that did not.
    """
    from .validate.constraints import SOFT, WARNING

    return [
        f"{finding.code}: {finding.message}"
        for finding in layout.findings
        if finding.tier in (SOFT, WARNING)
    ]


def _write_plates(plates, out_path: Path, config: Config) -> dict[str, Path]:
    """Write each packed plate as its own STL."""
    if not plates:
        return {}
    directory = out_path / PLATE_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for index, plate in enumerate(plates, start=1):
        name = f"plate_{index:02d}"
        path = directory / f"{name}.stl"
        path.write_bytes(trimesh.exchange.stl.export_stl(plate))
        written[name] = path
    return written
