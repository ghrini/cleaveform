"""Chunk-mode assembly validation (spec module 6).

Brick-mode research tooling does not accept arbitrary chunks, so chunk mode gets
its own checks: graph-based assembly ordering, connector-axis accessibility, and
collision-free insertion. None of them need Gurobi, a network, or any solver.

The ordering comes from the partition tree read bottom-up. That matters: a part
in the middle of a 2D grid of chunks touches neighbours on two axes at once and
cannot be seated by any single translation, so adding parts one at a time
deadlocks on most real models. Undoing the cuts in reverse instead gives the
order a person would use anyway - build each column, then join the columns -
and every join is a single rigid translation along one cut normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..config import Config
from ..decompose.graph import CutGraph
from ..decompose.partition import DecompositionResult, PartNode
from ..geometry import collision
from .constraints import HARD, WARNING, Finding

NO_ROTATION = "none (parts export in the model frame)"


@dataclass
class AssemblyStep:
    """One join: a moving group of parts seated onto a fixed group."""

    step_no: int
    moving_parts: list[str]
    fixed_parts: list[str]
    joint_ids: list[str]
    joint_type: str
    insertion_direction: list[float]
    insertion_axis: str
    rotation: str
    note: str | None = None

    @property
    def part_id(self) -> str:
        """Representative moving part, for the per-step metadata contract."""
        return self.moving_parts[0]

    def to_dict(self) -> dict:
        return {
            "step_no": self.step_no,
            "part_id": self.part_id,
            "moving_parts": list(self.moving_parts),
            "fixed_parts": list(self.fixed_parts),
            "attaches_to": list(self.fixed_parts),
            "joint_ids": list(self.joint_ids),
            "joint_type": self.joint_type,
            "insertion_direction": list(self.insertion_direction),
            "insertion_axis": self.insertion_axis,
            "rotation": self.rotation,
            "note": self.note,
        }


@dataclass
class AssemblyPlan:
    steps: list[AssemblyStep] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    base_part: str | None = None
    detached_groups: int = 0

    @property
    def ok(self) -> bool:
        return not any(f.tier == HARD for f in self.findings)


def plan_assembly(
    decomposition: DecompositionResult,
    graph: CutGraph,
    meshes: dict[str, trimesh.Trimesh],
    config: Config,
) -> AssemblyPlan:
    """Order the joins and verify each one can physically be made.

    Walks the partition tree in post-order so a cut is always undone after every
    cut beneath it. The result is deterministic: the tree, the child order and
    the sorted part ids fully determine the sequence.
    """
    plan = AssemblyPlan()
    interfaces_by_node: dict[str, list] = {}
    for interface in graph.interfaces:
        interfaces_by_node.setdefault(interface.node_path, []).append(interface)

    def assemble(node: PartNode) -> list[str]:
        if node.is_leaf:
            return [node.part_id] if node.part_id else []

        left = assemble(node.children[0])
        right = assemble(node.children[1])
        if not left or not right:
            return left + right

        joints = sorted(interfaces_by_node.get(node.path, []), key=lambda i: i.id)
        left_set, right_set = set(left), set(right)
        joints = [
            i
            for i in joints
            if {i.male_id, i.female_id} & left_set and {i.male_id, i.female_id} & right_set
        ]
        normal = np.asarray(node.cut.normal, dtype=float)

        # Move the smaller group; ties move the positive side, which travels
        # against the cut normal to meet the negative side.
        move_right = len(right) <= len(left)
        moving, fixed = (right, left) if move_right else (left, right)
        direction = -normal if move_right else normal

        if not joints:
            plan.detached_groups += 1
            plan.findings.append(
                Finding(
                    WARNING,
                    "detached_groups",
                    f"the group {_names(moving)} shares no connector with {_names(fixed)}; "
                    "they are separate sub-assemblies and will not hold together",
                    None,
                    {"moving": moving, "fixed": fixed},
                )
            )
        else:
            clear, overlap = collision.sweep_is_clear(
                [meshes[p] for p in moving],
                [meshes[p] for p in fixed],
                direction,
                config,
            )
            if not clear:
                plan.findings.append(
                    Finding(
                        HARD,
                        "insertion_blocked",
                        f"{_names(moving)} cannot be slid onto {_names(fixed)} along "
                        f"{_axis_text(direction)}: the path passes through solid "
                        f"geometry (worst overlap {overlap:.1f} mm3)",
                        moving[0],
                        {
                            "constraint": "collision_free_insertion",
                            "moving": moving,
                            "fixed": fixed,
                            "direction": [float(v) for v in direction],
                            "overlap_mm3": overlap,
                        },
                    )
                )

        _check_connector_axes(joints, normal, plan, config)

        plan.steps.append(
            AssemblyStep(
                step_no=len(plan.steps) + 1,
                moving_parts=moving,
                fixed_parts=fixed,
                joint_ids=[i.id for i in joints],
                joint_type=config.joint_type,
                insertion_direction=[float(v) for v in direction],
                insertion_axis=_axis_text(direction),
                rotation=NO_ROTATION,
                note=None if joints else "no connector between these groups",
            )
        )
        return sorted(left + right)

    all_parts = assemble(decomposition.root)
    plan.base_part = plan.steps[0].fixed_parts[0] if plan.steps else (
        all_parts[0] if all_parts else None
    )
    return plan


def _check_connector_axes(
    joints: list, normal: np.ndarray, plan: AssemblyPlan, config: Config
) -> None:
    """Connector-axis accessibility: every joint in a step shares one axis.

    Undoing one cut at a time makes this true by construction, so a violation
    means the cut graph disagrees with the partition tree and is worth surfacing
    rather than assuming away.
    """
    for interface in joints:
        axis = np.asarray(interface.normal, dtype=float)
        if abs(float(np.dot(axis, normal))) < 1.0 - config.length_epsilon_mm:
            plan.findings.append(
                Finding(
                    HARD,
                    "connector_axis_conflict",
                    f"joint {interface.id} runs along {_axis_text(axis)} but its step "
                    f"engages along {_axis_text(normal)}; it cannot be seated",
                    interface.male_id,
                    {
                        "constraint": "connector_axis_accessibility",
                        "joint": interface.id,
                    },
                )
            )


def _names(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return f"[{', '.join(parts)}]"


def _axis_text(vector) -> str:
    v = np.asarray(vector, dtype=float)
    labels = ("X", "Y", "Z")
    dominant = int(np.argmax(np.abs(v)))
    sign = "+" if v[dominant] >= 0 else "-"
    return f"{sign}{labels[dominant]}"
