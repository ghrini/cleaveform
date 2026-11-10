"""Module 7 - the plain-text assembly guide.

Phase 1 produces text only. Rendered per-step images arrive in Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..estimate import MINUTES_PER_HOUR

GUIDE_FILENAME = "assembly_guide.txt"
DIAGNOSTIC_FILENAME = "diagnostic.txt"
RULE_WIDTH = 78
RULE = "=" * RULE_WIDTH
THIN_RULE = "-" * RULE_WIDTH


def write_guide(
    out_dir: str | Path,
    config: Config,
    input_summary: dict[str, Any],
    parts: dict[str, dict[str, Any]],
    joints: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    base_part: str | None,
    constraints: dict[str, Any],
    warnings: list[str],
) -> Path:
    """Write ``assembly_guide.txt`` and return its path."""
    joints_by_id = {j["id"]: j for j in joints}
    lines: list[str] = []

    lines.append(RULE)
    lines.append("CLEAVEFORM ASSEMBLY GUIDE")
    lines.append(RULE)
    lines.append(f"Source model     : {input_summary.get('source', 'unknown')}")
    extents = input_summary.get("final_bbox_mm") or [0, 0, 0]
    lines.append(
        f"Model size       : {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm"
    )
    lines.append(f"Parts to print   : {len(parts)}")
    lines.append(f"Connectors       : {len(joints)} x {config.joint_type}")
    lines.append(
        f"Joint clearance  : {config.joint_clearance_mm:.2f} mm "
        "(set by the calibration print)"
    )
    lines.append(f"Material assumed : {config.material_name}")
    lines.append("")

    lines.append(THIN_RULE)
    lines.append("BEFORE YOU START")
    lines.append(THIN_RULE)
    lines.append("1. Slice every STL in parts/ with Bambu Studio.")
    lines.append(
        f"2. Print with a {config.nozzle_width_mm:.1f} mm nozzle at "
        f"{config.layer_height_mm:.2f} mm layers unless you have a reason not to."
    )
    lines.append(
        "3. Apply each part's recommended orientation from metadata.json before slicing."
    )
    lines.append(
        "4. Dry-fit each joint before gluing. Pegs should seat by hand without forcing."
    )
    lines.append("")

    lines.append(THIN_RULE)
    lines.append("PART LIST")
    lines.append(THIN_RULE)
    header = f"{'PART':<12}{'SIZE (mm)':<26}{'ORIENT':<12}{'SUPPORT':<9}EST TIME"
    lines.append(header)
    for part_id in sorted(parts):
        part = parts[part_id]
        size = part["bbox_mm"]["extents"]
        size_text = f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f}"
        orient = part["orientation"]["recommended"]
        support = "yes" if part["supports_required"] else "no"
        minutes = part["est_print_time_min"]
        lines.append(
            f"{part_id:<12}{size_text:<26}{orient:<12}{support:<9}"
            f"~{minutes / MINUTES_PER_HOUR:.1f} h"
        )
    lines.append("")
    lines.append("Print times are estimates from cleaveform, not slicer output.")
    lines.append("")

    lines.append(THIN_RULE)
    lines.append("ASSEMBLY STEPS")
    lines.append(THIN_RULE)
    lines.append(
        "Steps undo the cuts in reverse, so sub-assemblies are built first and then "
        "joined. Each step is a single straight push along one axis."
    )
    lines.append("")
    if base_part:
        lines.append(f"Start with {base_part} on the bench.")
        lines.append("")

    for step in steps:
        moving = _group(step["moving_parts"])
        fixed = _group(step["fixed_parts"])
        lines.append(f"Step {step['step_no']}: push {moving} onto {fixed}.")
        lines.append(
            f"    Direction : {step['insertion_axis']} "
            f"(vector {_vec(step['insertion_direction'])})"
        )
        lines.append(f"    Rotation  : {step['rotation']}")
        if step.get("note"):
            lines.append(f"    NOTE      : {step['note']}")
        for joint_id in step["joint_ids"]:
            joint = joints_by_id.get(joint_id)
            if joint is None:
                continue
            lines.append(
                f"    {joint_id}: {joint['peg_count']} x {joint['joint_type']} on the "
                f"{joint['axis'].upper()} face. Pegs on {joint['male_part']}, "
                f"sockets in {joint['female_part']} "
                f"(peg dia {joint['peg_radius_mm'] * 2:.1f} mm, "
                f"depth {joint['peg_depth_mm']:.1f} mm, "
                f"clearance {joint['clearance_mm']:.2f} mm)."
            )
        lines.append("")

    advisories = constraints.get("warnings", []) + constraints.get("soft", [])
    if advisories or warnings:
        lines.append(THIN_RULE)
        lines.append("NOTES AND ADVISORIES (none of these block printing)")
        lines.append(THIN_RULE)
        for note in warnings:
            lines.append(f"  - {note}")
        for finding in advisories:
            lines.append(f"  - [{finding['tier']}] {finding['message']}")
        lines.append("")

    lines.append(RULE)
    lines.append(
        "If a joint is tight or loose, re-run the calibration print "
        "(cleaveform calibrate) and update joint_clearance_mm."
    )
    lines.append(RULE)

    path = Path(out_dir) / GUIDE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_diagnostic(
    out_dir: str | Path, title: str, detail: dict[str, Any], notes: list[str]
) -> Path:
    """Write ``diagnostic.txt`` describing why a run could not complete."""
    lines = [RULE, "CLEAVEFORM DIAGNOSTIC", RULE, "", title, ""]
    if detail:
        lines.append("Detail:")
        for key in sorted(detail):
            lines.append(f"  {key}: {detail[key]}")
        lines.append("")
    if notes:
        lines.append("Context:")
        for note in notes:
            lines.append(f"  - {note}")
        lines.append("")
    lines.append("Suggested next steps:")
    lines.append("  - Relax the offending bound in config.py and re-run.")
    lines.append("  - Reduce peg_radius_mm or peg_edge_margin_mm if a face was too small.")
    lines.append("  - Raise max_parts or max_cut_depth if the model needs more cuts.")
    lines.append("  - Check the input mesh is a clean watertight solid.")
    lines.append("")
    lines.append(RULE)

    path = Path(out_dir) / DIAGNOSTIC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _group(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return f"the group [{', '.join(parts)}]"


def _vec(vector: list[float] | None) -> str:
    if vector is None:
        return "n/a"
    return "(" + ", ".join(f"{v + 0.0:+.0f}" for v in vector) + ")"
