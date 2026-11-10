"""The brick build guide: what to print, and where each brick goes.

Chunk mode's guide names every part in a sentence per step, which is right for
a dozen chunks and unreadable for a few hundred bricks. This writes the two
things a brick build actually needs: a bill of materials, and a map of each
course.

The maps letter bricks rather than number them. Every brick in a course is
given a letter such that no two touching bricks share one, using a greedy
four-colouring of the course's adjacency graph - four is always enough for a
map drawn on a plane. Unique ids per brick would be unreadable at any useful
size; what the eye needs is only to see where one brick stops and the next
starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..estimate import MINUTES_PER_HOUR
from ..export.guide import GUIDE_FILENAME, RULE, RULE_WIDTH, THIN_RULE
from .layout import BrickLayout

BILL_FILENAME = "bill_of_materials.txt"
GLYPHS = "ABCD"
EMPTY_GLYPH = "."


def write_brick_guide(
    out_dir: str | Path,
    layout: BrickLayout,
    config: Config,
    input_summary: dict[str, Any],
    estimates: dict[str, Any],
) -> Path:
    """Write the build guide and return its path."""
    lines: list[str] = []
    extents = input_summary.get("final_bbox_mm") or [0, 0, 0]
    grid = layout.grid

    lines.append(RULE)
    lines.append("CLEAVEFORM BRICK BUILD GUIDE")
    lines.append(RULE)
    lines.append(f"Source model    : {input_summary.get('source', 'unknown')}")
    lines.append(
        f"Model size      : {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm"
    )
    lines.append(
        f"Brick grid      : {grid.shape[0]} x {grid.shape[1]} studs, "
        f"{grid.shape[2]} courses"
    )
    lines.append(f"Stud pitch      : {config.brick_unit_mm:.1f} mm")
    lines.append(
        f"Bricks          : {layout.brick_count} from "
        f"{len(layout.counts_by_type)} shapes"
    )
    lines.append(f"Builds as       : {layout.piece_summary()}")
    if layout.change_summary():
        lines.append(f"Shape changes   : {layout.change_summary()}")
    lines.append(
        f"Clutch fit      : {config.brick_clutch_interference_mm:.2f} mm "
        f"interference, printer bias {config.brick_fit_bias_mm:+.2f} mm"
    )
    lines.append("")

    lines.append(THIN_RULE)
    lines.append("BILL OF MATERIALS")
    lines.append(THIN_RULE)
    lines.extend(_bill_lines(layout, estimates))
    lines.append("")

    lines.append(THIN_RULE)
    lines.append("BEFORE YOU START")
    lines.append(THIN_RULE)
    lines.append("1. Print the plates in plates/, or the individual shapes in bricks/.")
    lines.append("2. Print studs up, as exported. Do not re-orient: a brick stood on")
    lines.append("   its studs has nothing to hold it to the plate.")
    lines.append(
        f"3. Build one course at a time from the bottom. Each course is "
        f"{config.brick_unit_mm:.1f} mm tall."
    )
    lines.append("4. Bricks grip top to bottom only. Two bricks side by side are held")
    lines.append("   together by whatever sits across them in the next course up, so")
    lines.append("   follow the maps rather than improvising the layout.")
    lines.append("")

    lines.extend(_map_lines(layout))

    advisories = _advisories(layout)
    if advisories:
        lines.append(THIN_RULE)
        lines.append("NOTES (none of these stop you building)")
        lines.append(THIN_RULE)
        lines.extend(advisories)
        lines.append("")

    lines.append(RULE)
    lines.append(
        "If bricks are loose or too stiff, run `cleaveform calibrate --mode gauge` "
        "to measure your printer, then set brick_fit_bias_mm (and, if needed, "
        "brick_clutch_interference_mm - bigger is tighter). Both are separate "
        "from the chunk joint clearance, so neither affects chunk-mode output."
    )
    lines.append(RULE)

    path = Path(out_dir) / GUIDE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_bill_of_materials(
    out_dir: str | Path, layout: BrickLayout, estimates: dict[str, Any]
) -> Path:
    """Write the shopping list on its own, for printing alongside the plates."""
    lines = [RULE, "CLEAVEFORM BILL OF MATERIALS", RULE, ""]
    lines.extend(_bill_lines(layout, estimates))
    lines.append("")
    path = Path(out_dir) / BILL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _bill_lines(layout: BrickLayout, estimates: dict[str, Any]) -> list[str]:
    per_type = estimates.get("per_type", {})
    lines = [f"{'SHAPE':<14}{'QTY':>6}{'SIZE (studs)':>16}{'EST WEIGHT':>13}"]
    for type_id, count in layout.counts_by_type.items():
        detail = per_type.get(type_id, {})
        footprint = detail.get("footprint", "?")
        grams = detail.get("total_weight_g")
        weight = f"{grams:.0f} g" if grams is not None else "-"
        lines.append(f"{type_id:<14}{count:>6}{footprint:>16}{weight:>13}")
    lines.append("")
    lines.append(f"{'TOTAL':<14}{layout.brick_count:>6}")
    if "total_weight_g" in estimates:
        lines.append(
            f"Estimated filament: {estimates['total_weight_g']:.0f} g "
            f"over {estimates.get('plate_count', 0)} plate(s), "
            f"~{estimates.get('total_print_time_min', 0) / MINUTES_PER_HOUR:.1f} h"
        )
        lines.append("Estimates from cleaveform, not slicer output.")
    return lines


def _map_lines(layout: BrickLayout) -> list[str]:
    """One map per course, bottom first."""
    width, depth, courses = layout.grid.shape
    lines = [THIN_RULE, "COURSE MAPS (build bottom to top)", THIN_RULE]

    if width > RULE_WIDTH // 2:
        lines.append(
            f"Model is {width} studs wide, too wide to draw at this page width. "
            "Use a larger brick_unit_mm for a printable map."
        )
        lines.append("")
        return lines

    lines.append("Each letter is one brick; touching bricks never share a letter.")
    lines.append("")
    for z in range(courses):
        placements = [p for p in layout.placements if p.z == z]
        lines.append(f"Course {z + 1} of {courses}  ({len(placements)} bricks)")
        lines.extend(_course_map(placements, width, depth))
        lines.append("")
    return lines


def _course_map(placements: list, width: int, depth: int) -> list[str]:
    colours = _four_colour(placements)
    cells: dict[tuple[int, int], str] = {}
    for index, placement in enumerate(placements):
        glyph = GLYPHS[colours[index] % len(GLYPHS)]
        for cell in placement.footprint_cells:
            cells[cell] = glyph
    # Rows print with y descending so the map reads like a plan view.
    return [
        "    " + "".join(cells.get((x, y), EMPTY_GLYPH) for x in range(width))
        for y in range(depth - 1, -1, -1)
    ]


def _four_colour(placements: list) -> dict[int, int]:
    """Greedy colouring so no two touching bricks share a glyph."""
    occupied: dict[tuple[int, int], int] = {}
    for index, placement in enumerate(placements):
        for cell in placement.footprint_cells:
            occupied[cell] = index

    neighbours: dict[int, set[int]] = {i: set() for i in range(len(placements))}
    for (x, y), index in occupied.items():
        for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            other = occupied.get(cell)
            if other is not None and other != index:
                neighbours[index].add(other)
                neighbours[other].add(index)

    colours: dict[int, int] = {}
    for index in range(len(placements)):
        taken = {colours[n] for n in neighbours[index] if n in colours}
        colours[index] = next(c for c in range(len(GLYPHS) + 1) if c not in taken)
    return colours


def _advisories(layout: BrickLayout) -> list[str]:
    out: list[str] = []
    metrics = layout.metrics
    out.append(
        f"  - {metrics.get('bond_fraction', 0):.0%} of bricks span two or more "
        "bricks below them. Higher is a stronger model."
    )
    out.append(
        f"  - Weakest horizontal join: {metrics.get('weakest_link_cells', 0)} stud(s). "
        "That is where the model will break if it breaks."
    )
    for finding in layout.findings:
        if finding.tier != "hard":
            out.append(f"  - [{finding.tier}] {finding.message}")
    return out
