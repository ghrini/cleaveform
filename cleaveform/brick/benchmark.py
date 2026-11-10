"""Score brick mode against the shape catalogue, and write a scoreboard.

Brick mode used to be judged by one question - did it produce bricks - answered
on one vase. This module asks the questions that decide whether the tool is any
use on a model somebody actually cares about:

* **Does it build as one object?** (pieces, and the size of the biggest stray)
* **Does it look like the thing?** (how much of the original it covers, how much
  it adds that was never there)
* **Will it stand up?** (is the centre of mass over the footprint)
* **What does it cost?** (bricks, filament, print time, plates)

Everything runs on the cheap stages only - voxelize, tile, analyse - so a full
sweep of every shape at several brick sizes takes minutes and needs no printer.
"""

from __future__ import annotations

import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Point

from ..config import Config
from ..estimate import batch_summary
from ..geometry import repair
from . import shapes as catalogue
from .assembly import plate_count
from .layout import BrickLayout, build_layout
from .library import build_all
from .occupancy import OccupancyGrid, build_occupancy
from .tiling import TilingError

DEFAULT_SIZES_MM = (6.0, 8.0, 12.0, 16.0)
STANDARD_UNIT_MM = 8.0
# Resolution of the "truth" the bricks are compared against. Fixed rather than
# tied to the brick size, so a score means the same thing at every brick size.
FIDELITY_PITCH_MM = 2.0
LATTICE_MARGIN_CELLS = 2
PERCENT = 100.0
NOTE_WIDTH = 90
MINUTES_PER_HOUR = 60.0

PASS = "pass"
SPLIT = "split"
FAIL = "fail"
ERROR = "error"
VERDICT_ORDER = (PASS, SPLIT, FAIL, ERROR)
VERDICT_COLOURS = {PASS: "#3f9a5a", SPLIT: "#d9822b", FAIL: "#c0392b", ERROR: "#7b5ea7"}
VERDICT_MEANING = {
    PASS: "builds as the number of pieces the model has",
    SPLIT: "builds, but some bricks are not attached to the rest",
    FAIL: "cannot be built as one object",
    ERROR: "the tool refused or could not process it",
}

# Picture styling.
IMAGE_INCHES = (8.4, 3.8)
IMAGE_DPI = 105
MODEL_COLOUR = (0.85, 0.51, 0.17)
MAIN_COLOUR = (0.18, 0.43, 0.71, 1.0)
STRAY_COLOUR = (0.85, 0.18, 0.15, 1.0)
LIGHT_DIRECTION = (0.35, -0.55, 0.75)
SHADE_FLOOR = 0.42
VIEW_ELEVATION = 22
VIEW_AZIMUTH = -58
PICTURE_PADDING = 0.04
VOXEL_EDGE_WIDTH = 0.2


@dataclass
class Row:
    """One shape at one brick size."""

    shape: str
    category: str
    unit_mm: float
    size_mm: float
    verdict: str = ERROR
    note: str = ""
    cells: int = 0
    # Cells added (+) or removed (-) by simplification, against what was voxelized.
    cells_changed: int = 0
    bricks: int = 0
    shapes_used: int = 0
    pieces: int = 0
    expected_pieces: int = 1
    largest_stray: int = 0
    loose: int = 0
    bond_pct: float = 0.0
    singles: int = 0
    iou_pct: float = 0.0
    missing_pct: float = 0.0
    extra_pct: float = 0.0
    stands: bool = True
    stand_margin_mm: float = 0.0
    weight_g: float = 0.0
    print_hours: float = 0.0
    plates: int = 0
    seconds: float = 0.0


# --------------------------------------------------------------- fidelity
@dataclass(frozen=True)
class Truth:
    """What the model is meant to look like, sampled on a fixed fine grid."""

    filled: np.ndarray
    origin_mm: np.ndarray
    pitch_mm: float
    extents_mm: np.ndarray


def sample_truth(reference: trimesh.Trimesh, pitch_mm: float = FIDELITY_PITCH_MM) -> Truth:
    voxels = reference.voxelized(pitch=pitch_mm).fill()
    return Truth(
        filled=np.asarray(voxels.matrix, dtype=bool),
        # trimesh reports the centre of cell [0, 0, 0]; the corner is half a cell back.
        origin_mm=np.asarray(voxels.translation, dtype=float) - pitch_mm / 2.0,
        pitch_mm=pitch_mm,
        extents_mm=np.asarray(reference.extents, dtype=float),
    )


def _gather(array: np.ndarray, origin: np.ndarray, pitch: float, axes: list[np.ndarray]) -> np.ndarray:
    """Read ``array`` (cells of ``pitch`` anchored at ``origin``) at sample points ``axes``."""
    out = np.zeros(tuple(len(a) for a in axes), dtype=bool)
    sources, targets = [], []
    for axis, centres in enumerate(axes):
        index = np.floor((centres - origin[axis]) / pitch).astype(int)
        valid = (index >= 0) & (index < array.shape[axis])
        sources.append(index[valid])
        targets.append(np.nonzero(valid)[0])
    out[np.ix_(*targets)] = array[np.ix_(*sources)]
    return out


def fidelity(truth: Truth, grid: OccupancyGrid) -> tuple[float, float, float]:
    """``(iou, missing, extra)`` as fractions: bricks against the intended shape.

    ``missing`` is the share of the original the bricks do not cover; ``extra``
    is the amount of brick sitting where the model is not, as a share of the
    original. A model can score well on one and badly on the other, so both are
    reported next to the overlap.
    """
    grid_extent = np.asarray(grid.shape, dtype=float) * grid.pitch_mm + grid.origin_mm
    reach = np.maximum(truth.extents_mm, grid_extent)
    counts = np.ceil(reach / truth.pitch_mm).astype(int) + LATTICE_MARGIN_CELLS
    axes = [(np.arange(n) + 0.5) * truth.pitch_mm for n in counts]

    original = _gather(truth.filled, truth.origin_mm, truth.pitch_mm, axes)
    bricks = _gather(grid.filled, grid.origin_mm, grid.pitch_mm, axes)

    both = int((original & bricks).sum())
    either = int((original | bricks).sum())
    total = int(original.sum())
    if either == 0 or total == 0:
        return 0.0, 1.0, 0.0
    return (
        both / either,
        int((original & ~bricks).sum()) / total,
        int((bricks & ~original).sum()) / total,
    )


def stability(grid: OccupancyGrid) -> tuple[bool, float]:
    """Whether the centre of mass sits over the footprint, and by how much.

    Every cell weighs the same. The footprint is the convex hull of the bottom
    course, which is what the model actually rests on. Returns ``(stands,
    margin_mm)``; the margin is the distance from the centre of mass to the
    edge of the footprint, and is zero when it falls outside.
    """
    cells = np.argwhere(grid.filled)
    if len(cells) == 0:
        return False, 0.0
    centre = ((cells + 0.5) * grid.pitch_mm)[:, :2].mean(axis=0)
    bottom = np.argwhere(grid.filled[:, :, 0])
    if len(bottom) == 0:
        return False, 0.0
    corners = []
    for x, y in bottom:
        for dx in (0, 1):
            for dy in (0, 1):
                corners.append(((x + dx) * grid.pitch_mm, (y + dy) * grid.pitch_mm))
    hull = MultiPoint(corners).convex_hull
    point = Point(*centre)
    if not hull.contains(point):
        return False, 0.0
    return True, float(hull.exterior.distance(point))


# ---------------------------------------------------------------- scoring
def _verdict(layout: BrickLayout, expected: int) -> str:
    pieces = layout.piece_count
    if pieces == expected:
        return PASS
    if layout.ok:
        return SPLIT
    return FAIL


def score(
    shape: catalogue.Shape,
    mesh: trimesh.Trimesh,
    truth: Truth,
    unit_mm: float,
    size_mm: float,
    config: Config,
    brick_meshes: dict,
) -> tuple[Row, BrickLayout | None, OccupancyGrid | None]:
    """Run one shape at one brick size and measure everything."""
    row = Row(shape.name, shape.category, unit_mm, size_mm, expected_pieces=shape.expected_pieces)
    started = time.perf_counter()
    try:
        grid = build_occupancy(mesh, config)
        # Cells the ordinary vote kept; keep-thin additions count as changes.
        row.cells = grid.cell_count - grid.added_thin
        if row.cells == 0:
            row.note = "the model vanished: no cell was at least half full"
            return row, None, None
        layout = build_layout(grid, config)
    except TilingError as exc:
        row.note = str(exc).split(";")[0][:NOTE_WIDTH]
        row.seconds = time.perf_counter() - started
        return row, None, None
    row.seconds = time.perf_counter() - started

    # Everything below is measured on the grid the bricks were actually tiled
    # on, which differs from the voxelized one when simplification ran.
    tiled = layout.grid
    row.cells_changed = tiled.cell_count - row.cells
    metrics = layout.metrics
    counts = layout.counts_by_type
    row.bricks = layout.brick_count
    row.shapes_used = len(counts)
    row.pieces = layout.piece_count
    row.largest_stray = int(metrics["largest_stray_bricks"])
    row.loose = int(metrics["loose_brick_count"])
    row.bond_pct = float(metrics["bond_fraction"]) * PERCENT
    row.singles = int(counts.get("brick_1x1", 0))
    row.verdict = _verdict(layout, shape.expected_pieces)

    iou, missing, extra = fidelity(truth, tiled)
    row.iou_pct, row.missing_pct, row.extra_pct = iou * PERCENT, missing * PERCENT, extra * PERCENT
    row.stands, row.stand_margin_mm = stability(tiled)

    used = {t: brick_meshes[t] for t in counts}
    summary = batch_summary(used, counts, plate_count(counts, used, config), config)
    row.weight_g = float(summary["total_weight_g"])
    row.print_hours = float(summary["total_print_time_min"]) / MINUTES_PER_HOUR
    row.plates = int(summary["plate_count"])
    if layout.piece_count > shape.expected_pieces:
        row.note = layout.piece_summary()
    return row, layout, tiled


def run(
    shapes: list[catalogue.Shape],
    sizes_mm: tuple[float, ...] = DEFAULT_SIZES_MM,
    size_mm: float = catalogue.DEFAULT_SIZE_MM,
    config: Config | None = None,
    progress: Callable[[str], None] | None = None,
    keep_unit_mm: float = STANDARD_UNIT_MM,
) -> tuple[list[Row], dict]:
    """Score every shape at every brick size.

    Returns the rows and, for the standard brick size only, the meshes and
    layouts needed to draw pictures (keeping them for every size would hold
    hundreds of layouts in memory for no use).
    """
    base = (config or Config()).with_overrides(decomposition_mode="brick")
    say = progress or (lambda message: None)
    brick_cache: dict[float, dict] = {}
    rows: list[Row] = []
    kept: dict = {}

    for number, shape in enumerate(shapes, start=1):
        say(f"[{number}/{len(shapes)}] {shape.name}")
        # Exactly what the loader does with a real file: repair it, and refuse it
        # if it cannot be made watertight. Scoring the raw mesh would grade a
        # damaged file on a stage the real tool never lets it skip.
        mesh, loadable = repair.ensure_watertight(shape.mesh(size_mm), base)
        if not loadable:
            for unit in sizes_mm:
                rows.append(Row(shape.name, shape.category, unit, size_mm,
                                note="the loader could not make this watertight and would refuse it",
                                expected_pieces=shape.expected_pieces))
            continue
        truth = sample_truth(shape.reference_mesh(size_mm))
        for unit in sizes_mm:
            unit_config = base.with_overrides(brick_unit_mm=float(unit))
            if unit not in brick_cache:
                brick_cache[unit] = build_all(unit_config)
            row, layout, grid = score(shape, mesh, truth, unit, size_mm, unit_config, brick_cache[unit])
            rows.append(row)
            if unit == keep_unit_mm and layout is not None:
                kept[shape.name] = (mesh, grid, layout)
    return rows, kept


# --------------------------------------------------------------- pictures
def render_pair(mesh: trimesh.Trimesh, grid: OccupancyGrid, layout: BrickLayout, path: Path) -> bool:
    """Original beside bricks, with stray bricks in red. False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return False

    from .layout import _owner_grid

    light = np.asarray(LIGHT_DIRECTION, dtype=float)
    light = light / np.linalg.norm(light)
    shade = np.clip(mesh.face_normals @ light, 0.0, 1.0) * (1 - SHADE_FLOOR) + SHADE_FLOOR
    colours = np.clip(np.asarray(MODEL_COLOUR)[None, :] * shade[:, None], 0, 1)

    fig = plt.figure(figsize=IMAGE_INCHES)
    left = fig.add_subplot(1, 2, 1, projection="3d")
    left.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolors=colours, linewidths=0))
    right = fig.add_subplot(1, 2, 2, projection="3d")

    owner = _owner_grid(layout.placements, grid.shape)
    piece = np.array(layout.piece_of)[np.clip(owner, 0, None)]
    face = np.empty(grid.shape + (4,))
    face[:] = MAIN_COLOUR
    face[piece > 0] = STRAY_COLOUR
    right.voxels(grid.filled, facecolors=face, edgecolor=to_rgb("white"), linewidth=VOXEL_EDGE_WIDTH)

    half = float(max(mesh.extents.max(), np.asarray(grid.shape).max() * grid.pitch_mm)) / 2.0 * (1 + PICTURE_PADDING)
    centre = mesh.extents / 2.0
    left.set_xlim(centre[0] - half, centre[0] + half)
    left.set_ylim(centre[1] - half, centre[1] + half)
    left.set_zlim(0, 2 * half)
    right.set_box_aspect(tuple(grid.shape))
    for ax in (left, right):
        ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
        ax.set_axis_off()
    # The limits above span the same distance on every axis, so the box is a cube;
    # any other aspect would stretch the model.
    left.set_box_aspect((1, 1, 1))
    left.set_title("original", fontsize=9)
    stray = layout.piece_count - 1
    right.set_title(f"{layout.brick_count} bricks", fontsize=9)
    fig.text(
        0.5,
        0.0,
        f"builds as {layout.piece_summary()}" + ("  (strays in red)" if stray else ""),
        ha="center",
        va="top",
        fontsize=9,
        color="#c0392b" if stray else "#3f9a5a",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=IMAGE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# --------------------------------------------------------------- scoreboard
COLUMNS = [
    "shape", "category", "unit_mm", "verdict", "pieces", "largest_stray", "bricks", "cells_changed", "shapes_used",
    "loose", "bond_pct", "singles", "iou_pct", "missing_pct", "extra_pct", "stands", "weight_g",
    "print_hours", "plates", "seconds", "note",
]


def summary_counts(rows: list[Row]) -> dict[float, dict[str, int]]:
    out: dict[float, dict[str, int]] = {}
    for row in rows:
        out.setdefault(row.unit_mm, {v: 0 for v in VERDICT_ORDER})[row.verdict] += 1
    return out


def floating_counts(rows: list[Row]) -> dict[float, int]:
    """Rows per brick size that contain bricks resting on nothing.

    Whether a model holds together and whether it can be built course by course
    are different questions. A brick with nothing under it is attached from
    above, so the model can still be one piece - but you cannot lay it from the
    bottom up without something to rest it on. The verdict answers the first
    question; this answers the second.
    """
    out: dict[float, int] = {}
    for row in rows:
        out.setdefault(row.unit_mm, 0)
        if row.loose > 0:
            out[row.unit_mm] += 1
    return out


def write_scoreboard(rows: list[Row], out_dir: str | Path, images: dict[str, str] | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = images or {}
    written = {}

    csv_path = out / "scoreboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            data = asdict(row)
            writer.writerow([f"{data[c]:.1f}" if isinstance(data[c], float) else data[c] for c in COLUMNS])
    written["csv"] = csv_path

    json_path = out / "scoreboard.json"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    written["json"] = json_path

    md_path = out / "scoreboard.md"
    md_path.write_text(_markdown(rows), encoding="utf-8")
    written["markdown"] = md_path

    html_path = out / "scoreboard.html"
    html_path.write_text(_html(rows, images), encoding="utf-8")
    written["html"] = html_path
    return written


def _cell(row: Row) -> str:
    if row.verdict == ERROR:
        return "ERROR"
    text = f"{row.verdict.upper()} {row.pieces}p"
    if row.pieces > row.expected_pieces:
        text += f"/{row.largest_stray}s"
    text += f" {row.bricks}b"
    if row.loose:
        text += f" {row.loose}f"
    if row.cells_changed:
        text += f" {row.cells_changed:+d}c"
    return text


def _markdown(rows: list[Row]) -> str:
    units = sorted({r.unit_mm for r in rows})
    names = list(dict.fromkeys(r.shape for r in rows))
    lines = ["# Brick mode scoreboard", ""]
    counts = summary_counts(rows)
    floating = floating_counts(rows)
    lines.append("| brick size | " + " | ".join(VERDICT_ORDER) + " | with floating bricks |")
    lines.append("|---|" + "---|" * (len(VERDICT_ORDER) + 1))
    for unit in units:
        lines.append(
            f"| {unit:g} mm | " + " | ".join(str(counts[unit][v]) for v in VERDICT_ORDER)
            + f" | {floating[unit]} of {sum(counts[unit].values())} |"
        )
    lines += [
        "",
        "`Np` = pieces, `/Ns` = largest stray group in bricks, `Nb` = bricks, "
        "`Nf` = floating bricks (resting on nothing, so they cannot be laid from the bottom up), "
        "`+Nc`/`-Nc` = cells added or removed by shape simplification.",
        "",
    ]
    lines.append("| shape | " + " | ".join(f"{u:g} mm" for u in units) + " | fidelity @8mm |")
    lines.append("|---|" + "---|" * (len(units) + 1))
    index = {(r.shape, r.unit_mm): r for r in rows}
    for name in names:
        fid = index.get((name, STANDARD_UNIT_MM))
        fid_text = f"{fid.iou_pct:.0f}%" if fid and fid.verdict != ERROR else "-"
        cells = [_cell(index[(name, u)]) if (name, u) in index else "-" for u in units]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fid_text} |")
    return "\n".join(lines) + "\n"


def _html(rows: list[Row], images: dict[str, str]) -> str:
    units = sorted({r.unit_mm for r in rows})
    names = list(dict.fromkeys(r.shape for r in rows))
    index = {(r.shape, r.unit_mm): r for r in rows}
    category_of = {r.shape: r.category for r in rows}
    counts = summary_counts(rows)

    def badge(verdict: str) -> str:
        return f'<span class="b" style="background:{VERDICT_COLOURS[verdict]}">{verdict}</span>'

    head = "".join(f"<th>{u:g} mm</th>" for u in units)
    body = []
    for category, meaning in catalogue.CATEGORIES.items():
        members = [n for n in names if category_of[n] == category]
        if not members:
            continue
        body.append(f'<tr class="cat"><td colspan="{len(units) + 3}"><b>{html.escape(category)}</b> - {html.escape(meaning)}</td></tr>')
        for name in members:
            cells = []
            for unit in units:
                r = index.get((name, unit))
                if r is None:
                    cells.append("<td>-</td>")
                    continue
                detail = html.escape(r.note) if r.verdict == ERROR else (
                    f"{r.pieces} piece(s), stray {r.largest_stray}, {r.bricks} bricks, "
                    f"{r.loose} floating, shape changed by {r.cells_changed:+d} cell(s), "
                    f"fidelity {r.iou_pct:.0f}%"
                )
                cells.append(f'<td title="{detail}">{badge(r.verdict)}<br><small>{html.escape(_cell(r))}</small></td>')
            thumb = images.get(name)
            picture = f'<a href="{html.escape(thumb)}"><img src="{html.escape(thumb)}" width="170"></a>' if thumb else ""
            body.append(f"<tr><td><b>{html.escape(name)}</b></td>{''.join(cells)}<td>{picture}</td></tr>")

    legend = "".join(f"<li>{badge(v)} {html.escape(VERDICT_MEANING[v])}</li>" for v in VERDICT_ORDER)
    floating = floating_counts(rows)
    tally = "".join(
        f"<tr><td>{u:g} mm</td>" + "".join(f"<td>{counts[u][v]}</td>" for v in VERDICT_ORDER)
        + f"<td>{floating[u]} of {sum(counts[u].values())}</td></tr>"
        for u in units
    )
    tally_head = "".join(f"<th>{v}</th>" for v in VERDICT_ORDER) + "<th>with floating bricks</th>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Brick mode scoreboard</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2933}}
table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #d9dee3;padding:6px 10px;text-align:left;vertical-align:top}}
th{{background:#eef2f6}}.b{{color:#fff;border-radius:9px;padding:1px 8px;font-size:12px;font-weight:700}}
tr.cat td{{background:#f4f6f8;padding-top:14px}}small{{color:#52606d}}ul{{list-style:none;padding:0}}li{{margin:3px 0}}</style></head>
<body><h1>Brick mode scoreboard</h1>
<p>Every shape at every brick size. Hover a cell for detail; pictures are at {STANDARD_UNIT_MM:g} mm with stray bricks in red.</p>
<ul>{legend}</ul>
<table><tr><th>brick size</th>{tally_head}</tr>{tally}</table><br>
<table><tr><th>shape</th>{head}<th>picture</th></tr>{''.join(body)}</table></body></html>
"""
