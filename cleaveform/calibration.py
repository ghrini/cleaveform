"""The joint-fit calibration fixture (spec section 13, directive section 9).

``joint_clearance_mm`` cannot be derived analytically: it depends on the
printer, the filament and the profile. This module builds the physical test that
sets it, a two-chunk peg-and-socket cube printed at each candidate clearance.

The smallest clearance that assembles by hand without forcing and without slop
is the value to put in ``config.joint_clearance_mm``. Until that print happens,
the pipeline runs on the 0.20 mm starting default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .brick.library import COPLANAR_EPSILON_MM, BrickType, build_brick_solid
from .config import Config, parse_brick_footprint
from .geometry import booleans, solids

README_FILENAME = "CALIBRATION.txt"
RULE_WIDTH = 78


def build_pair(
    clearance_mm: float, config: Config
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Return the ``(male, female)`` halves of the calibration cube.

    The cube is split across X. The male half carries the pegs; the female half
    carries sockets at ``peg_radius_mm + clearance_mm``.
    """
    size = config.calibration_cube_mm
    half = size / 2.0
    normal = np.array([1.0, 0.0, 0.0])

    male = trimesh.creation.box(extents=(half, size, size))
    male.apply_translation((half / 2.0, half, half))
    female = trimesh.creation.box(extents=(half, size, size))
    female.apply_translation((half + half / 2.0, half, half))

    # Pegs sit either side of centre so the pair resists twisting as well as
    # pulling; twist is how you feel slop that a straight pull would hide.
    offset = size * config.calibration_peg_offset_fraction
    centers = [
        np.array([half, half - offset, half]),
        np.array([half, half + offset, half]),
    ]

    peg_config = config.with_overrides(joint_clearance_mm=clearance_mm)
    pegs = [solids.peg(c, normal, peg_config) for c in centers]
    sockets = [solids.socket(c, normal, peg_config) for c in centers]

    male = booleans.union([male, *pegs], peg_config)
    female = booleans.difference(female, sockets, peg_config)

    # The male peg tip sits at the +X end; the female socket opening sits at
    # the -X end (its solid back is at +X). A single shared rotation would
    # therefore print one half's mating feature facing up and the other's
    # facing down onto the bed - exactly the face whose accuracy the test is
    # measuring. Stand each half up so its own mating feature faces up: the
    # peg points at the ceiling, the socket opens like a cup.
    male = _stand_up(male, up_axis=normal, config=config)
    female = _stand_up(female, up_axis=-normal, config=config)
    return male, female


def _stand_up(mesh: trimesh.Trimesh, up_axis: np.ndarray, config: Config) -> trimesh.Trimesh:
    """Rotate so ``up_axis`` (in the part's own frame) points at +Z, then rest
    the part on the plate. Print-ready: no manual re-orientation needed."""
    transform = trimesh.geometry.align_vectors(up_axis, np.array([0.0, 0.0, 1.0]))
    out = mesh.copy()
    out.apply_transform(transform)
    # Rest on the plate and pull XY back to the positive quadrant so the part
    # previews sanely regardless of which way the rotation happened to swing.
    out.apply_translation(-out.bounds[0])
    return out


def write_calibration_set(out_dir: str | Path, config: Config) -> list[Path]:
    """Write a male/female STL pair for every candidate clearance, plus a readme."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for clearance in config.calibration_clearances_mm:
        male, female = build_pair(clearance, config)
        tag = f"{clearance:.2f}".replace(".", "p")
        for name, mesh in (("male", male), ("female", female)):
            path = out_path / f"calib_{tag}_{name}.stl"
            path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
            written.append(path)

    readme = out_path / README_FILENAME
    readme.write_text(_instructions(config), encoding="utf-8")
    written.append(readme)
    return written


def _instructions(config: Config) -> str:
    clearances = ", ".join(f"{c:.2f}" for c in config.calibration_clearances_mm)
    return "\n".join(
        [
            "CLEAVEFORM JOINT CLEARANCE CALIBRATION",
            "=" * RULE_WIDTH,
            "",
            "This is the physical gate for Phase 1. Nothing downstream is trustworthy",
            "until it has been run on the actual printer and filament you will use.",
            "",
            f"Cube size      : {config.calibration_cube_mm:.0f} mm",
            f"Peg diameter   : {config.peg_radius_mm * 2:.1f} mm",
            f"Peg depth      : {config.peg_depth_mm:.1f} mm",
            f"Clearances     : {clearances} mm",
            f"Nozzle assumed : {config.nozzle_width_mm:.1f} mm",
            f"Layer height   : {config.layer_height_mm:.2f} mm",
            "",
            "ORIENTATION",
            "-" * RULE_WIDTH,
            "Every STL here is already oriented print-ready: pegs point straight up,",
            "sockets open upward like a cup, both halves rest flat on the plate.",
            "Do not re-orient or let auto-arrange rotate them - if a preview shows a",
            "peg lying on its side, something upstream of the slicer changed it.",
            "",
            "PROCEDURE",
            "-" * RULE_WIDTH,
            "1. Slice every calib_*.stl in Bambu Studio with the profile you intend",
            "   to use for real parts. Do not change the profile between pairs.",
            "2. Print all pairs in one job so they share thermal conditions.",
            "3. Let them cool fully. Deburr the first layer if there is elephant foot.",
            "4. Dry-fit each pair by hand. Do not use tools, heat or force.",
            "",
            "READING THE RESULT",
            "-" * RULE_WIDTH,
            "  too tight : will not seat by hand, or needs force that flexes the part",
            "  correct   : seats fully by hand, holds its own weight, no side-to-side play",
            "  too loose : seats with visible slop or falls apart under its own weight",
            "",
            "The SMALLEST clearance that reads as 'correct' is your value.",
            "",
            "APPLYING THE RESULT",
            "-" * RULE_WIDTH,
            "Put it in a config file and pass it with --config, for example:",
            "",
            '    { "joint_clearance_mm": 0.15 }',
            "",
            "Every production run then uses that value. Recovery is not permitted to",
            "change it; only an explicitly experimental run may explore around it.",
            "",
        ]
    )


# --------------------------------------------------------------------------
# Brick clutch calibration
# --------------------------------------------------------------------------
#
# A separate fixture from the cube above, because it measures a different thing.
# The cube sets ``joint_clearance_mm`` for chunk mode's peg joints, where the
# engagement is ~6 mm deep and a clearance is the right parametrisation. A brick
# clutch engages over ~2 mm and grips by *interference*, so the number under
# test has the opposite sign and its own fixture.
#
# What this fixture is designed to avoid, having been learned the hard way:
#
#   1. Every rung is the SAME footprint. The first version varied the brick's
#      length to tell rungs apart, but length is also the dominant term in wall
#      stiffness, so a "looser" rung was partly just a floppier one.
#   2. The rungs are spaced in tenths, not hundredths. A ladder finer than the
#      printer's own repeatability returns the same feel at every rung, which is
#      exactly what four earlier print rounds reported.
#   3. Rungs are identified by raised dots on the END wall - never on the long
#      walls, which are the springs, and never by a dimension that participates
#      in the fit.

BRICK_README_FILENAME = "BRICK_CALIBRATION.txt"


MARKERS_PER_ROW = 3


def _marker_positions(count: int, config: Config) -> list[tuple[float, float]]:
    """Dot offsets from the centre of the end wall, laid out like dice pips.

    A single row overflows: five dots at a spacing wide enough not to fuse are
    wider than the 7.9 mm wall they sit on, and a dot hanging off the edge of
    the brick is a dot that fouls its neighbour on the plate.
    """
    pitch = config.brick_calibration_marker_pitch_mm
    rows = [
        min(MARKERS_PER_ROW, count - i * MARKERS_PER_ROW)
        for i in range((count + MARKERS_PER_ROW - 1) // MARKERS_PER_ROW)
    ]
    positions = []
    row_span = pitch * (len(rows) - 1)
    for row_index, row_count in enumerate(rows):
        v = row_index * pitch - row_span / 2.0
        col_span = pitch * (row_count - 1)
        for col in range(row_count):
            positions.append((col * pitch - col_span / 2.0, v))
    return positions


def _rung_markers(
    brick: BrickType, count: int, config: Config
) -> list[trimesh.Trimesh]:
    """``count`` raised dots on the brick's +X end wall.

    The end wall is chosen because it is structural. The long walls carry the
    ribs and have to flex by design, and a bump moulded into a spring changes
    the very thing being measured.
    """
    width, depth, height = brick.size_mm(config)
    radius = config.brick_calibration_marker_radius_mm
    positions = _marker_positions(count, config)

    reach = radius + max(
        (max(abs(h) for h, _ in positions), max(abs(v) for _, v in positions))
    )
    if reach > min(depth, height) / 2.0:
        raise ValueError(
            f"{count} markers do not fit on a {depth:.1f} x {height:.1f} mm end "
            "wall; lower brick_calibration_marker_pitch_mm or shorten "
            "brick_calibration_interferences_mm"
        )

    dots = []
    for horizontal, vertical in positions:
        dot = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        dot.apply_translation(
            (width, depth / 2.0 + horizontal, height / 2.0 + vertical)
        )
        dots.append(dot)
    return dots


def _integrated_brim(brick: BrickType, config: Config) -> trimesh.Trimesh | None:
    """A one-layer flange around the coupon's base.

    Not decoration. The wall of a brick is thin because it has to act as the
    spring that grips the stud, which leaves the open-bottomed shell standing on
    a ring about 0.9 mm wide - roughly 54 mm2 for a part 10.6 mm tall. Parts came
    off the plate mid-print at that contact area.

    The flange goes OUTSIDE the footprint and is one layer thick, so it peels off
    without loading the walls. It deliberately does not cross the cavity mouth:
    a membrane there would have to be cut out before the coupon could be tested,
    and cutting it is exactly the kind of thing that nicks a spring.
    """
    brim = config.brick_calibration_brim_mm
    if brim <= 0:
        return None

    width, depth, _ = brick.size_mm(config)
    height = config.layer_height_mm
    wall = config.brick_wall_mm

    outer = trimesh.creation.box(extents=(width + 2 * brim, depth + 2 * brim, height))
    outer.apply_translation((width / 2.0, depth / 2.0, height / 2.0))
    # Leave the cavity mouth clear.
    mouth = trimesh.creation.box(
        extents=(
            width - 2 * wall,
            depth - 2 * wall,
            height + 2 * COPLANAR_EPSILON_MM,
        )
    )
    mouth.apply_translation((width / 2.0, depth / 2.0, height / 2.0))
    return booleans.difference(outer, [mouth], config)


BATCH_BAR_MM = (4.0, 0.8, 0.7)
BATCH_BAR_DROP_MM = 2.6


def _batch_bars(brick: BrickType, config: Config) -> list[trimesh.Trimesh]:
    """Horizontal bars under the dot markers: which GENERATION a coupon is.

    Coupons from different geometry generations look nearly identical, and a
    test session that unknowingly mixes them measures nothing - the same
    failure that once made the chunk-mode calibration read backwards. Count
    the bars: no bars means the coupon predates the marking and should be
    recycled, not tested.
    """
    width, depth, height = brick.size_mm(config)
    bar_l, bar_t, bar_h = BATCH_BAR_MM
    bars = []
    for i in range(config.brick_calibration_batch):
        bar = trimesh.creation.box(extents=(bar_t, bar_l, bar_h))
        bar.apply_translation(
            (
                width,
                depth / 2.0,
                height / 2.0 - BATCH_BAR_DROP_MM - i * 2.0 * bar_h,
            )
        )
        bars.append(bar)
    return bars


def build_brick_rung(
    interference_mm: float, rung_index: int, config: Config
) -> trimesh.Trimesh:
    """One marked brick at one interference. Two of these make a testable pair."""
    rung_config = config.with_overrides(
        brick_clutch_interference_mm=interference_mm
    )
    brick = BrickType(*parse_brick_footprint(config.brick_calibration_footprint))
    solid = build_brick_solid(brick, rung_config)
    markers = _rung_markers(brick, rung_index, rung_config)
    bars = _batch_bars(brick, rung_config)
    brim = _integrated_brim(brick, rung_config)
    parts = [solid, *markers, *bars] + ([brim] if brim is not None else [])
    return booleans.union(parts, rung_config)


def write_brick_calibration_set(out_dir: str | Path, config: Config) -> list[Path]:
    """Two bricks per rung, so each rung can be pressed together on its own."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, interference in enumerate(
        config.brick_calibration_interferences_mm, start=1
    ):
        mesh = build_brick_rung(interference, index, config)
        for copy_number in (1, 2):
            path = out / (
                f"rung{index}_{interference:.2f}mm_copy{copy_number}.stl"
            )
            mesh.export(path)
            written.append(path)

    readme = out / BRICK_README_FILENAME
    readme.write_text(_brick_instructions(config), encoding="utf-8")
    written.append(readme)
    return written


def _brick_instructions(config: Config) -> str:
    lines = [
        "BRICK CLUTCH CALIBRATION".center(RULE_WIDTH),
        "=" * RULE_WIDTH,
        "",
        "Print all of these. Each rung is two identical bricks; push a rung's",
        "two bricks together and judge that rung on its own.",
        "",
        f"Footprint (every rung): {config.brick_calibration_footprint}",
        f"Brick unit:             {config.brick_unit_mm:.2f} mm",
        f"Wall:                   {config.brick_wall_mm:.2f} mm",
        "",
        "-" * RULE_WIDTH,
        f"{'dots':>6}  {'interference':>12}  file",
        "-" * RULE_WIDTH,
    ]
    for index, interference in enumerate(
        config.brick_calibration_interferences_mm, start=1
    ):
        dots = "*" * index
        lines.append(
            f"{dots:>6}  {interference:>9.2f} mm  "
            f"rung{index}_{interference:.2f}mm_copy1.stl / _copy2.stl"
        )
    lines += [
        "-" * RULE_WIDTH,
        "",
        "Count the raised dots on the short end wall to tell rungs apart.",
        "Every rung is the same size, on purpose - an earlier version of this",
        "test used brick length as the label, but length also sets how stiff",
        "the walls are, so it changed the fit it was trying to measure.",
        "",
        "WHAT TO LOOK FOR",
        "",
        "  Push two bricks of the same rung together, then pull them apart.",
        "",
        "    slips apart under its own weight ...... too loose, go up a rung",
        "    clicks, holds, comes apart by hand .... this is the one",
        "    needs real force, or creaks ........... too tight, go down a rung",
        "    will not seat, or splits .............. far too tight",
        "",
        "  Expect the low rungs to feel loose. The whole point of the wide",
        "  spacing is that the previous four ladders were all narrower than the",
        "  printer's own variation and every rung on them felt the same.",
        "",
        "WHEN YOU HAVE PICKED ONE",
        "",
        "  Put it in your printer profile as:",
        "",
        '      "brick_clutch_interference_mm": <the value from the table>',
        "",
        "  It is an interference, not a clearance: bigger means tighter.",
        "",
        "PRINT SETTINGS",
        "",
        "  These coupons carry their own brim - the thin flange around the base",
        "  is part of the model. Peel it off after printing; it is one layer",
        "  thick and comes away by hand. Do not cut it off with a blade near the",
        "  long walls, because those walls ARE the spring you are measuring.",
        "",
        "  It is there because a brick is a hollow shell open underneath, so its",
        "  first layer is only the wall outline - about 54 mm2 on a 3x1 that",
        "  stands 10.6 mm tall. The walls are thin because they have to act as",
        "  the spring, and that costs roughly 40% of the bed contact a solid",
        "  brick would have. At that contact area parts come off mid-print.",
        "",
        "  For REAL bricks later, switch a brim on in the slicer instead - you",
        "  do not want a moulded-on flange to remove from every one of them.",
        "",
        "  No supports. Studs up, exactly as exported - do not let the slicer",
        "  reorient them. Same filament and profile you will use for real",
        "  bricks, because the fit moves with both.",
        "",
        "=" * RULE_WIDTH,
    ]
    return "\n".join(lines) + "\n"

# --------------------------------------------------------------------------
# The fit gauge: measure the printer instead of guessing at it
# --------------------------------------------------------------------------
#
# Six rounds of coupon ladders failed to converge on a working clutch. The
# STL geometry was audited and is exact, so the whole error lives between the
# file and the plastic - slicer compensation, extrusion width, contact wear -
# and a ladder cannot separate those from the design number it is varying.
# This fixture measures the printer's NET bias directly: a bar of standard
# studs dropped into a comb of slots whose designed gaps straddle zero. The
# slot where the printed gap actually crosses zero IS the bias, read off by
# counting slots. No callipers needed.

GAUGE_README_FILENAME = "GAUGE.txt"
GAUGE_WALL_MM = 3.0          # measuring faces must not flex
GAUGE_BASE_MM = 2.8
GAUGE_LENGTH_MM = 28.0
GAUGE_TAB_MM = 4.0           # marks the tight end for counting
GAUGE_FLOOR_CLEARANCE_MM = 0.6
GAUGE_BAR_BODY_MM = (24.0, 6.0, 3.0)


def build_gauge_comb(config: Config) -> trimesh.Trimesh:
    """The slotted block. Each slot's designed gap is one gauge offset."""
    spans = [
        config.brick_stud_diameter_mm + offset
        for offset in config.brick_gauge_offsets_mm
    ]
    total_y = GAUGE_WALL_MM * (len(spans) + 1) + sum(spans)
    height = GAUGE_BASE_MM + config.brick_stud_height_mm + GAUGE_FLOOR_CLEARANCE_MM

    block = trimesh.creation.box(extents=(GAUGE_LENGTH_MM, total_y, height))
    block.apply_translation((GAUGE_LENGTH_MM / 2.0, total_y / 2.0, height / 2.0))

    # The tab: the tight-end wall runs longer than the block. Additive, so it
    # weakens no measuring face, and it makes "count from the tab" unambiguous.
    tab = trimesh.creation.box(extents=(GAUGE_TAB_MM, GAUGE_WALL_MM, height))
    tab.apply_translation((-GAUGE_TAB_MM / 2.0, GAUGE_WALL_MM / 2.0, height / 2.0))

    cutters = []
    y = GAUGE_WALL_MM
    depth = config.brick_stud_height_mm + GAUGE_FLOOR_CLEARANCE_MM
    for span in spans:
        cutter = trimesh.creation.box(
            extents=(GAUGE_LENGTH_MM + 2.0, span, depth + COPLANAR_EPSILON_MM)
        )
        cutter.apply_translation(
            (
                GAUGE_LENGTH_MM / 2.0,
                y + span / 2.0,
                height - (depth - COPLANAR_EPSILON_MM) / 2.0,
            )
        )
        cutters.append(cutter)
        y += span + GAUGE_WALL_MM

    solid = booleans.union([block, tab], config)
    return booleans.difference(solid, cutters, config)


def build_gauge_bar(config: Config) -> trimesh.Trimesh:
    """Three standard studs on a handle, printed studs-up like a brick.

    The studs are built by the same call with the same radius and segment
    count as a real brick's, so what the gauge measures is what a brick gets.
    """
    length, width, thick = GAUGE_BAR_BODY_MM
    body = trimesh.creation.box(extents=(length, width, thick))
    body.apply_translation((length / 2.0, width / 2.0, thick / 2.0))

    pitch = config.brick_unit_mm
    first_x = (length - 2.0 * pitch) / 2.0
    studs = [
        solids.cylinder_on_axis(
            np.array([first_x + i * pitch, width / 2.0, thick - COPLANAR_EPSILON_MM]),
            np.array([0.0, 0.0, 1.0]),
            config.brick_stud_radius_mm,
            config.brick_stud_height_mm + COPLANAR_EPSILON_MM,
            config,
        )
        for i in range(3)
    ]
    return booleans.union([body, *studs], config)


def write_brick_gauge_set(out_dir: str | Path, config: Config) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, mesh in (
        ("gauge_comb.stl", build_gauge_comb(config)),
        ("gauge_bar.stl", build_gauge_bar(config)),
    ):
        path = out / name
        mesh.export(path)
        written.append(path)
    readme = out / GAUGE_README_FILENAME
    readme.write_text(_gauge_instructions(config), encoding="utf-8")
    written.append(readme)
    return written


def _gauge_instructions(config: Config) -> str:
    stud = config.brick_stud_diameter_mm
    lines = [
        "THE FIT GAUGE".center(RULE_WIDTH),
        "=" * RULE_WIDTH,
        "",
        "This is not another fit test. It measures how your printer + slicer",
        "actually render a stud-in-slot fit, as one number, so the bricks can",
        "be corrected at the source instead of guessed at again.",
        "",
        "PRINT: both parts flat as exported, no supports, no brim needed,",
        "same filament and profile you use for bricks.",
        "",
        "READ IT:",
        "",
        "  The comb has slots from TIGHT to LOOSE. Slot 1 is the one next to",
        "  the tab that sticks out of the end. Hold the bar studs-down and try",
        "  each slot, starting from the tab end:",
        "",
        "    A = the first slot the bar DROPS INTO and slides freely",
        "    B = the first slot the bar can be PRESSED into with a firm push",
        "        and stays put when you lift the comb",
        "",
        "  Write down the two numbers (for example: A=5, B=4). That is the",
        "  whole measurement.",
        "",
        "-" * RULE_WIDTH,
        f"{'slot':>6}  {'designed gap':>13}  {'slot width':>11}",
        "-" * RULE_WIDTH,
    ]
    for index, offset in enumerate(config.brick_gauge_offsets_mm, start=1):
        lines.append(
            f"{index:>6}  {offset:>+10.2f} mm  {stud + offset:>8.2f} mm"
        )
    lines += [
        "-" * RULE_WIDTH,
        "",
        "The answer becomes brick_fit_bias_mm in your printer profile. After",
        "that, brick_clutch_interference_mm means the grip you actually get in",
        "the plastic, on this printer, with this profile.",
        "",
        "If the bar will not enter ANY slot, or rattles in slot 1, say so -",
        "that means the bias is outside this range and the next gauge will be",
        "shifted to find it.",
        "",
        "=" * RULE_WIDTH,
    ]
    return "\n".join(lines) + "\n"
