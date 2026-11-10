"""The shape catalogue and the benchmark that scores brick mode on it.

Brick mode was judged on a single vase until a probe on a cone showed it coming
out as nine pieces. These tests keep the yardstick itself honest: the shapes are
what they claim to be, the measurements measure what they say, and the verdicts
mean what the scoreboard says they mean. A benchmark that quietly mismeasures is
worse than none, because it gives a wrong answer with confidence.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cleaveform.brick import benchmark, shapes
from cleaveform.brick.assembly import pack_plates, plate_count
from cleaveform.brick.benchmark import ERROR, FAIL, PASS, SPLIT
from cleaveform.brick.layout import analyse, build_layout
from cleaveform.brick.library import build_all
from cleaveform.brick.occupancy import OccupancyGrid, build_occupancy
from cleaveform.brick.tiling import BrickPlacement
from cleaveform.config import Config

SIZE = 100.0


@pytest.fixture(scope="module")
def config() -> Config:
    return Config().with_overrides(decomposition_mode="brick")


class TestCatalogue:
    def test_names_are_unique_and_every_category_is_used(self):
        names = [s.name for s in shapes.SHAPES]
        assert len(names) == len(set(names))
        assert {s.category for s in shapes.SHAPES} == set(shapes.CATEGORIES)

    @pytest.mark.parametrize("shape", shapes.SHAPES, ids=lambda s: s.name)
    def test_each_shape_is_what_its_flags_say(self, shape):
        """A closed shape must be closed and right way out; the broken ones must
        be broken in exactly the way they claim, or they test nothing."""
        mesh = shape.mesh(SIZE)
        assert mesh.is_watertight == shape.watertight
        if shape.watertight:
            assert (mesh.volume < 0) == shape.inside_out
            assert abs(mesh.volume) > 0

    @pytest.mark.parametrize("shape", shapes.SHAPES, ids=lambda s: s.name)
    def test_each_shape_stands_on_the_plate_at_the_requested_size(self, shape):
        mesh = shape.mesh(SIZE)
        assert np.allclose(mesh.bounds[0], 0.0, atol=1e-6), "shape does not rest on the origin"
        assert float(mesh.extents.max()) == pytest.approx(SIZE, abs=1e-6)

    def test_a_shape_is_the_same_every_time(self):
        """The scoreboard compares runs, so the inputs cannot drift."""
        for name in ("blob", "vase", "table"):
            first, second = shapes.BY_NAME[name].mesh(SIZE), shapes.BY_NAME[name].mesh(SIZE)
            assert np.array_equal(first.vertices, second.vertices)

    @pytest.mark.parametrize("name", ["open_sphere", "inverted_sphere"])
    def test_broken_files_are_judged_against_what_they_were_meant_to_be(self, name):
        """Otherwise a sphere with a hole is scored on whatever the damaged
        surface happens to enclose, which says nothing about the tool."""
        shape = shapes.BY_NAME[name]
        reference = shape.reference_mesh(SIZE)
        assert reference.is_watertight and reference.volume > 0
        assert np.allclose(reference.bounds, shape.mesh(SIZE).bounds, atol=1e-6)

    def test_selection_by_name_and_category(self):
        assert [s.name for s in shapes.select(["cone", "cube"])] == ["cone", "cube"]
        assert {s.category for s in shapes.select(categories=["thin"])} == {"thin"}
        with pytest.raises(KeyError, match="unknown shape"):
            shapes.select(["not_a_shape"])
        with pytest.raises(KeyError, match="unknown categor"):
            shapes.select(categories=["not_a_category"])

    def test_two_bodies_are_expected_to_build_as_two_pieces(self):
        assert shapes.BY_NAME["two_bodies"].expected_pieces == 2


class TestFidelity:
    def _truth(self, unit):
        mesh = trimesh.creation.box(extents=(4 * unit, 4 * unit, 4 * unit))
        mesh.apply_translation(mesh.extents / 2)
        return benchmark.sample_truth(mesh)

    def test_bricks_that_exactly_fill_the_model_score_nearly_perfect(self, config):
        unit = config.brick_unit_mm
        grid = OccupancyGrid(np.ones((4, 4, 4), bool), unit, np.zeros(3))
        iou, missing, extra = benchmark.fidelity(self._truth(unit), grid)
        assert iou > 0.95 and missing < 0.05 and extra < 0.05

    def test_missing_bricks_show_up_as_missing_not_as_extra(self, config):
        unit = config.brick_unit_mm
        filled = np.ones((4, 4, 4), bool)
        filled[:, :, 2:] = False  # the top half is gone
        iou, missing, extra = benchmark.fidelity(self._truth(unit), OccupancyGrid(filled, unit, np.zeros(3)))
        assert missing == pytest.approx(0.5, abs=0.06)
        assert extra < 0.05
        assert iou == pytest.approx(0.5, abs=0.06)

    def test_bricks_where_the_model_is_not_show_up_as_extra(self, config):
        unit = config.brick_unit_mm
        filled = np.ones((8, 4, 4), bool)  # twice as long as the model
        iou, missing, extra = benchmark.fidelity(self._truth(unit), OccupancyGrid(filled, unit, np.zeros(3)))
        assert missing < 0.05
        assert extra == pytest.approx(1.0, abs=0.1)
        assert iou == pytest.approx(0.5, abs=0.06)

    def test_nothing_built_scores_zero(self, config):
        unit = config.brick_unit_mm
        grid = OccupancyGrid(np.zeros((4, 4, 4), bool), unit, np.zeros(3))
        iou, missing, _ = benchmark.fidelity(self._truth(unit), grid)
        assert iou == 0.0 and missing == 1.0


class TestStability:
    def test_a_plain_column_stands(self, config):
        grid = OccupancyGrid(np.ones((2, 2, 6), bool), config.brick_unit_mm, np.zeros(3))
        stands, margin = benchmark.stability(grid)
        assert stands and margin > 0

    def test_a_model_leaning_far_over_its_footprint_tips(self, config):
        """One cell of footprint carrying a long arm: the centre of mass is out
        past the edge of what it rests on."""
        filled = np.zeros((10, 1, 3), bool)
        filled[0, 0, 0] = True
        filled[:, 0, 1:] = True
        stands, margin = benchmark.stability(OccupancyGrid(filled, config.brick_unit_mm, np.zeros(3)))
        assert not stands and margin == 0.0


def _stack(type_id, w, d, x, y, courses):
    return [BrickPlacement(type_id, x, y, z, w, d, False) for z in courses]


def _layout_of(config, placements):
    nx = max(p.x + p.width_studs for p in placements)
    nz = max(p.z for p in placements) + 1
    filled = np.zeros((nx, 1, nz), bool)
    for p in placements:
        for cx, cy, cz in p.cells:
            filled[cx, cy, cz] = True
    return analyse(placements, OccupancyGrid(filled, config.brick_unit_mm, np.zeros(3)), config)


class TestVerdicts:
    def test_one_piece_where_one_is_expected_passes(self, config):
        layout = _layout_of(config, _stack("brick_4x1", 4, 1, 0, 0, range(10)))
        assert benchmark._verdict(layout, 1) == PASS

    def test_two_pieces_where_two_are_expected_passes(self, config):
        """A model made of two objects is two pieces by nature, and reporting a
        failure for it would punish the tool for being correct."""
        layout = _layout_of(
            config, _stack("brick_2x1", 2, 1, 0, 0, range(10)) + _stack("brick_2x1", 2, 1, 4, 0, range(10))
        )
        assert benchmark._verdict(layout, 2) == PASS
        assert benchmark._verdict(layout, 1) == FAIL

    def test_a_small_stray_builds_but_is_not_a_pass(self, config):
        main = _stack("brick_4x1", 4, 1, 0, 0, range(60))
        stray = _stack("brick_1x1", 1, 1, 5, 0, range(2))
        assert benchmark._verdict(_layout_of(config, main + stray), 1) == SPLIT

    def test_a_big_stray_group_fails(self, config):
        main = _stack("brick_4x1", 4, 1, 0, 0, range(60))
        stray = _stack("brick_1x1", 1, 1, 5, 0, range(config.brick_stray_hard_bricks + 1))
        assert benchmark._verdict(_layout_of(config, main + stray), 1) == FAIL


class TestScoring:
    def test_a_solid_cube_scores_as_a_clean_pass(self, config):
        cfg = config.with_overrides(brick_unit_mm=16.0)
        shape = shapes.BY_NAME["cube"]
        mesh = shape.mesh(SIZE)
        row, layout, grid = benchmark.score(
            shape, mesh, benchmark.sample_truth(shape.reference_mesh(SIZE)), 16.0, SIZE, cfg, build_all(cfg)
        )
        assert row.verdict == PASS and row.pieces == 1
        assert row.bricks == layout.brick_count > 0
        assert row.weight_g > 0 and row.print_hours > 0 and row.plates >= 1
        assert row.iou_pct > 70 and row.stands

    def test_a_model_too_small_to_fill_any_cell_is_reported_not_crashed(self, config):
        cfg = config.with_overrides(brick_unit_mm=16.0)
        shape = shapes.BY_NAME["cube"]
        mesh = shape.mesh(2.0)  # two millimetres across, against 16 mm bricks
        row, layout, grid = benchmark.score(
            shape, mesh, benchmark.sample_truth(shape.reference_mesh(2.0)), 16.0, 2.0, cfg, build_all(cfg)
        )
        assert layout is None and row.verdict == ERROR

    def test_the_brick_limit_is_reported_as_an_error_row(self, config):
        cfg = config.with_overrides(brick_unit_mm=6.0, max_bricks=10)
        shape = shapes.BY_NAME["cube"]
        row, layout, _ = benchmark.score(
            shape, shape.mesh(SIZE), benchmark.sample_truth(shape.reference_mesh(SIZE)), 6.0, SIZE, cfg, build_all(cfg)
        )
        assert layout is None and row.verdict == ERROR and "max_bricks" in row.note

    def test_the_plate_count_matches_what_packing_actually_produces(self, config):
        """The benchmark counts plates without building them, so the shortcut has
        to agree with the real packer."""
        meshes = build_all(config)
        for counts in ({"brick_1x1": 3}, {"brick_4x2": 120, "brick_2x1": 80}, {"brick_3x1": 1}):
            assert plate_count(counts, meshes, config) == len(pack_plates(counts, meshes, config))

    def test_every_brick_belongs_to_exactly_one_piece(self, config):
        shape = shapes.BY_NAME["cone"]
        cfg = config.with_overrides(brick_unit_mm=8.0)
        layout = build_layout(build_occupancy(shape.mesh(SIZE), cfg), cfg)
        assert len(layout.piece_of) == layout.brick_count
        assert layout.piece_of.count(0) == layout.piece_sizes[0]
        assert sorted(set(layout.piece_of)) == list(range(layout.piece_count))


class TestScoreboard:
    def _rows(self, config):
        cfg = config.with_overrides(brick_unit_mm=16.0)
        shape = shapes.BY_NAME["cube"]
        row, *_ = benchmark.score(
            shape, shape.mesh(SIZE), benchmark.sample_truth(shape.reference_mesh(SIZE)), 16.0, SIZE, cfg, build_all(cfg)
        )
        return [row]

    def test_all_four_files_are_written_and_name_the_shape(self, config, tmp_path):
        written = benchmark.write_scoreboard(self._rows(config), tmp_path)
        assert set(written) == {"csv", "json", "markdown", "html"}
        for path in written.values():
            assert "cube" in path.read_text(encoding="utf-8")

    def test_the_same_rows_give_the_same_files(self, config, tmp_path):
        """The scoreboard is compared run to run, so it cannot vary by itself."""
        rows = self._rows(config)
        first = benchmark.write_scoreboard(rows, tmp_path / "a")
        second = benchmark.write_scoreboard(rows, tmp_path / "b")
        for key in ("csv", "json", "markdown", "html"):
            assert first[key].read_bytes() == second[key].read_bytes()

    def test_a_picture_is_drawn_with_the_strays_marked(self, config, tmp_path):
        pytest.importorskip("matplotlib")
        cfg = config.with_overrides(brick_unit_mm=8.0)
        shape = shapes.BY_NAME["cone"]
        mesh = shape.mesh(SIZE)
        grid = build_occupancy(mesh, cfg)
        layout = build_layout(grid, cfg)
        path = tmp_path / "cone.png"
        assert benchmark.render_pair(mesh, grid, layout, path)
        assert path.stat().st_size > 5000

    def test_the_command_writes_a_scoreboard(self, tmp_path):
        from click.testing import CliRunner

        from cleaveform.cli import main

        result = CliRunner().invoke(
            main, ["bricktest", "-o", str(tmp_path), "--only", "cube", "--sizes", "16", "--no-images"]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "scoreboard.html").exists()
        assert "pass" in result.output


class TestOnlyRealBricks:
    """Every brick a layout names must be a brick the library can build.

    The connectivity repair once minted ``brick_1x2`` - a brick that is only ever
    the library's ``brick_2x1`` turned on its side. Nothing checked, so the
    layout looked fine until plate packing looked the brick up and failed with a
    bare KeyError. It was found by the benchmark, on a cup, and not by any test.
    """

    def test_a_bridge_across_a_split_uses_the_librarys_own_brick(self, config):
        from cleaveform.brick.layout import _apply_bridge, _owner_grid
        from cleaveform.brick.library import load_library

        library = load_library(config)
        placements = [
            BrickPlacement("brick_1x1", 0, 0, 0, 1, 1, False),
            BrickPlacement("brick_1x1", 0, 1, 0, 1, 1, False),
        ]
        grid = OccupancyGrid(np.ones((1, 2, 1), bool), config.brick_unit_mm, np.zeros(3))
        owner = _owner_grid(placements, grid.shape)
        across_y = _apply_bridge(placements, owner, (0, (0, 0), (0, 1)), grid, library, config)

        ids = {b.id for b in library}
        assert [p.type_id for p in across_y] == ["brick_2x1"]
        assert across_y[0].type_id in ids
        assert (across_y[0].width_studs, across_y[0].depth_studs) == (1, 2)
        assert across_y[0].rotated

    def test_a_library_with_no_pair_brick_is_left_alone_not_crashed(self, config):
        from cleaveform.brick.layout import repair_connectivity

        cfg = config.with_overrides(brick_library=("1x1",))
        placements = [
            BrickPlacement("brick_1x1", 0, 0, 0, 1, 1, False),
            BrickPlacement("brick_1x1", 2, 0, 0, 1, 1, False),
        ]
        grid = OccupancyGrid(np.array([[[True]], [[False]], [[True]]]), cfg.brick_unit_mm, np.zeros(3))
        assert repair_connectivity(placements, grid, cfg) == placements

    @pytest.mark.parametrize("unit", [6.0, 8.0, 16.0])
    def test_no_shape_in_the_catalogue_uses_a_brick_that_does_not_exist(self, config, unit):
        from cleaveform.brick.library import load_library
        from cleaveform.brick.tiling import TilingError

        cfg = config.with_overrides(brick_unit_mm=unit)
        real = {b.id for b in load_library(cfg)}
        for shape in shapes.SHAPES:
            try:
                layout = build_layout(build_occupancy(shape.mesh(SIZE), cfg), cfg)
            except TilingError:
                continue
            unknown = set(layout.counts_by_type) - real
            assert not unknown, f"{shape.name} at {unit:g} mm uses {sorted(unknown)}"


class TestLoadStageIsNotSkipped:
    """The real loader repairs a mesh and refuses it if it cannot.

    The first scoreboard voxelized raw meshes, so a sphere with one face missing
    scored as 244 pieces at 6 mm. That was a statement about a stage the real
    tool never lets a file skip, not about brick mode.
    """

    def test_a_sphere_with_a_hole_is_repaired_before_it_is_scored(self, config):
        rows, _ = benchmark.run(
            [shapes.BY_NAME["open_sphere"]], (6.0,), SIZE, config
        )
        assert rows[0].verdict == PASS, rows[0].note
        assert rows[0].pieces == 1

    def test_a_file_that_cannot_be_repaired_is_an_error_row_not_a_crash(self, config, monkeypatch):
        from cleaveform.geometry import repair

        monkeypatch.setattr(repair, "ensure_watertight", lambda mesh, cfg: (mesh, False))
        rows, kept = benchmark.run([shapes.BY_NAME["cube"]], (16.0, 8.0), SIZE, config)
        assert [r.verdict for r in rows] == [ERROR, ERROR]
        assert all("watertight" in r.note for r in rows)
        assert kept == {}

    def test_an_inside_out_file_is_turned_the_right_way_round(self, config):
        rows, _ = benchmark.run([shapes.BY_NAME["inverted_sphere"]], (8.0,), SIZE, config)
        assert rows[0].verdict == PASS and rows[0].iou_pct > 70


class TestFloatingBricks:
    """A model can be one piece and still not be buildable from the bottom up."""

    def test_floating_bricks_are_counted_per_size(self, config):
        base = dict(shape="s", category="solid", size_mm=SIZE, verdict=PASS)
        rows = [
            benchmark.Row(unit_mm=8.0, loose=0, **base),
            benchmark.Row(unit_mm=8.0, loose=5, **base),
            benchmark.Row(unit_mm=16.0, loose=0, **base),
        ]
        assert benchmark.floating_counts(rows) == {8.0: 1, 16.0: 0}

    def test_a_pass_with_floating_bricks_says_so_in_the_scoreboard(self, config):
        row = benchmark.Row("sphere", "solid", 8.0, SIZE, verdict=PASS, pieces=1, bricks=652, loose=68)
        assert benchmark._cell(row) == "PASS 1p 652b 68f"
        clean = benchmark.Row("cube", "solid", 8.0, SIZE, verdict=PASS, pieces=1, bricks=851)
        assert benchmark._cell(clean) == "PASS 1p 851b"

    def test_the_markdown_reports_how_many_rows_float(self, config):
        base = dict(shape="s", category="solid", size_mm=SIZE, verdict=PASS, pieces=1, bricks=10)
        rows = [benchmark.Row(unit_mm=8.0, loose=3, **base)]
        text = benchmark._markdown(rows)
        assert "with floating bricks" in text and "1 of 1" in text


BASELINE = __import__("json").loads(
    (__import__("pathlib").Path(__file__).parent / "fixtures" / "brick_baseline.json").read_text(encoding="utf-8")
)["rows"]
# Two sizes keep this quick; the full sweep is `cleaveform bricktest`.
GUARDED_UNITS = (12.0, 16.0)


class TestNothingThatPassedHasRegressed:
    """Fixing the tiler for one shape must not break another.

    The baseline was taken before the tiler was taught to track connectivity.
    Every shape that built correctly then has to keep doing so, and no shape may
    start needing a brick the library does not have. Shapes that fail in the
    baseline are not pinned: improving them is the point.
    """

    @pytest.mark.parametrize("unit", GUARDED_UNITS)
    def test_shapes_that_passed_still_pass(self, config, unit):
        passed = {r["shape"] for r in BASELINE if r["unit_mm"] == unit and r["verdict"] == PASS}
        selected = [s for s in shapes.SHAPES if s.name in passed]
        rows, _ = benchmark.run(selected, (unit,), shapes.DEFAULT_SIZE_MM, config)
        regressed = [(r.shape, r.verdict, r.note) for r in rows if r.verdict != PASS]
        assert not regressed, f"passed in the baseline, fail now at {unit:g} mm: {regressed}"


def _grid(courses, unit):
    """An occupancy grid from per-course ASCII pictures ('#' filled, '.' empty).
    Each picture is rows of the same width, indexed [y][x]."""
    nz = len(courses)
    ny, nx = len(courses[0]), len(courses[0][0])
    filled = np.zeros((nx, ny, nz), bool)
    for z, rows in enumerate(courses):
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                filled[x, y, z] = ch == "#"
    return OccupancyGrid(filled=filled, pitch_mm=unit, origin_mm=np.zeros(3))


class TestTilerKeepsTheModelAttached:
    """The tiler used to ask "does this brick sit on two different bricks?".

    That is not the goal. Two bricks already tied together by a lower course gain
    nothing from being spanned again, while two bricks in separate towers are what
    must be tied. A solid pyramid came out as eighteen pieces, one of them a
    31-brick tower up a face, and the bottom course - tiled with no objective at
    all - left bricks lying wholly in a rim with nothing over them.
    """

    @pytest.mark.parametrize("name,unit", [("pyramid", 8.0), ("pyramid", 12.0), ("pyramid", 16.0),
                                           ("cone", 12.0), ("cone", 6.0), ("mushroom", 8.0)])
    def test_shapes_that_used_to_come_apart_are_one_piece(self, config, name, unit):
        cfg = config.with_overrides(brick_unit_mm=unit)
        layout = build_layout(build_occupancy(shapes.BY_NAME[name].mesh(shapes.DEFAULT_SIZE_MM), cfg), cfg)
        assert layout.piece_count == 1, layout.piece_summary()
        assert layout.ok

    def test_a_pyramid_is_never_a_tower_of_strays(self, config):
        """The failure was a big stray GROUP, not a few loose bricks."""
        cfg = config.with_overrides(brick_unit_mm=8.0)
        layout = build_layout(build_occupancy(shapes.BY_NAME["pyramid"].mesh(shapes.DEFAULT_SIZE_MM), cfg), cfg)
        assert layout.metrics["largest_stray_bricks"] <= cfg.brick_stray_hard_bricks

    def test_an_overhang_is_held_by_a_brick_that_reaches_back_to_something(self, config):
        """One cell hangs past the course below. A brick lying wholly in the
        overhang rests on nothing; the brick covering it must also cover a cell
        that is supported, reaching inward from the edge."""
        from cleaveform.brick.tiling import tile_grid

        unit = config.brick_unit_mm
        # Found by searching small grids for one the old tiler got wrong: the cell
        # at the left of the top course hangs past the one pair below it, and the
        # old tiler laid it as a lone 1x1 resting on nothing.
        grid = _grid([[".##."], ["##.."]], unit)
        placements = tile_grid(grid, config)
        layout = analyse(placements, grid, config)
        assert layout.metrics["loose_brick_count"] == 0
        assert layout.piece_count == 1

    def test_the_bottom_course_does_not_leave_a_brick_nothing_can_hold(self, config):
        """Nothing is below the bottom course, so a brick there is held only if it
        sits under something. A brick lying wholly in a rim with nothing over it
        is a stray from the moment it is placed."""
        from cleaveform.brick.tiling import tile_grid

        unit = config.brick_unit_mm
        # Also found by search. The old tiler laid the bottom row as a 4x1 and a
        # lone 1x1 at the end, and the 1x1 has nothing over it: a stray from birth.
        grid = _grid([["#####"], ["###.."]], unit)
        placements = tile_grid(grid, config)
        layout = analyse(placements, grid, config)
        for placement in (p for p in placements if p.z == 0):
            assert any(grid.filled[x, y, 1] for x, y in placement.footprint_cells), (
                f"{placement.type_id} at {placement.x},{placement.y} has nothing over it"
            )
        assert layout.piece_count == 1

    def test_a_course_can_run_crosswise_to_the_one_below(self, config):
        """The pyramid's cause: a single-cell tip forces the first brick one way,
        which forces every later one the same way, so every course is parallel
        stripes whose seams never cross. Alternate courses must be free to run the
        other way, or the stripes can never be tied together."""
        from cleaveform.brick.tiling import tile_grid

        unit = config.brick_unit_mm
        diamond = [".#.", "###", ".#."]
        grid = _grid([diamond] * 6, unit)
        layout = analyse(tile_grid(grid, config), grid, config)
        assert layout.piece_count == 1

    def test_a_cube_does_not_need_more_bricks_than_before(self, config):
        """Scoring a tiling by the SUM of its bonds rewards using more bricks, and
        one cube at the finest size ran over the brick limit because of it."""
        cfg = config.with_overrides(brick_unit_mm=6.0)
        layout = build_layout(build_occupancy(shapes.BY_NAME["cube"].mesh(shapes.DEFAULT_SIZE_MM), cfg), cfg)
        assert layout.brick_count < cfg.max_bricks


class TestGroupTracking:
    def test_bricks_spanning_two_groups_join_them(self):
        from cleaveform.brick.tiling import NO_BRICK, BrickPlacement, _Groups

        groups = _Groups()
        shape = (4, 1)
        bottom = [BrickPlacement("brick_2x1", 0, 0, 0, 2, 1, False), BrickPlacement("brick_2x1", 2, 0, 0, 2, 1, False)]
        first = groups.advance(bottom, None, shape)
        assert first[0, 0] != first[2, 0], "side by side in one course are not joined"

        spanning = [BrickPlacement("brick_2x1", 1, 0, 1, 2, 1, False)]
        second = groups.advance(spanning, first, shape)
        assert second[1, 0] == second[2, 0]
        # both bottom bricks now belong to the same group as the one above
        assert groups.find(int(first[0, 0])) == groups.find(int(first[2, 0]))
        assert second[0, 0] == NO_BRICK

    def test_a_brick_over_nothing_starts_its_own_group(self):
        from cleaveform.brick.tiling import BrickPlacement, _Groups

        groups = _Groups()
        below = np.full((3, 1), -1, dtype=np.int32)
        out = groups.advance([BrickPlacement("brick_1x1", 1, 0, 1, 1, 1, False)], below, (3, 1))
        assert out[1, 0] >= 0


class TestCrosswiseStrategies:
    def test_the_plain_strategies_come_first_so_ties_change_nothing(self):
        from cleaveform.brick.tiling import BASE_STRATEGIES, SCAN_STRATEGIES

        assert SCAN_STRATEGIES[: len(BASE_STRATEGIES)] == tuple(s + (False,) for s in BASE_STRATEGIES)
        assert all(s[4] for s in SCAN_STRATEGIES[len(BASE_STRATEGIES):])

    def test_untransposing_swaps_the_axes_and_the_rotation_flag(self):
        from cleaveform.brick.tiling import BrickPlacement, _untranspose

        long_way = BrickPlacement("brick_4x1", 2, 5, 3, 4, 1, False)
        back = _untranspose(long_way)
        assert (back.x, back.y, back.width_studs, back.depth_studs) == (5, 2, 1, 4)
        assert back.rotated and back.type_id == "brick_4x1"
        assert _untranspose(back).rotated is False
        square = BrickPlacement("brick_2x2", 1, 2, 0, 2, 2, False)
        assert _untranspose(square).rotated is False

    @pytest.mark.parametrize("strategy", [(0, 0, False, False, True), (1, 1, True, False, True), (0, 0, True, True, True)])
    def test_a_transposed_course_still_covers_exactly_the_cells_it_was_given(self, config, strategy):
        from cleaveform.brick.library import load_library
        from cleaveform.brick.tiling import _tile_with

        mask = np.zeros((7, 5), bool)
        mask[1:6, 0:4] = True
        mask[2, 4] = True
        placed, _ = _tile_with(mask, 0, load_library(config), config, None, strategy)
        covered = np.zeros_like(mask, dtype=int)
        for p in placed:
            covered[p.x : p.x + p.width_studs, p.y : p.y + p.depth_studs] += 1
        assert (covered == mask.astype(int)).all(), "a cell is missing or covered twice"


class TestSimplification:
    """Some cells cannot be held by any tiling, so the SHAPE has to change.

    Each grid below was found by searching small grids for one that the plain
    tiler leaves in two pieces and that one kind of repair attaches. They are the
    smallest cases, so a repair that stops working fails here and not only on a
    140 mm vase.
    """

    @staticmethod
    def _pieces(courses, config, simplify):
        cfg = config.with_overrides(brick_simplify=simplify)
        return build_layout(_grid(courses, cfg.brick_unit_mm), cfg)

    def test_a_sliver_is_dropped(self, config):
        courses = [["...", ".#.", "..."], ["...", "...", "..#"]]
        assert self._pieces(courses, config, False).piece_count == 2
        layout = self._pieces(courses, config, True)
        assert layout.piece_count == 1
        assert layout.changes["dropped_cells"] > 0
        assert layout.grid.cell_count < 2

    def test_a_flat_rail_is_thickened(self, config):
        courses = [["..#...."], ["...###."]]
        assert self._pieces(courses, config, False).piece_count == 2
        layout = self._pieces(courses, config, True)
        assert layout.piece_count == 1
        assert layout.changes["thickened_cells"] > 0

    def test_a_seam_is_stitched(self, config):
        courses = [["#...#"], ["#####"], ["#...#"]]
        assert self._pieces(courses, config, False).piece_count == 2
        layout = self._pieces(courses, config, True)
        assert layout.piece_count == 1
        assert layout.changes["stitched_cells"] > 0

    def test_a_model_that_already_holds_together_is_not_touched(self, config):
        courses = [["###", "###", "###"]] * 3
        layout = self._pieces(courses, config, True)
        assert layout.piece_count == 1
        assert not layout.changes or not layout.changes.get("passes")
        assert layout.grid.cell_count == 27
        assert layout.change_summary() == ""

    def test_the_change_is_reported_not_hidden(self, config):
        layout = self._pieces([["#...#"], ["#####"], ["#...#"]], config, True)
        assert "stitched" in layout.change_summary()
        assert any(f.code == "shape_simplified" for f in layout.findings)
        assert layout.to_dict()["shape_changes"]["stitched_cells"] > 0

    def test_two_separate_bodies_are_never_joined(self, config):
        """An empty gap is a real separation however narrow. Joining the two would
        invent material, and the model would come out as one object."""
        courses = [["##.##"], ["##.##"], ["##.##"]]
        for simplify in (False, True):
            layout = self._pieces(courses, config, simplify)
            assert layout.piece_count == 2, (simplify, layout.piece_summary())
            assert not layout.changes or not layout.changes.get("stitched_cells")

    def test_simplifying_never_leaves_a_model_worse_attached(self, config):
        courses = [["#...#"], ["#####"], ["#...#"]]
        plain = self._pieces(courses, config, False)
        simple = self._pieces(courses, config, True)
        assert simple.piece_count <= plain.piece_count
        assert simple.metrics["largest_stray_bricks"] <= plain.metrics["largest_stray_bricks"]


class TestKeepThin:
    """The majority vote deletes a rail thinner than half a cell, and the posts it
    held come out as separate pieces. Keeping thin features must add material only
    where the model has some, never across empty space."""

    @staticmethod
    def _line(gap_fraction, gap=3):
        """Two filled cells at either end of a row, ``gap`` cells apart."""
        from cleaveform.brick.occupancy import preserve_connections

        n = gap + 2
        coarse = np.zeros((n, 1, 1), bool)
        coarse[0, 0, 0] = coarse[-1, 0, 0] = True
        fraction = np.zeros((n, 1, 1))
        fraction[1:-1, 0, 0] = gap_fraction
        fraction[coarse] = 1.0
        return coarse, preserve_connections(coarse, fraction, 0.15, 3)

    def test_a_thin_bar_through_material_is_bridged(self):
        coarse, kept = self._line(0.3)
        assert kept.all()
        assert (kept | coarse).sum() == kept.sum() and (kept >= coarse).all()

    def test_an_empty_gap_is_never_bridged(self):
        coarse, kept = self._line(0.0)
        assert (kept == coarse).all()

    def test_material_below_the_threshold_is_not_enough(self):
        coarse, kept = self._line(0.1)
        assert (kept == coarse).all()

    def test_a_gap_wider_than_the_limit_is_a_real_separation(self):
        coarse, kept = self._line(0.4, gap=4)
        assert (kept == coarse).all()

    def test_it_only_ever_adds_cells(self, config):
        cfg = config.with_overrides(brick_unit_mm=12.0)
        mesh = shapes.BY_NAME["fence"].mesh(shapes.DEFAULT_SIZE_MM)
        plain = build_occupancy(mesh, cfg)
        thin = build_occupancy(mesh, cfg.with_overrides(brick_keep_thin=True))
        assert plain.added_thin == 0
        assert thin.added_thin > 0
        assert thin.cell_count == plain.cell_count + thin.added_thin

    def test_a_fence_comes_apart_less_with_it(self, config):
        cfg = config.with_overrides(brick_unit_mm=12.0)
        mesh = shapes.BY_NAME["fence"].mesh(shapes.DEFAULT_SIZE_MM)
        plain = build_layout(build_occupancy(mesh, cfg), cfg)
        kept_cfg = cfg.with_overrides(brick_keep_thin=True)
        kept = build_layout(build_occupancy(mesh, kept_cfg), kept_cfg)
        assert kept.piece_count < plain.piece_count
        assert "thin-feature" in kept.change_summary()

    def test_a_value_that_would_keep_nothing_is_refused(self, config):
        with pytest.raises(ValueError):
            config.with_overrides(brick_thin_min_fill=0.0)
        with pytest.raises(ValueError):
            config.with_overrides(brick_thin_max_gap_cells=-1)


class TestTwoObjectsStayTwo:
    """The control: a file holding two separate objects is two pieces, and no option
    may make it one. Anything that did would be inventing a connection."""

    @pytest.mark.parametrize("unit", (8.0, 12.0, 16.0))
    @pytest.mark.parametrize("simplify,keep_thin", [(True, False), (False, True), (True, True)])
    def test_two_bodies(self, config, unit, simplify, keep_thin):
        cfg = config.with_overrides(
            brick_unit_mm=unit, brick_simplify=simplify, brick_keep_thin=keep_thin
        )
        mesh = shapes.BY_NAME["two_bodies"].mesh(shapes.DEFAULT_SIZE_MM)
        layout = build_layout(build_occupancy(mesh, cfg), cfg)
        assert layout.piece_count == 2, layout.piece_summary()


class TestOptionsNeverBreakWhatWorked:
    """Simplification and thin-feature keeping are for models that fail. A model
    that already built correctly must come out exactly as it did."""

    @pytest.mark.parametrize("unit", GUARDED_UNITS)
    def test_shapes_that_passed_still_pass_and_are_unchanged(self, config, unit):
        passed = {r["shape"] for r in BASELINE if r["unit_mm"] == unit and r["verdict"] == PASS}
        selected = [s for s in shapes.SHAPES if s.name in passed]
        cfg = config.with_overrides(brick_simplify=True, brick_keep_thin=True)
        rows, _ = benchmark.run(selected, (unit,), shapes.DEFAULT_SIZE_MM, cfg)
        worse = [(r.shape, r.verdict, r.note) for r in rows if r.verdict != PASS]
        assert not worse, f"passed in the baseline, fail with the options at {unit:g} mm: {worse}"


class TestTheHintAndTheReport:
    @staticmethod
    def _fence(config, **overrides):
        cfg = config.with_overrides(brick_unit_mm=12.0, **overrides)
        mesh = shapes.BY_NAME["fence"].mesh(shapes.DEFAULT_SIZE_MM)
        return build_layout(build_occupancy(mesh, cfg), cfg)

    def test_a_failure_points_at_the_option_that_could_help(self, config):
        layout = self._fence(config)
        assert not layout.ok
        assert "try --simplify" in " ".join(f.message for f in layout.findings)

    def test_with_the_option_on_the_hint_is_not_repeated(self, config):
        layout = self._fence(config, brick_simplify=True)
        assert "try --simplify" not in " ".join(f.message for f in layout.findings)

    def test_the_guide_says_what_changed(self, config, tmp_path):
        from cleaveform import pipeline

        cfg = config.with_overrides(
            cache_enabled=False, brick_unit_mm=16.0, brick_simplify=True, brick_keep_thin=True
        )
        mesh = shapes.BY_NAME["fence"].mesh(shapes.DEFAULT_SIZE_MM)
        path = tmp_path / "fence.stl"
        mesh.export(path)
        result = pipeline.run(path, tmp_path / "out", cfg)
        guide = result.guide_path.read_text(encoding="utf-8")
        assert "Shape changes" in guide
        assert "thickened" in guide or "stitched" in guide


class TestSimplifyIsCachedFaithfully:
    def test_a_warm_run_matches_a_cold_one(self, config, tmp_path):
        from cleaveform import pipeline

        mesh = shapes.BY_NAME["fence"].mesh(shapes.DEFAULT_SIZE_MM)
        path = tmp_path / "fence.stl"
        mesh.export(path)
        cfg = config.with_overrides(
            brick_unit_mm=16.0, brick_simplify=True, brick_keep_thin=True,
            cache_dir=str(tmp_path / "cache"),
        )
        cold = pipeline.run(path, tmp_path / "cold", cfg)
        warm = pipeline.run(path, tmp_path / "warm", cfg)
        assert warm.cache_stats["miss_count"] == 0
        assert cold.manifest["totals"] == warm.manifest["totals"]
        assert (tmp_path / "cold" / "bill_of_materials.txt").read_bytes() == (
            tmp_path / "warm" / "bill_of_materials.txt"
        ).read_bytes()
        assert "Shape changes" in warm.guide_path.read_text(encoding="utf-8")

    def test_the_options_are_part_of_the_cache_key(self, config):
        from cleaveform.cache import STAGE_FIELDS

        flat = set().union(*STAGE_FIELDS.values())
        for field in ("brick_simplify", "brick_keep_thin", "brick_thin_min_fill",
                      "brick_thin_max_gap_cells", "brick_simplify_passes", "brick_sliver_max_cells"):
            assert field in flat, field


class TestTheCommandLineSuggestsTheFix:
    """The flags are opt-in, so a user whose model comes apart has to be told they
    exist, at the moment it matters and not only in the README."""

    def _run(self, tmp_path, *flags, lenient=False):
        import json

        from click.testing import CliRunner

        from cleaveform.cli import main

        path = tmp_path / "fence.stl"
        shapes.BY_NAME["fence"].mesh(shapes.DEFAULT_SIZE_MM).export(path)
        args = ["split", str(path), "-o", str(tmp_path / "out"), "--mode", "brick",
                "--brick-unit", "16", "--no-cache"]
        if lenient:
            # Let a model that comes apart still be written, so the summary shows.
            settings = tmp_path / "lenient.json"
            settings.write_text(json.dumps({"brick_detached_fraction_hard": 1.0}))
            args += ["--config", str(settings)]
        return CliRunner().invoke(main, [*args, *flags])

    def test_a_refusal_names_the_flag_that_could_help(self, tmp_path):
        result = self._run(tmp_path)
        assert result.exit_code != 0
        assert "try --simplify" in result.output

    def test_a_model_in_several_pieces_is_pointed_at_the_flags(self, tmp_path):
        result = self._run(tmp_path, lenient=True)
        assert result.exit_code == 0, result.output
        assert "builds as 3 pieces" in result.output
        assert "try --simplify" in result.output

    def test_with_the_flags_on_it_builds_as_one_and_says_nothing_more(self, tmp_path):
        result = self._run(tmp_path, "--simplify", "--keep-thin")
        assert result.exit_code == 0, result.output
        assert "builds as 1 piece" in result.output
        assert "try --simplify" not in result.output
