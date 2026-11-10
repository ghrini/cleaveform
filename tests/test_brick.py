"""Brick mode: voxelize a model and build it from a small set of shapes.

The tests worth having here are the ones that caught real bugs while this was
being written, so each one names the failure it guards against:

- the occupancy grid must land on the model's own lattice, or the whole build
  is offset by half a brick, which no brick-count assertion would notice
- courses must not all break in the same place, or the model splits along that
  line and comes apart in the hand
- studs must fit sockets rather than jam against solid plastic
- estimates must charge layer overhead per plate, not per brick
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pytest
import trimesh

from cleaveform.brick.assembly import pack_plates, world_transform
from cleaveform.brick.layout import build_layout
from cleaveform.calibration import (
    BRICK_README_FILENAME,
    GAUGE_README_FILENAME,
    GAUGE_WALL_MM,
    build_brick_rung,
    build_gauge_bar,
    build_gauge_comb,
    write_brick_calibration_set,
    write_brick_gauge_set,
)
from cleaveform.validate.constraints import HARD, WARNING
from cleaveform.brick.library import (
    BrickType,
    build_brick_solid,
    grip_rail_faces_local,
    load_library,
    stud_centres_local,
    tube_centres_local,
)
from cleaveform.brick.occupancy import OccupancyGrid, build_occupancy
from cleaveform.brick.tiling import TilingError, tile_grid, tile_layer
from cleaveform.config import Config
from cleaveform.estimate import batch_print_time_min, print_time_min
from cleaveform.geometry import booleans


@pytest.fixture(scope="module")
def brick_config() -> Config:
    return Config().with_overrides(decomposition_mode="brick")


def _box_of_units(config: Config, width: int, depth: int, height: int):
    unit = config.brick_unit_mm
    return trimesh.creation.box(extents=(width * unit, depth * unit, height * unit))


class TestOccupancy:
    def test_a_box_of_whole_units_gives_exactly_that_many_cells(self, brick_config):
        grid = build_occupancy(_box_of_units(brick_config, 4, 3, 2), brick_config)
        assert grid.shape == (4, 3, 2)
        assert grid.cell_count == 24

    def test_the_grid_is_anchored_on_the_model_not_the_voxeliser(self, brick_config):
        """trimesh reports the CENTRE of cell (0,0,0), not its corner.

        Treating one as the other offsets the whole build by half a brick,
        which a test that only counts bricks would never catch.
        """
        mesh = _box_of_units(brick_config, 2, 2, 2)
        mesh.apply_translation((13.7, -5.2, 41.9))
        grid = build_occupancy(mesh, brick_config)
        assert np.allclose(grid.cell_origin_mm(0, 0, 0), mesh.bounds[0])

    def test_majority_vote_does_not_inflate_the_model(self, brick_config):
        """Surface voxelization marks every cell the surface passes through."""
        vase = trimesh.creation.cylinder(radius=30.0, height=70.0, sections=48)
        grid = build_occupancy(vase, brick_config)
        naive = np.asarray(
            vase.voxelized(pitch=brick_config.brick_unit_mm).fill().matrix, dtype=bool
        )
        assert grid.cell_count < int(naive.sum())

    def test_is_deterministic(self, brick_config):
        mesh = _box_of_units(brick_config, 3, 3, 3)
        first = build_occupancy(mesh, brick_config)
        second = build_occupancy(mesh, brick_config)
        assert np.array_equal(first.filled, second.filled)
        assert np.allclose(first.origin_mm, second.origin_mm)


class TestBrickGeometry:
    @pytest.mark.parametrize("footprint", [(1, 1), (2, 1), (2, 2), (4, 2)])
    def test_solids_are_watertight(self, footprint, brick_config):
        mesh = build_brick_solid(BrickType(*footprint), brick_config)
        assert mesh.is_watertight
        assert mesh.volume > 0

    def test_the_brick_is_hollow(self, brick_config):
        """A solid brick grips rigidly, so a few hundredths of first-layer
        squish decide whether it holds. Hollow walls flex instead."""
        config = brick_config
        mesh = build_brick_solid(BrickType(2, 2), config)
        outer_box = (
            (2 * config.brick_unit_mm - config.brick_fit_clearance_mm) ** 2
            * config.brick_unit_mm
        )
        assert abs(mesh.volume) < outer_box * 0.75

    def test_the_designed_fit_actually_interferes(self, brick_config):
        """A clutch grips by interference, so the solids must overlap.

        This assertion is the inverse of the one it replaces. The original
        demanded that a stacked pair overlap by nothing, which every printed
        ladder satisfied and every printed ladder rattled - a fit that touches
        at exactly zero is a slip fit the moment any real process error moves
        either half. The overlap is the plastic that has to flex.
        """
        config = brick_config
        assert config.brick_clutch_interference_mm > 0.0

        lower = build_brick_solid(BrickType(2, 2), config)
        upper = build_brick_solid(BrickType(2, 2), config)
        upper.apply_translation((0.0, 0.0, config.brick_unit_mm))
        overlap = booleans.intersection_volume(lower, upper, config)
        assert overlap > 0.0, "the designed fit does not touch, so it cannot grip"
        assert overlap < 5.0, f"{overlap:.2f} mm3 is a collision, not a fit"

    def test_the_grip_survives_the_process_bias_that_killed_the_old_fit(
        self, brick_config
    ):
        """The reason the interference is large rather than carefully small.

        Both halves of the fit move with the same process bias in opposite
        directions - an external stud prints small, an internal cavity prints
        large - so a designed interference loses roughly four times any bias.
        FDM holds around 0.05 mm, which erases anything under 0.2 mm. The whole
        printed ladder (0.19 down to -0.12) sat inside that error bar, which is
        why four rounds of it all felt the same.
        """
        bias = 0.05
        assert brick_config.brick_clutch_interference_mm > 4.0 * bias

    def test_zero_interference_cannot_grip(self, brick_config):
        """Recorded because five rounds of printing were spent learning it.

        0.19, 0.175 and 0.16 were all tried as clearances and all rattled. A
        fit that only just touches leaves the stud free before it meets
        anything, so it is a slip fit by definition and shrinking the number
        never makes a clutch.
        """
        loose = brick_config.with_overrides(brick_clutch_interference_mm=0.0)
        lower = build_brick_solid(BrickType(2, 2), loose)
        upper = build_brick_solid(BrickType(2, 2), loose)
        upper.apply_translation((0.0, 0.0, loose.brick_unit_mm))
        assert booleans.intersection_volume(lower, upper, loose) == 0.0

    def test_a_firmer_setting_grips_harder(self, brick_config):
        """The tuning knob has to move the fit monotonically, or a print round
        spent on it teaches nothing."""
        volumes = []
        for interference in (0.10, 0.20, 0.35):
            config = brick_config.with_overrides(
                brick_clutch_interference_mm=interference
            )
            lower = build_brick_solid(BrickType(2, 2), config)
            upper = build_brick_solid(BrickType(2, 2), config)
            upper.apply_translation((0.0, 0.0, config.brick_unit_mm))
            volumes.append(booleans.intersection_volume(lower, upper, config))
        assert volumes == sorted(volumes)
        assert volumes[0] < volumes[-1]

    def test_the_ribs_and_not_the_tubes_do_the_gripping(self, brick_config):
        """The tubes were demoted deliberately.

        A tube braces the roof, and a roof that flexes is a spring nobody
        specified. Making the tube grip as well put the fit on a feature whose
        stiffness is set by the roof above it, so the interference landed on
        whichever part of the brick happened to be softest. Now the tubes stand
        clear and the ribs alone carry the fit.
        """
        config = brick_config
        stud = stud_centres_local(BrickType(2, 2), config)[0]
        tube = tube_centres_local(BrickType(2, 2), config)[0]
        gap = (
            float(np.linalg.norm(tube - stud))
            - config.brick_stud_radius_mm
            - config.brick_tube_radius_mm
        )
        assert gap == pytest.approx(config.brick_tube_clearance_mm, abs=1e-6)
        assert gap > 0.0, "the tube touches the stud, so it is gripping"

    def test_the_grip_runs_the_whole_length_of_the_wall(self, brick_config):
        """Why the grip is a rail and not one bump per stud.

        The bumps this replaces sat at the stud positions, so joining two bricks
        required every stud to register into its own pair of bumps at the same
        instant. In the hand that is not a click, it is a jam - reported as
        "I can't get them to align". A rail has nothing to line up against
        along its length, so a stud meets it wherever it lands.
        """
        config = brick_config
        for brick in (BrickType(1, 1), BrickType(4, 1), BrickType(4, 2)):
            mesh = build_brick_solid(brick, config)
            width, depth, _ = brick.size_mm(config)
            near, far = grip_rail_faces_local(brick, config)

            # Sample along the cavity: the grip must be present everywhere, not
            # only where a stud happens to sit.
            inset = config.brick_wall_mm + config.min_feature_mm
            xs = np.linspace(inset, width - inset, 9)
            z = config.brick_stud_height_mm - config.min_feature_mm
            probe_near = np.column_stack(
                [xs, np.full_like(xs, near - 0.05), np.full_like(xs, z)]
            )
            probe_far = np.column_stack(
                [xs, np.full_like(xs, far + 0.05), np.full_like(xs, z)]
            )
            for probe, side in ((probe_near, "near"), (probe_far, "far")):
                hits = mesh.contains(probe)
                assert hits.all(), (
                    f"{brick.id} {side} rail is missing at "
                    f"{(~hits).sum()} of {len(hits)} points along the wall"
                )

    def test_a_stud_is_gripped_from_both_sides(self, brick_config):
        """Gripped on one side only pushes a stud off its lattice position
        instead of holding it on one."""
        config = brick_config
        for brick in (BrickType(1, 1), BrickType(4, 1), BrickType(4, 2)):
            _, depth, _ = brick.size_mm(config)
            near, far = grip_rail_faces_local(brick, config)
            assert near == pytest.approx(depth - far, abs=1e-6)
            assert near < depth / 2.0 < far

    def test_the_rails_stand_proud_of_the_wall_by_the_grip_offset(self, brick_config):
        """How far the rails protrude IS the fit."""
        config = brick_config
        gripped_span = config.brick_cavity_width_mm - 2.0 * (
            config.brick_stud_grip_offset_mm
        )
        assert gripped_span == pytest.approx(
            config.brick_stud_diameter_mm - config.brick_clutch_interference_mm,
            abs=1e-6,
        )

    def test_the_rails_stay_inside_the_brick(self, brick_config):
        """A rail is sunk into the wall, so a large one can break out through the
        OUTSIDE of the brick - a part wider than its own footprint, which fouls
        its neighbours on the grid instead of gripping anything."""
        config = brick_config
        for brick in (BrickType(1, 1), BrickType(3, 1), BrickType(4, 2)):
            mesh = build_brick_solid(brick, config)
            width, depth, _ = brick.size_mm(config)
            assert mesh.is_watertight
            assert mesh.bounds[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
            assert mesh.bounds[1][:2] == pytest.approx([width, depth], abs=1e-6)

    def test_every_first_layer_feature_is_printable(self, brick_config):
        """The first layer is the whole of a brick's grip on the build plate.

        A brick is a hollow shell open underneath, so its first layer is just
        the wall outline plus the tube rings - and the redesign thinned the wall
        from 1.58 mm to 0.90 mm, which cut that contact area by roughly 40%.
        Adhesion is therefore the design's known weak point and brick prints
        want a brim; see BRICK_CALIBRATION.txt.

        This test guards the other half of the problem: that whatever IS down
        there can actually be laid down by a nozzle. Worth stating plainly,
        because a previous version of this test was written to catch rib noses
        reaching the plate as unprintable specks, and that defect did not exist
        - the noses join the wall over about 0.57 mm, comfortably wider than a
        0.40 mm bead, so they were always fused. The measurement that suggested
        otherwise was of the rib's penetration PAST the wall face (0.16 mm),
        which is not the width of the join at all.
        """
        config = brick_config
        bead = config.nozzle_width_mm
        height = config.layer_height_mm / 2.0

        for brick in (BrickType(1, 1), BrickType(3, 1), BrickType(4, 2)):
            mesh = build_brick_solid(brick, config)
            section = mesh.section(
                plane_normal=[0.0, 0.0, 1.0], plane_origin=[0.0, 0.0, height]
            )
            assert section is not None, f"{brick.id} has no first layer at all"
            polygons = section.to_planar()[0].polygons_full

            expected = 1 + len(tube_centres_local(brick, config))
            assert len(polygons) == expected, (
                f"{brick.id} slices into {len(polygons)} first-layer islands, "
                f"expected {expected} (the wall ring plus its tubes)"
            )
            for polygon in polygons:
                assert not polygon.buffer(-bead / 2.0).is_empty, (
                    f"{brick.id} has a first-layer island narrower than one "
                    f"{bead} mm bead"
                )

    def test_the_rails_clear_the_elephant_foot_layer(self, brick_config):
        """Relief is bought out of engagement, so it is exactly one layer.

        Elephant foot squashes the first layer outward, so a rib nose printed in
        it comes out fatter than designed - the stud gets harder to start
        without the joint getting any stronger. One layer of relief removes
        that; more than one just spends grip that a 2.6 mm stud does not have.
        """
        config = brick_config
        assert config.brick_rib_relief_mm == pytest.approx(config.layer_height_mm)

    def test_the_rib_lead_in_keeps_the_grip_off_the_first_layer(self, brick_config):
        """The first printed layer is the one squashed by elephant foot, so a
        grip that starts there is calibrated on the least repeatable surface in
        the part. The rounded nose also stops the stud meeting a square step,
        which would shave or permanently set the rib on first assembly - a
        second way a designed press fit decays into a slip fit."""
        config = brick_config
        assert config.brick_rib_relief_mm >= config.layer_height_mm
        assert config.brick_rib_lead_in_mm == pytest.approx(
            config.brick_rib_relief_mm + config.brick_rail_chamfer_mm
        )
        assert 0.0 < config.brick_rib_lead_in_mm < config.brick_stud_height_mm

    def test_the_stud_still_has_something_to_grip(self, brick_config):
        """Relief is bought out of engagement, so it has to leave enough behind.

        A stud is only 2.6 mm long. Push the relief too high and the rib reaches
        full width above the stud tip, which is a brick that grips on nothing at
        all - the failure the relief was meant to prevent, arrived at from the
        other side.
        """
        config = brick_config
        full_width_from = config.brick_rib_lead_in_mm
        engagement = config.brick_stud_height_mm - full_width_from
        # Deliberately below the full 2.6 mm stud: the 45-degree lead-in ramp
        # is charged against engagement too, and it grows with the
        # interference, so a hard setting buys grip and spends entry depth. The
        # ramp still bears on the stud, so this floor is conservative.
        assert engagement > 1.2, (
            f"only {engagement:.2f} mm of the stud meets a full-width rail"
        )

    def test_too_much_relief_is_refused(self, brick_config):
        with pytest.raises(ValueError, match="never reaches the gripping part"):
            brick_config.with_overrides(brick_rib_relief_mm=3.0)

    def test_ribs_on_the_build_plate_are_refused(self, brick_config):
        with pytest.raises(ValueError, match="still land on the first layer"):
            brick_config.with_overrides(brick_rib_relief_mm=0.05)

    def test_one_wide_bricks_are_gripped_by_their_rails(self, brick_config):
        """No crossing exists on a 1xN brick, so it has no tube at all. It grips
        exactly like every other footprint, from the two rails."""
        config = brick_config
        assert tube_centres_local(BrickType(4, 1), config) == []
        assert len(grip_rail_faces_local(BrickType(4, 1), config)) == 2

    def test_tube_count_matches_the_stud_crossings(self, brick_config):
        """A 2x2 has one tube, a 2x4 has three - same as a real brick."""
        assert len(tube_centres_local(BrickType(2, 2), brick_config)) == 1
        assert len(tube_centres_local(BrickType(4, 2), brick_config)) == 3

    def test_the_brick_fit_is_independent_of_the_chunk_clearance(self, brick_config):
        """Retuning the brick clutch must never retune the calibrated chunk
        joints, which cost their own print rounds to land on 0.19 mm."""
        retuned = brick_config.with_overrides(brick_clutch_interference_mm=0.50)
        assert retuned.joint_clearance_mm == brick_config.joint_clearance_mm
        assert retuned.brick_stud_grip_offset_mm != (
            brick_config.brick_stud_grip_offset_mm
        )

    def test_studs_sit_on_the_global_stud_lattice(self, brick_config):
        """A stud must land over the socket of whatever sits below it."""
        config = brick_config
        centres = stud_centres_local(BrickType(2, 1), config)
        spacing = abs(centres[1][0] - centres[0][0])
        assert spacing == pytest.approx(config.brick_unit_mm)

    def test_library_prefers_larger_and_squarer_bricks(self, brick_config):
        order = [b.footprint for b in load_library(brick_config)]
        assert order[0] == "4x2"
        assert order.index("2x2") < order.index("4x1")


class TestStagger:
    """Courses must not all break in the same place."""

    def test_an_eight_wide_wall_breaks_differently_on_alternate_courses(
        self, brick_config
    ):
        """The case a reversed scan direction cannot fix.

        An 8-wide run tiles 4|4 scanning either way, so the seam lands at 4 in
        every course and the wall splits down the middle. Consecutive courses
        must not share a seam position.
        """
        grid = OccupancyGrid(
            filled=np.ones((8, 1, 4), dtype=bool),
            pitch_mm=brick_config.brick_unit_mm,
            origin_mm=np.zeros(3),
        )
        placements = tile_grid(grid, brick_config)
        seams = {}
        for placement in placements:
            edge = placement.x + placement.width_studs
            if edge != 8:
                seams.setdefault(placement.z, set()).add(edge)

        for lower, upper in zip(range(3), range(1, 4)):
            shared = seams.get(lower, set()) & seams.get(upper, set())
            assert not shared, (
                f"courses {lower} and {upper} share seam position(s) {shared}; "
                "nothing spans them and the wall splits there"
            )

    def test_a_solid_block_comes_out_as_one_connected_piece(self, brick_config):
        layout = build_layout(
            build_occupancy(_box_of_units(brick_config, 8, 8, 4), brick_config),
            brick_config,
        )
        assert layout.metrics["component_count"] == 1
        assert layout.ok

    def test_most_bricks_bond_two_below_them(self, brick_config):
        """A brick resting on exactly one brick ties nothing together."""
        layout = build_layout(
            build_occupancy(_box_of_units(brick_config, 8, 8, 4), brick_config),
            brick_config,
        )
        assert layout.metrics["bond_fraction"] > 0.5

    @pytest.mark.parametrize(
        "mesh_factory",
        [
            lambda: trimesh.creation.icosphere(subdivisions=3, radius=40.0),
            lambda: trimesh.creation.cylinder(radius=30.0, height=70.0, sections=48),
            lambda: trimesh.creation.annulus(r_min=15.0, r_max=35.0, height=40.0),
        ],
    )
    def test_curved_shapes_stay_connected(self, mesh_factory, brick_config):
        """A shifted scan origin is absorbed by a round footprint, so the
        tiler mirrors the grid and repairs whatever is still split."""
        layout = build_layout(
            build_occupancy(mesh_factory(), brick_config), brick_config
        )
        assert layout.metrics["component_count"] == 1


# Interference between stacked bricks is the clutch, so it is bounded rather
# than forbidden. One stud's worth of plastic is far more than the fit needs and
# far less than a stud driven into a solid roof, which is the failure this
# separates from the intended overlap.
MAX_CLUTCH_OVERLAP_MM3 = 20.0


class TestAssembledGeometry:
    def test_no_two_bricks_occupy_the_same_space(self, brick_config):
        """Two different rules, because the two cases mean opposite things.

        Bricks side by side in one course never touch - nothing holds them to
        each other, and any overlap there is a part that will not physically go
        where the layout says it goes. Bricks in adjacent courses are SUPPOSED
        to overlap: that sliver is the clutch, and demanding zero here is
        exactly the assertion that let five rounds of slip-fit bricks pass.
        So the vertical case is bounded rather than forbidden - large enough to
        grip, small enough that a stud is not being driven into solid plastic.
        """
        config = brick_config
        grid = build_occupancy(_box_of_units(config, 4, 4, 3), config)
        layout = build_layout(grid, config)
        meshes = {
            type_id: build_brick_solid(
                BrickType(*_footprint_of(type_id)), config
            )
            for type_id in layout.used_type_ids
        }

        placed = []
        for placement in layout.placements:
            copy = meshes[placement.type_id].copy()
            copy.apply_transform(world_transform(placement, grid, config))
            placed.append((placement, copy))

        for (first, mesh_a), (second, mesh_b) in itertools.combinations(placed, 2):
            if abs(first.z - second.z) > 1:
                continue
            overlap = booleans.intersection_volume(mesh_a, mesh_b, config)
            where = (
                f"{first.type_id}@{first.x},{first.y},{first.z} and "
                f"{second.type_id}@{second.x},{second.y},{second.z}"
            )
            if first.z == second.z:
                assert overlap <= config.volume_epsilon_mm3, (
                    f"{where} share a course and overlap by {overlap:.2f} mm3, "
                    "so they cannot both be placed"
                )
            else:
                assert overlap < MAX_CLUTCH_OVERLAP_MM3, (
                    f"{where} overlap by {overlap:.2f} mm3, which is a jam "
                    "rather than a clutch"
                )

    def test_rotated_bricks_land_on_their_grid_cell(self, brick_config):
        config = brick_config
        grid = build_occupancy(_box_of_units(config, 4, 4, 2), config)
        layout = build_layout(grid, config)
        for placement in layout.placements:
            mesh = build_brick_solid(
                BrickType(*_footprint_of(placement.type_id)), config
            )
            mesh.apply_transform(world_transform(placement, grid, config))
            expected = grid.cell_origin_mm(placement.x, placement.y, placement.z)
            assert np.allclose(mesh.bounds[0][:2], expected[:2], atol=0.2)


class TestBounds:
    def test_too_many_bricks_is_a_named_failure(self, brick_config):
        config = brick_config.with_overrides(max_bricks=10)
        with pytest.raises(TilingError) as excinfo:
            tile_grid(build_occupancy(_box_of_units(config, 8, 8, 4), config), config)
        assert excinfo.value.detail["limit"] == "max_bricks"

    def test_a_unit_too_small_to_print_is_refused(self):
        with pytest.raises(ValueError, match="wide between the walls"):
            Config().with_overrides(brick_unit_mm=4.0)

    def test_the_unit_size_is_tunable(self):
        """Brick size is meant to be changed, within what will print."""
        for unit in (6.0, 8.0, 12.0):
            config = Config().with_overrides(brick_unit_mm=unit)
            assert config.brick_wall_mm >= config.min_wall_mm
            assert build_brick_solid(BrickType(2, 1), config).is_watertight

    def test_a_smaller_unit_gives_more_bricks(self):
        coarse = Config().with_overrides(decomposition_mode="brick", brick_unit_mm=12.0)
        fine = Config().with_overrides(decomposition_mode="brick", brick_unit_mm=6.0)
        model = trimesh.creation.icosphere(subdivisions=3, radius=40.0)
        assert (
            build_layout(build_occupancy(model, fine), fine).brick_count
            > build_layout(build_occupancy(model, coarse), coarse).brick_count
        )


class TestEstimates:
    def test_layer_overhead_is_charged_per_plate_not_per_brick(self, brick_config):
        """500 bricks on 6 plates share their layers.

        Charging each brick its own layer changes would book about ten hours
        that never happen and make every other estimate untrustworthy.
        """
        config = brick_config
        mesh = build_brick_solid(BrickType(2, 1), config)
        naive = print_time_min(mesh, config) * 500
        batched = batch_print_time_min(
            {"brick_2x1": mesh}, {"brick_2x1": 500}, plate_count=6, config=config
        )
        assert batched < naive / 2

    def test_more_plates_cost_more_time(self, brick_config):
        mesh = build_brick_solid(BrickType(2, 1), brick_config)
        counts = {"brick_2x1": 100}
        one = batch_print_time_min({"brick_2x1": mesh}, counts, 1, brick_config)
        many = batch_print_time_min({"brick_2x1": mesh}, counts, 8, brick_config)
        assert many > one


class TestPlates:
    def test_every_brick_gets_a_place_on_a_plate(self, brick_config):
        config = brick_config
        meshes = {"brick_1x1": build_brick_solid(BrickType(1, 1), config)}
        plates = pack_plates({"brick_1x1": 40}, meshes, config)
        assert plates
        for plate in plates:
            extents = plate.bounds[1] - plate.bounds[0]
            assert extents[0] <= config.plate_size_mm[0]
            assert extents[1] <= config.plate_size_mm[1]

    def test_no_plates_without_bricks(self, brick_config):
        assert pack_plates({}, {}, brick_config) == []


def _footprint_of(type_id: str) -> tuple[int, int]:
    width, depth = type_id.removeprefix("brick_").split("x")
    return int(width), int(depth)


class TestBrickCalibration:
    """The fixture that sets the clutch, and the confounds it has to avoid."""

    def test_every_rung_is_the_same_size(self, brick_config):
        """The whole point of the redesigned fixture.

        The first version labelled rungs by brick length, but length is also
        the dominant term in how stiff a brick's walls are - a 1x1 is a closed
        braced box, while a 3x1 wall's nearest brace is a whole unit away. So
        the "looser" rungs were partly just floppier ones, and the test
        measured two things at once and neither of them cleanly.
        """
        sizes = set()
        for index, interference in enumerate(
            brick_config.brick_calibration_interferences_mm, start=1
        ):
            mesh = build_brick_rung(interference, index, brick_config)
            extents = mesh.bounds[1] - mesh.bounds[0]
            sizes.add(tuple(round(float(v), 6) for v in extents))
        assert len(sizes) == 1, f"rungs differ in size: {sorted(sizes)}"

    def test_rungs_are_told_apart_by_dots(self, brick_config):
        """Identification must not touch anything that participates in the fit,
        so it lives on the structural end wall as a countable feature."""
        volumes = [
            build_brick_rung(interference, index, brick_config).volume
            for index, interference in enumerate(
                brick_config.brick_calibration_interferences_mm, start=1
            )
        ]
        assert volumes == sorted(volumes)
        assert len(set(volumes)) == len(volumes)

    def test_each_rung_grips_harder_than_the_last(self, brick_config):
        """A ladder whose rungs do not differ measurably teaches nothing - which
        is exactly what four earlier printed ladders did."""
        overlaps = []
        for index, interference in enumerate(
            brick_config.brick_calibration_interferences_mm, start=1
        ):
            lower = build_brick_rung(interference, index, brick_config)
            upper = build_brick_rung(interference, index, brick_config)
            upper.apply_translation((0.0, 0.0, brick_config.brick_unit_mm))
            overlaps.append(
                booleans.intersection_volume(lower, upper, brick_config)
            )
        assert overlaps == sorted(overlaps)
        assert all(b > a for a, b in zip(overlaps, overlaps[1:]))

    def test_the_ladder_is_wider_than_the_printer_can_resolve(self, brick_config):
        """Every step must clear the process bias, or the rungs are
        indistinguishable in the hand no matter what the geometry says."""
        rungs = brick_config.brick_calibration_interferences_mm
        steps = [b - a for a, b in zip(rungs, rungs[1:])]
        assert min(steps) == pytest.approx(0.10, abs=1e-9) or min(steps) > 0.10

    def test_every_rung_is_printable(self, brick_config):
        for index, interference in enumerate(
            brick_config.brick_calibration_interferences_mm, start=1
        ):
            mesh = build_brick_rung(interference, index, brick_config)
            assert mesh.is_watertight
            assert mesh.volume > 0.0

    def test_markers_cannot_silently_hang_off_the_brick(self, brick_config):
        """A dot overflowing the end wall fouls its neighbour on the plate, so
        a ladder too long for the wall is a named error rather than a surprise
        in the slicer."""
        crowded = brick_config.with_overrides(
            brick_calibration_marker_pitch_mm=6.0
        )
        with pytest.raises(ValueError, match="do not fit"):
            build_brick_rung(0.35, 5, crowded)

    def test_the_optional_brim_earns_its_keep_when_enabled(self, brick_config):
        """Off by default, but it has to work when switched on.

        The open-bottomed shell stands on a ring about 0.9 mm wide - roughly
        54 mm2 for a part 10.6 mm tall. Parts once came off the plate at that
        contact area; that turned out to be a printer fault rather than a
        geometry limit, so the flange is disabled rather than deleted.
        """
        config = brick_config.with_overrides(brick_calibration_brim_mm=3.0)
        z = config.layer_height_mm / 2.0

        bare = build_brick_solid(
            BrickType(*_footprint_of("brick_" + config.brick_calibration_footprint)),
            config,
        )
        bare_area = sum(
            p.area
            for p in bare.section(
                plane_normal=[0, 0, 1], plane_origin=[0, 0, z]
            ).to_planar()[0].polygons_full
        )

        coupon = build_brick_rung(config.brick_clutch_interference_mm, 2, config)
        coupon_area = sum(
            p.area
            for p in coupon.section(
                plane_normal=[0, 0, 1], plane_origin=[0, 0, z]
            ).to_planar()[0].polygons_full
        )
        assert coupon_area > 3.0 * bare_area, (
            f"the flange only takes bed contact from {bare_area:.0f} to "
            f"{coupon_area:.0f} mm2, which is not worth moulding on"
        )

    def test_the_brim_does_not_change_the_fit_it_is_measuring(self, brick_config):
        """A fixture that alters what it measures is worse than no fixture."""
        config = brick_config
        config = config.with_overrides(brick_calibration_brim_mm=3.0)
        plain = config.with_overrides(brick_calibration_brim_mm=0.0)
        for interference in config.brick_calibration_interferences_mm:
            with_brim = build_brick_rung(interference, 1, config)
            without = build_brick_rung(interference, 1, plain)
            upper_a = build_brick_rung(interference, 1, config)
            upper_b = build_brick_rung(interference, 1, plain)
            upper_a.apply_translation((0.0, 0.0, config.brick_unit_mm))
            upper_b.apply_translation((0.0, 0.0, config.brick_unit_mm))
            grip_a = booleans.intersection_volume(with_brim, upper_a, config)
            grip_b = booleans.intersection_volume(without, upper_b, plain)
            assert grip_a == pytest.approx(grip_b, rel=1e-6)

    def test_the_brim_stays_clear_of_the_cavity_mouth(self, brick_config):
        """A membrane across the mouth would have to be cut out before the
        coupon could be pressed together, and cutting it is how you nick the
        spring you are trying to measure."""
        config = brick_config.with_overrides(brick_calibration_brim_mm=3.0)
        coupon = build_brick_rung(config.brick_clutch_interference_mm, 2, config)
        brick = BrickType(
            *_footprint_of("brick_" + config.brick_calibration_footprint)
        )
        width, depth, _ = brick.size_mm(config)
        # A point just inside the cavity, at the very first layer.
        inside_mouth = [[width / 2.0, depth / 2.0, config.layer_height_mm / 2.0]]
        assert not coupon.contains(inside_mouth)[0], (
            "the flange closes the cavity mouth"
        )

    def test_the_set_writes_two_bricks_per_rung(self, brick_config, tmp_path):
        """One brick cannot be pressed into anything, so a rung needs a pair."""
        written = write_brick_calibration_set(tmp_path, brick_config)
        stls = [p for p in written if p.suffix == ".stl"]
        expected = 2 * len(brick_config.brick_calibration_interferences_mm)
        assert len(stls) == expected
        assert (tmp_path / BRICK_README_FILENAME).exists()

    def test_the_instructions_name_the_field_to_set(self, brick_config, tmp_path):
        """A calibration whose result has nowhere to go is a wasted print."""
        write_brick_calibration_set(tmp_path, brick_config)
        text = (tmp_path / BRICK_README_FILENAME).read_text(encoding="utf-8")
        assert "brick_clutch_interference_mm" in text
        assert "interference, not a clearance" in text
        for interference in brick_config.brick_calibration_interferences_mm:
            assert f"{interference:.2f}" in text


class TestGauge:
    """The fixture that measures the printer instead of guessing at it.

    Six rounds of coupon ladders failed to converge while the STL geometry was
    verified exact, so the error lives between the file and the plastic. The
    gauge reads that error as one number, by counting slots.
    """

    def _slot_span(self, comb, config, index):
        spans = [
            config.brick_stud_diameter_mm + o for o in config.brick_gauge_offsets_mm
        ]
        y = GAUGE_WALL_MM * (index + 1) + sum(spans[:index]) + spans[index] / 2.0
        origin = np.array([[14.0, y, comb.bounds[1][2] - 1.0]])
        hi = comb.ray.intersects_location(origin, [[0, 1, 0]])[0]
        lo = comb.ray.intersects_location(origin, [[0, -1, 0]])[0]
        return hi[:, 1].min() - lo[:, 1].max()

    def test_gauge_parts_are_printable(self, brick_config):
        assert build_gauge_comb(brick_config).is_watertight
        assert build_gauge_bar(brick_config).is_watertight

    def test_each_slot_encodes_its_designed_gap(self, brick_config):
        comb = build_gauge_comb(brick_config)
        stud = brick_config.brick_stud_diameter_mm
        for index, offset in enumerate(brick_config.brick_gauge_offsets_mm):
            span = self._slot_span(comb, brick_config, index)
            # The boolean engine quantises coordinates around 1e-5; anything
            # under a micron is exact for a process that holds a tenth of a mm.
            assert span == pytest.approx(stud + offset, abs=1e-3), (
                f"slot {index + 1} is {span:.3f} mm, expected {stud + offset:.3f}"
            )

    def test_the_bar_studs_are_real_brick_studs(self, brick_config):
        """The gauge must measure the fit a brick actually gets, so its studs
        come from the same call with the same radius and segment count."""
        bar = build_gauge_bar(brick_config)
        mid_stud_z = 3.0 + brick_config.brick_stud_height_mm / 2.0
        origin = np.array([[4.0, 3.0, mid_stud_z]])
        a = bar.ray.intersects_location(origin, [[1, 0, 0]])[0]
        b = bar.ray.intersects_location(origin, [[-1, 0, 0]])[0]
        width = a[:, 0].min() - b[:, 0].max()
        assert width == pytest.approx(
            brick_config.brick_stud_diameter_mm, abs=0.01
        )

    def test_the_tab_marks_the_tight_end(self, brick_config):
        """Slots are read by counting from the tab, so the tab must sit
        beside slot 1 and nowhere else."""
        comb = build_gauge_comb(brick_config)
        assert comb.bounds[0][0] < -1.0
        overhang = comb.vertices[comb.vertices[:, 0] < -1.0]
        assert overhang[:, 1].max() <= GAUGE_WALL_MM + 1e-6

    def test_the_set_names_the_field_it_measures(self, brick_config, tmp_path):
        written = write_brick_gauge_set(tmp_path, brick_config)
        assert len([p for p in written if p.suffix == ".stl"]) == 2
        text = (tmp_path / GAUGE_README_FILENAME).read_text(encoding="utf-8")
        assert "brick_fit_bias_mm" in text
        assert "tab" in text

    def test_the_measured_bias_corrects_the_designed_geometry(self, brick_config):
        """Once the gauge reports, the interference field means what happens in
        the plastic, and the file quietly compensates for the printer."""
        tight_printer = brick_config.with_overrides(brick_fit_bias_mm=-0.20)
        # A printer that prints 0.20 tight needs 0.20 LESS designed
        # interference, so the rails recede by 0.10 each.
        assert tight_printer.brick_stud_grip_offset_mm == pytest.approx(
            brick_config.brick_stud_grip_offset_mm - 0.10
        )
        loose_printer = brick_config.with_overrides(brick_fit_bias_mm=0.20)
        assert loose_printer.brick_stud_grip_offset_mm == pytest.approx(
            brick_config.brick_stud_grip_offset_mm + 0.10
        )

    def test_coupons_carry_their_batch_mark(self, brick_config):
        """Coupons from different geometry generations look nearly identical;
        an unmarked mix reads like noise and burns a print round."""
        one = build_brick_rung(0.40, 1, brick_config)
        two = build_brick_rung(
            0.40, 1, brick_config.with_overrides(brick_calibration_batch=2)
        )
        assert two.volume > one.volume
        assert two.is_watertight


class TestBrickPipelineEndToEnd:
    """The whole brick run, model in and guide out.

    Brick mode once crashed on its very last step - writing the guide read a
    config field the clutch redesign had removed - and 263 tests still passed,
    because every test exercised a stage and none ran the stages together. A
    run that builds everything and then fails at the end is the worst kind of
    failure to ship, so this runs all of it.
    """

    def _box_stl(self, tmp_path, config):
        unit = config.brick_unit_mm
        mesh = trimesh.creation.box(extents=(3 * unit, 2 * unit, 2 * unit))
        mesh.apply_translation(mesh.extents / 2.0)
        path = tmp_path / "box.stl"
        mesh.export(path)
        return path

    def test_a_brick_run_completes_and_writes_its_guide(self, brick_config, tmp_path):
        from cleaveform import pipeline

        config = brick_config.with_overrides(cache_enabled=False)
        out = tmp_path / "out"
        result = pipeline.run(self._box_stl(tmp_path, config), out, config)

        guide = result.guide_path.read_text(encoding="utf-8")
        assert "BRICK BUILD GUIDE" in guide
        assert "COURSE MAPS" in guide
        assert (out / "bill_of_materials.txt").exists()
        assert list((out / "bricks").glob("*.stl"))

    def test_the_guide_reports_the_fit_actually_in_use(self, brick_config, tmp_path):
        """A guide that quotes a removed field, or a stale number, would send the
        next calibration round in the wrong direction."""
        from cleaveform import pipeline

        config = brick_config.with_overrides(
            cache_enabled=False,
            brick_clutch_interference_mm=0.33,
            brick_fit_bias_mm=-0.07,
        )
        result = pipeline.run(self._box_stl(tmp_path, config), tmp_path / "o", config)
        guide = result.guide_path.read_text(encoding="utf-8")
        assert "0.33 mm interference" in guide
        assert "-0.07" in guide
        assert "brick_stud_clearance_mm" not in guide


def _stacked(type_id, width, depth, x, y, courses):
    """A tower of identical bricks in a column, each course directly on the last."""
    from cleaveform.brick.tiling import BrickPlacement

    return [
        BrickPlacement(type_id, x, y, z, width, depth, False) for z in courses
    ]


def _grid_for(placements, unit):
    from cleaveform.brick.occupancy import OccupancyGrid

    nx = max(p.x + p.width_studs for p in placements)
    ny = max(p.y + p.depth_studs for p in placements)
    nz = max(p.z for p in placements) + 1
    filled = np.zeros((nx, ny, nz), dtype=bool)
    for p in placements:
        for cx, cy, cz in p.cells:
            filled[cx, cy, cz] = True
    return OccupancyGrid(filled=filled, pitch_mm=unit, origin_mm=np.zeros(3))


class TestConnectivityReporting:
    """Whether a model builds as ONE object has to be loud and correct.

    Brick mode once passed a cone that came out as nine pieces, one of them an
    eleven-brick tower running up its flank. Nothing failed, because the only
    rule was the TOTAL share of detached bricks, and eleven bricks in a large
    build is a small share. No test covered the disconnected-assembly rule or
    the detached-brick finding at all. These place bricks by hand so they pin
    the rule itself rather than whatever the tiler happens to do today.
    """

    def _layout(self, config, stray_height, main_height=60):
        from cleaveform.brick.layout import analyse

        main = _stacked("brick_4x1", 4, 1, 0, 0, range(main_height))
        stray = _stacked("brick_1x1", 1, 1, 5, 0, range(stray_height))
        placements = main + stray
        return analyse(placements, _grid_for(placements, config.brick_unit_mm), config)

    def test_one_connected_stack_is_one_piece(self, brick_config):
        layout = self._layout(brick_config, stray_height=0)
        assert layout.piece_count == 1
        assert layout.piece_summary() == "1 piece"
        assert layout.ok
        assert layout.metrics["largest_stray_bricks"] == 0

    def test_a_small_stray_is_reported_but_does_not_stop_the_build(self, brick_config):
        """A few edge bricks with nothing to grip are a property of the shape,
        so the model still builds - but it must say so rather than stay silent."""
        layout = self._layout(brick_config, stray_height=brick_config.brick_stray_hard_bricks)
        assert layout.ok
        assert layout.piece_count == 2
        assert layout.metrics["largest_stray_bricks"] == brick_config.brick_stray_hard_bricks
        finding = next(f for f in layout.findings if f.code == "detached_bricks")
        assert finding.tier == WARNING, (
            "a split model must never be reported below a warning"
        )

    def test_one_big_stray_group_is_a_hard_failure_whatever_the_total(self, brick_config):
        """The cone case. Five detached bricks in a 65-brick build are about 8 %,
        under the 10 % total, so the old rule let a tower that was never attached
        to the model pass as a success."""
        stray = brick_config.brick_stray_hard_bricks + 1
        layout = self._layout(brick_config, stray_height=stray)

        total = layout.metrics["stray_brick_count"] / layout.brick_count
        assert total < brick_config.brick_detached_fraction_hard, (
            "this case only means something if the TOTAL rule would have passed it"
        )
        assert not layout.ok
        finding = next(f for f in layout.findings if f.code == "disconnected_assembly")
        assert finding.tier == HARD
        assert f"{stray} bricks" in finding.message

    def test_the_total_share_rule_still_applies(self, brick_config):
        """Many small strays can fail the build even when no one group is big."""
        from cleaveform.brick.layout import analyse

        main = _stacked("brick_4x1", 4, 1, 0, 0, range(4))
        strays = []
        for column in range(5, 9):
            strays += _stacked("brick_1x1", 1, 1, column, 0, range(2))
        placements = main + strays
        layout = analyse(
            placements, _grid_for(placements, brick_config.brick_unit_mm), brick_config
        )
        assert max(layout.piece_sizes[1:]) <= brick_config.brick_stray_hard_bricks
        assert not layout.ok

    def test_the_summary_names_the_main_body_and_every_stray(self, brick_config):
        layout = self._layout(brick_config, stray_height=3)
        summary = layout.piece_summary()
        assert "2 pieces" in summary
        assert "main body of 60 bricks" in summary
        assert "3 brick(s)" in summary

    def test_the_stray_limit_must_be_positive(self, brick_config):
        with pytest.raises(ValueError, match="brick_stray_hard_bricks"):
            brick_config.with_overrides(brick_stray_hard_bricks=0)

    def test_the_headline_reaches_the_manifest_the_guide_and_the_cli(
        self, brick_config, tmp_path
    ):
        """A split model used to be invisible outside the manifest's metrics."""
        from cleaveform import pipeline

        config = brick_config.with_overrides(cache_enabled=False)
        unit = config.brick_unit_mm
        mesh = trimesh.creation.box(extents=(3 * unit, 2 * unit, 2 * unit))
        mesh.apply_translation(mesh.extents / 2.0)
        path = tmp_path / "box.stl"
        mesh.export(path)

        result = pipeline.run(path, tmp_path / "out", config)
        assert result.manifest["totals"]["piece_count"] == 1
        assert "Builds as       : 1 piece" in result.guide_path.read_text(encoding="utf-8")

    def test_layout_advisories_reach_the_command_line(self, brick_config):
        """The guide carried these; the CLI summary never did."""
        from cleaveform.pipeline import _layout_advisories

        layout = self._layout(brick_config, stray_height=2)
        lines = _layout_advisories(layout)
        assert any(line.startswith("detached_bricks:") for line in lines)
