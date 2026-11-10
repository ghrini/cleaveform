"""Connector generation (module 5) and chunk-mode assembly checks (module 6)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cleaveform.decompose import build_cut_graph, decompose
from cleaveform.geometry import analysis, solids
from cleaveform.io import load_model
from cleaveform.joints import JointError, generate_joints
from cleaveform.validate import HARD, plan_assembly


def _prepare(path, config):
    mesh = load_model(path, config).mesh
    result = decompose(mesh, config)
    graph = build_cut_graph(result, config)
    return result, graph


class TestPegSocketGeneration:
    def test_peg_count_scales_with_face_area(self, oversize_bar, config):
        _, graph = _prepare(oversize_bar, config)
        joints = generate_joints(graph, config)
        for placement in joints.placements:
            expected = min(
                max(
                    int(np.ceil(placement.face_area_mm2 / config.peg_area_per_peg_mm2)),
                    config.min_pegs_per_face,
                ),
                config.max_pegs_per_face,
            )
            assert placement.peg_count == expected

    def test_peg_count_is_clamped_to_the_configured_range(self, oversize_block, config):
        _, graph = _prepare(oversize_block, config)
        joints = generate_joints(graph, config)
        for placement in joints.placements:
            assert config.min_pegs_per_face <= placement.peg_count
            assert placement.peg_count <= config.max_pegs_per_face

    def test_socket_radius_is_peg_radius_plus_clearance(self, oversize_bar, config):
        _, graph = _prepare(oversize_bar, config)
        joints = generate_joints(graph, config)
        for placement in joints.placements:
            assert placement.socket_radius_mm == pytest.approx(
                placement.peg_radius_mm + config.joint_clearance_mm
            )

    def test_pegs_keep_clear_of_the_face_perimeter(self, oversize_bar, config):
        _, graph = _prepare(oversize_bar, config)
        joints = generate_joints(graph, config)
        by_id = {i.id: i for i in graph.interfaces}
        for placement in joints.placements:
            interface = by_id[placement.interface_id]
            inverse = np.linalg.inv(interface.to_3d)
            for center in placement.centers_mm:
                local = inverse @ np.array([*center, 1.0])
                distance = interface.polygon.exterior.distance(
                    __import__("shapely.geometry", fromlist=["Point"]).Point(
                        local[0], local[1]
                    )
                )
                assert distance >= config.peg_keepout_radius_mm - 1e-6

    def test_volume_change_matches_the_pegs_and_sockets_added(
        self, oversize_bar, config
    ):
        """A middle part is male on one face and female on another, so the net
        volume change is pegs added minus sockets removed."""
        result, graph = _prepare(oversize_bar, config)
        joints = generate_joints(graph, config)

        expected: dict[str, float] = {}
        for placement in joints.placements:
            peg_volume = (
                placement.peg_count
                * np.pi
                * config.peg_radius_mm**2
                * config.peg_depth_mm
            )
            socket_volume = (
                placement.peg_count
                * np.pi
                * config.socket_radius_mm**2
                * config.socket_depth_mm
            )
            expected[placement.male_id] = (
                expected.get(placement.male_id, 0.0) + peg_volume
            )
            expected[placement.female_id] = (
                expected.get(placement.female_id, 0.0) - socket_volume
            )

        assert expected
        for part_id, delta in expected.items():
            before = abs(graph.parts[part_id].mesh.volume)
            after = abs(joints.meshes[part_id].volume)
            # Cylinders are tessellated, so they hold slightly less than the
            # ideal solid of revolution.
            assert after - before == pytest.approx(delta, rel=0.02)

    @pytest.mark.parametrize("fixture_name", ["oversize_bar", "oversize_block"])
    def test_seated_parts_do_not_interfere(self, fixture_name, request, config):
        """Every peg fits inside its socket: the strongest single proof that the
        connectors actually mate rather than collide."""
        import itertools

        from cleaveform.geometry import booleans

        path = request.getfixturevalue(fixture_name)
        _, graph = _prepare(path, config)
        joints = generate_joints(graph, config)

        for a, b in itertools.combinations(sorted(joints.meshes), 2):
            overlap = booleans.intersection_volume(
                joints.meshes[a], joints.meshes[b], config
            )
            assert overlap <= config.volume_epsilon_mm3, (
                f"{a} and {b} overlap by {overlap} mm3 when seated"
            )

    def test_clearance_voids_account_for_the_missing_volume(self, oversize_bar, config):
        """Assembled volume falls short of the original by exactly the gaps."""
        mesh = load_model(oversize_bar, config).mesh
        _, graph = _prepare(oversize_bar, config)
        joints = generate_joints(graph, config)

        expected_gap = 0.0
        for placement in joints.placements:
            expected_gap += placement.peg_count * (
                np.pi * config.socket_radius_mm**2 * config.socket_depth_mm
                - np.pi * config.peg_radius_mm**2 * config.peg_depth_mm
            )
        assembled = sum(abs(m.volume) for m in joints.meshes.values())
        assert abs(mesh.volume) - assembled == pytest.approx(expected_gap, rel=0.02)

    def test_face_too_small_for_a_peg_is_a_hard_failure(self, narrow_plate, config):
        _, graph = _prepare(narrow_plate, config)
        with pytest.raises(JointError) as excinfo:
            generate_joints(graph, config)
        assert excinfo.value.detail["tier"] == HARD
        assert excinfo.value.node_path
        assert excinfo.value.candidate_index >= 0

    def test_walls_and_margins_respect_the_nozzle_minimum(self, config):
        assert config.min_wall_mm >= config.nozzle_width_mm * 2
        assert config.peg_edge_margin_mm >= config.min_wall_mm
        assert config.min_socket_backing_mm > config.socket_depth_mm


class TestSocketBacking:
    def test_shallow_female_part_is_rejected(self, config):
        """A socket that would punch through its part must be refused."""
        from cleaveform.decompose.graph import Interface
        from cleaveform.geometry import slicing
        from shapely.geometry import Polygon

        thickness = config.socket_depth_mm / 2.0
        male = trimesh.creation.box(extents=(20.0, 40.0, 40.0))
        male.apply_translation((-10.0, 0.0, 0.0))
        female = trimesh.creation.box(extents=(thickness, 40.0, 40.0))
        female.apply_translation((thickness / 2.0, 0.0, 0.0))

        normal = np.array([1.0, 0.0, 0.0])
        origin = np.zeros(3)
        _, to_3d = slicing.plane_transforms(origin, normal)
        interface = Interface(
            id="joint_001",
            male_id="part_001",
            female_id="part_002",
            axis=0,
            origin=origin,
            normal=normal,
            polygon=Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)]),
            to_3d=to_3d,
            area_mm2=1600.0,
            node_path="root",
            candidate_index=0,
        )
        from cleaveform.joints.peg_socket import _plan_interface

        with pytest.raises(JointError, match="material"):
            _plan_interface(interface, male, female, config, [])


class TestAssemblyPlanning:
    def test_single_axis_split_is_a_chain_of_pushes(self, oversize_bar, config):
        result, graph = _prepare(oversize_bar, config)
        joints = generate_joints(graph, config)
        plan = plan_assembly(result, graph, joints.meshes, config)
        assert plan.ok
        assert len(plan.steps) == result.part_count - 1
        for step in plan.steps:
            assert step.insertion_axis in ("+X", "-X")
            assert step.joint_ids

    def test_two_axis_grid_builds_subassemblies_first(self, oversize_block, config):
        """A 2x2 grid cannot be assembled one part at a time; it must nest."""
        result, graph = _prepare(oversize_block, config)
        joints = generate_joints(graph, config)
        plan = plan_assembly(result, graph, joints.meshes, config)
        assert plan.ok
        assert result.part_count == 4
        assert len(plan.steps) == 3
        # The last step joins two multi-part groups along the first cut axis.
        final = plan.steps[-1]
        assert len(final.moving_parts) == 2
        assert len(final.fixed_parts) == 2
        assert len(final.joint_ids) == 2

    def test_every_step_engages_a_single_axis(self, oversize_block, config):
        result, graph = _prepare(oversize_block, config)
        joints = generate_joints(graph, config)
        plan = plan_assembly(result, graph, joints.meshes, config)
        by_id = {i.id: i for i in graph.interfaces}
        for step in plan.steps:
            direction = np.asarray(step.insertion_direction)
            for joint_id in step.joint_ids:
                axis = np.asarray(by_id[joint_id].normal)
                assert abs(float(axis @ direction)) == pytest.approx(1.0)

    def test_ordering_is_deterministic(self, oversize_block, config):
        result, graph = _prepare(oversize_block, config)
        joints = generate_joints(graph, config)
        first = plan_assembly(result, graph, joints.meshes, config)
        second = plan_assembly(result, graph, joints.meshes, config)
        assert [s.to_dict() for s in first.steps] == [s.to_dict() for s in second.steps]


class TestOrientation:
    def test_recommended_orientation_minimises_support(self, config):
        # An L shape has an unambiguous best way up.
        base = trimesh.creation.box(extents=(60.0, 20.0, 20.0))
        base.apply_translation((0.0, 0.0, 0.0))
        upright = trimesh.creation.box(extents=(20.0, 20.0, 60.0))
        upright.apply_translation((20.0, 0.0, 40.0))
        shape = trimesh.util.concatenate([base, upright])
        shape.merge_vertices()

        _, _, oriented, support = analysis.best_orientation(shape, config)
        for _, transform in analysis.orientation_candidates():
            rotated = shape.copy()
            rotated.apply_transform(transform)
            assert support <= analysis.estimate_support_volume(rotated, config) + 1e-6

    def test_flat_box_needs_no_support(self, config):
        box = trimesh.creation.box(extents=(50.0, 50.0, 50.0))
        assert analysis.estimate_support_volume(box, config) == pytest.approx(0.0)

    def test_orientation_never_breaks_the_ceiling(self, config):
        long_bar = trimesh.creation.box(extents=(30.0, 290.0, 30.0))
        _, _, oriented, _ = analysis.best_orientation(long_bar, config)
        extents = oriented.bounds[1] - oriented.bounds[0]
        assert np.all(extents <= np.asarray(config.design_ceiling_mm) + 1e-9)


class TestSolids:
    def test_peg_sits_on_the_face_and_protrudes_along_the_normal(self, config):
        center = np.zeros(3)
        normal = np.array([0.0, 0.0, 1.0])
        peg = solids.peg(center, normal, config)
        assert peg.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
        assert peg.bounds[1][2] == pytest.approx(config.peg_depth_mm, abs=1e-6)

    def test_socket_is_deeper_than_the_peg(self, config):
        assert config.socket_depth_mm > config.peg_depth_mm
