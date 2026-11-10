"""The full six-term normalized cost function (spec section 10).

Phase 1 scored cuts on four terms. Phase 2 adds visible seam area, joint count,
print time and a strength penalty. These tests pin down what each new term is
supposed to reward, using fixtures whose right answer is obvious by eye: a
narrow neck should be a bad place to cut, a seam in a slot should count for
less than one across an open face.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cleaveform.cost import CutCandidate, baseline_print_time_min, score_candidate
from cleaveform.geometry import analysis, slicing
from cleaveform.io import load_model

N_AXES = 3


def _candidate(mesh, config, fraction=0.5, axis=0) -> CutCandidate:
    """Cut ``mesh`` at ``fraction`` along ``axis`` and package it for scoring."""
    normal = np.zeros(N_AXES)
    normal[axis] = 1.0
    low = float(mesh.bounds[0][axis])
    span = float(mesh.bounds[1][axis] - mesh.bounds[0][axis])
    origin = mesh.centroid.copy()
    origin[axis] = low + span * fraction

    negative, positive = slicing.split_by_plane(mesh, origin, normal, config)
    polygons, to_3d = slicing.cross_section_polygons(mesh, origin, normal, config)
    return CutCandidate(
        parent=mesh,
        pieces=[negative, positive],
        axis=axis,
        polygons=polygons,
        to_3d=to_3d,
    )


def _neck_shape() -> trimesh.Trimesh:
    """Two blocks joined by a narrow neck: an hourglass on its side."""
    left = trimesh.creation.box(extents=(40.0, 60.0, 60.0))
    left.apply_translation((-30.0, 0.0, 0.0))
    right = trimesh.creation.box(extents=(40.0, 60.0, 60.0))
    right.apply_translation((30.0, 0.0, 0.0))
    neck = trimesh.creation.box(extents=(20.0, 10.0, 10.0))
    return trimesh.boolean.union([left, neck, right], engine="manifold")


def _forked_shape() -> trimesh.Trimesh:
    """An I-beam profile: cutting near one end crosses two separate flanges."""
    lower = trimesh.creation.box(extents=(80.0, 40.0, 20.0))
    lower.apply_translation((0.0, 0.0, -30.0))
    upper = trimesh.creation.box(extents=(80.0, 40.0, 20.0))
    upper.apply_translation((0.0, 0.0, 30.0))
    web = trimesh.creation.box(extents=(20.0, 40.0, 80.0))
    return trimesh.boolean.union([lower, web, upper], engine="manifold")


class TestNormalization:
    def test_every_term_stays_within_unit_range(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        cost = score_candidate(
            _candidate(mesh, config), baseline_print_time_min(mesh, config), config
        )
        terms = cost.to_dict()
        terms.pop("total")
        for name, value in terms.items():
            assert 0.0 <= value <= 1.0, f"{name} = {value} escaped [0, 1]"

    def test_total_is_the_weighted_sum_of_its_terms(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        cost = score_candidate(
            _candidate(mesh, config), baseline_print_time_min(mesh, config), config
        )
        expected = (
            config.w_part_count * cost.norm_part_count
            + config.w_visible_seam_area * cost.norm_visible_seam_area
            + config.w_support_volume * cost.norm_support_volume
            + config.w_joint_count * cost.norm_joint_count
            + config.w_print_time * cost.norm_print_time
            + config.w_strength_penalty * cost.norm_strength_penalty
        )
        assert cost.total == pytest.approx(expected)

    def test_scoring_is_deterministic(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        baseline = baseline_print_time_min(mesh, config)
        first = score_candidate(_candidate(mesh, config), baseline, config)
        second = score_candidate(_candidate(mesh, config), baseline, config)
        assert first.to_dict() == second.to_dict()


class TestPartCountDrivesTheSearch:
    """Phase 1 had a bbox-feasibility term that fought the part-count term.

    It rewarded cuts that left less oversize behind, which is not the same as
    minimising parts: a 620 mm bar under a 274 mm ceiling wants an off-centre
    cut giving 3 parts, but feasibility preferred the balanced cut giving 4.
    The term is gone; these tests pin down that its job is covered.
    """

    def test_a_lopsided_cut_that_yields_fewer_parts_wins(self, oversize_bar, config):
        from cleaveform.decompose import decompose

        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        assert result.part_count == 3, "620 mm bar should need only three parts"
        assert result.cut_log[0]["fraction"] != 0.5

    def test_pieces_still_fit_without_a_feasibility_term(self, oversize_bar, config):
        """Feasibility is enforced by the recursion, not bought with a weight."""
        from cleaveform.decompose import decompose

        mesh = load_model(oversize_bar, config).mesh
        ceiling = np.asarray(config.decomposition_ceiling_mm)
        for node in decompose(mesh, config).parts:
            extents = node.mesh.bounds[1] - node.mesh.bounds[0]
            assert np.all(extents <= ceiling + 1e-9)

    def test_fewer_parts_scores_lower_on_the_part_count_term(
        self, oversize_bar, config
    ):
        mesh = load_model(oversize_bar, config).mesh
        baseline = baseline_print_time_min(mesh, config)
        lopsided = score_candidate(_candidate(mesh, config, 0.40), baseline, config)
        balanced = score_candidate(_candidate(mesh, config, 0.50), baseline, config)
        assert lopsided.norm_part_count < balanced.norm_part_count


class TestStrengthPenalty:
    """Do not cut the thinnest part of the model."""

    def test_cutting_a_narrow_neck_is_penalised(self, config):
        shape = _neck_shape()
        baseline = baseline_print_time_min(shape, config)
        through_neck = score_candidate(_candidate(shape, config, 0.5), baseline, config)
        through_block = score_candidate(
            _candidate(shape, config, 0.15), baseline, config
        )
        assert through_neck.norm_strength_penalty > through_block.norm_strength_penalty

    def test_full_width_cut_of_a_box_is_not_penalised(self, config):
        box = trimesh.creation.box(extents=(100.0, 40.0, 40.0))
        cost = score_candidate(
            _candidate(box, config), baseline_print_time_min(box, config), config
        )
        assert cost.norm_strength_penalty == pytest.approx(0.0, abs=1e-6)

    def test_penalty_raises_the_total_cost_of_the_weak_cut(self, config):
        """The term has to actually change the ranking, not just be reported."""
        shape = _neck_shape()
        baseline = baseline_print_time_min(shape, config)
        neck = _candidate(shape, config, 0.5)
        block = _candidate(shape, config, 0.15)

        without = config.with_overrides(w_strength_penalty=0.0)
        assert score_candidate(neck, baseline, config).total - score_candidate(
            neck, baseline, without
        ).total > score_candidate(block, baseline, config).total - score_candidate(
            block, baseline, without
        ).total


class TestPrintTime:
    def test_splitting_adds_overhead_against_the_unsplit_baseline(self, config):
        """Two pieces mean more total layers than one, so overhead is positive."""
        box = trimesh.creation.box(extents=(100.0, 40.0, 40.0))
        cost = score_candidate(
            _candidate(box, config), baseline_print_time_min(box, config), config
        )
        assert cost.norm_print_time > 0.0

    def test_baseline_is_the_whole_object_in_one_piece(self, config):
        from cleaveform.estimate import print_time_min

        box = trimesh.creation.box(extents=(100.0, 40.0, 40.0))
        assert baseline_print_time_min(box, config) == pytest.approx(
            print_time_min(box, config)
        )

    def test_zero_baseline_does_not_divide_by_zero(self, config):
        box = trimesh.creation.box(extents=(100.0, 40.0, 40.0))
        cost = score_candidate(_candidate(box, config), 0.0, config)
        assert cost.norm_print_time == 0.0


class TestJointCount:
    def test_a_cut_crossing_two_flanges_carries_more_joints(self, config):
        shape = _forked_shape()
        baseline = baseline_print_time_min(shape, config)

        two_patches = _candidate(shape, config, 0.15, axis=0)
        one_patch = _candidate(shape, config, 0.5, axis=0)
        assert len(two_patches.polygons) == 2, "fixture should cut two flanges"
        assert len(one_patch.polygons) == 1, "mid cut should cross the web only"

        assert score_candidate(two_patches, baseline, config).norm_joint_count > (
            score_candidate(one_patch, baseline, config).norm_joint_count
        )


class TestSeamVisibility:
    """Seam area weighted by whether the seam will actually be seen."""

    def test_ray_directions_are_deterministic_and_unit_length(self, config):
        first = analysis.sphere_directions(config.seam_visibility_rays)
        second = analysis.sphere_directions(config.seam_visibility_rays)
        assert np.array_equal(first, second)
        assert np.allclose(np.linalg.norm(first, axis=1), 1.0)

    def test_a_buried_seam_is_less_exposed_than_an_open_one(self, config):
        """A point down a slot has most of its escape directions walled off."""
        block = trimesh.creation.box(extents=(60.0, 60.0, 60.0))
        slot = trimesh.creation.box(extents=(6.0, 80.0, 40.0))
        slot.apply_translation((0.0, 0.0, 20.0))
        slotted = trimesh.boolean.difference([block, slot], engine="manifold")

        buried = np.array([[0.0, 0.0, 6.0]])
        open_face = np.array([[0.0, 0.0, -30.0]])
        assert analysis.seam_exposure(slotted, buried, config) < (
            analysis.seam_exposure(slotted, open_face, config)
        )

    def test_exposure_scales_the_seam_term(self, config):
        box = trimesh.creation.box(extents=(100.0, 40.0, 40.0))
        candidate = _candidate(box, config)
        cost = score_candidate(candidate, baseline_print_time_min(box, config), config)
        raw = candidate.cut_face_area_mm2 / float(box.area)
        assert cost.norm_visible_seam_area == pytest.approx(raw * cost.seam_exposure)
        assert 0.0 <= cost.seam_exposure <= 1.0

    def test_no_seam_points_reads_as_fully_exposed(self, config):
        """Fail safe: an unmeasurable seam is assumed visible, never free."""
        box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        assert analysis.seam_exposure(box, np.empty((0, N_AXES)), config) == 1.0

    def test_seam_points_lie_on_the_cutting_plane(self, config):
        box = trimesh.creation.box(extents=(100.0, 40.0, 40.0))
        candidate = _candidate(box, config)
        points = slicing.seam_points(candidate.polygons, candidate.to_3d, config)
        assert len(points) >= config.seam_sample_points - 1
        assert np.allclose(points[:, candidate.axis], points[0, candidate.axis])


class TestWeightSanity:
    """Spec section 13: each weight moves its own metric, and nothing else."""

    def test_weights_come_from_config(self, config):
        assert not hasattr(config, "w_bbox_feasibility")
        assert config.with_overrides(w_part_count=10.0).w_part_count == 10.0

    def test_spec_default_weights_are_the_documented_ones(self, config):
        assert config.w_part_count == 1.0
        assert config.w_support_volume == 0.5
        assert config.w_visible_seam_area == 0.8
        assert config.w_joint_count == 0.3
        assert config.w_print_time == 0.2
        assert config.w_strength_penalty == 0.6

    @pytest.mark.parametrize(
        "weight_name,term_name",
        [
            ("w_part_count", "norm_part_count"),
            ("w_visible_seam_area", "norm_visible_seam_area"),
            ("w_support_volume", "norm_support_volume"),
            ("w_joint_count", "norm_joint_count"),
            ("w_print_time", "norm_print_time"),
            ("w_strength_penalty", "norm_strength_penalty"),
        ],
    )
    def test_each_weight_moves_only_its_own_term(
        self, weight_name, term_name, oversize_bar, config
    ):
        mesh = load_model(oversize_bar, config).mesh
        baseline = baseline_print_time_min(mesh, config)
        candidate = _candidate(mesh, config)

        low = score_candidate(
            candidate, baseline, config.with_overrides(**{weight_name: 0.0})
        )
        high = score_candidate(
            candidate, baseline, config.with_overrides(**{weight_name: 10.0})
        )

        # The measured term is a property of geometry, not of the weights.
        assert getattr(low, term_name) == pytest.approx(getattr(high, term_name))
        # The weight shifts the total only when its term is actually nonzero.
        if getattr(low, term_name) > 0.0:
            assert high.total > low.total
        else:
            assert high.total == pytest.approx(low.total)
