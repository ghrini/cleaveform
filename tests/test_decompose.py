"""The deterministic decomposition rules (directive section 5)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cleaveform.config import Config
from cleaveform.cost import CutCandidate, baseline_print_time_min, score_candidate
from cleaveform.decompose import DecompositionError, build_cut_graph, decompose
from cleaveform.geometry import analysis, bbox, slicing
from cleaveform.io import load_model


class TestPartitionRules:
    def test_part_within_ceiling_is_returned_unchanged(self, small_cube, config):
        mesh = load_model(small_cube, config).mesh
        result = decompose(mesh, config)
        assert result.part_count == 1
        assert result.cut_log == []

    def test_cut_axis_is_the_worst_overshoot(self, oversize_bar, config):
        """The 620 mm X axis overshoots most, so the first cut is on X."""
        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        assert result.cut_log[0]["axis"] == "x"

    def test_candidate_fractions_come_from_config(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        for cut in result.cut_log:
            assert cut["fraction"] in config.cut_fractions

    def test_every_leaf_fits_the_decomposition_ceiling(self, oversize_block, config):
        mesh = load_model(oversize_block, config).mesh
        result = decompose(mesh, config)
        ceiling = np.asarray(config.decomposition_ceiling_mm)
        for node in result.parts:
            assert np.all(bbox.extents(node.mesh) <= ceiling + 1e-9)

    def test_decomposition_ceiling_reserves_room_for_pegs(self, config):
        for design, cutting in zip(
            config.design_ceiling_mm, config.decomposition_ceiling_mm
        ):
            assert design - cutting == pytest.approx(config.peg_depth_mm)

    def test_part_limit_produces_a_diagnostic(self, oversize_block, config):
        mesh = load_model(oversize_block, config).mesh
        tight = config.with_overrides(max_parts=2)
        with pytest.raises(DecompositionError) as excinfo:
            decompose(mesh, tight)
        assert excinfo.value.detail["limit"] == "max_parts"

    def test_depth_limit_produces_a_diagnostic(self, oversize_block, config):
        mesh = load_model(oversize_block, config).mesh
        shallow = config.with_overrides(max_cut_depth=1)
        with pytest.raises(DecompositionError) as excinfo:
            decompose(mesh, shallow)
        assert excinfo.value.detail["limit"] == "max_cut_depth"

    def test_pieces_are_watertight(self, tall_cylinder, config):
        mesh = load_model(tall_cylinder, config).mesh
        result = decompose(mesh, config)
        for node in result.parts:
            assert node.mesh.is_watertight

    def test_volume_is_conserved_across_the_cuts(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        total = sum(abs(node.mesh.volume) for node in result.parts)
        assert total == pytest.approx(abs(mesh.volume), rel=1e-6)


class TestRecoveryBlacklist:
    """The mechanism bounded recovery uses to explore a different cut."""

    def test_blacklisting_a_candidate_changes_the_chosen_cut(
        self, oversize_bar, config
    ):
        mesh = load_model(oversize_bar, config).mesh
        baseline = decompose(mesh, config)
        chosen = baseline.cut_log[0]["candidate_index"]

        alternative = decompose(mesh, config, blacklist={("root", chosen)})
        assert alternative.cut_log[0]["candidate_index"] != chosen

    def test_blacklisting_every_candidate_raises(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        blacklist = {("root", i) for i in range(len(config.cut_fractions))}
        with pytest.raises(DecompositionError, match="no valid cut"):
            decompose(mesh, config, blacklist=blacklist)


class TestCutGraph:
    def test_neighbouring_leaves_share_an_interface(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        graph = build_cut_graph(result, config)
        assert len(graph.interfaces) == result.part_count - 1

    def test_grid_decomposition_finds_every_interface(self, oversize_block, config):
        """A 2x2 chunk grid has four internal faces, not three."""
        mesh = load_model(oversize_block, config).mesh
        result = decompose(mesh, config)
        graph = build_cut_graph(result, config)
        assert result.part_count == 4
        assert len(graph.interfaces) == 4

    def test_male_side_is_the_negative_side_of_the_normal(self, oversize_bar, config):
        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        graph = build_cut_graph(result, config)
        for interface in graph.interfaces:
            male_center = graph.parts[interface.male_id].mesh.centroid
            female_center = graph.parts[interface.female_id].mesh.centroid
            along = float((female_center - male_center) @ interface.normal)
            assert along > 0

