"""Geometry kernel behaviour and export contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from cleaveform.estimate import print_time_min, summarize, weight_g
from cleaveform.export import ExportError, write_parts
from cleaveform.geometry import booleans, repair, slicing


class TestRepair:
    def test_open_mesh_is_closed(self, config):
        box = trimesh.creation.box(extents=(50.0, 50.0, 50.0))
        broken = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=box.faces[2:].copy(), process=False
        )
        assert not broken.is_watertight

        fixed, ok = repair.ensure_watertight(broken, config)
        assert ok
        assert fixed.volume == pytest.approx(box.volume)

    def test_repair_is_deterministic(self, config):
        box = trimesh.creation.box(extents=(50.0, 50.0, 50.0))
        broken = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=box.faces[2:].copy(), process=False
        )
        first, _ = repair.ensure_watertight(broken, config)
        second, _ = repair.ensure_watertight(broken, config)
        assert np.array_equal(first.vertices, second.vertices)
        assert np.array_equal(first.faces, second.faces)

    def test_unsalvageable_mesh_reports_failure(self, config):
        stray = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        _, ok = repair.ensure_watertight(stray, config)
        assert not ok

    def test_empty_mesh_is_detected(self, config):
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), int))
        assert repair.is_empty(empty, config)
        assert repair.as_watertight_or_none(empty, config) is None


class TestBooleans:
    def test_manifold_engine_is_available(self):
        assert booleans.BOOLEAN_ENGINE in trimesh.boolean.engines_available

    def test_union_adds_volume(self, config):
        a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b.apply_translation((10.0, 0.0, 0.0))
        assert booleans.union([a, b], config).volume == pytest.approx(2000.0)

    def test_difference_removes_volume(self, config):
        a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b = trimesh.creation.box(extents=(5.0, 5.0, 20.0))
        assert booleans.difference(a, [b], config).volume == pytest.approx(750.0)

    def test_disjoint_meshes_report_no_overlap(self, config):
        a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b.apply_translation((100.0, 0.0, 0.0))
        assert booleans.intersection_volume(a, b, config) == 0.0

    def test_overlapping_meshes_report_the_shared_volume(self, config):
        a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        b.apply_translation((5.0, 0.0, 0.0))
        assert booleans.intersection_volume(a, b, config) == pytest.approx(500.0)


class TestSlicing:
    def test_split_halves_a_box(self, config):
        box = trimesh.creation.box(extents=(100.0, 20.0, 20.0))
        negative, positive = slicing.split_by_plane(
            box, np.zeros(3), np.array([1.0, 0.0, 0.0]), config
        )
        assert negative.volume == pytest.approx(positive.volume)
        assert negative.is_watertight and positive.is_watertight
        assert negative.bounds[1][0] == pytest.approx(0.0)
        assert positive.bounds[0][0] == pytest.approx(0.0)

    def test_plane_frame_is_orthonormal(self):
        for normal in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            frame = slicing.plane_frame(normal)
            assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)

    def test_plane_frame_is_stable_for_the_same_normal(self):
        first = slicing.plane_frame([0.0, 0.0, 1.0])
        second = slicing.plane_frame([0.0, 0.0, 1.0])
        assert np.array_equal(first, second)

    def test_cross_section_area_matches_the_face(self, config):
        box = trimesh.creation.box(extents=(100.0, 20.0, 30.0))
        area = slicing.cross_section_area(
            box, np.zeros(3), np.array([1.0, 0.0, 0.0]), config
        )
        assert area == pytest.approx(20.0 * 30.0)

    def test_round_trip_between_plane_and_world(self):
        origin = np.array([1.0, 2.0, 3.0])
        normal = np.array([0.0, 0.0, 1.0])
        _, to_3d = slicing.plane_transforms(origin, normal)
        point = slicing.point_2d_to_3d((4.0, 5.0), to_3d)
        assert point[2] == pytest.approx(3.0)


class TestEstimation:
    def test_weight_exceeds_the_spec_formula_by_the_shell(self, config):
        """Spec section 12 writes weight as volume x density x infill.

        That omits the solid perimeters and top/bottom layers, which print at
        full density whatever the infill is set to, so it understates every
        part. Weight now uses the same extruded volume the calibrated time
        model uses.
        """
        from cleaveform.estimate import material_volume_mm3

        box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
        spec_formula = (
            (100.0**3 / 1000.0) * config.material_density_g_cm3 * config.infill_factor
        )
        actual = weight_g(box, config)
        assert actual > spec_formula
        assert actual == pytest.approx(
            material_volume_mm3(box, config)
            / 1000.0
            * config.material_density_g_cm3
        )

    def test_print_time_grows_with_volume(self, config):
        small = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        large = trimesh.creation.box(extents=(40.0, 40.0, 40.0))
        assert print_time_min(large, config) > print_time_min(small, config)

    def test_every_estimate_key_is_prefixed(self, config):
        box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        for key in summarize(box, config):
            assert key.startswith("est_")


class TestExportGuards:
    def test_non_watertight_part_is_refused(self, config, tmp_path):
        box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        broken = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=box.faces[2:].copy(), process=False
        )
        with pytest.raises(ExportError, match="watertight"):
            write_parts({"part_001": broken}, tmp_path, config)

    def test_parts_are_written_in_sorted_order(self, config, tmp_path):
        meshes = {
            f"part_{i:03d}": trimesh.creation.box(extents=(10.0, 10.0, 10.0))
            for i in (3, 1, 2)
        }
        written = write_parts(meshes, tmp_path, config)
        assert list(written) == ["part_001", "part_002", "part_003"]

    def test_manifest_is_valid_json(self, bar_run):
        text = bar_run.metadata_path.read_text(encoding="utf-8")
        assert json.loads(text)["tool"] == "cleaveform"
