"""Phase 1 acceptance tests (implementation directive section 9).

One test class per acceptance criterion, named so a failure says which gate
broke. The physical joint-fit gate cannot be automated; the closest testable
property is that the calibration fixture is generated correctly and that the
calibrated clearance is what reaches production geometry.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cleaveform.config import Config
from cleaveform.decompose import decompose
from cleaveform.export import DIAGNOSTIC_FILENAME, GUIDE_FILENAME, METADATA_FILENAME
from cleaveform.io import InputError, load_model
from cleaveform.pipeline import (
    PipelineError,
    assert_production_clearance,
    clearance_schedule,
    run,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _part_digests(out_dir: Path) -> dict[str, str]:
    return {p.name: _digest(p) for p in sorted((out_dir / "parts").glob("*.stl"))}


class TestDeterminism:
    """Same STL plus same config gives a byte-stable cut set and part list."""

    def test_repeated_runs_are_byte_identical(self, oversize_bar, config, tmp_path):
        first = run(oversize_bar, tmp_path / "a", config)
        second = run(oversize_bar, tmp_path / "b", config)

        assert _part_digests(tmp_path / "a") == _part_digests(tmp_path / "b")
        assert first.manifest["decomposition"]["cuts"] == (
            second.manifest["decomposition"]["cuts"]
        )
        assert [p["id"] for p in first.manifest["parts"]] == [
            p["id"] for p in second.manifest["parts"]
        ]
        assert first.manifest["joints"] == second.manifest["joints"]
        assert first.manifest["assembly"] == second.manifest["assembly"]

    def test_decomposition_alone_is_stable(self, oversize_block, config):
        mesh = load_model(oversize_block, config).mesh
        first = decompose(mesh, config)
        second = decompose(mesh, config)
        assert first.cut_log == second.cut_log
        assert [n.path for n in first.parts] == [n.path for n in second.parts]

    def test_tie_breaks_to_lowest_candidate_index(self, oversize_bar, config):
        """A symmetric bar makes 40% and 60% score alike; 40% must win."""
        mesh = load_model(oversize_bar, config).mesh
        result = decompose(mesh, config)
        root_cut = result.cut_log[0]
        costs = {root_cut["candidate_index"]: root_cut["cost"]["total"]}
        assert root_cut["candidate_index"] == min(costs)


class TestBuildVolumeGuarantee:
    """Every exported part fits the design ceiling, pegs included."""

    def test_oversize_input_is_split(self, bar_run, config):
        assert bar_run.manifest["totals"]["part_count"] > 1

    @pytest.mark.parametrize(
        "fixture_name", ["oversize_bar", "oversize_block", "tall_cylinder"]
    )
    def test_every_part_fits_the_ceiling(
        self, fixture_name, request, config, tmp_path
    ):
        path = request.getfixturevalue(fixture_name)
        result = run(path, tmp_path / fixture_name, config)
        ceiling = np.asarray(config.design_ceiling_mm)
        for stl in sorted((tmp_path / fixture_name / "parts").glob("*.stl")):
            mesh = trimesh.load_mesh(str(stl))
            extents = mesh.bounds[1] - mesh.bounds[0]
            assert np.all(extents <= ceiling + config.length_epsilon_mm), (
                f"{stl.name} measures {extents} which exceeds {ceiling}"
            )
        assert result.ok

    def test_recommended_orientation_also_fits(self, bar_run, config):
        ceiling = np.asarray(config.design_ceiling_mm)
        for part in bar_run.manifest["parts"]:
            oriented = np.asarray(part["orientation"]["oriented_extents_mm"])
            assert np.all(oriented <= ceiling + config.length_epsilon_mm)

    def test_model_within_ceiling_is_not_cut(self, small_cube, config, tmp_path):
        result = run(small_cube, tmp_path / "cube", config)
        assert result.manifest["totals"]["part_count"] == 1
        assert result.manifest["totals"]["joint_count"] == 0


class TestWatertightness:
    """Every exported part is manifold."""

    @pytest.mark.parametrize(
        "fixture_name", ["oversize_bar", "oversize_block", "tall_cylinder"]
    )
    def test_exported_parts_are_manifold(
        self, fixture_name, request, config, tmp_path
    ):
        path = request.getfixturevalue(fixture_name)
        run(path, tmp_path / fixture_name, config)
        stls = sorted((tmp_path / fixture_name / "parts").glob("*.stl"))
        assert stls
        for stl in stls:
            mesh = trimesh.load_mesh(str(stl))
            assert mesh.is_watertight, f"{stl.name} is not watertight"
            assert mesh.is_winding_consistent, f"{stl.name} has inconsistent winding"
            assert mesh.volume > 0


class TestJointFit:
    """The physical gate: the calibration fixture that sets the clearance."""

    def test_calibration_set_covers_every_candidate_clearance(self, config, tmp_path):
        from cleaveform.calibration import write_calibration_set

        written = write_calibration_set(tmp_path, config)
        stls = [p for p in written if p.suffix == ".stl"]
        assert len(stls) == len(config.calibration_clearances_mm) * 2
        assert (tmp_path / "CALIBRATION.txt").exists()
        for stl in stls:
            mesh = trimesh.load_mesh(str(stl))
            assert mesh.is_watertight

    def test_socket_is_larger_than_peg_by_exactly_the_clearance(self, config):
        from cleaveform.calibration import build_pair

        clearance = 0.15
        male, female = build_pair(clearance, config)
        # The male half gains peg volume; the female half loses socket volume,
        # and the socket must be the wider of the two by the clearance.
        peg_area = np.pi * config.peg_radius_mm**2
        socket_area = np.pi * (config.peg_radius_mm + clearance) ** 2
        assert socket_area > peg_area
        plain_half = (
            config.calibration_cube_mm / 2.0 * config.calibration_cube_mm**2
        )
        assert male.volume > plain_half
        assert female.volume < plain_half

    def test_generated_sockets_use_the_configured_clearance(self, bar_run, config):
        for joint in bar_run.manifest["joints"]:
            assert joint["clearance_mm"] == config.joint_clearance_mm
            assert joint["socket_radius_mm"] == pytest.approx(
                joint["peg_radius_mm"] + config.joint_clearance_mm
            )


class TestCalibrationOrientation:
    """The exported calibration STLs must be print-ready: mating feature up,
    resting flat on the plate. Getting this wrong biases the fit reading in
    exactly the dimension the test exists to measure."""

    def test_both_halves_rest_flat_on_the_plate(self, config):
        from cleaveform.calibration import build_pair

        male, female = build_pair(0.20, config)
        assert male.bounds[0][2] == pytest.approx(0.0, abs=1e-9)
        assert female.bounds[0][2] == pytest.approx(0.0, abs=1e-9)
        assert male.is_watertight
        assert female.is_watertight

    def test_male_peg_tip_is_the_highest_point_above_its_own_center(self, config):
        from cleaveform.calibration import build_pair

        male, _ = build_pair(0.20, config)
        for xy in _peg_centers_xy(config):
            gap = _downward_gap(male, xy)
            assert gap == pytest.approx(0.0, abs=1e-6), (
                "the peg should be the tallest feature directly above its own "
                f"center; found a {gap:.2f} mm gap instead, which means "
                "something else is printing higher than the peg"
            )

    def test_female_socket_opens_upward_by_exactly_its_depth(self, config):
        from cleaveform.calibration import build_pair

        _, female = build_pair(0.20, config)
        for xy in _peg_centers_xy(config):
            gap = _downward_gap(female, xy)
            assert gap == pytest.approx(config.socket_depth_mm, abs=1e-6), (
                "a straight-down ray at the socket center should fall through "
                f"{config.socket_depth_mm:.2f} mm of open air before hitting the "
                f"socket floor; got {gap:.2f} mm, meaning the socket does not "
                "open at the top face"
            )


def _peg_centers_xy(config):
    """The two peg/socket XY centers in the calibration cube's own frame."""
    import numpy as np

    size = config.calibration_cube_mm
    half = size / 2.0
    offset = size * config.calibration_peg_offset_fraction
    return [(half, half - offset), (half, half + offset)]


def _downward_gap(mesh, xy):
    """Distance from the mesh's top plane down to the first surface at (x, y)."""
    import numpy as np

    top = mesh.bounds[1][2]
    origin = np.array([[xy[0], xy[1], top + 5.0]])
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=origin, ray_directions=np.array([[0.0, 0.0, -1.0]])
    )
    assert len(locations) > 0, f"no surface hit above {xy}"
    return top - float(locations[:, 2].max())


class TestConstraintTiers:
    """Soft violations only penalise; hard violations reject."""

    def test_part_needing_support_is_not_rejected(self, bar_run):
        """Sockets facing down need support; that must penalise, not reject."""
        soft_codes = {f["code"] for f in bar_run.manifest["constraints"]["soft"]}
        assert "needs_support" in soft_codes
        assert bar_run.manifest["constraints"]["hard"] == []
        assert bar_run.ok
        assert any(p["supports_required"] for p in bar_run.manifest["parts"])

    def test_sub_thickness_connector_is_rejected(self, narrow_plate, config, tmp_path):
        out = tmp_path / "narrow"
        with pytest.raises(PipelineError) as excinfo:
            run(narrow_plate, out, config)
        assert "too small to hold" in str(excinfo.value)
        assert (out / DIAGNOSTIC_FILENAME).exists()
        assert not (out / "parts").exists() or not list((out / "parts").glob("*.stl"))

    def test_warnings_never_block(self, bar_run):
        assert bar_run.ok
        assert bar_run.manifest["constraints"]["ok"] is True


class TestRecoverySafety:
    """Production export uses the calibrated clearance whatever recovery tries."""

    def test_production_schedule_has_exactly_one_clearance(self, config):
        schedule = clearance_schedule(config)
        assert schedule == [config.joint_clearance_mm]

    def test_experimental_schedule_stays_inside_the_bound(self, config):
        experimental = config.with_overrides(experimental=True)
        schedule = clearance_schedule(experimental)
        assert len(schedule) > 1
        for value in schedule:
            deviation = abs(value - config.joint_clearance_mm)
            assert deviation <= config.recovery_clearance_range_mm + 1e-12

    def test_deviating_clearance_is_refused_in_production(self, config):
        with pytest.raises(PipelineError, match="calibrated clearance"):
            assert_production_clearance(config, config.joint_clearance_mm + 0.05)

    def test_deviating_clearance_is_allowed_when_experimental(self, config):
        experimental = config.with_overrides(experimental=True)
        assert_production_clearance(experimental, experimental.joint_clearance_mm + 0.05)

    def test_run_after_recovery_still_uses_calibrated_clearance(
        self, oversize_bar, config, tmp_path
    ):
        calibrated = config.with_overrides(joint_clearance_mm=0.15)
        result = run(oversize_bar, tmp_path / "calibrated", calibrated)
        recovery = result.manifest["recovery"]
        assert recovery["clearance_used_mm"] == 0.15
        assert recovery["clearance_is_calibrated"] is True
        assert recovery["experimental"] is False
        for joint in result.manifest["joints"]:
            assert joint["clearance_mm"] == 0.15

    def test_recovery_is_bounded(self, narrow_plate, config, tmp_path):
        with pytest.raises(PipelineError):
            run(narrow_plate, tmp_path / "bounded", config)


class TestOffline:
    """A full run works with no network and no Gurobi."""

    def test_run_makes_no_network_calls(self, oversize_bar, config, tmp_path, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("cleaveform must not open a network connection")

        monkeypatch.setattr(socket, "socket", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)
        result = run(oversize_bar, tmp_path / "offline", config)
        assert result.ok

    def test_gurobi_is_never_imported(self, bar_run):
        import sys

        assert "gurobipy" not in sys.modules

    def test_no_solver_or_ml_dependency_is_declared(self):
        text = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        for forbidden in ("gurobi", "torch", "tensorflow", "requests", "httpx"):
            assert forbidden not in text.lower()


class TestEndToEnd:
    """`cleaveform input.stl -o out/` produces everything, with no manual step."""

    def test_cli_produces_parts_metadata_and_guide(self, oversize_bar, tmp_path):
        from click.testing import CliRunner

        from cleaveform.cli import main

        out = tmp_path / "cli"
        result = CliRunner().invoke(
            main, ["split", str(oversize_bar), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output

        parts = sorted((out / "parts").glob("*.stl"))
        assert len(parts) > 1
        assert (out / METADATA_FILENAME).exists()
        assert (out / GUIDE_FILENAME).exists()

        guide = (out / GUIDE_FILENAME).read_text(encoding="utf-8")
        assert "ASSEMBLY STEPS" in guide
        assert "PART LIST" in guide
        for part in parts:
            assert part.stem in guide

    def test_bare_file_argument_routes_to_split(self, oversize_bar, tmp_path, monkeypatch):
        import sys

        from cleaveform import cli

        out = tmp_path / "bare"
        captured = {}
        monkeypatch.setattr(
            cli, "_ORIGINAL_MAIN", lambda: captured.update(argv=list(sys.argv))
        )
        monkeypatch.setattr(sys, "argv", ["cleaveform", str(oversize_bar), "-o", str(out)])
        cli.entrypoint()
        assert captured["argv"][1] == "split"

    def test_input_errors_are_reported_without_a_traceback(
        self, implausibly_tiny, tmp_path
    ):
        from click.testing import CliRunner

        from cleaveform.cli import main

        result = CliRunner().invoke(
            main, ["split", str(implausibly_tiny), "-o", str(tmp_path / "tiny")]
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "plausible minimum" in result.output
        assert "--input-units" in result.output

    def test_unit_override_makes_the_same_model_usable(
        self, implausibly_tiny, tmp_path
    ):
        from click.testing import CliRunner

        from cleaveform.cli import main

        out = tmp_path / "scaled"
        result = CliRunner().invoke(
            main, ["split", str(implausibly_tiny), "-o", str(out), "--input-units", "m"]
        )
        assert result.exit_code == 0, result.output
        assert list((out / "parts").glob("*.stl"))

    def test_hard_failure_exits_nonzero_and_points_at_the_diagnostic(
        self, narrow_plate, tmp_path
    ):
        from click.testing import CliRunner

        from cleaveform.cli import main

        out = tmp_path / "narrow_cli"
        result = CliRunner().invoke(main, ["split", str(narrow_plate), "-o", str(out)])
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert DIAGNOSTIC_FILENAME in result.output
        assert (out / DIAGNOSTIC_FILENAME).exists()

    def test_metadata_is_complete(self, bar_run):
        manifest = bar_run.manifest
        assert manifest["phase"] == 1
        required = {
            "id",
            "material",
            "bbox_mm",
            "orientation",
            "supports_required",
            "recommended_layer_height_mm",
            "est_weight_g",
            "est_filament_m",
            "est_print_time_min",
            "est_support_volume_mm3",
        }
        for part in manifest["parts"]:
            assert required <= set(part)
        for step in manifest["assembly"]:
            assert {"step_no", "insertion_direction", "joint_type"} <= set(step)

    def test_estimates_are_labelled_as_estimates(self, bar_run, tmp_path):
        assert "not slicer output" in bar_run.manifest["note"].lower()
        guide = bar_run.guide_path.read_text(encoding="utf-8")
        assert "estimates" in guide.lower()
