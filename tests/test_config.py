"""Config loading, validation and the no-magic-numbers rule."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cleaveform.config import Config

PACKAGE_ROOT = Path(__file__).parents[1] / "cleaveform"

# Numbers that carry no tuning meaning: axis and matrix indices, rounding
# precision, and the identity/zero literals arithmetic needs.
STRUCTURAL_FLOATS = {0.0, 1.0, 2.0, 0.5}
MAX_STRUCTURAL_INT = 10


class TestDefaults:
    def test_starter_values_match_the_directive(self):
        config = Config()
        assert config.design_ceiling_mm == (280.0, 300.0, 300.0)
        assert config.nozzle_width_mm == 0.4
        assert config.min_wall_mm == 0.8
        assert config.min_part_mm == 5.0
        assert config.max_parts == 50
        assert config.max_cut_depth == 8
        assert config.joint_clearance_mm == 0.20
        assert config.recovery_clearance_range_mm == 0.05
        assert config.peg_radius_mm == 3.0
        assert config.peg_depth_mm == 6.0
        assert config.min_pegs_per_face == 2
        assert config.max_pegs_per_face == 6
        assert config.peg_area_per_peg_mm2 == 1500.0
        assert config.peg_edge_margin_mm == 5.0
        assert config.material_density_g_cm3 == 1.24
        assert config.layer_height_mm == 0.2
        assert config.cut_fractions == (0.40, 0.50, 0.60)

    def test_estimation_constants_have_moved_off_the_starter_values(self):
        """The directive's estimation numbers were explicitly coarse guesses.

        They have since been fitted, in two stages, against real Bambu Studio
        measurements, so they deliberately no longer match section 8.
        ``test_estimation.py`` checks the fit itself; this only records that
        the change was intentional.
        """
        config = Config()
        assert config.infill_factor == 0.15  # directive said 0.20
        assert config.nominal_speed_mm_s != 150.0  # directive's starting guess
        assert config.layer_overhead_s != 2.0  # was a guessed 2.0
        assert config.shell_thickness_mm > 0  # not in the directive at all

    def test_brick_mode_is_available_alongside_chunk(self):
        """Brick mode was added in Phase 5; chunk stays the default."""
        assert Config().decomposition_mode == "chunk"
        assert Config().with_overrides(decomposition_mode="brick").decomposition_mode == (
            "brick"
        )

    def test_remaining_mode_flags_are_locked(self):
        config = Config()
        assert config.decomposition_mode == "chunk"
        assert config.joint_type == "peg_socket"
        assert config.generation_backend == "off"
        assert config.export_format == "stl"


class TestValidation:
    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"decomposition_mode": "spaghetti"}, "must be one of"),
            ({"joint_type": "dovetail"}, "peg_socket"),
            ({"generation_backend": "mesh"}, "off"),
            ({"min_wall_mm": 0.1}, "2 x nozzle_width_mm"),
            ({"peg_edge_margin_mm": 0.1}, "at least min_wall_mm"),
            ({"cut_fractions": (0.0, 1.0)}, "strictly between"),
            ({"max_parts": 0}, "non-negative"),
            ({"joint_clearance_mm": -0.1}, "non-negative"),
            ({"max_pegs_per_face": 1}, ">= min_pegs_per_face"),
        ],
    )
    def test_invalid_configuration_is_refused(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            Config().with_overrides(**overrides)

    def test_calibration_cube_too_small_to_back_its_socket_is_refused(self):
        with pytest.raises(ValueError, match="too thin to back"):
            Config().with_overrides(calibration_cube_mm=12.0)

    def test_calibration_cube_with_flexible_walls_is_refused(self):
        """A thin wall bulges under a tight peg and fakes a good fit."""
        with pytest.raises(ValueError, match="flexes"):
            Config().with_overrides(calibration_cube_mm=15.0)

    def test_default_calibration_cube_is_comfortably_valid(self):
        config = Config()
        half = config.calibration_cube_mm / 2.0
        offset = config.calibration_cube_mm * config.calibration_peg_offset_fraction
        assert half >= config.min_socket_backing_mm
        assert offset - config.socket_radius_mm >= config.min_wall_mm * 2

    def test_unknown_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown config key"):
            Config.load(overrides={"not_a_real_setting": 1})

    def test_json_file_overrides_are_applied(self, tmp_path):
        path = tmp_path / "clearance.json"
        path.write_text(json.dumps({"joint_clearance_mm": 0.15}), encoding="utf-8")
        config = Config.load(path)
        assert config.joint_clearance_mm == 0.15
        assert config.socket_radius_mm == pytest.approx(config.peg_radius_mm + 0.15)

    def test_cli_overrides_beat_the_file(self, tmp_path):
        path = tmp_path / "clearance.json"
        path.write_text(json.dumps({"joint_clearance_mm": 0.15}), encoding="utf-8")
        config = Config.load(path, {"joint_clearance_mm": 0.25})
        assert config.joint_clearance_mm == 0.25

    def test_round_trips_through_json(self):
        config = Config()
        restored = Config.load(overrides=config.to_dict())
        assert restored == config


class TestNoMagicNumbers:
    """Directive section 8: no numeric value appears inline in module code.

    Values live either in ``config.py`` (tunable at runtime) or in a named
    module-level constant (fixed physical or presentation facts, such as the
    millimetres in a metre). A bare number in the middle of an expression is
    what this rejects.
    """

    def test_module_code_holds_no_tuning_constants(self):
        offenders: list[str] = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if path.name == "config.py":
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            named = _named_constant_lines(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)
                ):
                    continue
                if node.lineno in named:
                    continue
                if _is_structural(node.value):
                    continue
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} -> {node.value}"
                )
        assert not offenders, "tunable literals outside config.py:\n" + "\n".join(
            offenders
        )

    def test_the_rule_actually_catches_a_violation(self, tmp_path):
        """Guard against the check silently passing everything."""
        offender = ast.parse("speed = velocity * 137.5")
        assert not _named_constant_lines(offender)
        assert not _is_structural(137.5)


def _named_constant_lines(tree: ast.Module) -> set[int]:
    """Line numbers covered by module-level ``UPPER_CASE = ...`` assignments."""
    lines: set[int] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not all(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            continue
        end = node.end_lineno or node.lineno
        lines.update(range(node.lineno, end + 1))
    return lines


def _is_structural(value: float | int) -> bool:
    if isinstance(value, int):
        return 0 <= value <= MAX_STRUCTURAL_INT
    if value in STRUCTURAL_FLOATS:
        return True
    # Float-comparison epsilons written inline in guards.
    return 0 < abs(value) <= 1e-6
