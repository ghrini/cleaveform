"""Unit policy and CLI scale overrides (spec module 1)."""

from __future__ import annotations

import numpy as np
import pytest

from cleaveform.io import InputError, load_model, parse_target_size
from cleaveform.io.loader import SUPPORTED_SUFFIXES


class TestUnitPolicy:
    def test_stl_coordinates_default_to_millimetres(self, small_cube, config):
        model = load_model(small_cube, config)
        assert model.scale_applied == 1.0
        assert np.allclose(model.final_extents_mm, (60.0, 60.0, 60.0))
        assert any("millimetres" in note for note in model.notes)

    def test_detected_bbox_is_always_reported(self, small_cube, config):
        model = load_model(small_cube, config)
        assert model.summary()["raw_bbox_mm"]
        assert model.summary()["final_bbox_mm"]

    def test_implausibly_tiny_model_is_rejected(self, implausibly_tiny, config):
        with pytest.raises(Exception, match="plausible minimum"):
            load_model(implausibly_tiny, config)

    def test_input_units_override_is_honoured(self, implausibly_tiny, config):
        model = load_model(implausibly_tiny, config, input_units="m")
        assert model.scale_applied == 1000.0
        assert np.allclose(model.final_extents_mm, (50.0, 50.0, 50.0))

    def test_scale_factor_override_is_honoured(self, small_cube, config):
        model = load_model(small_cube, config, scale_factor=2.0)
        assert np.allclose(model.final_extents_mm, (120.0, 120.0, 120.0))

    def test_target_size_override_is_honoured(self, small_cube, config):
        model = load_model(small_cube, config, target_size_mm=(30.0, 30.0, 30.0))
        assert np.allclose(model.final_extents_mm, (30.0, 30.0, 30.0))

    def test_overrides_are_mutually_exclusive(self, small_cube, config):
        with pytest.raises(InputError, match="at most one"):
            load_model(small_cube, config, input_units="cm", scale_factor=2.0)

    def test_unsupported_suffix_is_rejected(self, config, tmp_path):
        path = tmp_path / "model.obj"
        path.write_text("dummy", encoding="utf-8")
        with pytest.raises(InputError, match="accepts"):
            load_model(path, config)
        assert SUPPORTED_SUFFIXES == (".stl",)

    def test_missing_file_is_rejected(self, config, tmp_path):
        with pytest.raises(InputError, match="not found"):
            load_model(tmp_path / "absent.stl", config)


class TestTargetSizeParsing:
    def test_parses_three_values(self):
        assert parse_target_size("10,20,30") == (10.0, 20.0, 30.0)

    def test_none_passes_through(self):
        assert parse_target_size(None) is None

    @pytest.mark.parametrize("text", ["10,20", "a,b,c", "10,20,-5"])
    def test_rejects_malformed_input(self, text):
        with pytest.raises(InputError):
            parse_target_size(text)
