"""Print-time and material estimation, calibrated against real slices.

Spec section 13 sets the bar: estimated print time within 25% of a real slice on
3 to 5 samples. ``tests/fixtures/estimation/measured.json`` holds the actual
Bambu Studio output for five shapes; these tests assert the model still clears
that bar, so a later change to the estimation constants cannot quietly
decalibrate it.

The five shapes are deliberately dissimilar. The time model trades extruded
volume against layer count, and a tall column and a flat slab load those two
terms in opposite proportions. Fitting against five similar parts would pin one
constant well and leave the other guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from cleaveform.config import Config
from cleaveform.estimate import (
    filament_length_m,
    material_volume_mm3,
    print_time_min,
    weight_g,
)

ESTIMATION_DIR = Path(__file__).parent / "fixtures" / "estimation"
SPEC_TOLERANCE_PERCENT = 25.0


def _measured_times() -> dict[str, float]:
    data = json.loads((ESTIMATION_DIR / "measured.json").read_text(encoding="utf-8"))
    return data["print_time_min"]


def _measured_weights() -> dict[str, float]:
    data = json.loads((ESTIMATION_DIR / "measured.json").read_text(encoding="utf-8"))
    return data["weight_g"]


def _sample(name: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(str(ESTIMATION_DIR / f"{name}.stl"))


def _error_percent(predicted: float, actual: float) -> float:
    return (predicted - actual) / actual * 100.0


class TestCalibrationAgainstRealSlices:
    """Spec section 13: print time within 25% of a real slice.

    Calibration is two stages, and both are checked here. Fitting the shell
    term against time alone hit the 25% bar but baked in a ~1.2x systematic
    overestimate of filament weight: time conflates true extruded volume with
    per-layer travel overhead, so a time-only fit inflates the volume model to
    absorb that overhead. Weight, converted directly from measured mass, is not
    confounded that way, so the shell term is fitted against weight first and
    speed/overhead are fitted against time second, using that corrected volume.
    """

    @pytest.mark.parametrize("name,actual", sorted(_measured_times().items()))
    def test_each_sample_time_is_within_the_spec_tolerance(self, name, actual, config):
        predicted = print_time_min(_sample(name), config)
        error = _error_percent(predicted, actual)
        assert abs(error) <= SPEC_TOLERANCE_PERCENT, (
            f"{name}: predicted {predicted:.1f} min against a real slice of "
            f"{actual:.1f} min, off by {error:+.1f}%, beyond the "
            f"{SPEC_TOLERANCE_PERCENT}% bar"
        )

    @pytest.mark.parametrize("name,actual", sorted(_measured_weights().items()))
    def test_each_sample_weight_is_close_to_measured(self, name, actual, config):
        """Weight is fitted directly, so it should track much tighter than time."""
        predicted = weight_g(_sample(name), config)
        error = _error_percent(predicted, actual)
        assert abs(error) <= 5.0, (
            f"{name}: predicted {predicted:.1f} g against a measured "
            f"{actual:.1f} g, off by {error:+.1f}%; weight is fitted directly "
            "from mass and should stay tight, not just inside the time bar"
        )

    def test_the_time_fit_is_good_across_all_samples(self, config):
        errors = [
            abs(_error_percent(print_time_min(_sample(name), config), actual))
            for name, actual in _measured_times().items()
        ]
        mean_error = sum(errors) / len(errors)
        assert mean_error <= 15.0, f"mean time error {mean_error:.1f}% has drifted"

    def test_the_weight_fit_is_good_across_all_samples(self, config):
        errors = [
            abs(_error_percent(weight_g(_sample(name), config), actual))
            for name, actual in _measured_weights().items()
        ]
        mean_error = sum(errors) / len(errors)
        assert mean_error <= 2.0, f"mean weight error {mean_error:.1f}% has drifted"

    def test_the_measured_set_covers_the_spec_sample_count(self):
        assert 3 <= len(_measured_times()) <= 5
        assert set(_measured_times()) == set(_measured_weights())

    def test_samples_span_a_wide_range_of_shapes(self):
        """Guards the fit: similar samples would leave a constant unconstrained."""
        ratios = []
        for name in _measured_times():
            mesh = _sample(name)
            ratios.append(float(mesh.area) / float(abs(mesh.volume)))
        assert max(ratios) / min(ratios) > 2.0, (
            "the samples have become too alike in surface-area-to-volume; the "
            "shell and infill terms can no longer be told apart by the fit"
        )


class TestShellTerm:
    """The solid skin that infill density does not cover."""

    def test_a_thin_plate_is_mostly_shell(self, config):
        """Skin dominates when there is barely an interior to fill."""
        plate = trimesh.creation.box(extents=(200.0, 200.0, 2.0))
        volume = abs(plate.volume)
        assert material_volume_mm3(plate, config) > volume * config.infill_factor * 3

    def test_material_never_exceeds_a_solid_part(self, config):
        """A thin part is shell all the way through, never more than solid."""
        for extents in ((100.0, 100.0, 0.5), (50.0, 50.0, 1.0), (20.0, 20.0, 20.0)):
            mesh = trimesh.creation.box(extents=extents)
            assert material_volume_mm3(mesh, config) <= abs(mesh.volume) + 1e-6

    def test_a_bulky_part_is_mostly_infill(self, config):
        """Skin is a rounding error when the interior dominates."""
        block = trimesh.creation.box(extents=(200.0, 200.0, 200.0))
        infill_only = abs(block.volume) * config.infill_factor
        assert material_volume_mm3(block, config) < infill_only * 1.5

    def test_removing_the_shell_breaks_the_weight_fit(self, config):
        """The term earns its place: without it, weight is a lot worse."""
        without = config.with_overrides(shell_thickness_mm=0.0)
        errors = [
            abs(_error_percent(weight_g(_sample(name), without), actual))
            for name, actual in _measured_weights().items()
        ]
        assert max(errors) > 10.0


class TestEstimateShape:
    def test_print_time_grows_with_volume(self, config):
        small = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        large = trimesh.creation.box(extents=(40.0, 40.0, 40.0))
        assert print_time_min(large, config) > print_time_min(small, config)

    def test_taller_parts_cost_more_layer_overhead(self, config):
        """Same plastic, more layers, more time."""
        flat = trimesh.creation.box(extents=(100.0, 100.0, 10.0))
        tall = trimesh.creation.box(extents=(10.0, 100.0, 100.0))
        assert abs(flat.volume) == pytest.approx(abs(tall.volume))
        assert print_time_min(tall, config) > print_time_min(flat, config)

    def test_weight_and_filament_track_the_same_material_volume(self, config):
        mesh = _sample("s3_medium_cube")
        volume = material_volume_mm3(mesh, config)
        assert weight_g(mesh, config) == pytest.approx(
            volume / 1000.0 * config.material_density_g_cm3
        )
        assert filament_length_m(mesh, config) > 0

    def test_zero_speed_does_not_divide_by_zero(self, config):
        mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        assert print_time_min(mesh, config.with_overrides(nominal_speed_mm_s=0.0)) == 0.0
