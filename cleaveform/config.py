"""Central configuration for Cleaveform.

Every tunable numeric value in the project lives here. Module code must never
inline a magic number: it reads the value from a :class:`Config` instance.

Starter values follow the Phase 1 implementation directive section 8. They are
FDM/Bambu-reasonable starting points, not final recommendations. In particular
``joint_clearance_mm`` is expected to be replaced by the result of the physical
calibration print (``cleaveform calibrate``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

# Axis bookkeeping. These are structural indices, not tunable quantities.
AXIS_NAMES = ("x", "y", "z")
N_AXES = 3

DECOMPOSITION_MODES = frozenset({"chunk", "brick"})

BRICK_FOOTPRINT_SEPARATOR = "x"


def parse_brick_footprint(name: str) -> tuple[int, int]:
    """Parse a ``"4x2"`` library entry into ``(width_studs, depth_studs)``.

    Footprints are strings rather than nested tuples so a config file round-trips
    through JSON unchanged: JSON has no tuples, so a nested list would come back
    as a list and stop comparing equal to the default.
    """
    try:
        width, depth = name.lower().split(BRICK_FOOTPRINT_SEPARATOR)
        return int(width), int(depth)
    except ValueError as exc:
        raise ValueError(
            f"brick_library entry {name!r} must look like '4x2' (width x depth in studs)"
        ) from exc


@dataclass(frozen=True)
class Config:
    """Immutable run configuration.

    Construct with :meth:`load` so that file and CLI overrides are applied and
    validated consistently.
    """

    # ------------------------------------------------------------------
    # Locked Phase 1 mode flags (spec section 18)
    # ------------------------------------------------------------------
    decomposition_mode: str = "chunk"
    joint_type: str = "peg_socket"
    generation_backend: str = "off"
    export_format: str = "stl"
    cost_preset: str = "balanced"

    # ------------------------------------------------------------------
    # Target hardware (spec section 4)
    # ------------------------------------------------------------------
    design_ceiling_mm: tuple[float, float, float] = (280.0, 300.0, 300.0)
    build_volume_single_nozzle_mm: tuple[float, float, float] = (325.0, 320.0, 325.0)
    nozzle_width_mm: float = 0.4
    min_wall_mm: float = 0.8  # 2 x nozzle width
    min_feature_mm: float = 0.8  # 2 x nozzle width

    # ------------------------------------------------------------------
    # Decomposition bounds (directive section 8)
    # ------------------------------------------------------------------
    min_part_mm: float = 5.0
    max_parts: int = 50
    max_cut_depth: int = 8
    cut_fractions: tuple[float, ...] = (0.40, 0.50, 0.60)
    repair_max_attempts: int = 1

    # ------------------------------------------------------------------
    # Joints (directive sections 6 and 8)
    # ------------------------------------------------------------------
    joint_clearance_mm: float = 0.20
    recovery_clearance_range_mm: float = 0.05
    peg_radius_mm: float = 3.0
    peg_depth_mm: float = 6.0
    min_pegs_per_face: int = 2
    max_pegs_per_face: int = 6
    peg_area_per_peg_mm2: float = 1500.0
    peg_edge_margin_mm: float = 5.0
    socket_depth_margin_mm: float = 0.5
    peg_grid_pitch_mm: float = 1.0
    peg_cylinder_segments: int = 48
    min_joint_face_area_mm2: float = 100.0
    cut_face_probe_eps_mm: float = 0.1

    # ------------------------------------------------------------------
    # Estimation (spec section 12, directive section 8)
    # ------------------------------------------------------------------
    material_name: str = "PLA"
    material_density_g_cm3: float = 1.24
    # Must match what you actually slice at; it is a slicer setting, not a
    # printer property.
    infill_factor: float = 0.15
    layer_height_mm: float = 0.2
    filament_diameter_mm: float = 1.75
    # Fitted in two stages against five real Bambu Studio slices on an H2D (see
    # tests/fixtures/estimation/). Fitting shell_thickness_mm against print TIME
    # alone (worst 11.4%) baked in a systematic ~1.15x overestimate of filament
    # weight: time conflates true extruded volume with per-layer travel
    # overhead, so a time-only fit inflates the volume model to compensate.
    # Weight, converted directly from mass, is not confounded that way. Fitting
    # shell_thickness_mm against weight first (worst error on weight: 1.7%) and
    # then fitting speed and overhead against time using that corrected volume
    # (worst error on time: 15.7%, mean 6.0%) makes both numbers trustworthy
    # instead of trading one off against the other.
    nominal_speed_mm_s: float = 218.0
    layer_overhead_s: float = 1.54
    # Solid skin over the surface: perimeters plus top and bottom layers, which
    # print at full density regardless of the infill setting. Without this the
    # model underestimates anything with a lot of surface per unit volume.
    shell_thickness_mm: float = 0.729
    support_infill_factor: float = 0.15
    overhang_angle_deg: float = 45.0

    # ------------------------------------------------------------------
    # Cost weights, Phase 1 reduced cost (directive section 5 step 6)
    # ------------------------------------------------------------------
    # Spec section 10 weights: alpha part count, beta support, gamma seam,
    # delta joints, epsilon print time, zeta strength.
    w_part_count: float = 1.0
    w_support_volume: float = 0.5
    w_visible_seam_area: float = 0.8
    w_joint_count: float = 0.3
    w_print_time: float = 0.2
    w_strength_penalty: float = 0.6
    # Seam visibility sampling (see geometry.analysis.seam_exposure).
    seam_sample_points: int = 16
    seam_visibility_rays: int = 12
    seam_ray_offset_mm: float = 0.05

    # ------------------------------------------------------------------
    # Manufacturability soft thresholds (spec section 11)
    # ------------------------------------------------------------------
    support_volume_soft_fraction: float = 0.25
    print_time_soft_minutes: float = 1440.0
    seam_area_soft_fraction: float = 0.35
    warp_aspect_ratio_warn: float = 6.0
    tall_thin_aspect_warn: float = 4.0
    auto_repair_thin_walls: bool = False
    thin_wall_sample_faces: int = 2000
    thin_wall_ray_offset_mm: float = 0.01
    stability_contact_area_fraction_warn: float = 0.02

    # ------------------------------------------------------------------
    # Validation (spec module 6)
    # ------------------------------------------------------------------
    sweep_samples: int = 12
    sweep_distance_factor: float = 1.5
    collision_volume_tolerance_mm3: float = 1.0
    recovery_max_attempts: int = 8
    experimental: bool = False

    # ------------------------------------------------------------------
    # Units (spec module 1)
    # ------------------------------------------------------------------
    min_plausible_dim_mm: float = 1.0
    max_plausible_dim_mm: float = 5000.0

    # ------------------------------------------------------------------
    # Calibration fixture (spec section 13)
    # ------------------------------------------------------------------
    calibration_cube_mm: float = 24.0
    calibration_peg_offset_fraction: float = 0.25
    calibration_clearances_mm: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
    # The brick ladder is deliberately far wider than the chunk one above. Four
    # printed ladders spanning 0.19 down to -0.12 all felt identical, because
    # that whole span sits inside a single FDM process bias once it is
    # multiplied by four. A ladder narrower than the error bar cannot resolve
    # anything, so this one steps in tenths.
    # These numbers are NOT comparable to the ones tried before the grip
    # became a continuous rail. A discrete bump lets the wall dish locally
    # around it, which quietly gives back part of the interference; a rail
    # loads the whole wall at once and gives nothing back, so it reads tighter
    # in the hand at the same setting. 0.80 was already unassemblable on bumps
    # and would be worse on rails, so the ladder moves down rather than up.
    brick_calibration_interferences_mm: tuple[float, ...] = (0.40, 0.55, 0.70)
    # Every rung prints the same 3x1 footprint. The first attempt at this test
    # encoded the rung into the brick's LENGTH, which also happens to be the
    # largest single influence on how stiff its walls are - a 1x1 is a closed
    # braced box, while a 3x1 wall's nearest brace is a whole unit away. The
    # ladder therefore varied two things at once and measured neither.
    brick_calibration_footprint: str = "3x1"
    # Designed gap offsets for the gauge plate's slots, relative to an exact
    # stud fit. The bar's studs are dropped into each slot; the slot where the
    # printed gap crosses zero reveals brick_fit_bias_mm directly. The range is
    # wide because six rounds of coupons failed to find the fit inside a
    # narrower one.
    brick_gauge_offsets_mm: tuple[float, ...] = (
        -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30,
    )
    # Stamped on every calibration coupon as bars under the dot markers, and
    # bumped whenever the coupon geometry changes meaning. Coupons from
    # different geometry generations look nearly identical, and mixing them in
    # one test session produces exactly the unexplainable readings that burned
    # the chunk-mode calibration too.
    brick_calibration_batch: int = 1
    # Optional one-layer flange around the base of a calibration coupon, off by
    # default. It exists because a brick is a hollow shell open underneath, so
    # its first layer is only the wall outline - about 54 mm2 on a 3x1 that
    # stands 10.6 mm tall - and parts came off the plate at that contact area.
    # That turned out to be a printer fault rather than a geometry limit, and a
    # moulded flange is one more thing to peel off every coupon, so it is
    # disabled. Raise it again if bricks start lifting off the plate.
    brick_calibration_brim_mm: float = 0.0
    brick_calibration_marker_radius_mm: float = 0.75
    brick_calibration_marker_pitch_mm: float = 2.2

    # ------------------------------------------------------------------
    # Brick mode (spec module 4, brick branch)
    # ------------------------------------------------------------------
    # Stud pitch, and the cell size of the voxel grid. Isotropic: it is the
    # vertical course height too, because trimesh voxelizes on a single scalar
    # pitch. Everything else in this section is a fraction of it, so changing
    # this one number rescales the whole brick system coherently.
    brick_unit_mm: float = 8.0
    # Footprints in studs, largest area first: greedy tiling takes the first
    # that fits, so this order is the preference order.
    brick_library: tuple[str, ...] = ("4x2", "2x2", "4x1", "3x1", "2x1", "1x1")
    brick_stud_radius_fraction: float = 0.30
    # Taller than a real brick's stud, so real grip remains above the rib
    # lead-in relief. Course pitch is set by brick_unit_mm, so this is free.
    brick_stud_height_fraction: float = 0.325
    # HOW THE CLUTCH WORKS, and why it is not just a clearance number.
    #
    # Both sides of the fit move with the same process bias but in opposite
    # directions - an external stud prints small, an internal cavity prints
    # large - so the interference loses FOUR times any bias:
    #     interference = design - 4 * bias
    # A 0.06 mm design died at 15 microns of bias, well inside what FDM holds.
    # Chasing the number was hopeless; four printed ladders all felt the same.
    #
    # The fix is to make the fit insensitive rather than precise: a large
    # interference against a deliberately soft spring. A bad print day then
    # changes how firm the click is, not whether there is one.
    brick_clutch_interference_mm: float = 0.55
    # Thin, so the grip walls actually flex. This is the spring.
    brick_wall_mm: float = 0.90
    # Rounded vertical ribs carry the interference. Localising it onto a small
    # feature beats spreading it across a whole-wall dimension that accumulates
    # every process error in the part. The radius is derived rather than set:
    # it is bounded at both ends by the wall it springs from, and those bounds
    # move whenever the interference does.
    # Tubes are structure - they brace the roof bridge - not grip. They keep a
    # clearance so they never fight the ribs.
    brick_tube_clearance_mm: float = 0.30
    # Net dimensional bias of the printer at brick scale, measured with the
    # gauge plate (``cleaveform calibrate --mode gauge``): how much larger the
    # PRINTED gap between two mating faces comes out than the DESIGNED gap.
    # Negative means the printer prints fits tighter than the file. Once this
    # is set, ``brick_clutch_interference_mm`` means the interference wanted in
    # the actual plastic, and the geometry quietly compensates. Six rounds of
    # fit ladders failed to converge because this number was unknown and every
    # ladder guessed at it; measure it once instead.
    brick_fit_bias_mm: float = 0.0
    # How far the ribs are held clear of the build plate: exactly one layer, and
    # deliberately no more. Elephant foot squashes the first layer outward, and
    # a rib whose nose is printed in it starts life fatter than designed, which
    # makes the stud harder to enter without making the grip any stronger.
    #
    # It is one layer rather than several because relief is bought straight out
    # of engagement, and engagement is the scarce resource here - a stud is only
    # 2.6 mm long. An earlier value of 0.60 mm was chosen to fix a first-layer
    # defect that turned out not to exist (the rib bumps join the wall over
    # 0.57 mm, comfortably wider than a 0.40 mm bead, so they were always fused
    # and printable) and it cost 31% of the engagement for nothing.
    brick_rib_relief_mm: float = 0.20
    # Lateral gap between neighbouring bricks. They do not mate, they only need
    # to not interfere, so this is a plain clearance rather than a fit.
    brick_fit_clearance_mm: float = 0.10
    # Thicker than the walls it anchors. Previously it was thinner, which made
    # the roof - not the wall - 79% of the compliance: the spring nobody
    # designed was doing the springing.
    brick_roof_mm: float = 1.60
    brick_tube_wall_mm: float = 1.20
    brick_occupancy_threshold: float = 0.5
    brick_fill_interior: bool = True
    max_bricks: int = 2000
    # Below this share of bricks spanning two or more bricks beneath them,
    # the courses are stacked rather than interlocked and the model can split
    # along a course.
    brick_bond_fraction_warn: float = 0.5
    # Each pass bridges one split, so this bounds how fragmented a model may
    # be before it is reported as unbuildable rather than repaired.
    brick_repair_passes: int = 64
    # Bricks outside the main body, as a share of all bricks. Below this it
    # is a handful of edge bricks with nothing to grip and the model is still
    # buildable; above it the model genuinely comes apart.
    brick_detached_fraction_hard: float = 0.10
    # The total above says how much of the model is detached; this says how big
    # any ONE detached group is. They are different failures. Fifteen stray
    # bricks scattered one at a time are a few edge bricks to glue; one tower
    # of eleven running up the flank of a cone is a piece of the model that
    # was never attached to it, and as a share of a large build it can look
    # small enough to pass the fraction test. A single group bigger than this
    # is a hard failure whatever the total says.
    brick_stray_hard_bricks: int = 4
    # Some cells cannot be held by any tiling: nothing over them and nothing
    # under them, so they touch only their neighbours, and bricks never join
    # sideways. The interior of a one-brick-thick floor is the usual case.
    # Left alone they come out as stray bricks. With this on, the tiler uses
    # the strays as feedback and changes the SHAPE minimally to attach them -
    # dropping a sliver, thickening a flat floor by a course, or bridging an
    # island to the main body - then tiles again. Off by default because it
    # alters the model; what it changed is always reported.
    brick_simplify: bool = False
    brick_simplify_passes: int = 4
    # A stray group this small is a sliver, and is dropped rather than grown.
    brick_sliver_max_cells: int = 2
    # Features thinner than a brick vanish under the majority vote: a rail 5 mm
    # thick fills too little of a 12 mm cell to count, and the posts it joined
    # come out as separate pieces. With this on, the grid is sampled finer and
    # an empty cell holding at least ``brick_thin_min_fill`` of material is
    # added where it would join two otherwise-separate parts - never anywhere
    # else, so a model gains only the cells that keep it in one piece.
    brick_keep_thin: bool = False
    brick_thin_min_fill: float = 0.15
    # The widest run of empty-looking cells a thin feature may be bridged across.
    # Each cell on the run must itself hold ``brick_thin_min_fill`` of material, so
    # this never joins two parts that merely sit near each other: a gap with
    # nothing in it is a real separation, however narrow.
    brick_thin_max_gap_cells: int = 3
    plate_size_mm: tuple[float, float] = (250.0, 250.0)
    plate_margin_mm: float = 4.0

    # ------------------------------------------------------------------
    # Caching (spec section 7)
    # ------------------------------------------------------------------
    cache_enabled: bool = True
    cache_dir: str = ".cleaveform-cache"

    # ------------------------------------------------------------------
    # Numerical tolerances
    # ------------------------------------------------------------------
    area_epsilon_mm2: float = 1e-6
    volume_epsilon_mm3: float = 1e-6
    length_epsilon_mm: float = 1e-9

    # ------------------------------------------------------------------
    # Construction and validation
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
        """Build a config from defaults, an optional JSON file, then overrides."""
        values: dict[str, Any] = {}
        if config_file is not None:
            raw = json.loads(Path(config_file).read_text(encoding="utf-8"))
            values.update(_coerce(cls, raw))
        if overrides:
            values.update(_coerce(cls, {k: v for k, v in overrides.items() if v is not None}))
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def with_overrides(self, **overrides: Any) -> "Config":
        cfg = replace(self, **_coerce(type(self), overrides))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject internally inconsistent configurations early and loudly."""
        if self.decomposition_mode not in DECOMPOSITION_MODES:
            raise ValueError(
                f"decomposition_mode must be one of {sorted(DECOMPOSITION_MODES)}"
            )
        if self.joint_type != "peg_socket":
            raise ValueError("Phase 1 supports joint_type='peg_socket' only")
        if self.generation_backend != "off":
            raise ValueError("Phase 1 supports generation_backend='off' only")
        if self.export_format != "stl":
            raise ValueError("Phase 1 supports export_format='stl' only")
        if len(self.design_ceiling_mm) != N_AXES:
            raise ValueError("design_ceiling_mm must have three components")
        if any(v <= 0 for v in self.design_ceiling_mm):
            raise ValueError("design_ceiling_mm components must be positive")
        if self.min_wall_mm < self.nozzle_width_mm * 2:
            raise ValueError("min_wall_mm must be at least 2 x nozzle_width_mm")
        if self.peg_edge_margin_mm < self.min_wall_mm:
            raise ValueError("peg_edge_margin_mm must be at least min_wall_mm")
        if self.peg_radius_mm * 2 < self.min_feature_mm:
            raise ValueError("peg diameter is below the minimum printable feature size")
        if self.min_pegs_per_face < 1:
            raise ValueError("min_pegs_per_face must be at least 1")
        if self.max_pegs_per_face < self.min_pegs_per_face:
            raise ValueError("max_pegs_per_face must be >= min_pegs_per_face")
        if not self.cut_fractions:
            raise ValueError("cut_fractions must not be empty")
        if any(not 0.0 < f < 1.0 for f in self.cut_fractions):
            raise ValueError("cut_fractions must lie strictly between 0 and 1")
        if self.max_parts < 1 or self.max_cut_depth < 0:
            raise ValueError("max_parts and max_cut_depth must be non-negative bounds")
        if self.joint_clearance_mm < 0:
            raise ValueError("joint_clearance_mm must be non-negative")
        if self.recovery_clearance_range_mm < 0:
            raise ValueError("recovery_clearance_range_mm must be non-negative")

        self._validate_bricks()

        # The calibration fixture must be able to host a real production peg,
        # otherwise it measures nothing useful. Check it can, rather than
        # emitting a cube whose sockets punch out the back.
        if not 0.0 < self.calibration_peg_offset_fraction < 0.5:
            raise ValueError(
                "calibration_peg_offset_fraction must lie strictly between 0 and 0.5"
            )
        half = self.calibration_cube_mm / 2.0
        if half < self.min_socket_backing_mm:
            raise ValueError(
                f"calibration_cube_mm of {self.calibration_cube_mm} mm gives a "
                f"{half} mm half, too thin to back a {self.socket_depth_mm} mm socket; "
                f"it must be at least {self.min_socket_backing_mm * 2:.1f} mm"
            )
        offset = self.calibration_cube_mm * self.calibration_peg_offset_fraction
        wall = offset - self.socket_radius_mm
        if wall < self.min_wall_mm:
            raise ValueError(
                f"calibration_cube_mm of {self.calibration_cube_mm} mm leaves only "
                f"{wall:.2f} mm of wall beside each socket, below the "
                f"{self.min_wall_mm} mm minimum; a wall that thin flexes and would "
                "make a tight joint feel like a good fit"
            )

    def _validate_bricks(self) -> None:
        """Check the brick system is printable at the configured unit size.

        Brick size is meant to be tunable, so the failure mode of tuning it too
        far has to be a named error rather than bricks that split when a stud is
        pushed in.
        """
        if self.brick_unit_mm <= 0:
            raise ValueError("brick_unit_mm must be positive")
        if not self.brick_library:
            raise ValueError("brick_library must not be empty")
        for name in self.brick_library:
            width, depth = parse_brick_footprint(name)
            if width < 1 or depth < 1:
                raise ValueError(f"brick_library entry {name!r} must be at least 1x1")
        if not 0.0 < self.brick_stud_radius_fraction < 0.5:
            raise ValueError(
                "brick_stud_radius_fraction must lie strictly between 0 and 0.5"
            )
        if self.brick_stud_height_fraction <= 0:
            raise ValueError("brick_stud_height_fraction must be positive")
        if not 0.0 < self.brick_occupancy_threshold <= 1.0:
            raise ValueError("brick_occupancy_threshold must lie in (0, 1]")
        if self.max_bricks < 1:
            raise ValueError("max_bricks must be at least 1")
        if self.brick_stray_hard_bricks < 1:
            raise ValueError("brick_stray_hard_bricks must be at least 1")
        if self.brick_simplify_passes < 1:
            raise ValueError("brick_simplify_passes must be at least 1")
        if self.brick_sliver_max_cells < 0:
            raise ValueError("brick_sliver_max_cells cannot be negative")
        if self.brick_thin_max_gap_cells < 0:
            raise ValueError("brick_thin_max_gap_cells cannot be negative")
        if not 0.0 < self.brick_thin_min_fill < self.brick_occupancy_threshold:
            raise ValueError(
                "brick_thin_min_fill must be above 0 and below brick_occupancy_threshold, "
                "or it would add cells the majority vote already counts"
            )
        if self.brick_clutch_interference_mm < 0:
            raise ValueError(
                "brick_clutch_interference_mm is an interference, not a "
                "clearance; a negative value is a slip fit and cannot grip"
            )
        if self.brick_stud_grip_offset_mm <= 0:
            raise ValueError(
                f"the ribs would stand {self.brick_stud_grip_offset_mm:.2f} mm off "
                "the wall, so they never reach the stud; lower brick_wall_mm or "
                "raise brick_clutch_interference_mm"
            )

        if self.brick_wall_mm < self.min_wall_mm:
            raise ValueError(
                f"brick_wall_mm of {self.brick_wall_mm:.2f} mm is below the "
                f"{self.min_wall_mm} mm minimum; the wall is meant to be thin "
                "enough to spring, but below the nozzle's minimum it stops being "
                "a wall at all"
            )
        # The wall is a fixed thickness, so it no longer scales with the unit -
        # which means the unit is now bounded by the cavity instead. Shrink the
        # unit far enough and the gap between the walls closes below the stud
        # that has to enter it: not a tight fit but an impossible one, since the
        # stud would have to displace solid plastic rather than flex a wall.
        if self.brick_cavity_width_mm <= self.brick_stud_diameter_mm:
            raise ValueError(
                f"brick_unit_mm of {self.brick_unit_mm} mm leaves a cavity "
                f"{self.brick_cavity_width_mm:.2f} mm wide between the walls, but "
                f"the stud that enters it is {self.brick_stud_diameter_mm:.2f} mm "
                "across, so bricks could not be pushed together at all. Raise "
                "brick_unit_mm, or lower brick_wall_mm or "
                "brick_stud_radius_fraction."
            )
        if self.brick_roof_mm < self.min_wall_mm:
            raise ValueError(
                f"brick_roof_mm of {self.brick_roof_mm} mm is below the "
                f"{self.min_wall_mm} mm minimum wall; the studs would have nothing "
                "solid to stand on"
            )
        if self.brick_tube_wall_mm < self.min_wall_mm:
            raise ValueError(
                f"brick_tube_wall_mm of {self.brick_tube_wall_mm} mm is below the "
                f"{self.min_wall_mm} mm minimum wall"
            )
        if list(self.brick_gauge_offsets_mm) != sorted(self.brick_gauge_offsets_mm):
            raise ValueError(
                "brick_gauge_offsets_mm must run from tightest to loosest; the "
                "gauge instructions count slots from the tight end"
            )
        if self.brick_calibration_batch < 1:
            raise ValueError("brick_calibration_batch must be at least 1")
        if self.brick_rib_relief_mm <= 0:
            raise ValueError(
                "brick_rib_relief_mm must be positive; ribs that reach the build "
                "plate print as loose specks rather than as part of the wall"
            )
        if self.brick_rib_relief_mm < self.layer_height_mm:
            raise ValueError(
                f"brick_rib_relief_mm of {self.brick_rib_relief_mm} mm is under one "
                f"{self.layer_height_mm} mm layer, so the ribs would still land on "
                "the first layer"
            )
        if self.brick_rib_lead_in_mm >= self.brick_stud_height_mm:
            raise ValueError(
                f"the rib lead-in is {self.brick_rib_lead_in_mm:.2f} mm but a stud "
                f"is only {self.brick_stud_height_mm:.2f} mm tall, so the stud "
                "never reaches the gripping part of the rib"
            )
        if self.brick_tube_radius_mm <= self.brick_tube_wall_mm:
            raise ValueError(
                f"a tube of radius {self.brick_tube_radius_mm:.2f} mm cannot have "
                f"{self.brick_tube_wall_mm} mm walls; raise brick_unit_mm or lower "
                "brick_tube_wall_mm"
            )
        if self.brick_cavity_depth_mm <= self.brick_stud_height_mm:
            raise ValueError(
                f"the cavity is only {self.brick_cavity_depth_mm:.2f} mm deep but a "
                f"stud is {self.brick_stud_height_mm:.2f} mm tall, so bricks would "
                "not seat; lower brick_roof_mm or brick_stud_height_fraction"
            )

    def ceiling_for_axis(self, axis: int) -> float:
        return float(self.design_ceiling_mm[axis])

    @property
    def socket_radius_mm(self) -> float:
        """Socket radius is peg radius plus the calibrated clearance."""
        return self.peg_radius_mm + self.joint_clearance_mm

    @property
    def socket_depth_mm(self) -> float:
        return self.peg_depth_mm + self.socket_depth_margin_mm

    @property
    def peg_keepout_radius_mm(self) -> float:
        """Distance a peg centre must keep from the cut-face perimeter."""
        return self.socket_radius_mm + self.peg_edge_margin_mm

    @property
    def decomposition_ceiling_mm(self) -> tuple[float, float, float]:
        """Ceiling the partitioner cuts against.

        Pegs protrude ``peg_depth_mm`` beyond the cut face on the male side, so
        a part that exactly filled the design ceiling before joints would break
        the build-volume guarantee after them. Reserving that depth on every
        axis is conservative but keeps the guarantee unconditional.
        """
        reserved = self.peg_depth_mm
        return tuple(  # type: ignore[return-value]
            max(limit - reserved, self.min_part_mm) for limit in self.design_ceiling_mm
        )

    @property
    def min_socket_backing_mm(self) -> float:
        """Material depth the female part needs behind a socket."""
        return self.socket_depth_mm + self.min_wall_mm

    # ------------------------------------------------------------------
    # Brick geometry, all derived from brick_unit_mm
    # ------------------------------------------------------------------
    @property
    def brick_stud_radius_mm(self) -> float:
        return self.brick_unit_mm * self.brick_stud_radius_fraction

    @property
    def brick_stud_height_mm(self) -> float:
        return self.brick_unit_mm * self.brick_stud_height_fraction

    @property
    def brick_stud_grip_offset_mm(self) -> float:
        """How far a rib must stand off the wall to bite the stud.

        The stud nearest a wall sits half a unit in, so the rib has to cross
        the gap to it and then press in by half the interference.
        """
        stud_centre = self.brick_unit_mm / 2.0 - self.brick_fit_clearance_mm / 2.0
        stud_near_face = stud_centre - self.brick_stud_radius_mm
        # The designed interference is the wanted-in-plastic value plus the
        # measured print bias: a printer that prints gaps 0.2 mm tight needs
        # 0.2 mm LESS designed interference to land the same physical grip.
        designed = self.brick_clutch_interference_mm + self.brick_fit_bias_mm
        grip_face = stud_near_face + designed / 2.0
        return grip_face - self.brick_wall_mm

    @property
    def brick_stud_diameter_mm(self) -> float:
        return 2.0 * self.brick_stud_radius_mm

    @property
    def brick_cavity_width_mm(self) -> float:
        """Clear span between the two inner wall faces, before the ribs."""
        return (
            self.brick_unit_mm
            - self.brick_fit_clearance_mm
            - 2.0 * self.brick_wall_mm
        )

    @property
    def brick_rail_chamfer_mm(self) -> float:
        """Height of the 45-degree lead-in ramp on a rail's bottom edge.

        Equal to how far the rail stands proud, which makes the ramp 45 degrees
        at any setting. It is derived rather than set so that raising the
        interference cannot quietly steepen the entry - the harder the fit, the
        longer the ramp that guides the stud into it.
        """
        return self.brick_stud_grip_offset_mm

    @property
    def brick_rib_lead_in_mm(self) -> float:
        """Height at which a rib reaches its full width.

        The chamfered lead-in ramps from :attr:`brick_rib_relief_mm` up to
        here, so a stud meets a slope rather than a square step.
        """
        return self.brick_rib_relief_mm + self.brick_rail_chamfer_mm

    @property
    def brick_tube_radius_mm(self) -> float:
        """Outer radius of the roof-bracing tubes.

        These no longer grip. A closed cylinder is stiff in hoop, so putting a
        large interference on one makes a brick that cannot be pushed together;
        the compliant ribs do the gripping and the tubes keep clear of the
        stud so the two never fight.
        """
        diagonal = self.brick_unit_mm * math.sqrt(2.0) / 2.0
        return (
            diagonal - self.brick_stud_radius_mm - self.brick_tube_clearance_mm
        )

    @property
    def brick_cavity_depth_mm(self) -> float:
        """Depth of the open underside, from the bottom face up to the roof."""
        return self.brick_unit_mm - self.brick_roof_mm

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


def _coerce(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Validate override keys and convert sequences to tuples."""
    known = {f.name: f for f in fields(cls)}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"unknown config key: {key!r}")
        if isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out


DEFAULT_CONFIG = Config()
