"""Manufacturability constraints, tiered (spec section 11).

Separating "must reject" from "costs more" stops the tool from rejecting a
perfectly printable model just because part of it needs support.

- ``hard``    violation rejects the run
- ``soft``    violation is a cost penalty, never a rejection
- ``warning`` is advisory only and never blocks

Thin walls are flagged, never silently thickened. Geometry is only altered when
``auto_repair_thin_walls`` is enabled, and Phase 1 does not implement that path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..config import Config
from ..estimate import MINUTES_PER_HOUR, print_time_min
from ..geometry import analysis, bbox

HARD = "hard"
SOFT = "soft"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    tier: str
    code: str
    message: str
    part_id: str | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "code": self.code,
            "message": self.message,
            "part_id": self.part_id,
            "detail": self.detail,
        }


@dataclass
class ConstraintReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def of_tier(self, tier: str) -> list[Finding]:
        return [f for f in self.findings if f.tier == tier]

    @property
    def hard_failures(self) -> list[Finding]:
        return self.of_tier(HARD)

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "hard": [f.to_dict() for f in self.of_tier(HARD)],
            "soft": [f.to_dict() for f in self.of_tier(SOFT)],
            "warnings": [f.to_dict() for f in self.of_tier(WARNING)],
        }


def check_part(
    part_id: str,
    mesh: trimesh.Trimesh,
    oriented: trimesh.Trimesh,
    config: Config,
    joint_area_mm2: float = 0.0,
) -> list[Finding]:
    """Run every per-part manufacturability check and return the findings.

    ``mesh`` is the part as exported; ``oriented`` is the same part rotated into
    its recommended print orientation. Orientation-dependent checks (support,
    warp, aspect) use the latter, because that is how the part will be printed.
    """
    findings: list[Finding] = []
    findings.extend(_hard_checks(part_id, mesh, config))
    if any(f.tier == HARD for f in findings):
        # Geometry-level failures make the derived metrics meaningless.
        return findings
    findings.extend(_soft_checks(part_id, oriented, config, joint_area_mm2))
    findings.extend(_warnings(part_id, oriented, config))
    return findings


def _hard_checks(part_id: str, mesh: trimesh.Trimesh, config: Config) -> list[Finding]:
    findings: list[Finding] = []
    extents = bbox.extents(mesh)

    if len(mesh.faces) == 0 or abs(float(mesh.volume)) < config.volume_epsilon_mm3:
        findings.append(
            Finding(
                HARD,
                "degenerate_geometry",
                f"{part_id} has no usable volume",
                part_id,
                {"volume_mm3": float(mesh.volume)},
            )
        )
        return findings

    if not bbox.fits_ceiling(mesh, config):
        findings.append(
            Finding(
                HARD,
                "exceeds_design_ceiling",
                f"{part_id} measures {extents[0]:.1f} x {extents[1]:.1f} x "
                f"{extents[2]:.1f} mm, beyond the design ceiling "
                f"{config.design_ceiling_mm}",
                part_id,
                {
                    "extents_mm": [float(v) for v in extents],
                    "ceiling_mm": list(config.design_ceiling_mm),
                },
            )
        )

    if not mesh.is_watertight:
        findings.append(
            Finding(
                HARD,
                "not_watertight",
                f"{part_id} is not watertight after the repair pass",
                part_id,
                {},
            )
        )

    smallest = float(np.min(extents))
    if smallest < config.min_feature_mm:
        findings.append(
            Finding(
                HARD,
                "below_min_feature",
                f"{part_id} is only {smallest:.2f} mm on its smallest axis, below the "
                f"{config.min_feature_mm:.2f} mm minimum printable feature",
                part_id,
                {"smallest_mm": smallest, "minimum_mm": config.min_feature_mm},
            )
        )
    return findings


def _soft_checks(
    part_id: str, mesh: trimesh.Trimesh, config: Config, joint_area_mm2: float
) -> list[Finding]:
    findings: list[Finding] = []
    volume = float(abs(mesh.volume))
    support = analysis.estimate_support_volume(mesh, config)
    support_fraction = support / volume if volume > config.volume_epsilon_mm3 else 0.0

    if support > config.volume_epsilon_mm3:
        tier = SOFT
        findings.append(
            Finding(
                tier,
                "needs_support",
                f"{part_id} needs an estimated {support:.0f} mm3 of support "
                f"({support_fraction:.1%} of its volume); printable, penalised only",
                part_id,
                {
                    "support_volume_mm3": support,
                    "support_fraction": support_fraction,
                    "threshold_fraction": config.support_volume_soft_fraction,
                    "over_threshold": support_fraction
                    > config.support_volume_soft_fraction,
                },
            )
        )

    minutes = print_time_min(mesh, config, support)
    if minutes > config.print_time_soft_minutes:
        findings.append(
            Finding(
                SOFT,
                "long_print_time",
                f"{part_id} is estimated at {minutes / MINUTES_PER_HOUR:.1f} h, over the "
                f"{config.print_time_soft_minutes / MINUTES_PER_HOUR:.1f} h soft limit "
                "(estimate, not a slice)",
                part_id,
                {"est_print_time_min": minutes},
            )
        )

    area = float(mesh.area)
    if area > config.area_epsilon_mm2:
        seam_fraction = joint_area_mm2 / area
        if seam_fraction > config.seam_area_soft_fraction:
            findings.append(
                Finding(
                    SOFT,
                    "large_seam_area",
                    f"{part_id} devotes {seam_fraction:.1%} of its surface to visible "
                    "seams",
                    part_id,
                    {"seam_fraction": seam_fraction},
                )
            )

    extents = bbox.extents(mesh)
    footprint = np.sort(extents[:2])
    if footprint[0] > config.length_epsilon_mm:
        footprint_aspect = float(footprint[1] / footprint[0])
        if footprint_aspect > config.warp_aspect_ratio_warn:
            findings.append(
                Finding(
                    SOFT,
                    "warp_risk",
                    f"{part_id} has a {footprint_aspect:.1f}:1 footprint aspect ratio; "
                    "long thin footprints are warp-prone",
                    part_id,
                    {"footprint_aspect": footprint_aspect},
                )
            )

    aspect = bbox.aspect_ratio(mesh, config)
    if math.isfinite(aspect) and aspect > config.tall_thin_aspect_warn:
        findings.append(
            Finding(
                SOFT,
                "tall_thin",
                f"{part_id} has a {aspect:.1f}:1 overall aspect ratio and may need a "
                "brim or a different orientation",
                part_id,
                {"aspect_ratio": aspect},
            )
        )
    return findings


def _warnings(part_id: str, mesh: trimesh.Trimesh, config: Config) -> list[Finding]:
    findings: list[Finding] = []

    thickness = analysis.min_local_thickness(mesh, config)
    if math.isfinite(thickness) and thickness < config.min_wall_mm:
        note = (
            "flagged only; enable auto_repair_thin_walls to allow thickening"
            if not config.auto_repair_thin_walls
            else "auto thin-wall repair is enabled but not implemented in Phase 1"
        )
        findings.append(
            Finding(
                WARNING,
                "thin_wall",
                f"{part_id} has a region about {thickness:.2f} mm thick, below the "
                f"{config.min_wall_mm:.2f} mm minimum wall; {note}",
                part_id,
                {"min_thickness_mm": thickness, "min_wall_mm": config.min_wall_mm},
            )
        )

    return findings


def joint_face_on_bed_warning(
    part_id: str,
    joint_normals: list[np.ndarray],
    orientation_transform: np.ndarray,
    config: Config,
) -> Finding | None:
    """Warn when a mating face ends up flat on the build plate.

    Elephant foot on a plain outer surface is cosmetic. On a joint face it
    changes the fit, which is the one place first-layer compensation actually
    matters here.
    """
    rotation = np.asarray(orientation_transform)[:3, :3]
    for normal in joint_normals:
        rotated = rotation @ np.asarray(normal, dtype=float)
        if float(rotated @ np.array([0.0, 0.0, 1.0])) < -(1.0 - config.length_epsilon_mm):
            return Finding(
                WARNING,
                "joint_face_on_bed",
                f"{part_id} prints with a mating face flat on the build plate; enable "
                "elephant-foot compensation or the first layer will spread and tighten "
                "the joint",
                part_id,
                {"constraint": "first_layer_compensation"},
            )
    return None


def stability_warning(
    part_id: str, mesh: trimesh.Trimesh, joint_area_mm2: float, config: Config
) -> Finding | None:
    """Advisory contact heuristic for the joined structure."""
    area = float(mesh.area)
    if area <= config.area_epsilon_mm2 or joint_area_mm2 <= 0.0:
        return None
    fraction = joint_area_mm2 / area
    if fraction >= config.stability_contact_area_fraction_warn:
        return None
    return Finding(
        WARNING,
        "small_contact_area",
        f"{part_id} attaches through only {fraction:.2%} of its surface area; the joint "
        "may be weak relative to the part and could benefit from glue",
        part_id,
        {"contact_fraction": fraction},
    )
