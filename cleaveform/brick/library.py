"""The brick set: footprints, and the solid geometry of one brick.

Brick mode's whole premise is that output is a small number of repeated shapes
rather than many one-off parts, so a brick's mesh is built once per *type* and
printed many times. That is the reason brick mode does not reuse chunk mode's
per-part joint generation: there is nothing per-part to generate.

Bricks connect vertically only, stud into socket. Two bricks side by side in a
course do not touch each other; what binds them is a brick in the course above
overlapping both, exactly as in masonry. That makes stagger a structural
property rather than a cosmetic one - see :mod:`cleaveform.brick.tiling`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from ..config import Config, parse_brick_footprint
from ..geometry import booleans, solids

# Booleans between exactly coplanar faces produce slivers that the manifold
# backend rejects, so studs sink slightly into the body and socket cutters
# extend slightly past it. Small enough to be invisible at any print resolution.
COPLANAR_EPSILON_MM = 0.1

UP = np.array([0.0, 0.0, 1.0])
DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class BrickType:
    """One footprint in the brick set. Height is always one course."""

    width_studs: int
    depth_studs: int

    @property
    def id(self) -> str:
        return f"brick_{self.width_studs}x{self.depth_studs}"

    @property
    def footprint(self) -> str:
        return f"{self.width_studs}x{self.depth_studs}"

    @property
    def stud_count(self) -> int:
        return self.width_studs * self.depth_studs

    @property
    def is_square(self) -> bool:
        return self.width_studs == self.depth_studs

    def rotated(self) -> "BrickType":
        return BrickType(self.depth_studs, self.width_studs)

    def size_mm(self, config: Config) -> tuple[float, float, float]:
        gap = config.brick_fit_clearance_mm
        return (
            self.width_studs * config.brick_unit_mm - gap,
            self.depth_studs * config.brick_unit_mm - gap,
            config.brick_unit_mm,
        )

    def to_dict(self, config: Config) -> dict:
        width, depth, height = self.size_mm(config)
        return {
            "id": self.id,
            "footprint": self.footprint,
            "width_studs": self.width_studs,
            "depth_studs": self.depth_studs,
            "stud_count": self.stud_count,
            "size_mm": [width, depth, height],
        }


def load_library(config: Config) -> list[BrickType]:
    """Brick types from config, largest footprint first.

    Order is the tiler's preference order, so it is sorted rather than trusted:
    a config listing 1x1 first would otherwise tile everything in single studs.
    """
    types = [BrickType(*parse_brick_footprint(name)) for name in config.brick_library]
    # Squarer bricks win ties on stud count: a 2x2 ties its neighbours together
    # along both axes, a 4x1 only along one.
    types.sort(
        key=lambda t: (
            -t.stud_count,
            max(t.width_studs, t.depth_studs) / min(t.width_studs, t.depth_studs),
            t.footprint,
        )
    )
    return types


def stud_centres_local(brick: BrickType, config: Config) -> list[np.ndarray]:
    """Stud centres in the brick's own frame, origin at its min corner.

    Centres sit on the stud lattice regardless of the body shrink, so a stud
    always lands over the socket of the brick below.
    """
    unit = config.brick_unit_mm
    half_gap = config.brick_fit_clearance_mm / 2.0
    centres = []
    for i in range(brick.width_studs):
        for j in range(brick.depth_studs):
            centres.append(
                np.array([(i + 0.5) * unit - half_gap, (j + 0.5) * unit - half_gap])
            )
    return centres


def tube_centres_local(brick: BrickType, config: Config) -> list[np.ndarray]:
    """Tube centres: the crossings between four studs.

    Tubes brace the roof bridge; they no longer grip. A one-stud-deep brick has
    no crossing and therefore no tube, which is why its grip has to come from
    the walls.
    """
    unit = config.brick_unit_mm
    half_gap = config.brick_fit_clearance_mm / 2.0
    return [
        np.array([i * unit - half_gap, j * unit - half_gap])
        for i in range(1, brick.width_studs)
        for j in range(1, brick.depth_studs)
    ]


def grip_rail_faces_local(brick: BrickType, config: Config) -> list[float]:
    """The two y positions where a stud is gripped, in the brick's own frame.

    One rail per long wall, each running the full length of the cavity. The
    stud is caught between them.
    """
    _, depth_mm, _ = brick.size_mm(config)
    inner = config.brick_wall_mm + config.brick_stud_grip_offset_mm
    return [inner, depth_mm - inner]


def _rail_profile(y_outer: float, y_inner: float, config: Config) -> Polygon:
    """Cross-section of one rail in the (y, z) plane.

    The bottom inner edge is chamfered at 45 degrees. That chamfer is the whole
    reason this shape exists rather than a plain rectangle: it is what lets a
    stud find its way in from any position instead of having to arrive already
    lined up.
    """
    relief = config.brick_rib_relief_mm
    top = config.brick_stud_height_mm
    chamfer = config.brick_rail_chamfer_mm
    sign = 1.0 if y_inner > y_outer else -1.0
    return orient(
        Polygon(
            [
                (y_outer, relief),
                (y_inner - sign * chamfer, relief),
                (y_inner, relief + chamfer),
                (y_inner, top),
                (y_outer, top),
            ]
        )
    )


def _grip_rails(brick: BrickType, config: Config) -> list[trimesh.Trimesh]:
    """The two gripping rails, as solids in the brick's frame.

    Continuous rails, not one bump per stud. The bumps they replace were
    discrete and sat at the stud positions, so joining two bricks meant every
    stud had to register into its own pair of bumps at the same instant - which
    in the hand is not a click, it is a jam. A rail has nothing to line up
    against along its length, so a stud meets it wherever it happens to land.
    """
    width, depth, _ = brick.size_mm(config)
    wall = config.brick_wall_mm
    length = width - 2.0 * wall
    if length <= 0:
        return []

    # extrude_polygon builds in XY and extrudes along Z, so the profile comes
    # out as (y, z, x) and the axes are cycled back into place.
    to_brick_frame = np.array(
        [
            [0.0, 0.0, 1.0, wall],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    rails = []
    for y_outer, y_inner in ((0.0, grip_rail_faces_local(brick, config)[0]),
                             (depth, grip_rail_faces_local(brick, config)[1])):
        solid = trimesh.creation.extrude_polygon(
            _rail_profile(y_outer, y_inner, config), height=length
        )
        solid.apply_transform(to_brick_frame)
        rails.append(solid)
    return rails


def build_brick_solid(brick: BrickType, config: Config) -> trimesh.Trimesh:
    """One brick: a hollow shell, open underneath, studs on top, ribs inside.

    The grip is deliberately soft and deliberately over-sized. Both halves of
    the fit move with the same process bias in opposite directions - an
    external stud prints small, an internal cavity prints large - so a designed
    interference loses four times any bias. A small interference against a stiff
    wall therefore cannot survive FDM: fifteen microns of bias erases it, and
    four printed ladders all felt identical because of it.

    So the walls are thin enough to flex, the interference is large enough to
    outlive the process, and it is carried on ribs rather than on a whole-wall
    dimension that accumulates every error in the part. A bad print changes how
    firm the click is instead of whether there is one.

    Built at the origin with its min corner at (0, 0, 0) and studs pointing +Z.
    Print orientation is fixed studs-up: ``analysis.best_orientation`` minimises
    support and would happily stand a brick on its stud tips.
    """
    width, depth, body_height = brick.size_mm(config)
    wall = config.brick_wall_mm
    cavity_depth = config.brick_cavity_depth_mm

    body = trimesh.creation.box(extents=(width, depth, body_height))
    body.apply_translation((width / 2.0, depth / 2.0, body_height / 2.0))

    cavity = trimesh.creation.box(
        extents=(
            width - 2.0 * wall,
            depth - 2.0 * wall,
            cavity_depth + COPLANAR_EPSILON_MM,
        )
    )
    cavity.apply_translation(
        (width / 2.0, depth / 2.0, (cavity_depth - COPLANAR_EPSILON_MM) / 2.0)
    )
    shell = booleans.difference(body, [cavity], config)

    additions: list[trimesh.Trimesh] = []
    cutters: list[trimesh.Trimesh] = []

    for centre in stud_centres_local(brick, config):
        additions.append(
            solids.cylinder_on_axis(
                np.array([centre[0], centre[1], body_height - COPLANAR_EPSILON_MM]),
                UP,
                config.brick_stud_radius_mm,
                config.brick_stud_height_mm + COPLANAR_EPSILON_MM,
                config,
            )
        )

    rails = _grip_rails(brick, config)
    if rails:
        # Clip to the body. A rail is sunk into the wall it springs from, so
        # part of it lies behind the inner face by construction; clipping also
        # stops it breaking out through the OUTSIDE of the brick, which would
        # be a part wider than its own footprint that fouls its neighbours.
        clipped = booleans.intersection([booleans.union(rails, config), body], config)
        if clipped is not None:
            additions.append(clipped)

    for centre in tube_centres_local(brick, config):
        additions.append(
            solids.cylinder_on_axis(
                np.array([centre[0], centre[1], 0.0]),
                UP,
                config.brick_tube_radius_mm,
                cavity_depth,
                config,
            )
        )
        cutters.append(
            solids.cylinder_on_axis(
                np.array([centre[0], centre[1], -COPLANAR_EPSILON_MM]),
                UP,
                config.brick_tube_radius_mm - config.brick_tube_wall_mm,
                cavity_depth + COPLANAR_EPSILON_MM,
                config,
            )
        )

    solid = booleans.union([shell, *additions], config)
    return booleans.difference(solid, cutters, config) if cutters else solid


def build_all(config: Config, used: set[str] | None = None) -> dict[str, trimesh.Trimesh]:
    """Solids for every brick type, or only those a layout actually uses."""
    meshes: dict[str, trimesh.Trimesh] = {}
    for brick in load_library(config):
        if used is not None and brick.id not in used:
            continue
        meshes[brick.id] = build_brick_solid(brick, config)
    return meshes
