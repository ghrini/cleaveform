"""Module 5 - peg and socket connector generation.

At every cut face the part on the negative side of the plane normal gets pegs;
the part on the positive side gets matching sockets. Socket radius is the peg
radius plus the calibrated ``joint_clearance_mm``.

Placement is fully deterministic: candidate centres come from a fixed-pitch grid
inside the eroded face polygon, and the spread is chosen by farthest-point
selection seeded from the centroid, with ties broken by sorted index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

from ..config import Config
from ..decompose.graph import CutGraph, Interface
from ..geometry import analysis, booleans, slicing, solids


class JointError(RuntimeError):
    """A cut face cannot carry a connector that satisfies the hard constraints.

    Carries the originating cut so bounded recovery can rule that cut out and
    retry with a different one.
    """

    def __init__(self, message: str, node_path: str, candidate_index: int, detail: dict):
        super().__init__(message)
        self.node_path = node_path
        self.candidate_index = candidate_index
        self.detail = detail


@dataclass
class JointPlacement:
    """The connectors generated for one interface."""

    interface_id: str
    male_id: str
    female_id: str
    axis_name: str
    normal: np.ndarray
    centers_mm: list[np.ndarray]
    peg_radius_mm: float
    socket_radius_mm: float
    peg_depth_mm: float
    clearance_mm: float
    face_area_mm2: float

    @property
    def peg_count(self) -> int:
        return len(self.centers_mm)

    def to_dict(self) -> dict:
        return {
            "id": self.interface_id,
            "joint_type": "peg_socket",
            "male_part": self.male_id,
            "female_part": self.female_id,
            "axis": self.axis_name,
            "normal": [float(v) for v in self.normal],
            "peg_count": self.peg_count,
            "peg_radius_mm": self.peg_radius_mm,
            "socket_radius_mm": self.socket_radius_mm,
            "peg_depth_mm": self.peg_depth_mm,
            "clearance_mm": self.clearance_mm,
            "face_area_mm2": self.face_area_mm2,
            "centers_mm": [[float(v) for v in c] for c in self.centers_mm],
        }


@dataclass
class JointResult:
    meshes: dict[str, trimesh.Trimesh]
    placements: list[JointPlacement]
    warnings: list[str] = field(default_factory=list)


def generate_joints(graph: CutGraph, config: Config) -> JointResult:
    """Add pegs and sockets to every interface in the cut graph.

    Returns the modified part meshes keyed by part id. Parts with no interfaces
    are returned unchanged.
    """
    placements: list[JointPlacement] = []
    warnings: list[str] = []
    pegs_by_part: dict[str, list[trimesh.Trimesh]] = {}
    sockets_by_part: dict[str, list[trimesh.Trimesh]] = {}

    for interface in sorted(graph.interfaces, key=lambda i: i.id):
        male_mesh = graph.parts[interface.male_id].mesh
        female_mesh = graph.parts[interface.female_id].mesh
        placement = _plan_interface(interface, male_mesh, female_mesh, config, warnings)
        placements.append(placement)

        for center in placement.centers_mm:
            pegs_by_part.setdefault(interface.male_id, []).append(
                solids.peg(center, interface.normal, config)
            )
            sockets_by_part.setdefault(interface.female_id, []).append(
                solids.socket(center, interface.normal, config)
            )

    meshes: dict[str, trimesh.Trimesh] = {}
    for part_id in sorted(graph.parts):
        mesh = graph.parts[part_id].mesh
        pegs = pegs_by_part.get(part_id, [])
        sockets = sockets_by_part.get(part_id, [])
        try:
            if pegs:
                mesh = booleans.union([mesh, *pegs], config)
            if sockets:
                mesh = booleans.difference(mesh, sockets, config)
        except booleans.BooleanError as exc:
            raise JointError(
                f"boolean backend failed while fitting connectors to {part_id}: {exc}",
                node_path="",
                candidate_index=-1,
                detail={"part": part_id, "stage": "joint_boolean"},
            ) from exc
        meshes[part_id] = mesh

    return JointResult(meshes=meshes, placements=placements, warnings=warnings)


def _plan_interface(
    interface: Interface,
    male_mesh: trimesh.Trimesh,
    female_mesh: trimesh.Trimesh,
    config: Config,
    warnings: list[str],
) -> JointPlacement:
    """Choose peg centres on one cut face and check the hard constraints."""
    target_count = _peg_count(interface.area_mm2, config)

    eroded = interface.polygon.buffer(-config.peg_keepout_radius_mm)
    candidates = _grid_points(eroded, config)
    if len(candidates) < config.min_pegs_per_face:
        raise JointError(
            f"cut face {interface.id} between {interface.male_id} and "
            f"{interface.female_id} is {interface.area_mm2:.0f} mm2, too small to hold "
            f"{config.min_pegs_per_face} peg(s) of radius {config.peg_radius_mm} mm with "
            f"a {config.peg_edge_margin_mm} mm edge margin",
            node_path=interface.node_path,
            candidate_index=interface.candidate_index,
            detail={
                "interface": interface.id,
                "constraint": "connector_does_not_fit_cut_face",
                "tier": "hard",
                "face_area_mm2": interface.area_mm2,
            },
        )

    count = min(target_count, len(candidates))
    if count < config.min_pegs_per_face:
        count = config.min_pegs_per_face
    chosen_2d = _spread(candidates, count, config)

    centers: list[np.ndarray] = []
    for point in chosen_2d:
        center = slicing.point_2d_to_3d(point, interface.to_3d)
        _check_backing(interface, center, male_mesh, female_mesh, config)
        centers.append(center)

    return JointPlacement(
        interface_id=interface.id,
        male_id=interface.male_id,
        female_id=interface.female_id,
        axis_name=interface.axis_name,
        normal=np.asarray(interface.normal, dtype=float),
        centers_mm=centers,
        peg_radius_mm=config.peg_radius_mm,
        socket_radius_mm=config.socket_radius_mm,
        peg_depth_mm=config.peg_depth_mm,
        clearance_mm=config.joint_clearance_mm,
        face_area_mm2=interface.area_mm2,
    )


def _peg_count(area_mm2: float, config: Config) -> int:
    """Roughly one peg per ``peg_area_per_peg_mm2``, clamped to the configured range."""
    raw = math.ceil(area_mm2 / config.peg_area_per_peg_mm2)
    return int(min(max(raw, config.min_pegs_per_face), config.max_pegs_per_face))


def _grid_points(region, config: Config) -> np.ndarray:
    """Fixed-pitch lattice of legal peg centres inside the eroded face."""
    if region.is_empty or region.area <= config.area_epsilon_mm2:
        return np.empty((0, 2))
    min_x, min_y, max_x, max_y = region.bounds
    pitch = config.peg_grid_pitch_mm
    xs = np.arange(min_x, max_x + pitch, pitch)
    ys = np.arange(min_y, max_y + pitch, pitch)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2))
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside = np.array([region.contains(Point(p[0], p[1])) for p in points], dtype=bool)
    points = points[inside]
    if len(points) == 0:
        # A sliver region can contain no lattice point; fall back to its
        # representative interior point so a narrow face still gets one peg.
        fallback = region.representative_point()
        return np.array([[fallback.x, fallback.y]])
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


def _spread(points: np.ndarray, count: int, config: Config) -> list[np.ndarray]:
    """Farthest-point selection seeded from the point nearest the centroid."""
    if count >= len(points):
        return [points[i] for i in range(len(points))]
    centroid = points.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
    chosen = [first]
    distances = np.linalg.norm(points - points[first], axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distances))
        if distances[nxt] <= config.length_epsilon_mm:
            break
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(points - points[nxt], axis=1))
    chosen.sort()
    return [points[i] for i in chosen]


def _check_backing(
    interface: Interface,
    center: np.ndarray,
    male_mesh: trimesh.Trimesh,
    female_mesh: trimesh.Trimesh,
    config: Config,
) -> None:
    """Hard constraint: both sides need enough material behind the connector."""
    normal = np.asarray(interface.normal, dtype=float)

    female_depth = analysis.solid_depth_along(female_mesh, center, normal, config)
    if female_depth < config.min_socket_backing_mm:
        raise JointError(
            f"socket on {interface.female_id} at face {interface.id} would need "
            f"{config.min_socket_backing_mm:.2f} mm of material but only "
            f"{female_depth:.2f} mm is available",
            node_path=interface.node_path,
            candidate_index=interface.candidate_index,
            detail={
                "interface": interface.id,
                "part": interface.female_id,
                "constraint": "socket_backing_below_minimum",
                "tier": "hard",
                "available_mm": female_depth,
                "required_mm": config.min_socket_backing_mm,
            },
        )

    male_depth = analysis.solid_depth_along(male_mesh, center, -normal, config)
    if male_depth < config.min_wall_mm:
        raise JointError(
            f"peg on {interface.male_id} at face {interface.id} would sit on "
            f"{male_depth:.2f} mm of material, below the {config.min_wall_mm:.2f} mm "
            "minimum wall",
            node_path=interface.node_path,
            candidate_index=interface.candidate_index,
            detail={
                "interface": interface.id,
                "part": interface.male_id,
                "constraint": "peg_root_below_minimum_wall",
                "tier": "hard",
                "available_mm": male_depth,
                "required_mm": config.min_wall_mm,
            },
        )
