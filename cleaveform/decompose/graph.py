"""The cut graph: which finished parts touch, and across what face.

A recursive binary partition does not by itself tell you which leaves are
neighbours, because a cut made early is later subdivided on both sides. This
module recovers the real leaf-to-leaf interfaces geometrically: for every cut it
probes each leaf just off the plane and intersects the two footprints.

The result is the ``chunk set`` contract from spec section 8 and the input to
joint generation and assembly ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..config import AXIS_NAMES, Config
from ..geometry import slicing
from .partition import DecompositionResult, PartNode


@dataclass
class Interface:
    """A shared cut face between two finished parts.

    The part on the negative side of the plane normal carries the pegs (male);
    the positive side carries the sockets (female). Assigning sides from the
    plane normal keeps the choice deterministic.
    """

    id: str
    male_id: str
    female_id: str
    axis: int
    origin: np.ndarray
    normal: np.ndarray
    polygon: Polygon
    to_3d: np.ndarray
    area_mm2: float
    node_path: str
    candidate_index: int

    @property
    def axis_name(self) -> str:
        return AXIS_NAMES[self.axis]

    def other(self, part_id: str) -> str:
        return self.female_id if part_id == self.male_id else self.male_id

    def insertion_direction_for(self, part_id: str) -> np.ndarray:
        """Direction the given part travels to seat into its neighbour."""
        return -self.normal if part_id == self.female_id else self.normal

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "male_part": self.male_id,
            "female_part": self.female_id,
            "axis": self.axis_name,
            "origin_mm": [float(v) for v in self.origin],
            "normal": [float(v) for v in self.normal],
            "area_mm2": self.area_mm2,
        }


@dataclass
class CutGraph:
    parts: dict[str, PartNode]
    interfaces: list[Interface]
    warnings: list[str]

    def interfaces_for(self, part_id: str) -> list[Interface]:
        return [i for i in self.interfaces if part_id in (i.male_id, i.female_id)]

    def neighbours(self, part_id: str) -> list[str]:
        return sorted({i.other(part_id) for i in self.interfaces_for(part_id)})


def build_cut_graph(result: DecompositionResult, config: Config) -> CutGraph:
    """Recover leaf-to-leaf interfaces from the partition tree."""
    parts = {node.part_id: node for node in result.parts if node.part_id}
    interfaces: list[Interface] = []
    warnings: list[str] = []

    for node in sorted(result.root.internal_nodes(), key=lambda n: (n.depth, n.path)):
        cut = node.cut
        if cut is None or len(node.children) != 2:
            continue
        origin = np.asarray(cut.origin, dtype=float)
        normal = np.asarray(cut.normal, dtype=float)
        eps = config.cut_face_probe_eps_mm

        negative_leaves = sorted(node.children[0].leaves(), key=lambda n: n.part_id or "")
        positive_leaves = sorted(node.children[1].leaves(), key=lambda n: n.part_id or "")

        footprints_negative = {
            leaf.part_id: _footprint(leaf, origin - normal * eps, origin, normal, config)
            for leaf in negative_leaves
        }
        footprints_positive = {
            leaf.part_id: _footprint(leaf, origin + normal * eps, origin, normal, config)
            for leaf in positive_leaves
        }
        _, to_3d = slicing.plane_transforms(origin, normal)

        for male_id, male_face in footprints_negative.items():
            if male_face is None:
                continue
            for female_id, female_face in footprints_positive.items():
                if female_face is None:
                    continue
                shared = male_face.intersection(female_face)
                if shared.is_empty or shared.area < config.min_joint_face_area_mm2:
                    continue
                polygon = _largest_polygon(shared)
                if polygon is None:
                    continue
                interfaces.append(
                    Interface(
                        id=f"joint_{len(interfaces) + 1:03d}",
                        male_id=str(male_id),
                        female_id=str(female_id),
                        axis=cut.axis,
                        origin=origin,
                        normal=normal,
                        polygon=polygon,
                        to_3d=to_3d,
                        area_mm2=float(polygon.area),
                        node_path=node.path,
                        candidate_index=cut.candidate_index,
                    )
                )

    connected = {pid for i in interfaces for pid in (i.male_id, i.female_id)}
    for part_id in sorted(parts):
        if part_id not in connected and len(parts) > 1:
            warnings.append(
                f"{part_id} shares no cut face large enough for a connector "
                f"(minimum {config.min_joint_face_area_mm2:.0f} mm2); it is listed as a "
                "separate sub-assembly in the guide"
            )

    return CutGraph(parts=parts, interfaces=interfaces, warnings=warnings)


def _footprint(
    leaf: PartNode, probe_origin, frame_origin, normal, config: Config
) -> Polygon | None:
    """Union of the leaf's cross-section just off the cut plane."""
    polygons, _ = slicing.cross_section_polygons(
        leaf.mesh, probe_origin, normal, config, frame_origin=frame_origin
    )
    if not polygons:
        return None
    merged = unary_union(polygons)
    if merged.is_empty:
        return None
    return merged if merged.area > config.area_epsilon_mm2 else None


def _largest_polygon(geometry) -> Polygon | None:
    """Reduce a possibly multi-part intersection to its largest simple polygon."""
    if isinstance(geometry, Polygon):
        return geometry
    parts = [g for g in getattr(geometry, "geoms", []) if isinstance(g, Polygon)]
    if not parts:
        return None
    parts.sort(key=lambda p: (-p.area, round(p.bounds[0], 6), round(p.bounds[1], 6)))
    return parts[0]
