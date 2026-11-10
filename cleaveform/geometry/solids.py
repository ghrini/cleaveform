"""Parametric connector solids.

Pegs and sockets are cylinders of revolution generated with a fixed segment
count so that repeated runs produce byte-identical geometry.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..config import Config


def cylinder_on_axis(
    center, direction, radius: float, height: float, config: Config
) -> trimesh.Trimesh:
    """A closed cylinder whose base sits at ``center``, extruded along ``direction``.

    ``center`` is the centre of the base disc, not of the solid.
    """
    axis = np.asarray(direction, dtype=float)
    axis = axis / float(np.linalg.norm(axis))
    solid = trimesh.creation.cylinder(
        radius=radius, height=height, sections=config.peg_cylinder_segments
    )
    transform = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), axis)
    transform[:3, 3] = np.asarray(center, dtype=float) + axis * (height / 2.0)
    solid.apply_transform(transform)
    return solid


def peg(center, direction, config: Config) -> trimesh.Trimesh:
    """Peg solid protruding from the cut face at ``center`` along ``direction``."""
    return cylinder_on_axis(
        center, direction, config.peg_radius_mm, config.peg_depth_mm, config
    )


def socket(center, direction, config: Config) -> trimesh.Trimesh:
    """Socket cutter sunk into the cut face at ``center`` along ``direction``.

    Radius is the peg radius plus the calibrated clearance; the extra depth
    margin guarantees the peg bottoms out on the mating face rather than on the
    socket floor.
    """
    return cylinder_on_axis(
        center, direction, config.socket_radius_mm, config.socket_depth_mm, config
    )
