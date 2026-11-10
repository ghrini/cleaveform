"""Generate a small demo vase, watertight and ready for cleaveform.

Cleaveform has no text/image-to-3D generator (that is Phase 4, spec module 2,
GENERATION_BACKEND='off' in Phase 1). This script exists in its place: a
parametric revolved vase, hollowed by a boolean difference between an outer
and an inner solid of revolution.
"""

from __future__ import annotations

import numpy as np
import trimesh

# Silhouette: radius vs height, mm. Narrow foot, wide belly, necked-in
# shoulder, small flared mouth.
HEIGHTS_MM = np.array([0, 4, 15, 35, 55, 70, 82, 90, 96, 100], dtype=float)
RADII_MM = np.array([18, 20, 26, 32, 30, 24, 19, 17, 19, 21], dtype=float)

WALL_MM = 3.0
BASE_MM = 4.0  # solid floor thickness
SECTIONS = 96  # revolve smoothness


def capped_revolve(heights: np.ndarray, radii: np.ndarray) -> trimesh.Trimesh:
    """Revolve a wall profile into a closed solid with flat top and bottom caps.

    A bare revolve of a profile that does not touch the axis leaves an open
    tube, not a solid - trimesh's boolean backend refuses that as "not a
    volume". Prepending and appending a zero-radius point at the first and
    last height closes both ends with a flat disc.
    """
    points = [(0.0, heights[0])] + list(zip(radii, heights)) + [(0.0, heights[-1])]
    return trimesh.creation.revolve(np.array(points), sections=SECTIONS)


def build_vase() -> trimesh.Trimesh:
    outer = capped_revolve(HEIGHTS_MM, RADII_MM)

    inner_radii = np.clip(RADII_MM - WALL_MM, 1.0, None)
    above_base = HEIGHTS_MM >= BASE_MM
    wall_h = HEIGHTS_MM[above_base]
    wall_r = inner_radii[above_base]

    # Extend the cavity straight up past the outer top, so the difference
    # fully punches through outer's own top cap and leaves an open rim
    # instead of a flat washer at the mouth.
    extended_top = wall_h[-1] + 10.0
    inner_h = np.concatenate([[BASE_MM], wall_h, [extended_top]])
    inner_r = np.concatenate([[wall_r[0]], wall_r, [wall_r[-1]]])
    inner = capped_revolve(inner_h, inner_r)

    vase = trimesh.boolean.difference([outer, inner], engine="manifold")
    vase.merge_vertices()
    return vase


if __name__ == "__main__":
    from pathlib import Path

    vase = build_vase()
    assert vase.is_watertight, "vase is not watertight"
    extents = vase.bounds[1] - vase.bounds[0]
    print(f"size: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
    print(f"volume: {vase.volume:.0f} mm3")

    out = Path(__file__).parent / "small_vase.stl"
    vase.export(out)
    print(f"wrote {out}")
