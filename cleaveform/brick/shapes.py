"""A library of test shapes for brick mode, built from code.

Brick mode was developed against one vase, and a vase is the easy case: round,
hollow, roughly the same width top to bottom. Whether the tool really works is a
question about everything else - things that narrow to a point, loop back on
themselves, stand on thin legs, hang out over nothing, or arrive as a broken
mesh. This module is a catalogue of those cases.

The shapes are generated rather than stored as files for two reasons. Nothing
large is committed, and every shape can be rebuilt at any size, which matters
because whether a model survives depends on how many bricks span it, not on its
millimetres. The same catalogue feeds the benchmark (``benchmark.py``) and the
regression tests, so a shape that once failed stays covered.

Every shape stands on the build plate with its lowest point at z = 0 and its
largest dimension scaled to ``size_mm``. All dimensions below are fractions of
that size, kept in one ``P_<SHAPE>`` table each so no number is buried in a
builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh
from shapely.geometry import Polygon

from ..config import Config
from ..geometry import booleans

# Categories, in the order they are reported. Each names what the group stresses.
CATEGORIES = {
    "solid": "plain solids: slopes and curves",
    "hollow": "hollow and looping shapes",
    "thin": "features narrower than a brick",
    "overhang": "parts hanging out over nothing",
    "organic": "realistic stand-ins for things people actually make",
    "messy": "broken or awkward input files",
}

DEFAULT_SIZE_MM = 140.0
SECTIONS_ROUND = 48
SECTIONS_SMOOTH = 64
SECTIONS_WHEEL = 32
_CONFIG = Config()

# ---- dimensions, as fractions of the shape's size ----------------------------
P_PYRAMID = {"radius": 0.5}
P_CONE = {"radius": 0.45}
P_CAPSULE = {"height": 0.5, "radius": 0.25}
P_BOWL = {"radius": 0.5, "wall": 0.1}
P_CUP = {"radius": 0.3, "height": 0.55, "wall": 0.07, "handle_major": 0.17,
         "handle_minor": 0.04, "handle_gap": 0.1, "handle_z": 0.3}
P_TORUS = {"major": 0.35, "minor": 0.15}
P_ARCH = {"depth": 0.25, "height": 0.7, "gap_w": 0.5, "gap_d": 0.4, "gap_h": 0.5,
          "gap_x": 0.25, "gap_y": -0.05, "gap_z": -0.01}
# (radius, height) up the outside of the vase, then back down the inside.
P_VASE = {"profile": ((0.0, 0.0), (0.28, 0.0), (0.34, 0.25), (0.22, 0.7), (0.13, 0.86),
                      (0.17, 1.0), (0.11, 1.0), (0.07, 0.86), (0.16, 0.7), (0.27, 0.25),
                      (0.22, 0.07), (0.0, 0.07))}
P_TRAY = {"depth": 0.7, "height": 0.22, "in_w": 0.84, "in_d": 0.54, "in_h": 0.22,
          "in_x": 0.08, "in_y": 0.08, "floor": 0.07}
P_TABLE = {"top": 0.08, "top_z": 0.6, "leg": 0.1, "leg_h": 0.62, "near": 0.02, "far": 0.88}
P_CHAIR = {"seat": 0.6, "seat_t": 0.07, "seat_z": 0.45, "back_h": 0.5, "back_z": 0.5,
           "leg": 0.07, "leg_h": 0.47, "leg_far": 0.53}
P_BRIDGE = {"deck_d": 0.3, "deck_t": 0.07, "deck_z": 0.3, "pier": 0.1, "pier_h": 0.32,
            "pier_far": 0.9, "mid": 0.45}
P_FENCE = {"post": 0.06, "post_h": 0.5, "post_span": 0.94, "rail_d": 0.04, "rail_t": 0.05,
           "rail_y": 0.01, "rail_z": (0.15, 0.35)}
P_STAIRS = {"run": 0.2, "width": 0.4, "rise": 0.16}
P_LETTER_L = {"arm": 0.7, "thick": 0.2, "depth": 0.2}
P_LETTER_H = {"width": 0.8, "thick": 0.2, "bar_lo": 0.4, "bar_hi": 0.6, "depth": 0.2}
P_TOWER = {"width": 0.12}
P_MUSHROOM = {"stem_r": 0.1, "stem_h": 0.5, "cap_r": 0.45, "cap_squash": 0.55}
P_T_SHAPE = {"post": 0.15, "post_h": 0.7, "post_x": 0.425, "bar_t": 0.15, "bar_z": 0.6}
# Each crown ball is (radius, x, y, z).
P_TREE = {"trunk_r": 0.07, "trunk_h": 0.5,
          "crown": ((0.28, 0.0, 0.0, 0.62), (0.2, 0.2, 0.0, 0.5), (0.2, -0.2, 0.05, 0.5))}
P_SHELF = {"wall_t": 0.1, "wall_d": 0.6, "wall_h": 0.8, "plank_w": 0.8, "plank_t": 0.08,
           "plank_x": 0.05, "plank_z": 0.45}
P_BLOB = {"lobe_a": 0.22, "lobe_b": 0.15, "lobe_c": 0.08, "phase": 1.0}
P_CAR = {"body_d": 0.45, "body_h": 0.2, "body_z": 0.12, "cabin_w": 0.5, "cabin_d": 0.4,
         "cabin_h": 0.17, "cabin_x": 0.22, "cabin_y": 0.025, "cabin_z": 0.3,
         "wheel_r": 0.12, "wheel_w": 0.07, "wheel_x": (0.2, 0.8), "wheel_y": (0.02, 0.43)}
P_ANIMAL = {"body_r": 0.5, "body_scale": (0.5, 0.28, 0.28), "body_z": 0.5, "head_r": 0.12,
            "head_x": 0.32, "head_z": 0.66, "leg_r": 0.045, "leg_h": 0.38, "leg_x": 0.18,
            "leg_y": 0.07, "tail_r": 0.05, "tail_x": -0.27, "tail_z": 0.55}
P_HOUSE = {"depth": 0.7, "wall_h": 0.55, "eave": 0.05, "roof_h": 0.35, "roof_depth": 0.78,
           "roof_shift": 0.04, "door_w": 0.2, "door_d": 0.1, "door_h": 0.3,
           "door_x": 0.4, "door_y": -0.01, "door_z": -0.01}
P_BALL = {"radius": 0.5}
P_TWO_BODIES = {"radius": 0.2, "gap": 0.9}


@dataclass(frozen=True)
class Shape:
    name: str
    category: str
    note: str
    build: Callable[[float], trimesh.Trimesh]
    # A model made of two separate objects can only ever build as two pieces, so
    # that is the right answer for it rather than a failure.
    expected_pieces: int = 1
    # Whether the surface is closed. A closed surface can still be wrong (see
    # ``inside_out``), so the two are separate.
    watertight: bool = True
    # Closed, but every face points inward - the volume comes out negative.
    inside_out: bool = False
    # What a broken file was MEANT to be. Fidelity is scored against this, so a
    # sphere with a hole is judged on how well the tool recovered the sphere,
    # not on whatever the damaged surface happens to enclose.
    reference: Callable[[float], trimesh.Trimesh] | None = None

    def _transform(self, size_mm: float) -> tuple[np.ndarray, float]:
        """The shift and scale that stand this shape on the plate at ``size_mm``."""
        raw = self.build(size_mm)
        shift = -raw.bounds[0]
        extent = float(raw.extents.max())
        return shift, (size_mm / extent if extent > 0 else 1.0)

    def _placed(self, raw: trimesh.Trimesh, size_mm: float) -> trimesh.Trimesh:
        shift, scale = self._transform(size_mm)
        out = raw.copy()
        out.apply_translation(shift)
        out.apply_scale(scale)
        out.apply_translation(-out.bounds[0])
        return out

    def mesh(self, size_mm: float = DEFAULT_SIZE_MM) -> trimesh.Trimesh:
        return self._placed(self.build(size_mm), size_mm)

    def reference_mesh(self, size_mm: float = DEFAULT_SIZE_MM) -> trimesh.Trimesh:
        """The intended shape, placed exactly as :meth:`mesh` places the real one."""
        build = self.reference or self.build
        return self._placed(build(size_mm), size_mm)


# ------------------------------------------------------------------ helpers
def _union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return booleans.union(parts, _CONFIG)


def _box(w, d, h, x=0.0, y=0.0, z=0.0) -> trimesh.Trimesh:
    """A box with its minimum corner at (x, y, z)."""
    m = trimesh.creation.box(extents=(w, d, h))
    m.apply_translation((x + w / 2.0, y + d / 2.0, z + h / 2.0))
    return m


def _cyl(r, h, x=0.0, y=0.0, z=0.0, sections=SECTIONS_ROUND) -> trimesh.Trimesh:
    """An upright cylinder with the centre of its base at (x, y, z)."""
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_translation((x, y, z + h / 2.0))
    return m


def _ball(r, x, y, z, sx=1.0, sy=1.0, sz=1.0, subdivisions=3) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r)
    m.apply_scale((sx, sy, sz))
    m.apply_translation((x, y, z))
    return m


def _slab_below(z, s) -> trimesh.Trimesh:
    """A box covering everything at or below height ``z``, far wider than any shape."""
    return _box(4 * s, 4 * s, 2 * s, -2 * s, -2 * s, z - 2 * s)


def _stand(profile: Polygon, depth: float) -> trimesh.Trimesh:
    """Extrude a 2D outline in the x-z plane, so a letter or roof stands upright."""
    m = trimesh.creation.extrude_polygon(profile, height=depth)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    return m


def _lay_flat_cylinder(r, h, x, y, z) -> trimesh.Trimesh:
    """A cylinder lying along the y axis, centred at (x, y, z)."""
    m = trimesh.creation.cylinder(radius=r, height=h, sections=SECTIONS_WHEEL)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    m.apply_translation((x, y, z))
    return m


# ------------------------------------------------------------------- solids
def _cube(s):
    return _box(s, s, s)


def _pyramid(s):
    return trimesh.creation.cone(radius=P_PYRAMID["radius"] * s, height=s, sections=4)


def _cone(s):
    return trimesh.creation.cone(radius=P_CONE["radius"] * s, height=s, sections=SECTIONS_SMOOTH)


def _capsule(s):
    return trimesh.creation.capsule(height=P_CAPSULE["height"] * s, radius=P_CAPSULE["radius"] * s)


def _sphere(s):
    return trimesh.creation.icosphere(subdivisions=3, radius=P_BALL["radius"] * s)


# ------------------------------------------------------------------- hollow
def _bowl(s):
    p = P_BOWL
    r, t = p["radius"] * s, p["wall"] * s
    shell = booleans.difference(_ball(r, 0, 0, 0), [_ball(r - t, 0, 0, 0)], _CONFIG)
    return booleans.intersection([shell, _slab_below(0.0, s)], _CONFIG)


def _cup(s):
    p = P_CUP
    r, h, t = p["radius"] * s, p["height"] * s, p["wall"] * s
    body = booleans.difference(_cyl(r, h), [_cyl(r - t, h, z=t)], _CONFIG)
    handle = trimesh.creation.torus(major_radius=p["handle_major"] * s, minor_radius=p["handle_minor"] * s)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    handle.apply_translation((r + p["handle_gap"] * s, 0, p["handle_z"] * s))
    return _union([body, handle])


def _torus(s):
    return trimesh.creation.torus(major_radius=P_TORUS["major"] * s, minor_radius=P_TORUS["minor"] * s)


def _arch(s):
    p = P_ARCH
    outer = _box(s, p["depth"] * s, p["height"] * s)
    cut = _box(p["gap_w"] * s, p["gap_d"] * s, p["gap_h"] * s, p["gap_x"] * s, p["gap_y"] * s, p["gap_z"] * s)
    return booleans.difference(outer, [cut], _CONFIG)


def _vase(s):
    return trimesh.creation.revolve(np.array(P_VASE["profile"]) * s, sections=SECTIONS_SMOOTH)


def _tray(s):
    p = P_TRAY
    outer = _box(s, p["depth"] * s, p["height"] * s)
    cut = _box(p["in_w"] * s, p["in_d"] * s, p["in_h"] * s, p["in_x"] * s, p["in_y"] * s, p["floor"] * s)
    return booleans.difference(outer, [cut], _CONFIG)


# --------------------------------------------------------------------- thin
def _table(s):
    p = P_TABLE
    top = _box(s, s, p["top"] * s, 0, 0, p["top_z"] * s)
    legs = [
        _box(p["leg"] * s, p["leg"] * s, p["leg_h"] * s, x * s, y * s, 0)
        for x in (p["near"], p["far"])
        for y in (p["near"], p["far"])
    ]
    return _union([top, *legs])


def _chair(s):
    p = P_CHAIR
    seat = _box(p["seat"] * s, p["seat"] * s, p["seat_t"] * s, 0, 0, p["seat_z"] * s)
    back = _box(p["seat"] * s, p["seat_t"] * s, p["back_h"] * s, 0, 0, p["back_z"] * s)
    legs = [
        _box(p["leg"] * s, p["leg"] * s, p["leg_h"] * s, x * s, y * s, 0)
        for x in (0.0, p["leg_far"])
        for y in (0.0, p["leg_far"])
    ]
    return _union([seat, back, *legs])


def _bridge(s):
    p = P_BRIDGE
    deck = _box(s, p["deck_d"] * s, p["deck_t"] * s, 0, 0, p["deck_z"] * s)
    piers = [
        _box(p["pier"] * s, p["deck_d"] * s, p["pier_h"] * s, x * s, 0, 0)
        for x in (0.0, p["pier_far"], p["mid"])
    ]
    return _union([deck, *piers])


def _fence(s):
    p = P_FENCE
    posts = [
        _box(p["post"] * s, p["post"] * s, p["post_h"] * s, x, 0, 0)
        for x in np.linspace(0, p["post_span"] * s, 5)
    ]
    rails = [_box(s, p["rail_d"] * s, p["rail_t"] * s, 0, p["rail_y"] * s, z * s) for z in p["rail_z"]]
    return _union([*posts, *rails])


def _stairs(s):
    p = P_STAIRS
    steps = [
        _box(p["run"] * s, p["width"] * s, (i + 1) * p["rise"] * s, i * p["run"] * s, 0, 0)
        for i in range(5)
    ]
    return _union(steps)


def _letter_l(s):
    p = P_LETTER_L
    t, a = p["thick"] * s, p["arm"] * s
    return _stand(Polygon([(0, 0), (a, 0), (a, t), (t, t), (t, s), (0, s)]), p["depth"] * s)


def _letter_h(s):
    p = P_LETTER_H
    t, w = p["thick"] * s, p["width"] * s
    lo, hi = p["bar_lo"] * s, p["bar_hi"] * s
    outline = [(0, 0), (t, 0), (t, lo), (w - t, lo), (w - t, 0), (w, 0), (w, s),
               (w - t, s), (w - t, hi), (t, hi), (t, s), (0, s)]
    return _stand(Polygon(outline), p["depth"] * s)


def _tower(s):
    w = P_TOWER["width"] * s
    return _box(w, w, s)


# ----------------------------------------------------------------- overhang
def _mushroom(s):
    p = P_MUSHROOM
    stem = _cyl(p["stem_r"] * s, p["stem_h"] * s)
    cap = _ball(p["cap_r"] * s, 0, 0, p["stem_h"] * s, sz=p["cap_squash"])
    cap = booleans.difference(cap, [_slab_below(p["stem_h"] * s, s)], _CONFIG)
    return _union([stem, cap])


def _t_shape(s):
    p = P_T_SHAPE
    post = _box(p["post"] * s, p["post"] * s, p["post_h"] * s, p["post_x"] * s, 0, 0)
    bar = _box(s, p["post"] * s, p["bar_t"] * s, 0, 0, p["bar_z"] * s)
    return _union([post, bar])


def _tree(s):
    p = P_TREE
    trunk = _cyl(p["trunk_r"] * s, p["trunk_h"] * s)
    crown = [_ball(r * s, x * s, y * s, z * s) for r, x, y, z in p["crown"]]
    return _union([trunk, *crown])


def _shelf(s):
    p = P_SHELF
    wall = _box(p["wall_t"] * s, p["wall_d"] * s, p["wall_h"] * s)
    plank = _box(p["plank_w"] * s, p["wall_d"] * s, p["plank_t"] * s, p["plank_x"] * s, 0, p["plank_z"] * s)
    return _union([wall, plank])


# ------------------------------------------------------------------ organic
def _blob(s):
    p = P_BLOB
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v = np.asarray(m.vertices)
    theta = np.arccos(np.clip(v[:, 2], -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    k = 0.5 * s * (
        1
        + p["lobe_a"] * np.sin(2 * phi + p["phase"]) * np.sin(theta)
        + p["lobe_b"] * np.cos(3 * theta)
        + p["lobe_c"] * np.sin(5 * phi)
    )
    return trimesh.Trimesh(v * k[:, None], m.faces, process=False)


def _car(s):
    p = P_CAR
    body = _box(s, p["body_d"] * s, p["body_h"] * s, 0, 0, p["body_z"] * s)
    cabin = _box(p["cabin_w"] * s, p["cabin_d"] * s, p["cabin_h"] * s,
                 p["cabin_x"] * s, p["cabin_y"] * s, p["cabin_z"] * s)
    wheels = [
        _lay_flat_cylinder(p["wheel_r"] * s, p["wheel_w"] * s, x * s, y * s, p["body_z"] * s)
        for x in p["wheel_x"]
        for y in p["wheel_y"]
    ]
    return _union([body, cabin, *wheels])


def _animal(s):
    p = P_ANIMAL
    sx, sy, sz = p["body_scale"]
    body = _ball(p["body_r"] * s, 0, 0, p["body_z"] * s, sx=sx, sy=sy, sz=sz)
    head = _ball(p["head_r"] * s, p["head_x"] * s, 0, p["head_z"] * s)
    legs = [
        _cyl(p["leg_r"] * s, p["leg_h"] * s, x * s, y * s, 0)
        for x in (-p["leg_x"], p["leg_x"])
        for y in (-p["leg_y"], p["leg_y"])
    ]
    tail = _ball(p["tail_r"] * s, p["tail_x"] * s, 0, p["tail_z"] * s, sx=2.0)
    return _union([body, head, *legs, tail])


def _house(s):
    p = P_HOUSE
    walls = _box(s, p["depth"] * s, p["wall_h"] * s)
    roof = _stand(
        Polygon([(-p["eave"] * s, 0), ((1 + p["eave"]) * s, 0), (0.5 * s, p["roof_h"] * s)]),
        p["roof_depth"] * s,
    )
    roof.apply_translation((0, (p["roof_depth"] - p["roof_shift"]) * s, p["wall_h"] * s))
    door = _box(p["door_w"] * s, p["door_d"] * s, p["door_h"] * s,
                p["door_x"] * s, p["door_y"] * s, p["door_z"] * s)
    return booleans.difference(_union([walls, roof]), [door], _CONFIG)


# -------------------------------------------------------------------- messy
def _open_sphere(s):
    """A sphere with one face missing: the kind of hole a bad export leaves."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=P_BALL["radius"] * s)
    return trimesh.Trimesh(m.vertices, m.faces[1:], process=False)


def _inverted_sphere(s):
    """A sphere whose faces all point inward, as a flipped export would."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=P_BALL["radius"] * s)
    return trimesh.Trimesh(m.vertices, m.faces[:, ::-1], process=False)


def _two_bodies(s):
    p = P_TWO_BODIES
    a = _ball(p["radius"] * s, 0, 0, p["radius"] * s)
    b = _ball(p["radius"] * s, p["gap"] * s, 0, p["radius"] * s)
    return trimesh.util.concatenate([a, b])


SHAPES: list[Shape] = [
    Shape("cube", "solid", "the baseline: a plain block", _cube),
    Shape("pyramid", "solid", "four sloping faces narrowing to a point", _pyramid),
    Shape("cone", "solid", "narrows to a point: the tip is smaller than a brick", _cone),
    Shape("capsule", "solid", "a rounded pill standing on end", _capsule),
    Shape("sphere", "solid", "curved on every side", _sphere),
    Shape("bowl", "hollow", "a thin curved shell open at the top", _bowl),
    Shape("cup", "hollow", "a hollow cylinder with a handle looping off one side", _cup),
    Shape("torus", "hollow", "a ring: connected only by going all the way round", _torus),
    Shape("arch", "hollow", "a span held up by two legs, open underneath", _arch),
    Shape("vase", "hollow", "the original test shape, redrawn procedurally", _vase),
    Shape("tray", "hollow", "a shallow open box with thin walls", _tray),
    Shape("table", "thin", "a flat top on four thin legs", _table),
    Shape("chair", "thin", "legs, a seat and a back", _chair),
    Shape("bridge", "thin", "a long thin deck on three piers", _bridge),
    Shape("fence", "thin", "thin posts joined by thin rails", _fence),
    Shape("stairs", "thin", "five steps, each higher than the last", _stairs),
    Shape("letter_L", "thin", "a letter made of narrow arms", _letter_l),
    Shape("letter_H", "thin", "two uprights joined by a narrow bar", _letter_h),
    Shape("tower", "thin", "a tall column barely wider than a brick", _tower),
    Shape("mushroom", "overhang", "a wide cap on a narrow stem", _mushroom),
    Shape("t_shape", "overhang", "a bar balanced on a post", _t_shape),
    Shape("tree", "overhang", "a trunk carrying a round crown", _tree),
    Shape("shelf", "overhang", "a plank sticking out from a wall", _shelf),
    Shape("blob", "organic", "a lumpy rounded form with no flat sides", _blob),
    Shape("car", "organic", "a body, a cabin and four wheels", _car),
    Shape("animal", "organic", "a body with a head, four legs and a tail", _animal),
    Shape("house", "organic", "walls, a pitched roof and a door recess", _house),
    Shape("open_sphere", "messy", "a sphere with a face missing", _open_sphere,
          watertight=False, reference=_sphere),
    Shape("inverted_sphere", "messy", "a sphere with every face flipped inward", _inverted_sphere,
          inside_out=True, reference=_sphere),
    Shape("two_bodies", "messy", "two separate balls: can only ever be two pieces", _two_bodies,
          expected_pieces=2),
]

BY_NAME = {shape.name: shape for shape in SHAPES}


def select(names: list[str] | None = None, categories: list[str] | None = None) -> list[Shape]:
    """Shapes by name or category, in catalogue order. No filter means all of them."""
    chosen = SHAPES
    if names:
        missing = [n for n in names if n not in BY_NAME]
        if missing:
            raise KeyError(f"unknown shape(s) {missing}; choose from {sorted(BY_NAME)}")
        chosen = [BY_NAME[n] for n in names]
    if categories:
        unknown = [c for c in categories if c not in CATEGORIES]
        if unknown:
            raise KeyError(f"unknown categor(ies) {unknown}; choose from {sorted(CATEGORIES)}")
        chosen = [s for s in chosen if s.category in categories]
    return list(chosen)
