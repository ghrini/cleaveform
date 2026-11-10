# Cleaveform

Take a 3D model that is too big to print, split it into pieces that fit a Bambu
Lab H2D bed, put peg-and-socket connectors at every cut, and get printable STLs
plus an assembly guide. A person prints and builds. No robotics.

Two modes, both deterministic and offline:

```bash
cleaveform model.stl -o out/                # chunk: a few large custom pieces
cleaveform model.stl -o out/ --mode brick   # brick: many small standard bricks
```

**Chunk mode** cuts a model into a handful of one-off pieces, and only when it
is too big for the bed. A 64 mm vase comes back uncut, as one part.

**Brick mode** voxelizes the model and builds it from a small set of repeated
interlocking bricks - the same 64 mm vase becomes 108 bricks from 6 shapes. Not
compatible with real LEGO; this is its own brick system.

```
out/
  parts/part_001.stl ...   one STL per part
  metadata.json            per-part bbox, orientation, estimates, joints, cut log
  assembly_guide.txt       numbered steps: what goes where, which way it pushes
  diagnostic.txt           written instead, if the run cannot complete
```

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[repair,dev]"
```

`manifold3d` is the boolean backend and is required. `open3d` is optional; it
provides a second-chance repair pass for non-manifold input meshes.

## Before you trust the output: calibrate

The joint clearance cannot be derived from geometry. It depends on your printer,
filament and profile, so it is set by a physical print. This is a gate, not a
suggestion: **until you run it, every joint uses the 0.20 mm starting default.**

```bash
cleaveform calibrate -o calib/
```

Print all eight STLs in one job, then dry-fit each pair by hand. The smallest
clearance that seats fully without forcing and without slop is your value. Put
it in a config file:

```json
{ "joint_clearance_mm": 0.15 }
```

```bash
cleaveform model.stl -o out/ --config myprinter.json
```

## How it decides where to cut

Recursively, on each part:

1. Take the bounding box. If it fits the ceiling on every axis, the part is done.
2. Otherwise cut across the axis that overshoots its ceiling limit by the most.
3. Try planes at 40%, 50% and 60% of that axis, and throw out any that leave a
   piece empty, unrepairable, or thinner than `min_part_mm`.
4. Score the survivors on six normalised terms: part count, support volume,
   visible seam area, joint count, print-time overhead, and a strength penalty
   for cutting through a narrow section. All are dimensionless and land in
   [0, 1], so no term can dominate on raw magnitude alone.
5. Take the cheapest. Ties go to the lowest candidate index, which is what keeps
   runs reproducible.
6. Recurse. Stop with a diagnostic if `max_parts` or `max_cut_depth` is hit.

The ceiling used for cutting is the design ceiling minus `peg_depth_mm`, because
pegs stick out past the cut face. That is what makes the build-volume guarantee
unconditional rather than usually-true.

### What the cost terms are for

`part_count` and `print_time` push toward fewer, larger pieces. `support_volume`
avoids cuts that leave overhangs. `joint_count` penalises a cut whose
cross-section breaks into several patches, since each patch needs its own
connector. `strength_penalty` compares the cut's cross-section against the
bounding-box slice it sits in, so cutting an hourglass at the waist scores badly
and the tool stops handing back parts that snap at the joint. `visible_seam_area`
weights seam area by how visible the seam actually is: rays are fired from points
along the seam, and the fraction that escape without hitting the model again is
its exposure, so a seam buried in a slot costs less than one across an open face.

There is deliberately no bounding-box feasibility term. Phase 1 had one, and it
turned out to fight the part-count term: it rewarded leaving less oversize
behind, which is not the same as needing fewer parts, and it split a 620 mm bar
into four parts where three fit. Pieces are kept inside the ceiling by the
recursion, which cuts until they fit, not by anything in the cost.

## Brick mode

Voxelize the model onto a stud lattice, tile each course with bricks from a
small library, and export **one STL per shape** plus packed build plates, a bill
of materials, and a map of every course. Six files describe a hundred bricks,
because the bricks repeat.

```
out/
  bricks/brick_2x1.stl ...   one file per shape, print many
  plates/plate_01.stl ...    copies arranged on build-plate-sized sheets
  bill_of_materials.txt      how many of each
  assembly_guide.txt         a letter map of every course
```

Brick size is one number: `brick_unit_mm`, the stud pitch, defaulting to 8 mm.
Everything else - stud radius, socket depth, library footprints - derives from
it, so `--brick-unit 6` rescales the whole system coherently. Below about
5.5 mm the wall beside each socket drops under the minimum printable thickness
and config validation refuses it by name rather than letting you find out in
PLA. Bricks of different unit sizes do not interoperate; changing it starts a
second collection rather than extending the first.

### Why the courses are staggered

Bricks grip top to bottom only. Two bricks side by side in a course have no
connection to each other at all - what binds them is a brick in the course above
overlapping both, exactly as in masonry. So if every course breaks in the same
place, nothing ties across that line and the model splits along it.

Two approaches were tried and measured before the current one. Alternating the
scan direction on odd courses does nothing: an 8-wide run tiles `4|4` in both
directions, and any run that is a multiple of the longest brick is
mirror-symmetric. Preferring seam positions at a phase offset works along one
axis, but the x and y preferences cancel into ties that resolve identically in
every course, and a solid block still split cleanly in half.

What works is to stop proxying the goal: when choosing a brick, first prefer one
covering two or more different bricks in the course below. That is the
definition of a bonded course. Each course also tries several scan origins and
mirrorings and keeps the best-bonded result, because a shifted origin is
absorbed by a round footprint - the scan slides back to the same first filled
cell and reproduces the course below. Whatever is still split afterwards is
bridged directly by merging the two bricks either side of the split.

`metadata.json` reports what that achieved: `bond_fraction` (share of bricks
spanning two below - the number that actually means interlocked),
`seam_alignment_fraction`, and `weakest_link_cells`, which names the plane where
the model would break.

### Known limits

- Brick mode coarsens. At 8 mm an object 64 mm across is 8 cells wide, and
  detail below the unit size is gone. Larger models brick better.
- A feature one cell wide running diagonally has nothing holding it on. Bricks
  at an edge with nothing above them are reported rather than silently shipped;
  if enough of the model is detached it is a hard failure.
- The stud clutch has its own calibrated value, `brick_clutch_interference_mm`,
  separate from the chunk joints' `joint_clearance_mm`. Note the name: it is an
  **interference**, not a clearance, so a larger number is a tighter fit and
  zero cannot grip at all. It has its own fixture - see *Calibrating the brick
  clutch* below.
- Bricks are hollow shells, open underneath. Rounded vertical ribs on the long
  walls grip the stud; the tubes at the stud crossings only brace the roof.
  Filament drops about 8% against a solid brick: much less than the 40% volume
  drop implies, because the interior surface it adds prints as solid skin.

### Does it build as one piece?

Every brick run states it plainly, in the command-line summary and at the top of
the guide:

```
builds as 1 piece
builds as 9 pieces (largest stray group: 11 brick(s))
```

Bricks join only top to bottom, so a model can come out as a main body plus
groups that nothing holds on. A single stray group larger than
`brick_stray_hard_bricks` (default 4) is a hard failure, whatever share of the
build it is: an eleven-brick tower running up the flank of a cone is a piece of
the model that was never attached to it, and as a fraction of a large build it
used to look small enough to pass. Smaller strays are reported as warnings,
never silently.

### How the tiler keeps a model attached

Bricks join only top to bottom, so the tiler tracks which connected group every
cell of the course below belongs to and chooses each brick by what it attaches:
a brick that spans several groups (it merges them) beats one that spans one (it
is merely supported) beats one that rests on nothing. On the bottom course, where
nothing is below, a brick that sits under cells of the course above beats one that
does not. Cells with nothing to hold them are tiled first, each in a brick that
also covers something attached, reaching inward from the edge.

Two findings are worth knowing. A pyramid's footprint is a diamond whose tip is a
single cell, which forces the first brick one way, which forces every later brick
the same way: every course came out as parallel stripes whose seams never crossed.
So courses may run crosswise to the one below, as in masonry. And the repair pass
tries every candidate bridge across a split instead of stopping at the first that
does not help.

What it cannot fix is a cell with nothing over it and nothing under it, such as
the interior of a one-brick-thick floor: it can only touch its neighbours sideways
and bricks never join sideways. Those are reported as stray bricks, and need the
shape changed (thickened or trimmed), not a cleverer tiler.

### When the shape itself has to change

Two opt-in flags, both off by default, for models that otherwise come out in
several pieces. Neither touches a model that already builds as one piece.

```bash
cleaveform split model.stl -o out/ --mode brick --simplify
cleaveform split fence.stl -o out/ --mode brick --keep-thin
```

`--simplify` takes the smallest change that attaches each group of stray bricks,
using the tiler's own report of what it could not hold: a sliver of a cell or two
is dropped; a flat group (a floor one brick thick) is thickened by a course so the
course above can span it; and a rail that meets a post across a seam gets the one
or two cells over (or under) the seam that let a brick cross it. The best result
over a few passes is kept, so it can never leave a model worse attached than it
found it.

`--keep-thin` is for features thinner than a brick cell, which the majority vote
deletes: a fence's rails, a handle's neck. It adds a cell back only along a path of
cells that each hold some of the model (`brick_thin_min_fill`, default 15 %), at
most `brick_thin_max_gap_cells` (3) long, and only where that joins two parts
otherwise apart.

Neither ever joins two separate objects: a gap with nothing in it is a real
separation however narrow, and a file holding two bodies comes out as two pieces
with every combination of the flags. Whatever was changed is counted and shown
under "Shape changes" in the guide and the manifest, and as a `shape_simplified`
warning. `cleaveform bricktest` takes the same flags, and shows `+Nc`/`-Nc` on each
row for cells added or removed.

### Testing brick mode on many shapes

Brick mode was first developed against one vase. To see how it does on
everything else, there is a catalogue of 30 generated shapes (solids, hollow
shapes, thin and overhanging parts, organic stand-ins and broken input files)
and a benchmark that scores them. It needs no printer and no model files.

```bash
cleaveform bricktest -o scores/                       # everything, at 6/8/12/16 mm bricks
cleaveform bricktest -o scores/ --only cone,pyramid   # a few shapes
cleaveform bricktest -o scores/ --category thin --sizes 8,12
```

It writes `scoreboard.html` (with a side-by-side picture of each shape, stray
bricks in red), plus `.csv`, `.json` and `.md`. Each row reports pieces and the
largest stray group, fidelity (overlap with the original, what is missing, what
was added), whether the centre of mass sits over the footprint, floating bricks
(bricks resting on nothing, which cannot be laid from the bottom up), brick
count and mix, filament, print time and plates. Shapes go through the same
repair the loader applies, so a damaged file is scored as the real tool would
treat it.

`tests/fixtures/brick_baseline.json` freezes the first scoreboard. A test fails
if any shape that built correctly then stops doing so.

### Why the clutch is designed the way it is

Five printed rounds went into this, and the first four all failed the same way,
so the reasoning is worth recording.

A stud and the cavity it enters are an external feature and an internal one.
Process bias moves them in **opposite** directions - an external feature prints
small, an internal cavity prints large - and each half of the fit has two
surfaces. So a designed interference loses about four times any bias:

```
actual interference  =  designed  -  4 x bias
```

FDM holds roughly 0.05 mm. That erases anything under 0.2 mm outright. The
entire first ladder ran from 0.19 down to -0.12 - a span narrower than the
machine's own error bar, which is why every rung on it felt identical and why
tightening the number never helped.

Two things follow, and both are in the design:

- **The interference is large** (0.35 mm default), so that it survives the bias
  instead of being consumed by it. A bad print then changes how firm the click
  is, rather than whether there is one.
- **The compliance is deliberate and located.** The gripping walls are thin
  (0.90 mm) so they act as the spring, the roof is thicker (1.60 mm) so it does
  not. In the earlier design this was backwards: the roof was thinner than the
  wall it anchored, so 79% of the flex came from the roof rotating - the
  intended spring was rigid and the actual spring was an accident nobody had
  specified or measured.

The interference is carried on ribs rather than on a whole-wall dimension. A
wall dimension accumulates every error in the part; a rib is a small feature
that can be tuned on its own. The ribs have a rounded lead-in so the stud never
meets a square step, which would shave or permanently set them on first
assembly - a second, independent way a designed press fit decays into a slip
fit.

### Calibrating the brick clutch

```bash
cleaveform calibrate --mode brick -o brick_fit_test/
```

Six bricks, about a quarter of an hour on one plate. Three rungs, two bricks
each so a rung can be pressed together on its own. Push a pair together, pull them apart,
and keep the rung that clicks and holds but comes apart by hand:

```json
{ "brick_clutch_interference_mm": 0.40 }
```

Every rung is the **same 3x1 footprint**; they are told apart by raised dots on
the short end wall. That is deliberate. An earlier version of this fixture
labelled its rungs by brick length - but length is also the largest single
influence on how stiff a brick's walls are (a 1x1 is a closed braced box, while
a 3x1 wall's nearest brace is a whole unit away), so it varied two things at
once and cleanly measured neither. The dots sit on the end wall rather than the
long walls because the long walls are the springs, and a bump moulded into a
spring changes the thing being measured.

The rungs step in tenths, not hundredths, for the reason above: a ladder finer
than the printer's own repeatability returns the same answer at every rung. The
ladder is short because both ends are already known - four rounds at 0.19 and
below all slipped, and past about 0.50 the wall stops springing and starts
fighting, which the tightest rung tells you without needing its own print.
- Printing studs-up means the roof bridges the cavity. Spans are short and the
  tubes support the middle on wider bricks.

## How it decides the assembly order

Not one part at a time. A part in the middle of a 2×2 grid of chunks touches
neighbours on two axes at once and cannot be seated by any single translation,
so adding parts sequentially deadlocks on most real models.

Instead the cuts are undone in reverse, deepest first. That produces the order a
person would use anyway (build each column, then push the columns together), and
every step is a single rigid push along one axis. Each push is checked for
collision by sampling the approach path against the parts already placed.

## What the numbers mean

Every `est_*` value is a heuristic from cleaveform, not slicer output. Your
slicer is authoritative.

Both print time and filament weight have been calibrated against five real
Bambu Studio slices spanning a wide range of shapes. `tests/test_estimation.py`
holds the measured data and fails if a change decalibrates either one.

Calibration also changed the model, not just its constants. Extruded volume is
infill plus a solid skin over the surface, because perimeters and top and bottom
layers print at full density whatever the infill is set to. Without that term the
model underestimates high-surface-area parts and overestimates bulky ones.

Fitting is two stages, in this order, because they are not interchangeable:

1. **Shell thickness fits against weight.** Weight converts directly from
   measured mass, so it is a clean read on the true extruded volume.
2. **Speed and layer overhead fit against time**, using that already-correct
   volume. Time conflates volume with per-layer travel overhead, so fitting the
   volume model against time first (which was tried) baked in a systematic ~20%
   overestimate of weight - accurate on the axis being measured, wrong on the
   one riding along with it.

Result: weight within 2% worst case, time within 16% worst case (spec bar: 25%).

`infill_factor` must match what you actually slice at. It is a slicer setting,
not a printer property, and the calibration assumed 15%.

Manufacturability findings are tiered, so a part is never rejected just for
needing support:

| Tier | Meaning | Examples |
|---|---|---|
| hard | rejects the run | exceeds the ceiling, not watertight, connector below minimum thickness |
| soft | penalised, still printed | needs support, long print, warp-prone footprint, large seam |
| warning | advisory only | thin wall detected, mating face on the build plate |

Thin walls are flagged, never silently thickened. For a functional model,
holding the dimension usually matters more than printability.

## Recovery, and the one thing it may not touch

When a cut produces a face too small for a connector, or a part that cannot be
inserted, that cut is ruled out and the decomposition is retried, bounded by
`recovery_max_attempts`.

Recovery may not change the joint clearance. A production run's clearance
schedule contains exactly one value, the calibrated one, and export refuses to
write parts made with anything else. Only an explicit `--experimental` run may
explore within `recovery_clearance_range_mm`, and it says so loudly.

## Caching

Stage results are cached on disk, keyed by the content hash of the stage input
plus the config fields that actually affect that stage. Keys chain, so changing
a parameter invalidates its own stage and everything downstream, and nothing
above it. On by default; the cache lives in `.cleaveform-cache`.

```bash
cleaveform model.stl -o out/ --no-cache      # recompute everything
cleaveform model.stl -o out/ --clear-cache   # wipe it first
cleaveform model.stl -o out/ --cache-dir /somewhere/else
```

Measured on a 5,120-face model at 660 mm, splitting into 12 parts:

| | Time | Why |
|---|---|---|
| Cold | 6.6 s | nothing cached |
| Identical rerun | 0.09 s | every stage reused, ~70x |
| Changed `joint_clearance_mm` | 6.0 s | cuts reused, connectors rebuilt |
| Changed `cut_fractions` | full recompute | the decomposition itself changed |

The clearance case is worth being straight about: it skips the decomposition,
but that is only about 10% of a run. Regenerating the connectors is roughly 60%,
and a new clearance genuinely means new sockets, so the saving is small. The
large win is the repeat run, which is the common case while you are iterating on
everything except the geometry.

A warm run is guaranteed to produce byte-identical output to a cold one, and a
test asserts it. That is less obvious than it sounds: the natural way to store a
mesh is PLY or STL, and both round-trip vertices through `float32`, shifting them
by about 1e-6 mm. Invisible on a print, but enough to make cached output differ
from freshly computed output and quietly break the determinism guarantee.
Meshes are stored as exact `float64` arrays instead.

## Configuration

Every tunable number lives in [`cleaveform/config.py`](cleaveform/config.py) and
nowhere else; a test enforces that no tuning constant appears inline in module
code. Override with a JSON file (`--config`) or the CLI flags.

```bash
cleaveform config          # print the active configuration
```

Values you are most likely to change: `joint_clearance_mm` (from calibration),
`design_ceiling_mm`, `peg_radius_mm` / `peg_depth_mm` / `peg_edge_margin_mm`
(if cut faces are being rejected as too small), and `max_parts` /
`max_cut_depth`.

## Layout

```
cleaveform/
  geometry/    module 0: repair, booleans, slicing, bbox, collision, units, analysis
  io/          module 1: STL load, unit policy
  decompose/   modules 4 and 4b: recursive partition, cut graph
  joints/      module 5: peg and socket generation
  validate/    module 6: constraint tiers, assembly planning
  export/      module 7: STL writer, manifest, assembly guide
  cache.py     module 7: content-hash cache, one layer per stage
  stages.py    cached wrappers around the expensive stages
  brick/       brick mode: occupancy, tiling, layout, plates, guide
  cost.py      module 10: the six-term normalised cost
  estimate.py  module 12: print time and material heuristics
  config.py    every tunable value
  pipeline.py  end to end, plus bounded recovery
  cli.py       entrypoint
```

The decomposition interface is deliberately narrow. A stronger optimizer (beam
search, ILP, a learned policy) can replace `decompose()` without any neighbouring
module changing.

## Tests

```bash
.venv/Scripts/python -m pytest
```

`tests/test_acceptance.py` maps one class to each Phase 1 acceptance criterion:
determinism, build-volume guarantee, watertightness, joint fit, constraint tiers,
recovery safety, offline operation, end to end. Fixtures are generated, not
committed by hand:

```bash
python tests/fixtures/generate_fixtures.py
```

## Known limits in Phase 1

- STL input only. Text and image input, OBJ, STEP and 3MF are later phases.
- Chunk mode and `peg_socket` only. No brick mode, dovetails or snap fits.
- A cut side that falls into several disconnected bodies stays grouped as one
  part, with a warning. Splitting it would create parts with no connector.
- Insertion checking samples the approach path rather than sweeping it
  continuously, so a thin interlocking feature could slip between samples.
  Raise `sweep_samples` if you are cutting something intricate.
- Print-time and weight estimates are calibrated on one printer (a Bambu H2D
  at 0.2 mm layers, 15% infill). On a very different profile, re-fit them.
