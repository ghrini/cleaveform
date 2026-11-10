# Cleaveform — Phase 1 Implementation Directive

**Project name:** Cleaveform (the tool splits an object into chunks and the chunks cleave back together). Package `cleaveform`, CLI command `cleaveform`.

**Read this first. Build only what this directive scopes.** The file `fabrication-platform-build-spec-v3.md` is architectural context — do not implement Phases 2–5 from it. Where this directive and the spec differ on Phase 1, this directive wins.

---

## 1. Directive

Build a deterministic, offline, CLI tool that takes a single 3D file (STL), automatically partitions it into pieces that each fit the Bambu Lab H2D print bed, adds peg-and-socket connectors at every cut, and outputs one STL per part plus a plain-text assembly guide. No machine learning, no network calls, no Gurobi, no robotics. It must be reproducible: same input + same config → identical output.

## 2. Exact Phase 1 scope

**In:** STL input; mesh repair/scale; deterministic recursive partition (chunk mode); peg-socket joints; chunk-mode buildability checks (collision-free insertion, connector-axis accessibility, graph-based assembly order); hard/soft/warning manufacturability tiers; bounded recovery that never changes production clearance; STL-per-part export + per-part metadata + text assembly guide.

**Out (do not build in Phase 1):** text/image generation; brick mode; StableLego/ASP; Gurobi; TRELLIS/Hunyuan3D; rendered instruction images; dual-material; generic/Bambu `.3mf`; UI; caching (optional, not required).

**Locked config (from spec §18):** `DECOMPOSITION_MODE=chunk`, `JOINT_TYPE=peg_socket`, `GENERATION_BACKEND=off`, STL export, one "balanced" cost preset.

## 3. Required stack

- Python 3.11+.
- `trimesh` (mesh IO/ops), `manifold3d` (robust booleans), `numpy`, `open3d` (repair/analysis), `build123d` or OpenSCAD via `solidpython` (parametric peg/socket solids), `click`/`argparse` (CLI).
- No GPU, no network, no solver dependencies.

## 4. Repository layout

```
cleaveform/
  geometry/        # Module 0: repair, booleans, offset, bbox, collision, units, voxel↔mesh
  io/              # Module 1: STL load, unit policy, CLI flag parsing
  decompose/       # Module 4 + 4b: deterministic recursive partition
  joints/          # Module 5: peg_socket generation
  validate/        # Module 6: chunk-mode checks + bounded recovery
  export/          # Module 7: STL writer, metadata, text assembly guide
  cost.py          # Module 10: normalized Phase-1 cost (part_count, bbox, cut-face area, support est)
  config.py        # all flags/thresholds; no magic numbers inline
  cli.py           # entrypoint
tests/
  fixtures/        # sample STLs (incl. one larger than the bed)
  ...
```

## 5. Deterministic decomposition rules (must implement exactly)

Design ceiling: `280 × 300 × 300 mm`. Recurse per part:

1. Compute the AABB.
2. If AABB ≤ ceiling on all axes → part is final; return it.
3. Else pick the axis with the greatest `dimension / ceiling_limit` ratio.
4. Generate candidate cut planes ⟂ that axis at **40%, 50%, 60%** of its extent.
5. Split each candidate via `geometry`; reject any candidate whose pieces are empty, non-watertight after one repair attempt, or smaller than `MIN_PART_MM`.
6. Score valid candidates with the Phase-1 cost only: `norm_part_count`, bbox feasibility, cut-face area, estimated support volume (all normalized).
7. Choose the lowest-cost candidate (ties → lowest index, for determinism).
8. Recurse on each resulting piece.
9. If `MAX_PARTS` or `MAX_CUT_DEPTH` is exceeded, stop and return a diagnostic report.

## 6. Connector rules (peg_socket)

At each cut face: pegs on the face on the "male" side, matching sockets on the "female" side. Socket radius = peg radius + `JOINT_CLEARANCE_MM`. Peg count/pattern derived deterministically from face area. All peg/socket walls and edge margins ≥ 2× nozzle width (0.8 mm @ 0.4 nozzle). No feature within the configured margin of the face perimeter. `JOINT_CLEARANCE_MM` is a config value set by the calibration test — never hard-coded, never changed by recovery in a production run.

## 7. Unit handling

STL coordinates = mm by default; always print the detected bbox. Reject implausibly tiny/huge models unless `--input-units`, `--target-size`, or `--scale-factor` is supplied. Honor those overrides.

## 8. Starter constants (initial defaults — put these in `config.py`)

Concrete starting values so the tool runs day one. These are FDM/Bambu-reasonable defaults, not final; the calibration test tunes clearance and the print-time model.

- `DESIGN_CEILING_MM = (280, 300, 300)`
- `NOZZLE_WIDTH_MM = 0.4` ; `MIN_WALL_MM = 0.8` (2× nozzle)
- `MIN_PART_MM = 5.0` (reject slivers below this on any axis)
- `MAX_PARTS = 50` ; `MAX_CUT_DEPTH = 8` (safety caps → diagnostic if exceeded)
- `JOINT_CLEARANCE_MM = 0.20` (initial; replaced by the calibration result)
- `RECOVERY_CLEARANCE_RANGE_MM = 0.05` (experimental runs only)
- Peg-socket defaults: `PEG_RADIUS_MM = 3.0`, `PEG_DEPTH_MM = 6.0`, `MIN_PEGS_PER_FACE = 2`, ~1 peg per 1500 mm² of cut-face area (cap at 6), `PEG_EDGE_MARGIN_MM = 5.0`
- Estimation: `MATERIAL_DENSITY_G_CM3 = 1.24` (PLA), `INFILL_FACTOR = 0.20`, `LAYER_HEIGHT_MM = 0.2`, `NOMINAL_SPEED_MM_S = 150` (coarse; calibrate to a real slice)

No numeric value appears inline in module code — all live here and are overridable.

## 9. Phase 1 acceptance tests

- **Determinism:** same STL + config → byte-stable cut set and part list across runs.
- **Build-volume guarantee:** every exported part bbox ≤ design ceiling. Test with a fixture larger than the bed (must split; parts must fit).
- **Watertightness:** every exported part is manifold.
- **Joint fit (physical gate):** print the two-chunk calibration cube at 0.10/0.15/0.20/0.25 mm; the smallest that assembles by hand without forcing/slop sets `JOINT_CLEARANCE_MM`.
- **Constraint tiers:** a part needing support (soft) is not rejected; a sub-thickness connector (hard) is.
- **Recovery safety:** production export uses the calibrated clearance regardless of recovery attempts.
- **Offline:** full run with no network and no Gurobi.
- **End-to-end:** `cleaveform input.stl -o out/` → per-part STLs + `assembly_guide.txt`, no manual step.

## 10. Output

Per run: `out/parts/*.stl`, `out/metadata.json` (per-part: id, bbox, est_volume, est_filament, est_print_time, orientation, supports_required), `out/assembly_guide.txt` (numbered steps: part, attaches-to, insertion direction, joint type). On failure: `out/diagnostic.txt` with the offending part/step and constraint.

---

**When Phase 1 passes all §9 tests, stop and hand back for review before starting Phase 2.**
