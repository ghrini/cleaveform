# Cleaveform — Handoff Package

**Project:** Cleaveform — a tool that takes a 3D file (later: text/image), splits it into
pieces that fit a Bambu Lab H2D print bed, generates snap/peg connectors at every cut, and
outputs printable STLs + an assembly guide. A human prints and builds. No robotics.

## How to use this package (read in this order)

1. **`phase1-implementation-directive.md`** — THE BUILD INSTRUCTION. Build exactly this and
   nothing more. It defines Phase 1 scope, the required stack, repo layout, the deterministic
   decomposition algorithm, connector rules, starter constants, and acceptance tests.

2. **`fabrication-platform-build-spec-v3.md`** — ARCHITECTURE CONTEXT ONLY. The full 5-phase
   product design. Use it as reference for anything the directive doesn't spell out. **Do NOT
   implement Phases 2–5 from it now.** If the directive and the spec disagree on Phase 1, the
   directive wins.

## Ground rules for the build

- Phase 1 is offline: no ML, no network, no Gurobi, no cloud.
- Deterministic: same input + same config → identical output.
- Every numeric value lives in `config.py` (starter defaults are in directive §8). No magic
  numbers inline.
- `manifold3d` is the boolean backend — mesh cutting/joint subtraction is the main technical
  risk; if geometry errors appear, look there first.
- The joint clearance (`JOINT_CLEARANCE_MM`) is finalized by a physical print test on the H2D,
  not by the agent. Build with the 0.20 mm default; the human calibrates after first print.

## Stop condition

When Phase 1 passes all acceptance tests (directive §9), STOP and hand back for review before
starting Phase 2.
