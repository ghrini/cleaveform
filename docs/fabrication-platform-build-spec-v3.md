# Fabrication Platform — Build Specification (v3)

**One-line goal:** Given either a text prompt ("a vase") or a dropped file (2D image or 3D model), produce a set of 3D-printable part files that a person prints on a Bambu Lab H2D and assembles by hand into the object, plus a numbered assembly guide.

**Status:** Master product specification and architectural source of truth. Phase 1 (§15) is the first implementation milestone and the **only required scope for the initial engineering build**. This document is context; the separate *Phase 1 Implementation Directive* is the build prompt.

**Release boundaries (unambiguous):**
- **Initial release / MVP = Phase 1 only.**
- **Product v1 = Phases 1–3.**
- **Future releases = Phases 4–5.**

Changes from v2 are in §17.

---

## 1. Scope

**In scope:** text→3D, image→3D, direct 3D file input; converting a 3D shape into printable, hand-assembleable parts; two representations (bricks, custom chunks); optimization of cuts (Module 4b, the differentiator); mode-specific validation with bounded recovery; export of per-part files + metadata + assembly guide.

**Explicitly out of scope:**
- Robotic assembly (no robot arms, motion planning, LeRobot, or APEX-MR). A human prints and builds.
- **Slicing / G-code.** Output is printable geometry; the user slices in Bambu Studio. All print-time/filament/support figures are **estimates** (§12), not slicer output.
- Cloud infra, auth, billing, multi-user. v1 is single-user local/desktop.

**Format-support caveats (do not overpromise):**
- **STEP** requires a CAD-kernel import path (OpenCascade via `build123d`/`cadquery`), tessellated to the Mesh contract before Module 3. `trimesh` alone does not reliably ingest STEP.
- **`.3mf`** support is phased (§ Module 7): generic 3MF geometry ≠ a full Bambu Studio project with printer-specific settings.

---

## 2. Two-pipeline architecture

Two near-independent products sharing one handoff. The Geometry Pipeline is fully useful alone, ships first, and needs no ML.

**Geometry Pipeline (ships standalone, no ML):**
```
[3D file] → (0) Geometry Kernel (used throughout)
          → (1) Input → (3) Repair/scale/voxelize
          → (4) Decomposition → (4b) Manufacturing Optimizer
          → (5) Joint generation → (6) Validation + recovery
          → (7) Export
```
**Intelligence Pipeline (additive):**
```
[text]/[2D image] → (2) 3D generation → mesh → hands off at Module 3
```
The pipelines connect only through the **Mesh** contract (§8). High-risk generative 3D lives entirely in the Intelligence Pipeline and can be delayed or swapped freely.

---

## 3. Glossary

Geometry Kernel (shared geometry ops); Voxel; Brick mode; Chunk mode; Joint/connector; Cost function (§10); Manufacturability constraints (§11, tiered); Assembly sequence; Clearance (calibrated mating gap). Hard constraint = reject if violated; Soft constraint = cost penalty; Warning = advisory only.

---

## 4. Target hardware constraints (fixed)

Printer: **Bambu Lab H2D** (dual-nozzle FDM).

| Property | Value | Consequence |
|---|---|---|
| Build volume, single nozzle | 325 × 320 × 325 mm | Absolute chunk ceiling |
| Build volume, dual nozzle | 300 × 320 × 325 mm | Ceiling with 2 materials |
| **Design ceiling to cut against** | **280 × 300 × 300 mm** | Margin for brim/warp/X-reduction |
| Nozzle diameters | 0.2/0.4/0.6/0.8 mm (0.4 default) | Min feature; walls ≥ 2× nozzle width |
| Dual nozzle | rigid+flexible, or dedicated support | Flexible snap-fits; clean support |
| Export target | Bambu Studio | STL always; 3MF phased |

**Tolerance:** `JOINT_CLEARANCE_MM` = **0.15–0.25 mm**, set by calibration (§13). Never hard-coded. Production export always uses the calibrated value (§ Module 6 recovery rules).

---

## 5. Module 0 — Geometry Kernel (shared)

Single library all geometry-touching modules depend on; prevents drift/duplication.
- **Responsibilities:** mesh repair (hole-fill, manifold, normals); booleans (union/diff/intersect); offsetting; fillets/chamfers; AABB & oriented bbox; collision/interference; unit & scale management; voxel↔mesh conversion.
- **Tools:** `trimesh`, `open3d`, `manifold3d` (robust booleans), `build123d`/OpenSCAD (parametric solids), OpenCascade (STEP).
- **Rule:** any geometry utility a second module needs lives here.

---

## 6. Modules 1–7

### Module 1 — Input handler
- **Inputs:** text; image (`.png`/`.jpg`); 3D file (`.stl`/`.obj`/`.3mf`/`.step`).
- **Routing:** text/image → Module 2; 3D file → Module 3 (STEP first through the CAD-kernel tessellation path in Module 0).
- **Unit policy (required — STL has no units):**
  - STL: treat coordinates as **mm by default**; always report the detected bbox.
  - OBJ: use imported metadata if present, else mm.
  - STEP / 3MF: **preserve declared units**.
  - Reject implausibly small/large dimensions unless the user confirms or supplies overrides.
- **CLI overrides:** `--input-units mm|cm|m|in`, `--target-size X,Y,Z`, `--scale-factor FLOAT`.

### Module 2 — 3D generation (Intelligence Pipeline)
- `GENERATION_BACKEND = mesh` → Hunyuan3D (text/image/multi-view) or TRELLIS (single-image, voxel-native); `= bricks` → LegoGPT (text→stable brick layout, fixed set, 20³ grid, brick mode only); `= off`.
- Cached on `hash(prompt|image, model, params)` (§7).

### Module 3 — Mesh processing / voxelization
- Clean, scale, optionally voxelize (via Module 0). Output: repaired watertight mesh + optional voxel grid.
- `VOXEL_PITCH_MM`: coarse → fewer chunkier parts; fine → more detail/parts.

### Module 4 — Decomposition (core)
- **Brick mode:** fill voxel grid with standard bricks → StableLego task graph `{brick_id,x,y,z,ori}` + `lego_library.json`.
- **Chunk mode:** partition mesh into pieces ≤ design ceiling → sub-meshes + cut graph.
- Candidates are scored by the **normalized** cost function (§10); Module 4b drives the search.

### Module 4b — Manufacturing Optimizer (differentiator)
- **Purpose:** choose/refine cuts to jointly optimize printability, strength, assembly, aesthetics — the concentrated IP.
- **v1 = deterministic reference algorithm (chunk mode):**
  1. Compute the AABB.
  2. If AABB ≤ design ceiling on all axes → part is final; return it.
  3. Else pick the axis with the greatest `dimension / ceiling_limit` ratio.
  4. Generate candidate cut planes ⟂ that axis at **40%, 50%, 60%** of its extent.
  5. Reject candidates producing empty, non-watertight (after one repair attempt), or sub-`MIN_PART_MM` pieces.
  6. Score valid candidates with the Phase-1 reduced cost: normalized part_count, bbox feasibility, cut-face area, estimated support volume.
  7. Select the lowest-cost candidate.
  8. Recurse on each resulting piece.
  9. Abort with a diagnostic if `MAX_PARTS` or `MAX_CUT_DEPTH` is exceeded.
- The interface is fixed so a stronger optimizer (beam search / ILP / learned policy) can replace this later without touching neighbors.
- **Output:** chosen decomposition + per-part recommended print orientation.

### Module 5 — Joint generation (chunk mode only)
- **`JOINT_TYPE`:** `peg_socket` (simplest; glue/friction), `dovetail` (captured, orientation-constrained), `snap_fit` (cantilever; best flexible).
- **Per cut (peg_socket reference):** place pegs on face A, matching sockets on face B; socket radius = peg radius + `JOINT_CLEARANCE_MM`; peg/socket walls and edge margins ≥ 2× nozzle width; keep features off the face perimeter by a set margin.
- **`DUAL_MATERIAL_JOINTS`** (with `snap_fit`): tag clip bodies for the flexible nozzle → dual-material `.3mf` (Phase 5).

### Module 6 — Validation + recovery (mode-specific)
Do **not** assume brick-specific research code accepts arbitrary chunks. Validation differs by mode:

- **Brick mode:**
  - Stability — StableLego (`intelligent-control-lab/StableLego`).
  - Assembly planning — PhysicsAwareCombinatorialASP (`intelligent-control-lab/PhysicsAwareCombinatorialASP`).
  - Both need **Gurobi** (free academic license).
- **Chunk mode (no brick tools):**
  - Collision-free insertion check per part.
  - Connector-axis accessibility (can the joint be engaged along its axis without obstruction?).
  - Contact/overhang stability heuristic for the joined structure.
  - Graph-based assembly ordering from the cut graph.
- **All modes:** manufacturability check against §11 tiers.

**Recovery (bounded, safe):**
- Soft-constraint failures → try alternate part orientation or an alternate valid cut via Module 4b within configured bounds, then retry.
- Trapped part → move seam / re-partition, retry.
- **Clearance is protected:** recovery may test values only within `RECOVERY_CLEARANCE_RANGE_MM` (e.g. ±0.05) and **only when the run is marked experimental**. **Production export always uses the calibrated `JOINT_CLEARANCE_MM`.** Prefer reserving clearance changes for calibration, not routine recovery.
- On exhausting the retry budget → return an **actionable report** (part/step, constraint, attempts).

### Module 7 — Export + instructions
- **Phased file output:**
  - **Required (Phase 1):** one **STL per part**.
  - **Optional/experimental:** generic **3MF per part**.
  - **Deferred (Phase 5):** fully arranged **Bambu Studio project** `.3mf` with printer-specific per-object settings and dual-material assignments.
- **Per-part metadata:** `{ id, material, est_weight_g, est_filament_m, est_print_time_min, orientation, supports_required, est_support_volume, recommended_layer_height, bbox }` (estimates per §12).
- **Assembly guide (per step):** `{ step_no, part_id, attaches_to, insertion_direction, rotation, joint_type, force_hint(optional/qualitative), image }`. `insertion_direction`/`rotation` computed from joint geometry; `force_hint` best-effort, never a blocker; rendered PNG per step (Phase 3).

---

## 7. Caching (cross-cutting)

Keys = content hash of stage input + affecting params. Layers: `mesh_cache`, `voxel_cache`, `chunk_cache`, `validation_cache`. Changing a param invalidates only its stage + downstream. `CACHE_ENABLED` toggle.

---

## 8. Data contracts

- **Mesh:** `trimesh.Trimesh`, mm, watertight, +Z up (sole cross-pipeline handoff).
- **Voxel grid:** `{ grid:np.ndarray[bool](X,Y,Z), pitch_mm, origin_mm }`.
- **Brick layout:** ordered `[{brick_id,x,y,z,ori}]` + `lego_library.json`.
- **Chunk set:** `[{ id, mesh, orientation, neighbors:[{id, cut_plane:[point,normal]}] }]`.
- **Assembly sequence:** ordered `[{part_id, pose, insertion_direction, rotation}]`.
- **Part metadata / Export manifest:** as in Module 7.

---

## 9. Configuration flags

| Flag | Values | v1 default | Owner decision |
|---|---|---|---|
| `GENERATION_BACKEND` | `mesh`/`bricks`/`off` | `off` | Which generator |
| `DECOMPOSITION_MODE` | `brick`/`chunk` | `chunk` | Bricks vs chunks |
| `VOXEL_PITCH_MM` | float | — | Detail vs part count |
| `JOINT_TYPE` | `peg_socket`/`dovetail`/`snap_fit` | `peg_socket` | Connector style |
| `JOINT_CLEARANCE_MM` | float | calibrated (§13) | Fit tolerance |
| `RECOVERY_CLEARANCE_RANGE_MM` | float | 0.05 | Recovery exploration bound (experimental only) |
| `AUTO_REPAIR_THIN_WALLS` | bool | `false` | Allow geometry-altering thickening |
| `DUAL_MATERIAL_JOINTS` | bool | `false` | Flexible clips |
| `COST_WEIGHTS` | α…ζ (§10) | see §10 | What "better" means |
| `MANUFACTURABILITY` | tiered thresholds (§11) | see §11 | Printability limits |
| `MAX_PARTS`, `MAX_CUT_DEPTH`, `MIN_PART_MM` | int/float | set | Decomposition bounds |
| `EXPORT_BUNDLE` | bool | `false` (Phase 5) | Bundled project |
| `CACHE_ENABLED` | bool | `true` | Cache on/off |
| `--input-units`/`--target-size`/`--scale-factor` | CLI | — | Unit handling |

Defaults exist only so the pipeline runs; they are starting points, not recommendations. Surface any run that depends on an unset owner decision.

---

## 10. Decomposition cost function (normalized)

All terms are made **dimensionless** before weighting, so weights are meaningful and no raw-magnitude term dominates. Lower is better.

```
Cost = α·norm_part_count
     + β·norm_support_volume
     + γ·norm_visible_seam_area
     + δ·norm_joint_count
     + ε·norm_print_time
     + ζ·norm_strength_penalty
```
Normalizations (examples):
- `norm_support_volume = total_support_volume / object_volume`
- `norm_visible_seam_area = weighted_visible_seam_area / total_surface_area`
- `norm_print_time = est_print_time / baseline_unsplit_print_time` (baseline = hypothetical single-piece estimate, even if unprintable)
- `norm_part_count = part_count / MAX_PARTS`; `norm_joint_count = joint_count / part_count`
- `norm_strength_penalty ∈ [0,1]` from cut cross-section vs. detected thin/load-bearing features.

Default weights (starting point; calibrate on §13 samples): α=1.0, β=0.5, γ=0.8, δ=0.3, ε=0.2, ζ=0.6. Presets (e.g. "structural" vs "display") are an owner decision.

---

## 11. Manufacturability constraints (tiered)

Separating "must reject" from "costs more" prevents rejecting printable models just because they need support.

**Hard constraints (reject if violated):**
- Exceeds build/design volume.
- Non-watertight geometry (after repair attempt).
- Connector/wall below minimum printable thickness (≥ 2× nozzle width).
- Feature below minimum printable size.
- Empty/degenerate/invalid geometry.

**Soft constraints (cost penalty, feed §10 — not rejection):**
- Overhang beyond 45° (needs support, still printable).
- Support volume / percentage.
- Seam visibility.
- Estimated print time.
- Warp risk (footprint aspect × material).
- Tall/thin aspect ratio.

**Warnings (advisory only):**
- Elephant-foot / first-layer compensation.
- Estimated material concerns.
- Recommended layer height.

**Thin walls:** never silently thicken user geometry. Default = **flag the thin region only**. Geometry is altered **only** when `AUTO_REPAIR_THIN_WALLS=true`, and every modification is recorded in the run report. For functional models, preserving dimensions can matter more than printability.

---

## 12. Print-time & material estimation (heuristic — NOT slicing)

Estimates for the cost function, metadata, and display; the user's slicer is authoritative.
- `est_volume` from mesh; `est_filament_g = est_volume_cm3 × material_density × infill_factor`.
- `est_support_volume` from overhang analysis (§11).
- `est_print_time_min ≈ f(est_volume, surface_area, est_support_volume, layer_height, nominal_speed)` — coarse, calibrated once against real Bambu Studio slices (§13) to fix the ratio. Labeled as estimates everywhere.

---

## 13. Acceptance criteria & tests

- **Tolerance calibration (Phase 1 gate):** two-chunk peg-socket cube at 0.10/0.15/0.20/0.25 mm; smallest that assembles by hand without forcing/slop sets `JOINT_CLEARANCE_MM`. Blocks later phases.
- **Determinism:** identical input + params → identical cuts (reproducible Phase 1 algorithm).
- **Build-volume guarantee:** every exported part bbox ≤ design ceiling (automated).
- **Manufacturability tiers:** hard violations reject; soft violations only penalize; warnings never block.
- **Estimation calibration:** est. print time within ±25% of a real slice on 3–5 samples.
- **Recovery safety:** production runs never deviate from calibrated clearance; experimental recovery stays within `RECOVERY_CLEARANCE_RANGE_MM`.
- **Cost-weight sanity:** each weight moves its own metric in the expected direction.
- **Cache:** unchanged upstream param → no recompute.
- **Watertightness:** every exported part manifold (assert on export).
- **Offline:** Phase 1 runs with no network and no Gurobi.
- **End-to-end per phase:** entry input → printable files + guide, no manual step.

---

## 14. External dependencies & references

- StableLego — `github.com/intelligent-control-lab/StableLego` (brick mode).
- PhysicsAwareCombinatorialASP — `github.com/intelligent-control-lab/PhysicsAwareCombinatorialASP` (brick mode).
- LegoGPT — `avalovelace1.github.io/LegoGPT/`, demo `huggingface.co/spaces/cmu-gil/LegoGPT-Demo`.
- Hunyuan3D (Tencent), TRELLIS (Microsoft) — text/image→3D, GitHub + Hugging Face.
- Libraries: `trimesh`, `open3d`, `manifold3d`, `build123d`/OpenSCAD, OpenCascade (STEP), `lib3mf`, `pyrender`.
- **Gurobi** — brick-mode solver (free academic license). Not needed for Phase 1.

---

## 15. Build phases (dependency-ordered)

- **Phase 1 — Spine (no ML), chunk mode:** 3D file → Module 0/3 → Module 4 chunk → Module 4b deterministic algorithm (§6) → Module 5 `peg_socket` → Module 6 chunk-mode checks + manufacturability + bounded recovery → Module 7 STL + text guide. Gate: tolerance calibration. Solves a real maker problem alone: import STL → auto-partition → connectors → guide → print. **No Gurobi, no StableLego/ASP, no network.**
- **Phase 2 — Estimation + metadata + caching + normalized cost wired into 4b.**
- **Phase 3 — Full validation depth + rendered stepwise instructions.**
- **Phase 4 — Intelligence Pipeline:** Module 2 `mesh` backend → feeds Module 3.
- **Phase 5 — Brick mode + polish:** brick backend + LegoGPT (+Gurobi/StableLego/ASP); dual-material snap-fits; dovetails; generic→Bambu `.3mf` bundling; stronger optimizer; UI.

---

## 16. Risks & mitigations

FDM fit (calibration gate); estimates vs real slices (calibrate to ±25%, label as estimates); optimizer scope creep (simple deterministic v1 behind stable interface); brick tools misapplied to chunks (mode-specific validation, §6/#10); silent geometry damage (thin-wall flag-only default); loose joints from recovery (clearance protected); non-manifold generative meshes (mandatory repair); Gurobi licensing (Phase 5 only); scope creep to robotics (fenced by §1).

---

## 17. What changed from v2 (second review integration)

1. **v1 vs Phase 1 disambiguated** — status + release boundaries at top.
2. **Deterministic Phase 1 decomposition algorithm** — Module 4b, step-by-step.
3. **Cost terms normalized** — §10, all dimensionless.
4. **Constraints tiered** — §11 hard / soft / warning.
5. **No silent thin-wall thickening** — flag-only default, `AUTO_REPAIR_THIN_WALLS` gate, logged.
6. **Unit policy + CLI overrides** — Module 1.
7. **STEP via CAD kernel** — §1 caveat + Module 0/1.
8. **`.3mf` phased** — Module 7 (STL required, generic 3MF optional, Bambu project Phase 5).
9. **Recovery clearance protected** — Module 6, `RECOVERY_CLEARANCE_RANGE_MM`, production uses calibrated value.
10. **Mode-specific validation** — Module 6; brick tools (StableLego/ASP) confined to brick mode; chunk mode uses collision/accessibility/ordering checks.

---

## 18. Recommended Phase 1 configuration (owner-confirmable)

A concrete lock so the first build is testable. Flip any before implementation:

- chunk mode only; `peg_socket` only; file input only; CLI; STL export required; generic 3MF deferred; no Gurobi; no StableLego; no ASP; no rendered instructions; no dual material; one "balanced" cost preset.

## 19. Open questions for the product owner

1. Confirm the §18 Phase 1 lock, or adjust.
2. Default cost-weight profile: single "balanced", or structural/display presets.
3. Which generation backend (Hunyuan3D vs TRELLIS) when Phase 4 begins.
4. How much intelligence lives in Module 4b (optimizer) vs. hard gates in Module 6 — the optimize-vs-reject boundary.
5. UI surface for later phases (CLI/desktop/web).
