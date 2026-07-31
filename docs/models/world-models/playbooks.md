# World-model failure playbooks

When hitting one of these failure classes, follow the playbook instead of
improvising. Append newly discovered gotchas to the relevant section.

## 1. GPU test flakes (ANV / Meteor Lake, device 0x7d55)
- Gating test runs are `BRAIN_DEVICE=cpu` + `MOE_SKIP_GPU_TESTS=1` — GPU
  results never gate CI.
- Known driver bug: compute-compute pipeline barriers across SLICED (sub-range)
  descriptor bindings mishandled by ANV (root-caused in the vulkan-tiled-binding
  work). Workaround: serialize the affected dispatches into fenced batches
  (`submit` + `poll_wait` between them).
- A stale `DISPLAY=:0` without X auth breaks Vulkan device enumeration (up to
  SIGSEGV inside enumeration). Headless GPU work must run with `DISPLAY=`.
- Never absorb a driver bug by loosening a numeric tolerance.

## 2. wgsl-cpu JIT unsupported constructs
The CPU backend compiles the SAME WGSL via naga -> Cranelift. Its subset is
strict; violations are compile-time errors. KNOWN-UNSUPPORTED (append here):
- `%` on f32 (float modulo) — use `x - y * floor(x / y)`.
- `clamp(v, lo, hi)` — use `max(min(v, hi), lo)`.
- `floor(v)` — for `v >= 0` use `f32(u32(v))` (u32 truncation == floor there).
- Bare builtin vector values — index them (`gid.x`), never pass whole vectors.
- Local arrays of non-scalar or non-constant size.
- Workgroup kernels: only a single top-level `workgroupBarrier()`; no array
  locals inside workgroup kernels. (wm kernels are barrier-free by design.)
Rules: NEVER fork kernel text per backend. If a construct is truly unavoidable,
extending crates/wgsl-cpu becomes its own TDD unit. The gate that catches all
of this early: `cargo test -p brain-wgsl-cpu --test compile_all` (every registered
kernel must JIT-compile).

## 3. Gradcheck tolerance failures (fp32 noise vs real bug)
Global tolerances: eps=5e-3, atol=4e-3, rtol=8e-2 (crates/gradcheck). Ladder:
1. Rerun with different directions/seed. Intermittent => conditioning; shrink
   the test problem (fewer channels, smaller spatial), don't touch tolerances.
2. Look at failure structure: sign flip or clean factor (2x, missing term) =>
   real bug; return to implement with the diagnosis.
3. Marginal miss (just over atol): recompute ONE direction's analytic
   derivative in f64 on the host from the spec math; compare.
4. NEVER loosen the global tolerances. A per-check override needs reviewer
   sign-off + a note in the unit spec.
VQ/argmin caveat: finite differences must not cross an assignment boundary —
tiny-config gradchecks use well-separated codebook inits (see check_moe's
smoothing trick for the precedent).

## 4. OOM (30 GB box)
- Tests: tiny shapes only (<= 4x8x8-scale tensors); real-checkpoint runs are
  one-at-a-time, never concurrent with builds.
- Importers stream tensor-by-tensor; never materialize a whole 500M-param
  model twice.

## 5. Checkpoint downloads
- Preflight: `df` must show download_size*1.5 + 10 GB headroom.
- Raw downloads -> gitignored scratchpad/wm-checkpoints/; delete raw after
  conversion to .safetensors; converted artifacts also gitignored (out/).

## 6. minWM-derived training pitfalls (for when models train)
- Controllability is capped by action<->frame alignment precision — verify the
  dataset alignment invariant tests before suspecting model code.
- Small batch kills controllability before it kills loss: budget grad-accum.
- A world model looks uncontrollable early in training; don't diagnose "WASD
  ignored" as a bug before the overfit gates and alignment tests are green.
