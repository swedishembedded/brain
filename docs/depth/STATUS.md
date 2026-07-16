# Depth workstream — status

Goal: monocular depth support in brain — train / quantize / infer **ZipDepth**
(6.1M, pure conv, MIT) and **Depth Anything 3** on CPU/NPU/GPU, load open
pretrained weights, and run `brain depth --camera` / `--image` rendering
colorized depth in an SDL window in realtime. Reference material:
`/data/workspace/resources/depth-models/` (read-only), in particular
`docs/brain-gap-analysis.md` (the verified gap audit) and
`docs/losses-and-eval.md` (formula-level losses + the honest-eval protocol).

Mode: direct implementation on this branch, gated by the repo's standard
discipline — `make build`, headless
`BRAIN_DEVICE=cpu MOE_SKIP_GPU_TESTS=1 make test` (with `DISPLAY=` unset: a
stale X11 DISPLAY breaks Vulkan enumeration), `make gradcheck`, tests written
from the specs in `docs/depth/specs/`.

## Done

### P0 — engine unblocks
- `wgsl-cpu`: `floor`/`ceil`/`trunc`/`round`/`fract`/`clamp`/`saturate`/`mix`/
  `sign` lowered to Cranelift (previously a hard `Err`, which blocked **bilinear
  resize** — needed by both ZipDepth's `UltraLightFusion` and DA3's DPT head —
  and would have broken the "same WGSL, both backends" invariant). 6 numeric
  tests; `round` is pinned to **ties-to-even**, where both `floor(x+0.5)` and
  `f32::round` (half-away) are silently wrong.
- `checkpoint::torchpt`: `LongStorage` (int64) → f32. Every `nn.BatchNorm2d`
  serializes an int64 `num_batches_tracked`, and the dtype error aborted the
  **whole file**, so *any* conv net with BN was unreadable. Both released
  ZipDepth checkpoints now parse: `zipdepth_base.pth` 278 tensors / 6,802,927
  elements / 43 `num_batches_tracked`; `zipdepth_base_npu.pth` 283 / 6,801,324 /
  44 — matching an independent Python-side inspection exactly. New env-gated
  `TORCHPT_REAL_PT` test reads a genuine torch file (every other fixture in that
  suite is synthetic, i.e. tests our *model* of torch's writer, not torch).
- `kernels`: **`gelu_erf_bwd`** — the exact-GELU derivative. `gelu_erf` had no
  backward at all, so anything training through it had to borrow `gelu_bwd`, the
  derivative of the *tanh approximation*. Measured: the two disagree by at most
  **8.7e-4**, which is under gradcheck's ATOL (4e-3) *on its own* — so
  `within(atol, rtol)` accepts the wrong derivative at **0/81 sample points, for
  any rtol**. The mispairing is not "usually missed"; it **cannot** be caught.
  DA3/DINOv2 uses exact GELU, so this had to exist before DA3 trains.

## Decisions

- **`gradcheck::directional_check_step` (per-tensor eps): considered, REFUTED,
  dropped.** The idea: `directional_check` perturbs every element at once, so the
  weight-space step is `eps·√numel` — hence yolo's hand-tuned `eps=5e-4`
  (`yolo/tests/p3_gradcheck.rs:36-44`). Holding the *step norm* constant instead
  looked like it would remove per-model tuning. It does not: GPT passes at
  constant `eps=5e-3` but **fails** at constant `target_step=0.05`
  (`blocks.1.attn.out.bias`, 16 elements, rel-err 12.3%) — the step grew from
  0.02 to 0.05 and the central difference degraded. Conditioning depends on local
  curvature, not step norm, so a constant step needs tuning too; it merely
  relocates the magic number. Revisit at P7.4 only if DA3's eps tuning is
  genuinely painful, with two real data points instead of one hypothesis.

## Next

P1 — `crates/vision`: lift the conv blocks out of `crates/yolo` (they are generic
but trapped behind hardcoded kernel-index consts, `blocks.rs:45-49`), resolving
kernel ids **by name** via `ConvKernelIds::resolve`. Proof obligations: yolo's
pinned fingerprint (297 tensors / 3,167,776 params) + p3 gradcheck + a new
forward-value pin all unchanged.
