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

### P1 — `crates/vision`, the shared conv layer
- **`yolo/tests/p1_forward_pin.rs`** landed FIRST, before any refactor, so it
  captures the pre-change baseline: 45,696 cls/box logits pinned bitwise in both
  train and eval mode. This is the gate that matters — p8 pins only names/counts
  and p3 only backward-vs-forward consistency, so both stay green through a
  silent numeric shift.
- **`crates/vision`**: `ConvKernelIds::resolve` looks kernels up **by name**,
  killing the positional-index class of bug structurally (`wm-diamond` declares
  `K_NLC_NCHW = 8` in one module and `= 18` in another for the same kernel).
  Absent kernels → `NONE`, and `need()` panics naming the kernel.
- **Moved out of yolo, as pure relocations** (verified by diffing against the
  originals — only the const-import block, `ctx.ids.*` rewrites, and one doc
  link): `ConvBN`/`Bottleneck`/`C2f`/`SPPF` (900 → 895 lines) and the neck
  plumbing `Up`/`Cat`/`Acc` (previously private). yolo keeps its `PIPELINES` and
  frozen index order (its checkpoint contract), plus config/head/loss/assign/
  boxmath/nms/infer.
- Proof at every step: forward pin bitwise-identical, p8 297 tensors / 3,167,776
  params unchanged, p3 gradcheck green, full workspace suite 178/178.

### P2 — the depth kernel family
- 17 kernels, registry 180 → 200, all JIT-compiling on both backends.
- **ReLU cost nothing**: `leaky_relu(slope=0)` is exactly ReLU in both
  directions. ConvTranspose2d deferred (DPT needs it, ZipDepth doesn't).
- Tested by **adjointness**, not FD: every `*_dx` here is the adjoint of a linear
  op, so `<A(x),y> == <x,Aᵀ(y)>` holds to round-off rather than a tolerance, and
  it catches exactly this family's failure modes (dropped edge tap, transposed
  group index, off-by-one window). 14 tests.
- Backwards are all **gathers** (brain is atomic-free). Where inverting the index
  map is subtle, the candidate range is loose and each candidate re-evaluates the
  *forward's own* predicate — one definition, evaluated twice.

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

## Traps found by the gates (do not re-learn these)

- **`target` is a WGSL reserved keyword.** `masked_l1` would not parse.
- **The wgsl-cpu JIT rejects user-defined functions** — everything inlines into
  `main`. A `src_coord()` helper compiled on wgpu and hard-failed on CPU, i.e. it
  silently broke the two-backend invariant. Same reason `gelu_erf.wgsl` inlines
  its erf. Any new kernel must be helper-free.
- **`backend-cpu` binds fast paths BY KERNEL NAME** (`find("conv2d")`,
  `lib.rs:127-133`) into AVX2/winograd. Naming a grouped conv `conv2d` would
  silently inherit a DENSE path that ignores `groups` and computes wrong results
  with **no error**. Hence `conv2d_gd`, and hence it is ~1 ULP off `conv2d`
  (generic JIT vs vectorized summation order) — the test allows ≤4 ULP.
- **`AGENTS.md`'s "≤4 storage buffers/kernel" is already false**: `router_bwd`,
  `mla_scores` and `layernorm_dgamma` bind 5, and WebGPU guarantees 8.
- The p3 yolo gradcheck takes **~29 min** under contention; budget for it.

## Next

P3 — `crates/depth`: the ZipDepth model. Blocks compose from `crates/vision`
(`ConvBN` + the new `conv2d_gd`), plus the ZipDepth-specific modules
(`QARepBlock`, SE, `StripPoolingAttention`, `GlobalContextBlock`,
`MinimalMultiScale`/`CrossScale`, `LightweightSPPF`, `UltraLightFusion`,
`FastConvexUpsample`). `maxpool5` IS LightweightSPPF exactly.

Still needed before the model closes (identified while writing P2, not yet
built): a broadcast-add for StripPoolingAttention (`[B,C,H,1] + [B,C,1,W]`), the
GlobalContextBlock's per-image `[B,C,HW]x[B,HW,1]` weighted pool and its
`[B,C,1,1]` residual add (`bias_add` is `[C]` shared across N, so it does not
fit), a softmax over the **9 axis** for `FastConvexUpsample` (distinct from
`softmax_hw`), and a generic-scale `resize_nearest` for `MinimalCrossScale`
(`upsample2` is 2×-hardcoded).
