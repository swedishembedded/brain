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

### P2 — the depth kernel family (COMPLETE for ZipDepth)
- **24 kernels, registry 180 → 207**, all JIT-compiling on both backends.
  `conv2d_gd{,_dx,_dw}`, `resize_bilinear{,_dx}`, `resize_nearest{,_dx}`,
  `avgpool2d{,_dx}`, `broadcast_add_hw{,_da}`, `softmax_k{,_dx}`,
  `weighted_gap{,_dx,_dm}`, `add_chan_bcast{,_dv}`, `pixel_shuffle{,_dx}`,
  `convex_upsample{,_dmask,_dd}`, `sigmoid{,_bwd}`, `masked_l1{,_grad}`.
  **Everything ZipDepth's default path executes now exists.**

- **ReLU cost nothing**: `leaky_relu(slope=0)` is exactly ReLU in both
  directions. ConvTranspose2d deferred (DPT needs it, ZipDepth doesn't).
- Tested by **adjointness**, not FD: every `*_dx` here is the adjoint of a linear
  op, so `<A(x),y> == <x,Aᵀ(y)>` holds to round-off rather than a tolerance, and
  it catches exactly this family's failure modes (dropped edge tap, transposed
  group index, off-by-one window). 18 tests.
  ⚠️ An adjoint identity only holds for the operator you actually applied: where
  the forward SUMS two broadcasts (`broadcast_add_hw`), each argument must be
  isolated by zeroing the other, or the test reports a broken adjoint for a
  correct kernel.
- Backwards are all **gathers** (brain is atomic-free). Where inverting the index
  map is subtle, the candidate range is loose and each candidate re-evaluates the
  *forward's own* predicate — one definition, evaluated twice.
- Fixed a latent **SIGPIPE race in `scripts/kernels-regen.sh`**
  (`grep | head -1` under `set -o pipefail`): it worked at 180/200 kernels and
  began aborting the regen at 211, reading as "nothing happened".

### Reuse audit — 4 kernels written, then deleted as redundant

Asked whether the new kernels duplicated ones brain already had, the answer was
partly **yes**, and it was settled by test rather than by reading:

- `strip_pool{,_dx}` is **bit-identical** to `avgpool2d{,_dx}` with a degenerate
  axis (`Ho=H,Wo=1` = mean over W; `Ho=1,Wo=W` = mean over H), backward included.
  **Deleted.**
- `softmax_hw{,_dx}` is **bit-identical** to `softmax_k{,_dx}` at `M=1` — a stride
  of 1 over `K=HW` *is* a contiguous softmax over the map. **Deleted.**
- `resize_nearest` also subsumes the pre-existing `upsample2` — but `upsample2`
  has a **name-bound vectorized CPU path** (`fast_ops::upsample2`,
  backend-cpu:418-425) that yolo's neck and wm-diamond run on, so both are kept
  and a test pins that they agree exactly at 2×. **That duplication is earned;
  the other two were not.**

**Reused rather than rewritten**: `maxpool5` *is* LightweightSPPF exactly (K/pad
are params); `bias_add` for the head; `scale_chan`/`mul`/`add2`/`concat2` for the
attention gates and residuals; `leaky_relu(0)` for ReLU; `mse_value_w`'s
weighted-loss shape as the precedent for `masked_l1`; the `gradnorm_sq`
host-reduce split for every global reduction; `crates/vision` for every conv
block; `model::Model` + the blanket `CheckModel` impl for P3's gradcheck.

**The standing rule this establishes:** before adding a kernel, check whether an
existing one covers it under degenerate arguments — and if you keep both, pin
their equivalence with a test so the duplication cannot drift.

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

## Next — P3, `crates/depth`: the ZipDepth model

The kernel layer is complete; this is now purely composition. Everything below is
verified against the two released checkpoints (`strict=True`, zero missing/
unexpected keys) — see `resources/depth-models/README.md` and the spec notes in
this repo's plan.

**Config (base/balanced, the 6.1M model):** `dims=[48,96,192,384]`,
`depths=[2,2,6,2]`, `dec_ch=96`, `half_dec_ch=32`.

**Module → kernel map** (all present):
| module | kernels |
|---|---|
| `ConvBN` | `conv2d_gd` + `bn_*` + `leaky_relu(0)` |
| `QARepBlock` | 2× `conv2d_gd` + `bn_*` + `add2` + relu. **Identity branch has NO BN** → its fuse adds no bias term |
| `ChannelAttention` (SE) | `avgpool2d`(1,1) + 2× `conv2d_gd`(1×1, no bias) + relu + `sigmoid` + `mul` |
| `StripPoolingAttention` | `strip_pool`×2 + `broadcast_add_hw` + depthwise `conv2d_gd` + `bn` + `sigmoid` + `mul` |
| `GlobalContextBlock` | `conv2d_gd`(C→1, bias) + `softmax_hw` + `weighted_gap` + 2× `conv2d_gd` + `bn` + relu + `add_chan_bcast` |
| `MinimalMultiScale` | 2× depthwise `conv2d_gd` (dilation 1 and 2) + `add2` + `bn` + `add2` |
| `MinimalCrossScale` | grouped 1×1 `conv2d_gd` + `resize_nearest` + `avgpool2d` + `scale_chan`(0.3) + `add2` |
| `LightweightSPPF` | `ConvBN` + 3× **`maxpool5`** (exact match, K/pad are params) + `concat2` + `ConvBN` |
| `UltraLightFusion` | `resize_bilinear`(align=**false**) + 2× grouped 1×1 + `add2` + `bn` + relu |
| `head_half` | `conv2d_gd`(3×3) + `bias_add` |
| `FastConvexUpsample` (unfold) | `ConvBN` + `conv2d_gd`(→36, bias) + `softmax_k`(K=9) + `convex_upsample` |
| `FastConvexUpsample` (npu) | 1×1 + `bn` + relu + depthwise 5×5 + `bn` + relu + 1×1 + `sigmoid` + blend of `resize_nearest`/`resize_bilinear` |

**Gotchas already established:**
- ImageNet normalize is **inside** the model; `mean`/`std` are `(1,3,1,1)` buffers
  **in the state_dict**.
- Output is `[B,1,H,W]` at **exactly** the input resolution, final **ReLU** →
  unbounded non-negative **inverse depth**, relative only.
- `_pick_groups(max_g=4)` must be replicated exactly — it determines weight shapes.
- `align_corners` is inconsistent **by design**: `false` inside the decoder,
  `true` in the predictor's final upsample to source resolution. Match both.
- The 6.1M figure is **post-fusion**; the checkpoint stores the 6.79M unfused form.
- Dead code — do not port: `decoder.forward`'s `size` arg, `edge_ch`,
  `use_half_res=False`/`head_direct`, `ZipDepthDecoder.fuse()`.
- Gradcheck at **eps=5e-4** (not 5e-3), per `yolo/tests/p3_gradcheck.rs:36-44`.

Then: P4 (data/train/eval), P5 (demo), P6 (NPU), P7 (DA3), P8 (docs).
