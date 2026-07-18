# Depth workstream — status

Goal: monocular depth support in brain — train / quantize / infer **ZipDepth**
(6.1M, pure conv, MIT) on CPU/NPU/GPU, load open pretrained weights, and run
`brain depth --camera` / `--image` rendering colorized depth in an SDL window
in realtime. (Depth Anything 3 was in the original goal; it is **dropped**.) Reference material:
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
are params); **`conv_bias`** for every biased conv (NOT `bias_add` — see the trap
below); `scale_chan`/`mul`/`add2`/`concat2` for the attention gates and
residuals; `leaky_relu(0)` for ReLU; `mse_value_w`'s weighted-loss shape as the
precedent for `masked_l1`; the `gradnorm_sq` host-reduce split for every global
reduction; `crates/vision` for every conv block; `model::Model` + the blanket
`CheckModel` impl for P3's gradcheck.

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
- **`bn_eval` takes `x, mv, gb, out` — the SAME four buffers as `bn_train`**, with
  RUNNING stats in `mv`. It is NOT the collapsed `scale|bias` form; `sb` exists
  only for the fused `conv_act*` kernels. Binding `sb` reads binding 3 OOB and
  **SEGFAULTS** rather than erroring, because the CPU JIT compiles with
  `MemFlags::trusted()` — there are no bounds checks anywhere in that backend.
- **ZipDepth is NOT "grouped conv end to end"** — a wrong assumption of mine that
  `ConvKernelIds::need()` caught by naming the kernel. The stem, most
  QARepBlocks, SPPF, SE and GlobalContextBlock are all `groups=1`.
- **`bias_add` is a LINEAR-layer bias, NOT an NCHW conv bias.** It computes
  `out[idx] += bias[idx % n]` — `[M,N]` row-major with the biased dim TRAILING.
  In NCHW the channel is not trailing, so it silently indexes garbage (caught by
  a test failing at 7.24). Use **`conv_bias`** (fused conv + per-channel bias);
  every biased conv in ZipDepth is dense, so it covers all of them. `bias_grad`
  IS still the right backward, via the `[M=N, N=C*HW]` view + a host spatial
  reduce. All documented at `yolo/src/head.rs:14-29` — read it before wiring a
  biased conv.
- The p3 yolo gradcheck takes **~29 min** under contention; budget for it.

### P3 — `crates/depth` (in progress)

- **`config.rs` — the parameter layout, VERIFIED against both real checkpoints**,
  key-by-key and shape-by-shape, with no GPU and no forward pass:
  `zipdepth_base.pth` 235 float tensors + 43 counters → 0 missing / 0 extra /
  0 shape-mismatched; `zipdepth_base_npu.pth` 239 + 44 → likewise. Plus
  unconditional unit tests pinning the counts and element totals
  (6,802,927−43 and 6,801,324−44).
  This was built first on purpose: it is pure structure, so the config, channel
  derivations, `_pick_groups`, BN placement, bias decisions and naming are all
  proven before any arithmetic exists to debug on top of them.
  Param names are the **reference's own state_dict keys** (following yolo ↔
  Ultralytics), so import is a 1:1 name match rather than a translation table.
- **`fuse.rs` — RepVGG reparameterization.** Host-side only. Verified equivalent
  to the unfused three-branch forward against an independent host convolution,
  across residual / no-residual / downsample / grouped / depthwise.
- **`fold_bn` moved `crates/npu` → `crates/vision`.** It is a property of the conv
  block (what `Conv`'s eval path already does), not of quantization; it had zero
  imports and three consumers, only one of which is the NPU. The move lets
  `crates/depth` reuse it without depending on npu's OpenVINO runtime. npu
  re-exports it, so `topology.rs`/`sim.rs` are untouched.

- **`net.rs`** — PIPELINES + `ids()`. Registers **both** conv forms: ZipDepth's
  dense units (stem, most QARepBlocks, SPPF, SE, GCB — all `groups=1`) route to
  `conv2d` for the AVX2 fast path; only the genuinely grouped/dilated ones go to
  `conv2d_gd`. `ConvSpec::is_dense()` picks per unit.
- **`init.rs`** — deterministic init. BatchNorm is classified **structurally**
  (a sibling `running_var` exists) rather than by name pattern; misclassifying
  BN's scale as a conv weight inits it Gaussian-around-0 instead of 1, and the
  model then builds, runs, and trains to garbage. Asserted at exactly 43 BN
  groups. `head_half.bias` inits to **0.5**, not 0 — the output is ReLU'd, so
  zero parks every pixel on the flat side and the head gets no gradient.
- **`vision::Conv` is now spec-driven** (`ConvSpec{groups,dilation,act}`) — the
  deferred half of P1, done because ZipDepth needs grouped/dilated + ReLU while
  yolo needs dense + SiLU, and the ~900 lines around them are identical. One conv
  unit now serves both. yolo stays **bitwise identical** (forward pin 3/3).
  This gave `bn_eval` its first consumer: registered and tested for ages, but
  nothing dispatched it because yolo always fuses.

- **`blocks.rs` — `QARepBlock`** (15 of the base config's blocks): two
  `vision::Conv` units at `Act::None`, summed, plus a raw identity when
  `cin==cout && stride==1`, then one ReLU. FD-gated over residual /
  channel-change / stride-2, and — the property RepVGG exists for — the **fused
  single conv is checked against the block's own eval forward** (<2e-4), not just
  against `fuse.rs`'s host reference, since those two could agree with each other
  and both disagree with the model.
- **`vision::ConvNames`** makes the block's five tensor names data rather than a
  hardcoded `format!`. Three styles: `brain` (yolo/Ultralytics), `torch_conv_bn`,
  and `torch_seq(P, ci, bi)` — needed because inside an `nn.Sequential` torch
  indexes conv and BN **positionally** (`branch_3x3.0`/`.1`), with no `.conv`/`.bn`
  at all. `prefix` stays separate: it is the ActTap key and must equal the
  exported ONNX node name, which identifies the conv SITE, not its weights.

- **`blocks.rs` — `ChannelAttention` (SE)**: `hidden = max(dim/8,4)`, both 1×1s
  bias-free with no BN. The gate is `[N,C,1,1]` — **per image** — and
  `scale_chan` indexes by `(idx/inner) % c`, so the obvious `(c=C, inner=H*W)`
  would apply image 0's gate to the whole batch. Applying the reuse rule instead
  of writing a kernel: **`c = N*C, inner = H*W`** makes the index `n*C + c`,
  exactly the per-image gate. Pinned by a test where the two images have opposite
  sign. Backward reuses `mul` + `add_chan_bcast_dv` (which *is* the
  adjoint-of-a-broadcast) + `avgpool2d_dx`; the FD test checks `d_in` too, since a
  missing x-path is the classic SE bug and weight grads would not catch it.

- **`vision::BatchNorm`** — a standalone BN unit, because `MinimalMultiScale` is
  `x + BN(dw₁(x) + dw₂(x))`: ONE BN over the SUM of two convs, which a unit whose
  BN is welded to its own conv cannot express. `Conv` KEEPS its own deliberately —
  its BN is fusion-aware (the eval fast path is `conv_act_reg`, one kernel doing
  conv+BN+act from a collapsed `scale|bias`, where BN never runs as a separate
  dispatch), while a standalone BN can never fuse and has no tap site. What is
  shared and must not drift — the eps, the `mv`/`gb`/`mvg` layouts, the
  `bn_stats`→host→`bn_train` interleave — is documented at both sites.
- **`ConvSpec::norm`** (`Norm::{None,Bn}`) — a conv without its own BN. Needed by
  `MinimalMultiScale`'s branches (the reference has exactly one `bn`, over their
  sum) and SE's two 1×1s. At `Norm::None` the forward is a raw conv → act in both
  modes, and `param_list` omits the four BN tensors.
- **`blocks.rs` — `MinimalMultiScale`**: two `Norm::None` depthwise branches
  (dilation 1 and 2, both shape-preserving) + the shared BN + residual, no
  activation anywhere. FD-gated over both branch weights and the shared BN's gamma.
- Fixed `param_list` computing the weight as `[cout, cin, k, k]`; grouped convs
  are `[cout, cin/groups, k, k]`. The earlier blocks were dense, so it had not bitten.

- **`vision::Act::Sigmoid`** — a conv unit whose output is a GATE, not a feature
  map (`StripPoolingAttention` ends `Conv->BN->Sigmoid` and multiplies into `x`).
  sigmoid's uniform is `[total]`, same as silu's, so only `act_pair` grew a line.
- **`vision::ConvSpec::bias`** — a learned per-channel conv bias, dispatched as the
  fused `conv_bias`. Independent of `norm`, because `GlobalContextBlock.transform.0`
  is a biased conv followed by BN. `conv_bias` is DENSE-only (no groups/dilation in
  its uniform), so `bias => is_dense()` is asserted in the ctor. Every forward path
  now routes through ONE `conv_step` helper, so a biased unit cannot lose its bias
  on one of the three paths (raw / train / unfused-eval). The backward is
  `bias_grad` on the `[N, C*HW]` view + a host spatial reduce — **exactly what
  `yolo/src/head.rs:177-201` hand-rolls**; that copy is now refactorable onto this.
- **`blocks.rs` — `StripPoolingAttention`, `GlobalContextBlock`**.

### Blocks: 10 of 10 DONE

`QARepBlock` (x15), `ChannelAttention` (SE), `MinimalMultiScale`,
`StripPoolingAttention`, `GlobalContextBlock`, `LightweightSPPF`,
`MinimalCrossScale`, `UltraLightFusion`, `FastConvexUpsample` (both variants).

**Not one of the ten needed a new kernel.** Every one composed from `crates/vision`
plus the P2 family. Two additions to `vision` were forced, and both are genuine
generalizations rather than depth-specific hooks:
- **`Act::Sigmoid`** — a conv whose output is a gate, not a feature map.
- **`ConvSpec::bias`** — a learned per-channel conv bias (`conv_bias`), independent
  of `norm`. Its backward is `bias_grad` on the `[N, C*HW]` view + a host spatial
  reduce, **exactly what `yolo/src/head.rs:177-201` hand-rolls**; that copy is now
  refactorable onto it.
- **`SPPF::with_spec` + `NameStyle` + `SppfSpec`** — `LightweightSPPF` IS
  `vision::SPPF`, differing only in the hidden width (`c1/4` from the INPUT
  channels, not `c_out/2` from the output), ReLU vs SiLU, and torch vs brain names.
  The pool chain, the concat fold and the whole backward are shared verbatim, and
  yolo's forward pin stayed bitwise identical through the refactor.
- **`axpy`** registered — `x + 0.3*delta` and `a*nn + (1-a)*bi` both fall out of
  `out += s*in` (at `s = -1` it is also the subtract), so the scaled residual and
  the blend need no scale kernel.

### Findings

- **`GlobalContextBlock` carries two MATHEMATICALLY DEAD parameters.** The loss is
  exactly invariant to `context_weight.bias` (one scalar added to every position of
  the softmax axis — `softmax(z+b) == softmax(z)`) and to `transform.0.bias` (BN
  subtracts the mean right after: the classic `bias=False`-before-BN redundancy,
  left `True` upstream). Both are in the checkpoint and must load; neither can ever
  learn. Measured: shifting either by ±0.5 moves the loss by **exactly 0.0f32**, and
  ±5.0 by 2 ULP (softmax's max-subtraction round-off), where a live parameter would
  move it by ~850. **Neither can be FD-checked** — FD divides that round-off by
  2·eps and reports noise (fd = 1.5e-1 against an analytic 4.2e-5). They are pinned
  by the invariance itself, which is the stronger claim anyway.
- **Buffer sizing caused THREE separate out-of-bounds bugs**, all of the same shape
  and all invisible until a fixture broke the coincidence:
  1. `GlobalContextBlock.d_gap` sized `n*c`, but `hidden = max(c/reduction, 8)`
     EXCEEDS `c` whenever `c < 32`. Equal at `c=32`; heap corruption at `c=4`.
  2. `MinimalCrossScale` reused `d_l2h` (32 elements) as scratch for a 64-element
     add — the projection buffers are sized to the PROJECTIONS' outputs, which are
     neither scale's own numel.
  3. `FastConvexUpsample` used `acc` (the block's output size) to receive the grad
     wrt `c0`'s output, which lives on the half-res grid at `hidden` channels.

  The CPU JIT uses `MemFlags::trusted()`, so each was silent heap corruption
  (`free(): invalid next size`) or a SIGSEGV — never an error message pointing at
  the cause. **Standing rule: size every backward buffer from the PRODUCING unit's
  own `out_shape`. Never re-derive it from the block's nominal channel count.**
- **FD is the wrong instrument on a maxpool chain, and yolo's seed is a lottery.**
  SPPF's directional FD does not converge as eps shrinks (16.8, 11.9, 12.4, 20.5,
  30.1 at eps 1e-5..1e-2 against an analytic 11.4) — the signature of a kink, not of
  a wrong gradient. At the worst element the one-sided slopes are 0.83 / 1.42 and
  the analytic is 1.395: it matches the LEFT slope exactly and the central
  difference splits them. At yolo's own parameters (N=4, eps=5e-3) this config fails
  on **3 of 5 seeds**, while yolo's hand-picked `assert_grads(&h, 707, "sppf")`
  passes. `maxpool5` caches its argmax, so brain's gradient is the FROZEN-argmax
  gradient — a **subgradient**, which must lie within the envelope of the two
  one-sided slopes. That is what the test asserts, with a derived noise floor
  (`8*|L|*f32::EPSILON/eps`) below which the slopes are round-off.
- **An FD fixture must not carry a large constant.** `FastConvexUpsample`'s output is
  a depth map (~2.0 everywhere), so `sum(out*r)` with uncentered `r` came to -87.74,
  whose f32 ULP is 7.6e-6. The signal is ~1e-2, so `lp-lm` was FIVE ULPs and every FD
  value came out an exact multiple of the ULP — pure quantization, converging only
  at eps=1e-2. **Centering the loss weights** removes the constant that carries no
  gradient and restores ~4 orders of FD headroom. This will matter again for the
  real `ZipDepthLoss`.

### P3 — the ZipDepth model (COMPLETE)

- **All 10 blocks** compose into `ZipDepth` (`model.rs`): encoder (stem, 4 stages
  with their attention tails, SPPF, cross-scale) + decoder (proj4, 4 fusions, head,
  convex upsampler), forward + backward + `param_list` + eval/train toggle.
- **Layout matches the released `.pth` EXACTLY**, verified against the real file
  with brain's own reader — and the check is now three-way: `ZipConfig::param_list`
  (from the reference source), the BUILT graph's `param_list` (block-by-block), and
  the file all agree. base: 235 float tensors, npu: 239, zero missing/extra/mismatch.
- **Master gradcheck** (`p3_gradcheck`): both variants gradcheck end to end,
  element-wise on an eps ladder (directional FD is unusable on a deep post-ReLU
  depth loss — the analytic is right, proven by single-element FD at ~2%; the
  instrument needs per-magnitude eps). 15/16 spanning tensors at 86-100% / median
  <0.04. Sabotage-verified: a dropped dominant path spikes the median >1.0.
- **Import** (`import.rs`): loads a released checkpoint 1:1 by name (allowed to be
  trivial because the layout is pre-verified), strict — rejects the wrong variant
  and any missing/extra tensor by name. Env-gated test confirms the REAL
  `zipdepth_base.pth` loads completely and a model on those weights runs a forward.

**brain can now load pretrained ZipDepth and run it.** What remains before the
camera demo: P4 (data/train/eval), P5 (capture + HFSM + SDL CLI), P6 (NPU quant).

Buffer-sizing bugs found assembling the model (all the same class, all silent heap
corruption / SIGSEGV under the CPU JIT's trusted stores): `scale_chan` given a
2-field uniform (its ABI is `[total,c,inner]`, so it divided by inner=0 → SIGFPE on
the first op); `d_fhalf_acc` reusing a buffer sized to the model output (8192) for
f_half's grad (65536); and the recurring lesson holds — **size every backward
buffer from the producing unit's own `out_shape`.**

### P5 (partial) — the demo runs on real weights, in a window

`brain depth --image <ppm> --weights <pth>` loads pretrained ZipDepth and shows
RGB | colorized depth side-by-side (Esc quits, `[`/`]` cycle colormaps without
re-inference; `--headless` writes a PPM + content hash). **Verified visually on a
real photo: foreground reads red/near, background deep blue/far, smooth between.**

- `depth::viz` — Colormap (blue->red anchor ramp), robust p2/p98 Bounds (a lone
  specular spike must not swing the hue), side-by-side composite. 9 tests.
- `depth::Predictor` — letterbox in, eval forward, unwarp the depth onto the
  frame's own grid (bilinear), like `Yolo::detect`. 2 tests.
- `cli depth_cli` — the CLI, device-aware (`Gpu::new` honours `--device
  cpu|vulkan` / `BRAIN_DEVICE`; CPU output is bit-identical before/after the
  switch). Env-gated smoke test: deterministic headless run, 2x-width composite.
- Makefile: `depth/demo`, `depth/smoke`.

### P5 camera — `brain depth --camera` (V4L2/YUYV), structurally complete

`crates/capture`: hand-rolled V4L2 ioctl FFI (no `v4l`/`bindgen` dep), split so the
subtle parts are testable with ZERO hardware —
- `convert::yuyv_to_rgb` (BT.601), pinned: neutral chroma -> greyscale, +V pushes
  red, +U pushes blue (catches the classic R/B swap). Odd width panics.
- `slot::FrameSlot`: a single-slot latest-frame buffer (producer always wins,
  drops counted) — NOT a channel (mpsc rebuilds a backlog; sync_channel(1) stalls
  the producer and the driver silently drops with no counter). Send+Sync, lossy
  not blocking, thread-shared. 6 tests.
- `v4l2::Device`: mmap-streaming open/S_FMT(force YUYV)/REQBUFS/QBUF/DQBUF/STREAMON.
  Every ioctl number, struct size and field offset was PRINTED from a C program
  against the live `<linux/videodev2.h>` (never hand-computed) and `tests/abi.rs`
  re-derives them, failing on any drift. Byte-buffer + offset technique like
  wm-display's SDL_Event. Rejects an MJPEG-only camera by name.

The `--camera` loop: capture thread -> FrameSlot -> take-latest -> Predictor ->
EMA-smoothed bounds -> colorize -> composite -> SDL window, with a fps/infer-ms/drop
HUD. Esc quits, `[`/`]` cycle colormaps live.

**CANNOT be end-to-end validated in this environment** (the /dev/video* devices
exist but are permission-denied, and there is no display) — exactly the R4
hardware dependency the plan flagged. The pure pieces are fully tested, the ABI is
pinned against the live header, and the device-open path errors gracefully. **Needs
the user's actual laptop webcam to confirm it streams** (and that the cam exposes
YUYV, not MJPEG-only). GPU/NPU full-model execution is wired (`--device`) but
likewise untested here.

**Not started**: P4 (train/eval from scratch — the other half of the goal), P6
(NPU quant). **P7 (DA3) is DROPPED** — decided 2026-07-18; ZipDepth is the
depth model, full stop.

### P5.5 — GPU performance: ~3000 ms/frame → ~170 ms/frame (wgpu, Intel Arc MTL)

The demo ran at ~30 fps on the NPU but ~1–3 s/frame on the GPU backends — 50–100x
too slow. Measured first (single cold image, 719x467 → 608x384 model input,
contended box), then fixed TWO independent root causes:

| backend | cold baseline | steady state after | best-observed |
|---|---|---|---|
| cpu | 855 ms | ~350 ms median | 226 ms |
| gpu (wgpu) | 2901 ms | ~470 ms median | **170 ms** |
| vulkan | 9364 ms | ~520 ms median | 345 ms |

(The box is heavily contended — medians swing 2x between runs; the minima are
the signal. `brain depth --bench N` was added to measure steady state: the
single-image number conflates one-time model build + BN packing — 175 readbacks
on frame 1 — with per-frame cost.)

**Root cause 1 — the Vulkan backend did 2 blocking queue submits PER DISPATCH.**
`make_uniform` allocated every transient uniform DEVICE_LOCAL, so its zero +
upload each ran a one-off command buffer with submit + fence-wait (~600 blocking
GPU round trips per ~300-dispatch frame), plus a `vkAllocateMemory` per uniform
per frame, growing forever (a 30 fps camera leaks ~7k buffers+sets/s). Fix:
transient uniforms are HOST_VISIBLE (written by direct map — zero submits) and
recycled through size-keyed pools after each flush's fence-wait, descriptor sets
through per-pipeline pools ⇒ steady state is 1 submit + 1 readback copy per
frame. `backend-vulkan/tests/perf_contract.rs` pins all three properties
(step-building performs NO queue submits / O(1) submits per frame / transient
pool bounded across frames) via a `queue_submits()` counter, red→green.
Transient steps (`step`/`step_sliced`) are now submit-once by contract;
hold-and-resubmit code must use `uniform_dynamic` + `step_buf` (all in-repo
code already did).

**Root cause 2 — ZipDepth could not fuse a single conv.** The fused
`conv_act*` kernels hardcoded SiLU, so every one of ZipDepth's (ReLU) conv+BN
units ran naive conv2d + bn_eval + leaky_relu — three full-tensor passes, no
register tiling. Fixes, each red→green-tested:

- **Act selector in `conv_act`/`conv_act_reg`/`conv_act_tiled`** (`p.act`:
  0 identity, 1 relu, 2 silu, 3 sigmoid — an 11th uniform word; the branch is
  uniform ⇒ coherent). `can_fuse` is now act-agnostic, so every dense+BN unit
  (any act) is ONE register-tiled dispatch. yolo's SiLU path stays bitwise
  (forward pin green); fused==unfused pinned per act on CPU and on the real GPU
  (`depth/tests/p3_fused_eval.rs`). CPU uniforms are now padded to 16B like the
  GPU backends, so a grown Params struct reads 0 from a stale caller instead of
  out of bounds.
- **`conv_bias_reg` dispatched** (existed, never used): dense biased convs
  (head, GCB) take the register-tiled kernel.
- **`conv2d_gd_reg`** — NEW grouped/dilated register-tiled kernel with
  GROUP-ALIGNED 8x4 octets (all 8 lanes share the group ⇒ shared input loads;
  masked tail `nc = min(8, cout_g - oc*8)`; depthwise degenerates to 1x4).
  After the fusion round, `conv2d_gd` was **56% of a CPU frame** (the grouped
  1x1 fusion projections + dilated depthwise branches ran as scalar JIT loops).
- **CPU fast path for `conv2d_gd`** (`fast_conv::conv2d_gd`): depthwise gets a
  dedicated channel-parallel loop; general grouped runs the existing AVX2 GEMM
  per (image, group) on contiguous channel slices; im2col grew dilation. CPU
  binds `conv2d_gd` and `conv2d_gd_reg` to the same path (names exact-matched,
  so the dense-fast-path name trap does not apply). conv2d_gd: ~910 → ~18
  ms/frame on CPU; the depth master gradcheck itself dropped 281 s → 161 s.
- **`leaky_relu` CPU fast path** (~40 dispatches/frame ran as scalar JIT).
- GPU-vs-CPU parity for the grouped kernel pinned on real hardware over the
  straddle-prone shapes (grouped 1x1 cout_g=12, depthwise, depthwise dilated).

**Remaining follow-ups (not blocking the demo):**
- QARep RepVGG-fused EVAL path: `fuse.rs` already computes the collapsed single
  conv per block (verified); dispatching it in eval would halve the encoder's
  conv work + drop the add/relu dispatches.
- A proper GPU GEMM conv (im2col / vec4 loads) if low-tens-of-ms is needed at
  384-input; alternatively `--input 256` quarters the work.
- First-frame BN packing does 175 readbacks (one-time); could pack host-side at
  import instead.
- Vulkan `storage()` still zero-fills DEVICE_LOCAL buffers with one blocking
  submit each at model build (~hundreds, one-time per resolution).

### P6 (measurement done) — the INT8 decision, with a plan-changing finding

`brain depth calib --report` runs pretrained ZipDepth on real images and reports
each conv's `outlier_ratio = absmax / p99.99` — the INT8-hostility signal — with
NO NPU and NO OpenVINO, entirely on `--device cpu`. This is the plan's "run FIRST"
step, and it did its job: it **removed** work rather than added it.

**FINDING (real zipdepth_base.pth, 17 images):** ENCODER mean outlier_ratio 4.99,
DECODER mean 3.98. **The decoder is NOT the quant-sensitive part** — which
CONTRADICTS the QuartDepth prior (a ViT-L/DPT result) for this 6.1M pure-conv net.
The worst layers (ratio 15-21) are all in the ENCODER's deep downsampling path:
`down4`, `cross_scale.high_to_low`, and the GCB softmax `context_weight`. So the
plan's expensive FP-decoder ablation is **unnecessary**; the INT8 policy should
keep a handful of NAMED high-tail layers in FP, encoder or decoder alike.

Mechanics: `vision::Conv::apply_tap` — every conv now taps at its input, including
the Norm::None raw convs (fusion projections, head, GCB's raw convs) the eval path
skipped before (which would have blinded the decoder analysis). yolo's forward pin
stays bitwise identical. `depth::quant::ActStatsCollector` is an observe-only
ActTap. SE is the one intentional non-tap (raw conv2d on a [N,C,1,1] descriptor).

### P6 (fp32 NPU DEPLOYMENT DONE — the NPU was here all along)

The NPU IS available in this environment (Intel AI Boost, Core Ultra 7 155H,
`/dev/accel/accel0`, OpenVINO 2026.2.1 auto-discovered from the venv). An earlier
claim that it wasn't was WRONG.

**ZipDepth runs on the Intel NPU end to end with exact parity.**
`npu::depth_topology::build_depth_graph` walks `depth::ZipDepth`'s graph and emits
fp32 ONNX for the blend/where_conv (NPU) variant — its upsampler is
Conv/BN/Relu/Sigmoid/Resize/Mul/Add only (no unfold/softmax-9/pixel-shuffle), so
every op is a standard NPU node. `fuse_qarep` collapses each RepVGG block to one
biased 3x3, BN folds into every conv.

Measured (`tests/depth_onnx.rs`, env-gated on ZIPDEPTH_NPU_PTH):
  OpenVINO CPU: cosine(brain-CPU, ONNX) = **1.00000**, max|Δ| = 0.0000 (graph EXACT)
  Intel NPU:    cosine = **0.99998**, max|Δ| = 0.0003 (fp16 internal precision)
Every block matched brain's forward on the FIRST parity run. Visually confirmed on
the ZipDepth sample. `npu::tests::npu_live` separately proves a brain-built ONNX
compiles+runs on the device.

`brain depth --image <ppm> --weights <npu.pth> --variant npu --infer npu` runs the
demo on the NPU through the CLI. **The demo now runs on all three targets:** CPU +
GPU via brain's engine (`--device cpu|vulkan`), Intel NPU via `--infer npu`.

**REMAINING in P6:** INT8 quantization — emit QDQ convs (yolo's topology already
has the pattern) keeping the handful of high-tail layers (from the outlier report)
in FP, and measure the accuracy. The fp32 path is the harder, now-done part.

## Reference — the ZipDepth spec

Everything below is verified against the two released checkpoints
(`strict=True`, zero missing/unexpected keys).

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

**Gotchas already established** (all now enforced by tests in `config.rs`/`fuse.rs`):
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

## Reuse rules established this workstream

1. **Before adding a kernel, check whether an existing one covers it under
   degenerate arguments** — and settle it with a test, not by reading.
   `strip_pool` and `softmax_hw` were written, shown bit-identical to
   `avgpool2d`/`softmax_k`, and deleted.
2. **If you keep a duplicate, it must be earned and pinned.** `resize_nearest` vs
   `upsample2` is kept only because `upsample2` has a name-bound vectorized CPU
   path; a test pins that they agree so they cannot drift.
3. **A pure function must not gate a model crate on a hardware backend.**
   `fold_bn` had zero imports and lived in `crates/npu`; it moved to
   `crates/vision` so `crates/depth` could reuse it without inheriting OpenVINO.
4. **Structure before arithmetic.** The param layout was verified against the real
   checkpoint before a single kernel was dispatched.
