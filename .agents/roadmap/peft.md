# peft - roadmap

brain has a correctness-oriented vanilla LoRA implementation (host-side
`model::lora::Pair` used by 8 diffusion/audio crates, plus a deliberately
separate device-side `ParamStore` `.lora_a`/`.lora_b` representation used by
qwen3/qwen35/qwen35moe/deepseek2/kronos/rl) but no generic PEFT abstraction
and none of the parameter-efficient methods that now define the field. This
campaign builds a shared `AdapterKind` substrate once (Phase 1, in
`crates/model/src/adapter/`) and stages every other method as a phase with a
falsifiable done-number, so "what does brain support" is derivable from the
`AdapterKind` registry and the `Adapter.kind` strings a loader accepts,
never a hand-maintained doc table (see lesson #54 - three such tables were
already found wrong in two directions).

Triggered by an external review that rated brain's PEFT support 4/10
overall (vanilla LoRA itself ~8/10) and found LoKr completely absent -
brain's third-party adapter loader (`model::lora::read_external_adapter`)
recognizes exactly three conventional LoRA key spellings and hard-errors on
anything else, so it cannot load a LyCORIS file at all.

---

## How this ledger was built, and what it is measured against

Every finding below was re-derived from the tree on 2026-09-07 by three
independent research/design passes (not carried forward from the external
review, which was treated as a hypothesis to verify, not a source of truth -
several of its specific file:line claims were wrong and are corrected where
relevant). Hardware this ledger's numbers are measured against: 2x Tesla P40
(SM 6.1, DP4A yes, no tensor cores, no bf16) + Xeon E5-2690 v3 (AVX2, no
VNNI) + ~177 GB RAM. Diffusion targets: FLUX.2 klein-4b (fp32 frozen base
fits one card at 13.93 GiB) and klein-9b (36.2 GB fp32 - does not fit one
card without quantization or sharding).

## Decisions this ledger encodes

1. **Diffusion first, LLM second.** Every milestone's done-number is
   measured on a real FLUX.2 klein block where possible, not a synthetic
   tiny config. A tiny config is the *gradcheck* harness, never the fidelity
   gate (lesson #40 - tiny configs can numerically starve a normalization op
   and produce a hollow-but-passing check).
2. **INT8 frozen base before NF4.** NF4 is a strictly harder version of the
   same problem (`Dtype::NF4` has no `Weight::` arm in `Ops::bind` at all,
   `crates/model/src/ops.rs:664`) and int8 is the one with a real
   measurement behind it (`crates/flux2/tests/int8_base_grads.rs`).
3. **A quantized training path is gated device-vs-fp32-device, not on
   `backend-cpu`.** `matmul_i8_dyn` is GPU-only. Lesson #5's both-backends
   rule stays satisfied by keeping the *fp32* path's existing both-backend
   gates green untouched, and gating the new int8 path against the fp32
   device path instead - stated explicitly in each such test's doc, never
   silently dropping a backend.
4. **Loading a third-party adapter format and training it are separate
   products.** Loading is a fold-once host operation with no perf
   constraint; training is a per-step device cost that for LoHa is ~2x the
   targeted linear's FLOPs. Ship loading (Phase 4) before committing to
   training any exotic format (Phase 8).
5. **One `AdapterKind` trait spans both LoRA families** (host `Pair` and
   device `ParamStore`), living in `crates/model` (layer 3, per
   `scripts/gates/check-crate-layers.sh`) - no new crate.
6. **No hand-maintained per-method docs column** (lesson #54). What brain
   supports is derived from code, not prose.

---

# Phase 1 - the generic adapter substrate (done, except the two gaps below)

Full design detail lived in the session plan that opened this campaign
(`AdapterKind`/`TargetSpec`/`TargetHp`/`AdapterSet`/`TargetSelector` in
`crates/model/src/adapter/`; migration of the 6 host crates with real
consumers onto it - supir, cosyvoice, s3dit, ltxv + av_lora, wan, flux2;
device param-list/role/save/fold kind-awareness; rsLoRA; LoRA+; LoRA-FA;
per-layer rank/alpha on the host substrate). Status and milestone-by-
milestone verification numbers are tracked in this repo's normal commit
history and test suite, not duplicated here - see `crates/model/src/adapter/`
for what has landed. Phases 2-9 below assume Phase 1 exists and is stable.

**Two things scoped out of Phase 1, on purpose, and why:**

1. **`minimaxmusic3`'s three LoRA modules were not migrated.** They are a
   genuinely different substrate today (no Adam, no alpha, no
   serialization, `delta` allocates a fresh `Vec` instead of accumulating
   in place) with zero consumers outside their own tests - migrating them
   would change what those tests measure for no product benefit. Cheap
   whenever it happens: their `name -> shape` functions are already a
   `linear_sites()` body.
2. **Three capabilities have real, tested substrate support but no wired
   consumer, and were deliberately left unwired rather than built as
   unreachable code ahead of one:**
   - **Per-target rank/alpha on the DEVICE family** (qwen3/qwen35/
     qwen35moe/deepseek2/kronos) - `AdapterPlan`/`AdapterSet::
     build_planned` exist and are tested on the host substrate
     (`crates/model/tests/adapter_per_target_rank.rs`), but none of those
     four crates' `LoraCfg` exposes anything but a single scalar `rank`,
     so there is no real caller to size mixed-rank device scratch buffers
     against yet.
   - **Device-side LoRA-FA** (`Role::Frozen` specifically for `.lora_a`,
     leaving `.lora_b` trainable) - the host substrate's `freeze_a` is
     structural (`Pair::freeze_a`, gated by
     `crates/model/tests/adapter_lora_fa.rs`), but none of the four
     device-family crates' configs expose a `freeze_a` flag.
   - **LoRA dropout on the device family** - the host path now hard-errors
     on a nonzero `TargetHp::dropout` (`crates/model/tests/
     adapter_dropout.rs` - `LoraPair::project` consumes a dense
     `dL/dW_eff` and never sees the adapter's input `x`, so masking it
     there is not expressible), and the device family is where it
     genuinely belongs (a trainer with `x` at the adapter's input can mask
     it before `A·x` and re-derive the identical mask in the backward's
     recompute of that same activation) - but building it means a NEW
     WGSL kernel (a counter-based mask hash, no PRNG kernel exists in this
     tree today) plus touching qwen3/qwen35/qwen35moe/deepseek2's
     `lora_fwd`/`proj_bwd` dispatch, for a config knob nothing sets yet.
   All three are real work, not abandoned - each gets a config field (a
   per-crate `LoraCfg.freeze_a: bool` / per-target rank map / `dropout:
   f32`) and its own device-side wiring the day a caller (most likely
   M12's CLI/capability surface, or self-improve's LoRA-in-RL-loop
   workstream) actually needs one. Building the device dispatch first,
   with no caller to gate it against, is exactly the "ported and
   unit-tested is not reachable" shape this campaign's own decisions
   section (#5, no hand-maintained support claims) argues against
   documenting as done.

---

# Phase 2 - INT8 frozen-base training (the priority; the long pole)

**Unlocks:** klein-9b LoRA on ONE 24 GiB card; first-step latency from >1 h /
~100 GB RSS to minutes / <8 GB RSS (`flux2::finetune`'s current
`read_dit_tensors` materializes the whole Q8_0 DiT as host fp32); the
prerequisite machinery (transposed quantized copy, quantized-base backward
seam) for every later quantized milestone (NF4/QLoRA, LoftQ, DoRA-on-int8).

## What the math actually is

Forward, per targeted linear, is **unchanged** from the existing inference
path:
```
y[t,o] = sx[t] * sum_g ( sum_{i in g} xq[t,i]*wq[o,i] ) * sw[o,g]      (matmul_i8_dyn)
```
`sw[o,g]` factors out of the DP4A accumulator because `g` indexes the
**contraction** axis - that is the whole reason the forward works unmodified.

Backward w.r.t. the activation contracts over the *output* axis instead:
```
dx[t,i] = sum_o dY[t,o] * W[o,i]
```
`sw[o,g(i)]` now depends on the contraction index `o`, so it cannot leave the
sum. **The fix is not a new kernel - it is a second, transposed operand.**
Define `Wt[i,o] := W[o,i]` (shape `[in, out]`, row-major) and requantize it
group-wise: `(wtq, swt) := int8::quantize_weight(Wt, in, out)`. Then running
`matmul_i8_dyn` with `x := dY` (`M=t, K=out`), `w := wtq` (`N=in, K=out`),
`sw := swt` computes exactly `dx[t,i]`, with the group scale now indexed by
`i` (this GEMM's output) and `g` running over `o` (the contraction axis).
Constraint: `out` must be a multiple of 32 (true of every klein/LTX/Wan
linear).

## Milestones

### M2.0 - Measure the ACTIVATION-quantization term on its own. KILL GATE.

`crates/flux2/tests/int8_base_grads.rs` already measures the *weight*-
quantization term (worst adapter-gradient cosine 0.999530, rel_l2 3.076e-2)
but explicitly does not cover activation quantization - and the existing
roadmap note said this number is missing and required before an int8
trainer can be trusted. Build it first; it is cheap and can kill the phase.

**Design.** Take one real klein-9b `double_blocks.0`. Quantize `W` once
(`group_scales`/`pack_row`, GROUP=32), then dequantize it back to f32 and use
*that* as a common reference base. Run (A) fp32 devgrad on the dequantized-
int8 weights, and (B) the int8 devgrad on the packed form of the identical
weights (`matmul_i8_dyn` forward + the transposed-copy `dx`). A and B
multiply by bit-identical weight values, so every difference between them is
the activation-quantization term plus DP4A re-association - the isolated
number nothing else in this repo can produce.

New test: `crates/flux2/tests/int8_act_grads.rs`, `#[ignore]`, needs
`BRAIN_DEV_GPU=1` + `BRAIN_FLUX2_DIT`. Reports, per targeted linear, cosine
**and** rel_l2 (never cosine alone - it is scale-invariant and cannot see a
dropped scale factor) on `dA`, `dB`, and the block input `dx`, plus a
three-way table: A-vs-B (activation term), A-vs-fp32-original (weight term,
cross-checking the existing 3.076e-2 figure), B-vs-fp32-original (total).

**Done number, and a real kill gate:**
- PROCEED if the isolated activation term is worst-case cosine >= 0.9990
  and worst-case rel_l2 <= 5.0e-2, and the total (B vs fp32) is worst-case
  cosine >= 0.9985 / rel_l2 <= 7.0e-2.
- **KILL the int8 trainer** if the total is worse than cosine 0.99 or
  rel_l2 1.5e-1 - at that point the existing two-card fp32 split
  (`devtrain::new_multi`, bit-identical-gated) is the right answer, and this
  ledger records why rather than shipping a lossy trainer nobody asked for.
- Also tap `dY` with `model::actstats::Collector` (`outlier_ratio =
  absmax/p99.99`) in the same test: if `dY`'s outlier ratio is far worse
  than a normal activation's, the fallback is a per-token-per-group scale
  for `dY` only, or keeping `dx` in fp32 for the last N blocks (error grows
  toward the front of a block per the existing weight-term measurement, so a
  mixed ladder is defensible).

### M2.1 - The transposed group-wise quantizer, streamed

**Build (host, no WGSL):** `model::int8::quantize_transposed_from(source,
name, n, k) -> Option<(Vec<u32>, Vec<f32>)>`, returning the `[k,n]` packed +
`[k, n/32]` scales for `W^T`.

**The streaming trick that bounds memory.** Group `g` of `W^T`'s row `i`
covers exactly one 32-row block of the source (`W[32g..32g+32, i]`). Stream
the source in GROUP-sized (32) row blocks: each block yields, for every
column `i`, one complete scale `swt[i,g]` and 8 packed words at
`wtq[i, 8g..8g+8]`. Peak host allocation is `32*k` floats plus the output -
never `n*k`. Note the existing Q8_0 byte-repack fast path
(`DitWeights::try_i8_rect`) cannot serve this direction (its blocks run
along rows, not columns) - the 32-row-block route dequantizes Q8_0 to f32 32
rows at a time instead, which is the bounded fallback.

**Done number:** a host unit test in `crates/model/src/int8.rs`'s `mod
tests` asserting `quantize_transposed(W)` is bit-identical (`assert_eq!` on
raw `u32`/`f32` bits) to `quantize_weight(transpose(W), k, n)` for randomized
`[n,k]` at `n,k in {32,64,96,3072}`. Plus a `#[ignore]`d real-checkpoint run
asserting peak RSS stays under 8 GB while converting the whole klein-9b DiT.

### M2.2 - Wire int8 into `flux2::devgrad`, forward and dx

`devgrad::LinDev`'s weight field gains an `I8 { wq, sw, wtq, swt }` arm
alongside the existing fp32 one; `BlockDev::lin` gains an `lin_i8` sibling
that calls `int8::upload_quantized` + the new `upload_quantized_transposed`,
so the trainer never materializes an fp32 `[out,in]` at all (removing the
existing >1h/~100GB-RSS dequantize). `lin_fwd`'s first step becomes
`I8Scratch::quant_rows(x)` + `mm8_rows_off` (the existing production int8
GEMV helper, already used by `flux1::model`/`flux2::model`); `lin_bwd`'s
`dx = dy*W` step becomes the same call against the transposed pair. The
low-rank adapter steps (`xa = x*A^T`, `y += xa*B^T`, `dA`, `dB`) stay fp32 -
rank-width, trainable, tiny.

**Accumulate, first cut:** `matmul_i8_dyn`'s epilogue has no accumulate
flag. Write into a small `[m, inn]` fp32 scratch and follow with the
existing `add_inplace` kernel rather than adding an accumulate flag to the
int8 kernel's contract up front - zero kernel changes, ships fast, gateable
immediately. Revisit only if a profile shows the extra pass matters.

**Identity-at-init.** At `B=0` the adapter contributes exactly `0.0` in
every precision tier, unchanged from the existing fp32 LoRA path - so the
existing `a_fresh_adapter_is_a_device_no_op`-style assertion must be checked
against the **int8 base's own loss**, bit-for-bit, never against the fp32
base's loss (that is a fidelity question, not an identity one - conflating
them produces a gate that either can't pass or doesn't test anything).

**Done numbers:**
1. `crates/flux2/tests/dev_grad.rs` gets an int8 sibling gated at whatever
   M2.0 measured (proposed: worst cosine > 0.999 and worst rel_l2 < 5e-2,
   both printed) - **not** the existing fp32 gate's 0.9999999/1e-5, which is
   an exact-arithmetic bar int8 cannot meet. The existing fp32 gates stay
   untouched on both backends.
2. `assert_eq!` bit-identity between an int8 run with no adapter and one
   with a freshly-initialized (`B=0`) adapter.
3. The existing honest-learning-bound shape (`early < 0.95*l0`, `lmin <
   0.7*l0`, `last < 0.85*l0`) reproduced on the int8 path.

### M2.3 - Memory budget, single card, klein-9b

**Budget:** two int8 copies of 9.05 G params = 18.10 GB packed **+ 2.26 GB
of f32 scale planes = 20.36 GB** (group-wise int8 costs 1.125 B/elem, not 1
- do not drop the scale-plane term), + ~2.4 GB activations at 1536 joint
tokens + a small dx scratch + adapter/grad (~0.3 GB). ~23.1 GB against a
25.77 GB card - under 3 GB headroom, and wgpu's per-buffer staging overhead
eats into that further.

**Done number:** a `#[ignore]`d test that builds the klein-9b `DeviceTrainer`
on one card, runs 3 steps, and asserts measured resident VRAM < 23.5 GB,
step 1 reached in < 10 minutes from a cold Q8_0 GGUF, peak host RSS < 8 GB -
the three numbers that together prove or disprove the memory case for
building this phase at all.

### M2.4 - `--cards N` composition

int8 is orthogonal to the existing block-stack sharding (`devtrain::
new_multi`) - it changes what each `LinDev` holds, not which card holds it.
Gate: the existing two-card-split bit-identity test must still pass on the
int8 path (int8 quantization is deterministic host arithmetic and DP4A
accumulation is exact integer arithmetic, so this is a stronger and
achievable claim, not a relaxed one). With int8, `--cards 2` on klein-9b
frees ~11.5 GB/card - headroom for ~2x the token count.

### M2.5 (optional, measure before building) - streamed transposed scratch

**Hypothesis:** the second copy need not be resident - rebuild `W^T` for one
block at a time, on device, just before that block's backward. Drops
memory from 20.36 GB to ~11.5 GB at an estimated ~2% step-time cost. **Not
the first cut**: requantizing from the already-quantized `wq` is a second
rounding (expect ~sqrt(2)x M2.0's error), and it needs a real new kernel - a
device group-wise dequantize+transpose+requantize with a per-32-row-block
reduction, which needs the barrier-free + cooperative kernel pair per lesson
#5. Build M2.1's resident copy first; only take this if M2.3's headroom
proves too tight in practice.

**Phase 2 difficulty:** medium-high. No new GEMM kernel, one new host
quantizer, one enum arm, ~6 dispatch sites. The two real risks are (a) `dY`'s
dynamic range (M2.0 kills or passes it), (b) the ~3 GB headroom.

---

# Phase 3 - DoRA

**Why here:** highest value-per-line method after LoRA for diffusion
(consistently reported better than LoRA at equal rank on style/subject
fine-tunes), and it decomposes algebraically into GEMM shapes brain already
has - far cheaper than a naive "materialize `W+BA` and take its row norm"
reading would suggest. Depends on Phase 2 only for its int8 arm (M3.3).

## The math brain implements

Match **PEFT's** convention (per-output-row magnitude, `m in R^out`,
reducing the *input* axis of a `[out,in]` weight) rather than the DoRA
paper's literal per-column form - PEFT/diffusers/ComfyUI interop decides it,
and getting this axis wrong is a uniform mis-scale that cosine cannot see.

```
W' = diag(m/n) * (W + s*B*A),   s = alpha/r,   n[o] = ||W[o,:] + s*(BA)[o,:]||_2
y  = x*W'^T = (m/n) elementwise_col ( x*W^T + s*(x*A^T)*B^T )
```
The forward never needs `W'` - it is the ordinary LoRA output rescaled per
output channel. Only `n` needs an `[out,in]`-shaped reduction, and **it
decomposes because `W` is frozen**:
```
n[o]^2 = c[o] + 2s*sum_k B[o,k]*P[o,k] + s^2*sum_k B[o,k]*(BG)[o,k]
  c  = rowsum(W elementwise W)   [out]     - CONSTANT, computed once at load
  P  = W * A^T                   [out, r]  - existing GEMM shape
  G  = A * A^T                   [r, r]    - existing GEMM shape, tiny
  BG = B * G                     [out, r]  - existing GEMM shape (G symmetric)
```
Backward, with `g = m/n`, `u[o] = dL/d(n^2)[o] = dL/dn[o] / (2n[o])`:
```
dy'      = g elementwise_col dy
dm[o]    = (sum_t dy[t,o]*(y_base + s*y_lora)[t,o]) / n[o]
dL/dn[o] = -dm[o]*m[o]/n[o]
dB      += 2*u elementwise_row ( s*P + s^2*BG )
dA      += 2s*( (u elementwise B)^T * W ) + 2s^2*( ((u elementwise B)^T * B) * A )
```
plus the ordinary LoRA `dA`/`dB` contribution from `dy'`. `(u elementwise
B)^T * W` is exactly the existing `matmul_dw`-shaped GEMM with no transpose
of `W` needed.

## What exists vs what must be built

| Need | Status |
|---|---|
| `P=WA^T`, `G=AA^T`, `BG=BG`, `(u.B)^T*W` | exists - existing GEMM/`mm_rows_off` shapes |
| `sum_k B[o,k]*P[o,k]` (both quadratic terms) | exists - `row_dot.wgsl` |
| per-output-channel forward scale + its gain gradient | exists - `scale_chan.wgsl` / `scale_chan_dg.wgsl` (`inner=1, C=out`; `dx` is the forward kernel itself) |
| the `.lora_m` tensor in the param list | near-free - the `lin` param-list closure already emits a variable tensor count per leaf; the role filter and save filter are the same `ends_with` shape a third suffix slots into |
| `n = sqrt(c + 2s*d1 + s^2*d2)` and `g = m/n` | **new kernel** |
| the `dm -> dn -> u` chain | **folded into the same kernel's backward** |

**New WGSL: exactly two small kernels**, both one-thread-per-output-row, no
reduction, no barrier - lesson #5's barrier-free/cooperative pair
requirement does not apply to either:
- `dora_rownorm.wgsl` - `n[o] = sqrt(max(c[o]+2s*d1[o]+s^2*d2[o], eps));
  g[o] = m[o]/n[o]`.
- `dora_rownorm_bwd.wgsl` - from `dg[o]` (produced by `scale_chan_dg`) emit
  `dm[o] = dg[o]/n[o]` and `u[o] = -dg[o]*m[o]/(2*n[o]^3)`.

Both need full `@what/@how/@opt/@cpu/@gpu/@npu/@quant/@dtype` headers, then
`make kernels-regen` and `make kernels-table`.

## Identity at init, and what is exactly zero

- **Init:** `A ~ init()`, `B=0`, `m := n|_{B=0} = sqrt(c)`. Then `g=1` and
  `W'=W`. **Not bit-exact as stated** - `sqrt(c[o])` computed twice (once to
  set `m`, once inside the norm kernel from the same `c`) is `1.0` only to
  fp32 rounding of a division. **Fix: compute `m` at init using the same
  kernel that computes `n`, from the same `c` buffer**, so the two are
  literally the same float and `m/n` is exactly `1.0` - turning a tolerance
  gate into a `to_bits()` gate for free. Do this.
- **Exactly zero at init:** only `dA`'s LoRA-path contribution (same
  `B=0` mechanism as plain LoRA). Everything else is **live at init** -
  `dg[o] = sum_t dy[t,o]*y_base[t,o]` is nonzero even at `B=0`, so `dm`,
  `u`, and `dB`'s new `2s*u*P` term are all nonzero from step 1. DoRA's
  dead-gradient set at init is `{lora_a}` only - strictly smaller than
  plain LoRA's. Warm-up for gradcheck: drive `B` non-zero as usual, **and**
  perturb `m` away from `sqrt(c)` so the `m/n != 1` branch of `dn`'s `-dg*m/
  n^2` term is actually exercised (otherwise it is probed only at the point
  where it's multiplied by a fixed ratio - a lesson-#40-shaped hollow check).

## Gradcheck shape

`directional_check` **cannot** see a partial error in the `n` path - it
folds `dA`/`dB` contributions across three stages (`P`/`BG` -> `row_dot` ->
`sqrt`), exactly the shape documented as invisible to it. `check_flux2_dora`
**must** carry `elementwise_check` siblings on `lora_m` and on `lora_a`.

## Fold / merge semantics

**Non-additive**: `W' = diag(m/n)*(W + s*BA)` is not expressible through the
existing `Pair::delta`/`delta_strided` or `device_adapter::fold_delta` - a
new `fold_dora(w, a, b, m, r, scale)`. `Adapter.kind` dispatch handles this
with no schema change (`kind` is already a free-form string). On an **int8**
base the fold must requantize the row-scaled result, which is cheap - a
uniform row scale interacts benignly with a group-wise weight scale.

## On-disk format and interop

- **brain's own:** `<name>.lora_m` alongside `.lora_a`/`.lora_b`,
  `Adapter.kind = "dora"`. Never `.safetensors` as the adapter's own
  extension (that extension is reserved for recognizing third-party
  ComfyUI/ai-toolkit LoRAs elsewhere in the pipeline).
- **Third-party keys to recognize in `read_external_adapter`:** `<stem>.
  dora_scale` (Kohya/ComfyUI/diffusers - high confidence) and `<stem>.
  lora_magnitude_vector.weight` (PEFT - high confidence on the name, medium
  on the exact `.weight` suffix), both `base_model.model.`-prefixable on
  PEFT dumps (high confidence, strip alongside the existing
  `diffusion_model.` strip).
- **Verify against a real checkpoint before implementing:** the exact
  `dora_scale` shape (`[out]` vs `[1,out]` vs `[out,1]`), and whether the
  file stores `m` or `m/||W+dW||` pre-divided - getting this wrong silently
  rescales every output channel (cosine-invisible, rel_l2-visible).

## Done numbers

1. `gradcheck::check_flux2_dora(seed)` passes the standard grad gate
   (4e-3/8e-2, no dead gradients) after the `B!=0`/`m!=sqrt(c)` warm-up,
   **plus** `elementwise_check` on `lora_m` and `lora_a` at the same
   tolerance.
2. Device-vs-host: worst cosine > 0.9999999, worst rel_l2 < 1e-5, on
   **both** backends (every kernel involved is `@cpu yes`, so this is
   achievable - do not let it silently become GPU-only).
3. Fresh-adapter no-op: `to_bits()` equality (via the shared-`c` trick).
4. The standard honest learning bound on klein-4b.
5. **Cost bound:** measured step time on klein-4b at 512px within 1.15x the
   plain-LoRA step at the same rank (analytically ~2% FLOPs + ~97ms of extra
   `W` reads at r=16/out=1536 on klein-4b). If it exceeds 1.15x, a profile -
   not this document - explains why before shipping.

### M3.3 - DoRA on an int8 base (depends on Phase 2)

**The one genuine blocker:** `P = W*A^T` has `W` as the *left* operand with
`A` fp32, but the existing int8 GEMM's activation contract is one scale per
row while `W` is packed with 32 scales per row - `P` cannot reuse it
directly. **Recommendation:** a device group-wise dequantize into a small
per-linear fp32 scratch (one linear at a time), then the existing fp32 GEMM
- because the same scratch-dequantize kernel is also what Phase 5 (NF4) and
M2.5 want, so it's the one worth building once. Do not attempt M3.3 before
M3.1 is green on fp32 - otherwise a DoRA bug and an int8 bug are
indistinguishable. Difficulty: medium.

---

# Phase 4 - External-adapter interop: LyCORIS + PEFT loading, IA3, textual inversion

**Independent of Phases 2-3 - can run concurrently with either.** Highest
user-value-per-engineering-hour in this whole roadmap: it makes brain able
to *consume* the community's diffusion-adapter ecosystem, which is where
real FLUX/SDXL/Wan LoRAs actually live.

### M4.1 - LyCORIS/PEFT external adapter loading (fold-only)

`read_external_adapter` currently recognizes exactly three key spellings and
hard-errors on anything else - **deliberately**, and that policy (a loader
that quietly drops keys returns a run that looks successful and isn't) must
survive unchanged. Extend the *recognized set*, never the error policy.

**Keys, with confidence, from general LyCORIS/PEFT knowledge - verify
against a real checkpoint before trusting any "medium" row:**

| Key | Method | Shape | Confidence |
|---|---|---|---|
| `lokr_w1` | LoKr, undecomposed factor 1 | `[out_l, in_m]` | high |
| `lokr_w1_a`, `lokr_w1_b` | LoKr, factor 1 decomposed | `[out_l,r]`,`[r,in_m]` | high |
| `lokr_w2` | LoKr, undecomposed factor 2 | `[out_k, in_n]` | high |
| `lokr_w2_a`, `lokr_w2_b` | LoKr, factor 2 decomposed - **the common case** | `[out_k,r]`,`[r,in_n]` | high |
| `lokr_t2` | LoKr conv, Tucker core | `[r,r,k1,k2]` | medium |
| `hada_w1_a`,`hada_w1_b`,`hada_w2_a`,`hada_w2_b` | LoHa | `[out,r]`,`[r,in]` x2 | high |
| `hada_t1`,`hada_t2` | LoHa conv Tucker | `[r,r,k1,k2]` | medium |
| `oft_blocks` | OFT | `[n_blocks,d,d]` | medium-high (newer files use `oft_R`/`oft_r`; both exist in the wild) |
| `dora_scale` | DoRA (Kohya/ComfyUI) | `[out]` or `[1,out]` | high name / medium shape |
| `lora_magnitude_vector[.weight]` | DoRA (PEFT) | `[out]` | medium-high |
| `base_model.model.` prefix | PEFT `save_pretrained` | - | high |
| `.alpha` | all | scalar | already handled |

**Needs checking against a real checkpoint before implementing:** the
`lokr_t2`/`hada_t*` Tucker mode ordering; whether LoKr's `alpha` scales the
whole `kron(w1,w2)` or only a decomposed factor (LyCORIS sets `scale =
alpha/lora_dim` only when a factor is low-rank-decomposed, ~70% confident -
getting it wrong is a uniform mis-scale cosine cannot see); the
`oft_blocks`-vs-`oft_R` split and whether stored blocks are raw `S`, skew
`Q`, or materialized `R`.

**The factorization, for loading (not creating) a LoKr file:** brain does
not need to derive `factor` at all - read `w1.shape`/`w2.shape` from the
file and validate `w1.shape[0]*w2.shape[0] == out` and `w1.shape[1]*
w2.shape[1] == in`; a shape that doesn't multiply out is a hard error naming
the tensor. `dW[a*out_k+c, b*in_n+d] = w1[a,b]*w2[c,d]`. **LoHa:** `dW =
(W1a*W1b) elementwise (W2a*W2b)`, scale `alpha/r`.

**Implementation:** generalize `ExternalPair` into an enum `ExternalDelta {
Lora{..}, LoKr{..}, LoHa{..}, Oft{..} }` with one `materialize(out,inn) ->
Vec<f32>` method; `fold_external_adapter` keeps its existing
validate-the-whole-adapter-before-writing-anything structure verbatim. OFT's
arm is multiplicative (`w = R*w`), which is why `apply` must own the write
rather than returning an additive delta.

**Done numbers:**
1. Fold a real LyCORIS LoKr file and a real LoHa file; compare the folded
   `[out,in]` against a straight-line host reference implementing the
   Kronecker/Hadamard formula directly: max_abs < 1e-6, rel_l2 < 1e-7.
2. An unrecognized key in a synthetic file still `Err`s naming that key.
3. A `lokr_w1` whose shapes don't multiply to `[out,in]` errors naming the
   tensor and both shapes.
4. Render with the folded adapter at `lora_scale in {0.0, 0.5, 1.0}`; `0.0`
   reproduces the base image bit-for-bit.

**Difficulty:** low-medium. **Risk:** the alpha/scale convention per method
- mitigate with one checked-in fold-parity fixture per format with a known
expected `max_abs`, not by reasoning about it in review.

### M4.2 - IA3

`y = (x*W^T) elementwise l`, `l in R^out`, init `l=1`. Composes entirely
from `scale_chan.wgsl` (forward and `dx`) and `scale_chan_dg.wgsl` (`dl`) -
**zero new kernels**. Param-list entry `.ia3_l` via the existing `lin`
closure pattern.

- **Identity at init:** `l=1` -> exact `to_bits()` no-op.
- **Exactly zero at init:** nothing - `dl[c] = sum x*dy != 0` at `l=1`. IA3
  is the one method with no dead gradient and no warm-up requirement -
  which makes it the right first non-additive `AdapterKind` to validate the
  Phase-1 trait against, before DoRA/OFT add real complexity.
- **Fold:** `W' = diag(l)*W` - a per-row multiply, merges cleanly even into
  an int8 base (row-uniform rescale of group scales, no repack).
- **Interop:** PEFT's `ia3_l` under `base_model.model.` prefixing.
- **Done:** clean grad gate with no warm-up; dev-vs-host at 0.9999999/1e-5
  on both backends; `to_bits()` no-op at `l=1`; adapter file < 1 MB for
  klein-4b.

### M4.3 - Trainable token embeddings (textual inversion)

For a diffusion-first repo, "trainable-token tuning" **is** textual
inversion, and it is a first-class community format. Mechanically: a
separate small `[n_new, d]` trainable tensor spliced into the text
encoder's embedding lookup, with the base embedding fully frozen -
`ParamStore` roles are per-tensor today, so this is a new small tensor, not
a sliced role on the existing one.

- **Identity at init:** initialize new rows from an existing token's
  embedding, so a fresh adapter reproduces that token's behavior bit-for-
  bit - not a no-op, a defined starting point.
- **Exactly zero at init:** nothing - the embedding gradient is live from
  step 1.
- **Fold:** not foldable into `W`, but - unlike prefix tuning - servable
  without touching the KV cache, prefix cache, or scheduler token
  accounting; it is extra vocabulary rows. This distinction is what earns
  it a place on the roadmap and is the reason prefix tuning is killed
  below.
- **Done:** loss-drop bound on a 5-image single-concept overfit; round-trip
  a community embedding file and reproduce its reference output at rel_l2
  < 1e-5.

---

# Phase 5 - NF4 / QLoRA, then LoftQ

Depends on Phase 2 (the transposed-copy pattern, the quantized-base
backward seam, the M2.0 measurement harness).

- `Dtype::NF4`/`F4E2M1` exist as tiers with host LUTs and a device GEMV
  kernel, but no `Weight::NF4` arm in `Ops::bind` and no tiled `_dyn` NF4
  GEMM (DiT training is the M~1536 tiled regime, not decode). NF4 training
  needs a `matmul_nf4_dyn` sibling of the existing `matmul_q4_dyn`, plus a
  transposed NF4 copy, plus double-quantization of the scale plane if the
  memory win is to be real.
- **Memory:** NF4 is 0.5 B/elem + scales. Two copies of klein-9b ~11.3 GB -
  comfortable, but that is the *only* thing it buys over int8's 20.4 GB,
  and int8 already fits.
- **Honest call:** NF4 buys headroom for a hypothetical ~18B diffusion
  model, not for anything on the current target list. **Sequence it after
  Phase 6, and gate its start on a named workload int8 cannot hold** -
  recording that condition is more useful than scheduling a date.
- **LoftQ belongs here, not with the initializer bucket.** It alternates
  `Q(W - BA)` and an SVD of the residual so `Q(W_res) + BA ~= W`. Its done
  number is unambiguous: `||Q(W_res) + s*BA - W||_F / ||W||_F` at init,
  versus `||Q(W) - W||_F / ||W||_F` - target >= 2x reduction at r=16 on real
  klein tensors. Needs the SVD primitive from Phase 7 first.

---

# Phase 6 - OFT (Cayley-Neumann), block-diagonal only

## The math

`W' = R*W`, `R = blockdiag(R_1 .. R_n)`, each `R_b` a `d x d` orthogonal
block, `d = out/n` (choose `d in {16,32,64}`).

Parameterize each block by a raw `S_b`, form the skew part `Q_b = (S_b -
S_b^T)/2` and approximate the Cayley transform with a truncated Neumann
series rather than a matrix inverse:
```
R_b ~= (I + Q_b) * (I + Q_b + Q_b^2 + ... + Q_b^J),   J = 5 typical
```
No solve, no factorization, no atomics - just `J` batched `d x d` matmuls.
Orthogonality is then approximate and must be **measured**, not assumed
(done number below). Adopt COFT's `||Q|| <= eps` ball constraint by default
to keep the series inside its radius of convergence.

**Apply the rotation to the output activation, not to `W`:** `(R*W)*x =
R*(W*x)`, so `y'[t, b*d+p] = sum_q R_b[p,q] * y[t, b*d+q]`. Cost `m*out*d`
vs the base GEMM's `m*in*out` - roughly 2% at `d=64, in=3072` - and **no
`[out,in]` scratch anywhere**, materially better than rotating `W` itself
(which would cost `out*in*d` plus a full write every step).

## What exists vs what must be built

The existing batched-matmul kernel (with `trans_a`/`trans_b` flags and flat
element offsets, no 256-byte alignment constraint) covers four of five
pieces with zero new code: the Neumann series' `Q^j` powers, the rotation of
`y`, the `dy' -> dy` back-rotation, and `dR_b = sum_t dy'[t,b,:] (x)
y[t,b,:]`. **Must build: one kernel, `oft_skew.wgsl`** - `Q[b,p,q] =
(S[b,p,q]-S[b,q,p])/2` plus its backward, one thread per element, no
reduction (an optional COFT norm clamp would need a real per-block
reduction and the barrier-free/cooperative pair per lesson #5 - prefer
computing that norm with the existing two-pass gradnorm shape instead if it
is implemented at all).

## Identity at init, and what is exactly zero

- `S=0` -> `Q=0` -> Neumann sum `=I` -> `R=I` **exactly** (every term is
  `I` or contains a zero matrix) - the best identity story of any method
  here, a genuine `to_bits()` gate.
- **Exactly zero at init:** nothing live from `dR/dQ` (nonzero from step 1)
  - **but** the symmetric part of `S` has a gradient that is exactly zero
  *forever*, since `Q` depends only on the antisymmetric part. **Fix at the
  representation, not the gate**: store only the strictly-lower triangle
  (`d(d-1)/2` values per block). Storing a full `d x d` and whitelisting
  half of it in the dead-gradient check is exactly the hand-maintained-
  exception shape lesson #54 warns about.

## Merge semantics

**Multiplicative** - merges into a plain weight (serving unaffected after
folding), but: existing additive-delta fold functions cannot express it (a
new `apply` that owns the write); composition order with an additive
adapter must be pinned and asserted, never left to iteration order (`R*(W +
s*BA) != R*W + s*BA`); un-merging needs `R^T`, only exact if `R` is actually
orthogonal, which under Neumann truncation it is not - say so; on an int8
base the merge requires requantizing `R*W` (unlike IA3's row scale, `R`
mixes channels within a block).

## Done numbers

1. **Orthogonality:** `||R_b^T R_b - I||_F / sqrt(d) < 1e-3` per block,
   measured at the *trained* `S` magnitude after a real 500-step run (not
   at init, where it's trivially 0) - the number that says whether `J=5` is
   enough.
2. Clean grad gate with no dead gradients (the strictly-lower-triangle
   storage delivers this), **plus** `elementwise_check` on `oft_blocks` -
   the Neumann product folds across `J` stages, exactly the shape
   `directional_check` is blind to.
3. Device-vs-host at 0.9999999/1e-5 on both backends.
4. `to_bits()` no-op at `S=0`.
5. **Cost bound:** step time within 1.10x the LoRA step at `d=64`.

## KILLED (conditionally): BOFT

BOFT replaces the block-diagonal `R` with a butterfly product over `L =
log2(n)` stages for dense-orthogonal coverage at `O(n log n)` parameters.
**Record as killed unless OFT first demonstrates a measured quality win
over LoRA on a real klein fine-tune.** It multiplies OFT's cost by `L`
(~5-6 at `out=3072, d=64`) and serializes it (`L` dependent stages, no
overlap) - turning OFT's 2% overhead into 10-12% plus `L` extra dispatch
barriers - and needs `L` butterfly-permutation gather kernels brain does
not have for this stride pattern, a real new indexing-bug surface. The
published quality gap between OFT and BOFT is small relative to the gap
between LoRA and OFT; building the second before measuring the first
repeats an ordering mistake this repo has already paid for elsewhere.
**Falsifier that would revive it:** OFT beating LoRA by a margin brain can
measure on its own eval, plus a profile showing OFT's rotation is not on
the step's critical path.

---

# Phase 7 - One SVD primitive, then PiSSA / OLoRA

brain has no SVD anywhere, and every advanced initializer on the original
list (PiSSA, OLoRA, EVA, CorDA, LoRA-GA, LoftQ, AdaLoRA) needs one. Build
**one** host randomized-range-finder SVD in `crates/model` (random `Omega
[in, r+p]`, `Y = W*Omega`, QR of `Y`, `B = Q^T*W`, small SVD of `B`, `U =
Q*U~`) - roughly 200 lines, no new dependency, existing `par::rows_mut` for
the GEMMs.

- **PiSSA:** `A,B` from the top-r singular triplets of `W`; the frozen base
  becomes `W_res = W - s*BA`. **Consequences that must be written down:**
  (a) PiSSA *modifies the frozen base*, so the adapter is only valid
  against the exact base it was derived from - the card's `variant_of`
  field must carry a hash, not just a name; (b) the fresh-adapter no-op
  degrades from `to_bits()` to `rel_l2 < 1e-6` (`W_res + s*BA = W` only in
  exact arithmetic); (c) on an int8 base, `Q(W_res) + s*BA` is a
  *different* approximation of `W` than `Q(W)` - measurably better or
  worse, and that measurement is the milestone.
- **OLoRA:** `A,B` from a QR of `W`, same base-modification consequence,
  ~30 lines once the QR exists.
- **Done number for the whole phase:** on real klein tensors at r=16, loss
  after 100 steps is lower with PiSSA init than the default `B=0` init by a
  margin larger than the seed-to-seed spread over 3 seeds. **If it isn't,
  say so and stop** - this is a technique class that reliably shows up in
  papers and unreliably shows up in 1000-step diffusion fine-tunes.

**Defer, with reasons:** EVA (SVD of *activations* - brain's
`model::actstats::Collector` gives it a cheap calibration harness, so it's
the natural next step if PiSSA pays off), CorDA and LoRA-GA (each need a
calibration pass plus an SVD plus their own gradient statistics - three new
harnesses for an unmeasured margin; build only if PiSSA's measurement is
positive).

---

# Phase 8 - LyCORIS training (LoKr, LoHa) - only if Phase 4 shows demand

Loading (M4.1) is what users need first. Training is a separate, much more
expensive product, and the cost asymmetry between the two methods should
drive the decision:

- **LoHa's `dW = P1 (x) P2` does not factor through the activation** - the
  Hadamard product destroys the low-rank structure, so `x*(P1(x)P2)^T`
  cannot decompose into rank-r GEMMs. Training LoHa means materializing a
  full `[out,in]` scratch per targeted linear per step and running a second
  full-width GEMM: **~+100% forward FLOPs and +100% backward** on every
  targeted linear, versus LoRA's ~+2%. State this cost before anyone starts.
- **LoKr's `dW = W1 (x) W2` DOES factor**, via the vec trick: reshape `x`
  and run two GEMMs through the two Kronecker factors with a permute
  between them. Cost is roughly `sqrt(out)`x cheaper than the base GEMM
  (~55x at out=3072). Needs two permute (gather-shaped) kernels and careful
  index bookkeeping.
- **Gradcheck:** both factorizations fold across stages -
  `elementwise_check` mandatory alongside `directional_check` on every
  factor.
- **"Exactly zero at init" is where LoHa bites hardest.** LyCORIS zeros the
  second Hadamard factor (and LoKr's decomposed second factor) at init, so
  `dW=0` - but then the gradient of *both* halves of the first factor is
  exactly zero too, because they're multiplied by the zeroed second factor.
  This is a **two-sided** dead gradient, strictly worse than plain LoRA's
  one-sided case; the warm-up must drive the second factor non-zero before
  anything about the first factor can be FD-probed, or the gradcheck is
  hollow-but-passing by construction.
- **Recommendation:** build **LoKr training** if Phase 4 shows real LoKr
  usage (it's cheap via the vec trick). **Do not build LoHa training** -
  its 2x step cost buys an effective-rank increase a plain rank-2r LoRA
  gets for ~+4%. Record as a should-be-killed hypothesis; falsifier: a
  measured quality comparison at matched *step time*, not matched parameter
  count.

---

# Phase 9 - VeRA (rides on Phase 1's LoRA-FA)

`dW = diag(b)*B*diag(d)*A` with `A`,`B` **frozen shared random matrices**
(one pair for the whole model, generated from a seed) and only the
diagonals `b in R^out`, `d in R^r` trainable.

- **Composes from:** Phase 1's LoRA-FA (frozen `A`, and by extension a
  frozen `B` too here) plus two applications of the existing per-channel
  scale kernel. Near-zero new code.
- **Identity at init:** `b=0` -> exact `to_bits()` no-op.
- **Exactly zero at init:** `d`'s gradient (the r-vector) is exactly zero
  at `b=0`, same structure as LoRA's `dA`. Same warm-up shape.
- **The whole value proposition is file size:** `(out + r)` floats per
  linear plus a seed - klein-4b goes from ~100 MB to under 1 MB, genuinely
  attractive for distribution.
- **The whole risk is the seed contract:** the seed *is* the adapter. Any
  change to the shared random-init function or its generation order
  silently invalidates every VeRA adapter ever saved. **Gate:** a checked-
  in fixture asserting the shared `A`/`B` generated from a pinned seed are
  bit-identical to a checked-in hash, forever. Do not ship VeRA without
  this test.

---

# Appendix: killed and should-be-killed hypotheses

| Hypothesis | Verdict | Reason |
|---|---|---|
| Prompt / prefix / P-tuning | **KILLED** | Prefix tuning prepends trainable K/V to every layer, forcing brain's paged KV cache, prefix caching, continuous batching, and capability token accounting to all become adapter-aware - a cross-cutting serving-path change for a method whose only diffusion use case is already covered by textual inversion (M4.3), which needs none of it. Prompt tuning changes sequence length and cannot be folded, breaking the "fold once, serve like every other model" shape every other served adapter in this repo follows. **Falsifier:** an LLM instruction-tuning workload needing per-request adapter swap at serve time where folding is impossible. |
| 8-bit optimizer state | **KILLED** | Shrinks `m`/`v`, which in a LoRA run is ~0.5% of the model - saves nothing that matters. In a full fine-tune, the existing host-RAM AdamW offload already puts `m`/`v` in 177 GB of RAM - a strictly larger saving with zero quantization error. Building an 8-bit optimizer here builds a worse version of something already shipped. |
| Paged optimizer state | **KILLED**, same reason | Exists upstream to survive OOM spikes on a box with no host-offload path; brain has one, deterministic and measured. ("Paged" in this tree already means paged KV cache - reusing the word would be a naming collision on top of a redundant feature.) |
| AdaLoRA | **SHOULD BE KILLED** | Buys an adaptive rank budget via an SVD parameterization + orthogonality regularizer + importance-EMA + budget scheduler - four new moving parts, each its own gradcheck surface, for brain's actual workload (a 1-3k-step diffusion fine-tune on 20-50 images) where rank is rarely the binding constraint and a scheduler would never leave its warm-up. Phase 1's static per-layer rank already gives most of this value without the machinery. **Falsifier:** a measured run where fixed-rank LoRA demonstrably underfits some layers and overfits others by a margin larger than seed noise. |
| GaLore | **KILLED for diffusion** | Reduces optimizer state, not weights - klein-9b's fp32 weights alone (36.2 GB) exceed a 24 GB card regardless, and for klein-4b the existing host-RAM offload already removes optimizer state entirely. Also requires a full dense `dW`, which brain's LoRA path deliberately never forms. Revisit only for LLM full fine-tuning, after offload is measured insufficient. |
| Q-GaLore before GaLore | **KILLED outright** | Quantizing the projections of a technique whose unquantized form hasn't been shown to help brain's targets is the exact ordering mistake this ledger otherwise avoids. |
| APOLLO | **KILLED with GaLore** | Same family, same prerequisite, same reasoning. |
| BAdam / LOMO | **DEFER, not kill** | LOMO's fuse-backward-with-update removes the gradient buffer entirely (a real, simple 1x-model saving, no SVD, no projection) - but it is a full-fine-tuning technique, and every current diffusion target either fits with the existing offload or does not fit at all. Revisit if an LLM full-FT campaign starts; it would then be the first thing to build in this family, ahead of GaLore. |
| BOFT | **CONDITIONALLY KILLED** | See Phase 6. |
| LoHa training | **SHOULD BE KILLED** | 2x step cost on every targeted linear for an effective-rank gain a rank-2r LoRA delivers for ~+4%. Falsifier: a matched-step-time quality comparison. LoHa *loading* still ships in M4.1. |
| EVA / CorDA / LoRA-GA | **DEFER** | Three new calibration harnesses for an unmeasured margin. PiSSA is the cheap probe of whether this whole initializer class pays off in brain's regime; if it doesn't, none of these will either. |

---

## Dependency graph

```
P2 int8 base ----+--> P3.3 DoRA-on-int8
                 +--> P5 NF4/QLoRA --> LoftQ --(needs)--> P7 SVD
                 +--> klein-9b single card, larger-token two-card

P3 DoRA (fp32) --------> P3.3 (also needs P2)
P4 interop / IA3 / textual inversion -- independent, parallelizable
P6 OFT -- independent (needs only the Phase-1 AdapterKind trait) --> [BOFT, gated on OFT's measured win]
P7 SVD --> PiSSA, OLoRA, LoftQ, [EVA/CorDA/LoRA-GA if PiSSA pays off]
P8 LoKr training -- gated on P4 showing demand
P9 VeRA -- needs Phase 1's LoRA-FA only
```

The long pole in the whole roadmap is `model::int8::quantize_transposed_from`
and its streaming loader (M2.1) - a host data-movement problem, not a
kernel problem. Phase 2's first cut needs zero new WGSL.
