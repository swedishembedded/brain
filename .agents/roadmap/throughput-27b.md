# 27B LoRA / QLoRA training throughput - plan to 1,000-2,000 trained tokens/s

Scope: Qwen3.8-27B LoRA/QLoRA training throughput.

Ledger and execution plan for making one Qwen3.8-27B LoRA step (crate `qwen35`,
frozen bf16 base and frozen NF4 base, rank 16 on all 12 projections, recompute on)
train at **at least 1,000 tokens/s on one GH200, with 2,000 as the stretch**, on
the real checkpoint through `brain qwen35 finetune --batch 1`. Today it is about
20 tokens/s on a quiet card and 2-4 tokens/s on the shared card (below). Companion
to `training-27b.md` (its "Throughput plan" section is superseded by this file),
`cuda.md` (provider chain, M6-M11) and `gh200.md` (machine ceilings).

Every number is labelled **VERIFIED-BY-RUN** (measured in the run named next to it)
or **DERIVED** (arithmetic from code, shapes and the measured ceilings). Update this
file in the commit that lands or changes an item, with the before/after numbers of
the protocol in section 7.

---

## 0. Status (lane T1, 2026-10-16)

What has landed since the plan was written, with the deviations from it. Numbers are in
section 8; the entry that records the runs is knowledge #310.

| item | state | deviation from the plan |
|---|---|---|
| Q1 no MLP replay in the backward | landed | also skips the `down` product of each chunk rebuild (nothing reads it): the chunked backward does no more GEMM work than the whole-sequence MLP's |
| Q2 poll_wait per MLP chunk | landed | one wait per MLP pass instead of one per row chunk (knowledge 312: real NF4 621 to 635 tokens/s) |
| Q3 sizing of segment and MLP chunk | landed as a footprint rule | by base tier (4-bit and bf16 2048/2048, fp32 128/512; `finetune.rs` `long_settings`, knowledge 311) rather than by free device memory: no backend reports free memory through `Gpu` yet |
| Q4 tiled rmsnorm rows | landed | both twins registered for training; also the cooperative grad-norm, a block-per-row `ce_stats` and a block-per-head GDN decay gradient (single-thread reductions the profile found on the default path) |
| Q5 probe as an instrument | landed | `--seed`, dispatches per step, warm tokens/s, other compute contexts, `--gemm-precision`, `--profile` (device time per kernel plus a launch census) |
| K1 bf16 GEMMs on the vendor library | landed | `libcublas` `cublasGemmEx` / `cublasGemmStridedBatchedEx` rather than `cublasLt`: the heuristic is the same library path with no descriptor plumbing; the step records through a placeholder native kernel so fencing and lifetimes are unchanged, and a submission holding one is issued step by step (no graph capture). The tier is `GemmPrecision::{Auto, Fp32Reference, Bf16TensorCore}`; `Auto` is the default and falls back to the fp32 kernels, with the reason logged once, on any device that cannot run the library (no CUDA, compute capability below 8.0, library absent, handle failure, a shape outside the 16-byte-row rule) |
| K2 NF4 through the same GEMM | landed | decode into a bf16 scratch per call; no weight-reuse cache |
| K3 LoRA on library GEMMs | partly | the five products per linear are single fp32 library GEMMs with the scale in alpha and the accumulation in beta; the shared-input A concatenation is not done |
| K4 fused head CE gradient | not done | the head GEMMs themselves run on the library (windows are whole 64-row groups) |
| K5 tensor-core flash | landed in a different form | blockwise attention around library GEMMs plus three small element-wise kernels (softmax with causal block skipping, score gradient, row dot) instead of a fused `mma.sync` kernel: memory bounded by `SCORE_BLOCK_BYTES`, no bespoke MMA code |
| K6 native GDN training kernels | forward (fp32 and, per T1 below, tensor-core) and the backward sweep landed | the forward recurrence and both histories are one native launch (bit-identical to the walk); the backward computes everything that does not depend on the running state gradient for all chunks at once, and its sequential state-gradient sweep is one native launch per layer (`gdn_bwd_sweep_f32`, one block per head, state in dynamic shared memory; the host walk of six dispatches per chunk remains the fallback; knowledge 311). The whole-tensor backward phases are not fused |
| T1 tensor-core GDN chunk forward | landed (forward) | not the plan's bf16 `mma.sync`: three-pass TF32 (`a_hi*b_hi + a_hi*b_lo + a_lo*b_hi`, fp32 accumulate), which keeps the products within about 1e-6 of fp32 so the plan's loose bf16 tolerances are not needed. Three native kernels, all behind `GdnFastIds::tensor_core` (a caller flag, qwen35 only) and the provider's compute-capability floor 8.0: `gdn_chunk_prep_tc` (steps 1-9 and the intra scores in one launch per (head, chunk)), its token-major forms `gdn_chunk_prep_tok(_train)_tc` (query/key L2 norm, head repeat and the five chunk-major permutes folded into the loads) and `gdn_chunk_loop(_train)_tc` (the across-chunk recurrence, four 32-column blocks per head). Older devices, the CPU and Vulkan backends, `BRAIN_NO_PROVIDER=cuda` and any declined launch keep the portable sequence unchanged (knowledge 314) |
| T2 tensor-core GDN chunk backward | landed | two launches per layer pass instead of about sixty: `gdn_bwd_sweep_tc` (the reverse sweep restated so a head splits over four 32-column blocks, 192 blocks; the one product that contracts over columns, `d_decayed_k`, moves out because it does not feed the recurrence, fed by a history of the state gradient the sweep writes) and `gdn_chunk_bwd_post_tc` (everything independent across chunks, one block per (head, chunk)). Same opt-in and fallback as T1 (`GdnFastIds::tensor_core`, `GdnBwdScratchBufs::new_for` allocates the history only then). Parity: every gradient within 3.1e-6 of the portable path (`gdn_chunk_fwd_train_tc`). Knowledge 316 |
| T2 mixer layout fold | landed | the sweep and post kernels read `d_out` token-major and write `d_value`, `d_raw_g`, `d_beta` token-major; `gdn_qk_bwd_tok` sums each key head's value heads and applies the l2norm adjoint (2 permutes + 2 `kv_expand_bwd` + 2 `l2norm_scale_dx` before). `gdn_conv_silu_tok` / `gdn_conv_silu_bwd_tok` / `gdn_conv_hist_tok`: the depthwise convolution, SiLU and qkv split on the token-major rows (no `nlc_nchw` / `nchw_nlc` / `concat_split` / `concat2` around it); frozen conv weight and four taps only, otherwise the old chain |
| T2 kernel tuning | landed | masked TF32 split instead of two `cvt.rna.tf32` per value; unrolled staging loops (the loads of a loop in flight together); the post kernel streams its 128 x 128 operands through `cp.async` slabs and keeps `d_query` / `d_key` in registers; the prefix's triangular solve in outer-product form (bit-identical to the recurrence). Evidence is block cycles read with `clock64` (knowledge 316) |
| T4 glue folds and decode cache | landed (by count and parity) | `swiglu_bwd_fused`, `add_rmsnorm_fwd` / `rmsnorm_dx_add`, `silu_mul_bf16`, the token-major recurrence output, and `DecodedWeightScope` (NF4 weights decoded once per layer pass). Not folded: the MLP row-chunk copies (`take_rows` / `put_rows`) and the GEMM-prologue decode (the vendor library reads bf16 from memory). Knowledge 317; idle-card timing pending |
| K7, K8, Q6 | not done | |
| plumbing | landed | the native `concat_split`/`chan_place`/`concat2` launched one block per element for the channel windows of a row-major matrix (`H = W = 1`): 2.6 M blocks for one MLP chunk window, 8 ms a call; planes under 32 floats are now copied flat |

---

## 1. Where a step's time goes today

### 1.1 What the code does (verified by reading the tree at c8a33d4e)

Shapes: d 5120, ff 17408, vocab 248320, 64 layers (48 Gated DeltaNet, 16 GQA),
GQA 24 q heads / 4 kv heads / head_dim 256, GDN 16 key heads and 48 value heads of
128. Base parameters that take part in GEMMs: 383.3 M per GDN layer, 372.3 M per GQA
layer (both including the 267.4 M MLP), 24.36 G for the stack, plus the 1.27 G LM
head.

**GEMMs never touch a tensor core.** Every float GEMM (f32, bf16 and NF4 base;
forward, dX, LoRA, head) runs generated or native fp32 code on the CUDA cores,
compiled with `--fmad=false`, so each multiply-add is a separate `mul` and `add`
(`crates/backend-cuda/src/nvrtc.rs:263-272`). The bf16 and NF4 tiers are
`kernels::template::dtype_variant` / `nf4_weight_variant`
(`crates/kernels/src/template.rs:147,243`) rewrites of `matmul_reg3` /
`matmul_dx_reg` (128x128 tile, BK 8, 8x8 register block): the weight is decoded to
f32 at load time, staged in shared memory as f32, and computed in fp32. These
`#w=bf16|nf4` variants never get a native kernel, because the native redirect
matches exact registered names (`crates/gpu-core/src/native_upgrade.rs:965`), and
`CudaProvider::accepts` declines bf16, NF4 and every backward
(`crates/gpu-core/src/provider/cuda.rs:304`). The only tensor-core kernels in the
tree are `matmul_i8_mma`, `moe_i8_grouped_mma` (int8 serving) and
`flash_prefill_f16_hd256` (inference prefill). No cuBLAS / cuBLASLt / CUTLASS is
linked; `libcublasLt.so.13` and `libcudnn.so.9` are present in the toolkit lane
(`~/project/shared/containers/cuda/cuda-13.0.2/lib`).
- Ceiling of that design (DERIVED): the card's fp32 rate is 67 TFLOP/s counting an FMA
  as two FLOPs; without contraction a multiply-add costs two issue slots, so no fp32
  non-FMA GEMM can exceed about 33 TFLOP/s. At 155-190 GFLOP per token that is a hard
  ceiling of about **175-215 tokens/s for the GEMMs alone**. The 1,000 tokens/s target is
  therefore impossible on the current GEMM stack, however well tuned.

**Base GEMMs run more often than the memory plan needs** (`crates/qwen35/src/model.rs`):
- `backward` (`:2852`) replays the whole `layer_forward` (`:2521`) for every layer.
  With `mlp_rows` set (the CLI default for rows of 2048 tokens or more), that replay's
  MLP output goes into the throwaway `recompute_out` buffer. `mlp_bwd_saved` (`:2802`)
  then rebuilds every MLP chunk a second time, so **the MLP forward runs three times per
  step, and one of those runs is wasted** (DERIVED: 34.2 GFLOP/token, 18% of all GEMM work).
- The LM head runs forward twice (loss, then recomputed in the backward,
  `crates/model/src/head_ce.rs:377-434`) plus dX.

GEMM work per trained token at T = 8192 (DERIVED):

| | forward | recompute | MLP rebuild | dX | total GFLOP/token |
|---|---:|---:|---:|---:|---:|
| layers today | 48.7 | 48.7 | 34.2 | 48.7 | 180.3 |
| head today | 2.5 | 2.5 | - | 2.5 | 7.6 |
| LoRA (rank 16, all 12 sites) | | | | | about 1.5 |
| **today** | | | | | **about 189** |
| after Q1 (no wasted MLP replay) and K4 (no head recompute) | | | | | **about 153** |
| floor without any recompute (does not fit at 8K) | | | | | about 104 |

**LoRA uses the naive one-thread-per-output kernels.** `lora_fwd` (`model.rs:1772`)
issues `matmul` (x·Aᵀ), `matmul` (·Bᵀ), then `axpy` into y. `proj_bwd` (`:1904`) adds 7
more dispatches per projection: a recomputed x·Aᵀ, two `grad_scale`, two `matmul_dw`
(each output is a serial loop over all T rows), two `matmul_dx`. None of these kernels
has a tiled or native upgrade.

**About 0.9 M kernel launches per 8K step, most of them from the GDN chunk loop.**
- The GDN training forward walks the chunks on the host: 11 dispatches per 64-token chunk,
  never native (`crates/model/src/gdn.rs:1073`). The backward uses 25 per chunk plus a
  29-dispatch whole-tensor phase (`gdn.rs:1305`).
- With segments of 128 tokens (the CLI's long-row default,
  `crates/qwen35/src/finetune.rs:130,166`), each segment adds about 209 dispatches per GDN
  layer per step (VERIFIED-BY-RUN, the two 4-layer profiles below differ by 18,810 dispatches
  over 90 segment-layers).
- The 64-layer, 8,192-token step therefore issues about 0.87 M launches (DERIVED from the
  measured 4-layer counts). The native inference recurrence `gdn_chunk_loop_f32.cu`, one
  launch for the whole across-chunk walk and bit-identical to the walk, exists but is used
  only by the non-training forward.
- Each eager launch also uploads its uniform with a `cuMemcpyHtoDAsync`
  (`crates/backend-cuda/src/backend.rs:912,947`). A stream bubble of about 4.8 us per
  launch was measured in `cuda.md` ("Prefill on tensor cores"), and the empty-kernel launch
  latency is 5.3 us (`gh200.md`).

**About 3,100 blocking `poll_wait`s per step.** There is one after each 512-row MLP chunk in
the forward, in the replay and in `mlp_bwd_saved` (`model.rs:2794,2815`), and one per layer in
the backward (`:2985`). Every intermediate is a fresh `g.storage` (an allocation plus a
memset; there is no arena on the training path).

**Attention is the fp32 scalar flash trio.** The trio runs at about 4 TFLOP/s (knowledge
240): 187 ms per layer at T = 4096 and 2.8 s at 16384, growing as T². At 8K it is about
15 s per step over the 16 GQA layers (DERIVED from 240), which caps the step at about
540 tokens/s even if everything else were free.

**These are not bottlenecks:**
- The optimizer: one `adamw` per tensor (992 tensors), with device-side clipping and no
  readback (`crates/optim/src/lib.rs:198-267`).
- The base load: streamed once at construction (161 s for NF4,
  VERIFIED-BY-RUN in this session's L1 log), then resident. Nothing is re-uploaded per step.
- The loss readback: one per micro-step.

### 1.2 Profile runs (this session, 2026-10-16, release build of c8a33d4e)

Command (worktree `~/project/shared/containers/worktrees/tp`; raw logs and
`bucket.py` in the planner's scratch directory):

```
BRAIN_BACKEND=cuda BRAIN_PROFILE=1 qwen35_train_probe --tokens 4096 --layers 4 --rank 16 \
    --steps 2 --bf16-residuals on --mlp-rows 512 --base nf4|bf16 --gdn-segment 128|2048
```

There are 4 layers (3 GDN + 1 GQA) at real width, random weights, and the real 248,320
vocabulary head.

**The card was shared for every run** and the gpu lock was held by another lane's real-27B
run, so these ran unlocked (they are small, 12-14 GiB). `nvidia-smi pmon` showed **four other
compute contexts active** (one at 61-74% SM). Per-launch timing on a time-sliced card bills
the neighbour's slice to whichever kernel was running (`.agents/rules/kernels.md` §E.0a). The
signature is unmistakable: every small kernel (`mul`, `sigmoid`, `concat_split`,
`region_copy`, ...) read **2.0-2.9 ms per call regardless of its size**. Real device costs are
in the microseconds. So the tables below RANK by dispatch count and GEMM time; they are not
the step's exclusive-card composition.

VERIFIED-BY-RUN (step 2 of each run; one run each, shared card):

| run | dispatches/step | step s | tokens/s (4 layers) | step MiB/token |
|---|---:|---:|---:|---:|
| NF4, seg 128, profiled | 29,888 | 104.9 | 39 | 0.548 |
| NF4, seg 128, not profiled | 29,888 | 83.8 / 107.0 | 49 / 38 | 0.547 |
| bf16, seg 128, profiled | 29,888 | 84.0 | 49 | 0.548 |
| NF4, seg 2048, profiled | 11,078 | 42.1 | 97 | 1.041 |
| (L1 lane, same NF4 seg-128 config, earlier, quieter card) | 29,888 | 44.0 | 93 | 0.547 |

Profile buckets for the NF4, seg 128 run, step 2 (VERIFIED-BY-RUN; ms are time-slice-inflated
as noted above):

| bucket | ms | share | calls | ms/call |
|---|---:|---:|---:|---:|
| GDN chunk math (`bmm_tiled`, `gdn_*`, `row_dot`, `exp`, `sub`, `l2norm`, `kv_expand`, `conv1d`, `scale_row`) | 42,961 | 43.6% | 15,744 | 2.73 |
| layout/copy glue (`concat_split`, `splice_add`, `region_copy`, permutes, `concat2`) | 25,853 | 26.2% | 8,786 | 2.94 |
| elementwise (`mul`, `silu`, `sigmoid`, rope, `add2`) | 6,946 | 7.0% | 2,501 | 2.78 |
| base GEMM forward, `matmul_reg3#w=nf4` | 6,829 | 6.9% | 326 | 20.9 |
| LoRA (`matmul`, `matmul_dx`, `matmul_dw`, `grad_scale`, `axpy`) | 5,135 | 5.2% | 1,787 | 2.87 |
| bf16 GEMMs (head here) | 5,097 | 5.2% | 36 | 141.6 |
| base GEMM dX, `matmul_dx_reg#w=nf4` | 3,056 | 3.1% | 115 | 26.6 |
| norms | 1,163 | 1.2% | 416 | 2.80 |
| head CE | 575 | 0.6% | 48 | 12.0 |
| optimizer and clip | 499 | 0.5% | 125 | 4.0 |
| flash attention (T = 4096, 1 layer) | 495 | 0.5% | 4 | 123.8 |

In the earlier, quieter run by the L1 lane (same config, 37.0 s of device time) the GEMM rows
(fwd, dX and head) were 30% and the GDN and glue rows 54%.

What these runs establish:
1. **On a shared card, step time is proportional to the launch count.** Cutting the
   dispatches 2.7x (segment 128 to 2048) cut the step 2.5x (104.9 s to 42.1 s) at the same
   GEMM work (VERIFIED-BY-RUN).
2. **The GEMMs run at 5-9 TFLOP/s on the shared card** (VERIFIED-BY-RUN, the GEMM rows'
   FLOPs over their device time: the NF4 forward 33.7 TFLOP in 5.0-6.8 s, the head forward
   20.8 TFLOP in 2.4 s). The DERIVED exclusive rate is 12-18 TFLOP/s. cuBLASLt bf16 on this
   card measures 833 TFLOP/s at 8192³ (`gh200.md`, VERIFIED by that probe): a 45-70x gap on
   the work that dominates an exclusive-card step.
3. **The real 27B CLI step at 8K (L1 lane, NF4, seg 128) had not finished one step after about
   60 minutes** on the shared card (VERIFIED-BY-RUN, still in its first step at 12:40,
   training since 11:41 after a 161 s load), i.e. below 3 tokens/s. That fits 0.87 M launches at
   about 2.5 ms of time-sliced cost each (DERIVED).

### 1.3 The 64-layer, 8K step on an exclusive card (DERIVED)

These are estimates from the shapes, the measured rates and the counts above. The "19-20
tokens/s" of knowledge 234 is consistent with them.

| component | today, s/step at T = 8192 | why |
|---|---:|---|
| base and head GEMMs, generated fp32 | 85-130 | 1.55 PFLOP at 12-18 TFLOP/s |
| LoRA, naive kernels | 10-40 | serial T-long loops in `matmul_dw`; T·r threads in `matmul` |
| flash trio, fp32 | about 15 | knowledge 240 scaled by T² |
| launches (0.87 M x 5-10 us) | 4-9 | per-dispatch upload and launch |
| 3.1 K `poll_wait` drains and fresh allocations | 1-3 | |
| **total** | **115-195** | **42-70 tokens/s**, about 20 when also shared or with the head, glue and dispatch costs higher |

---

## 2. Time budget at the target

Workload W8K: 64 layers, T = 8192, batch 1, rank 16 on all 12 sites, recompute on, real
vocabulary. GEMM work after Q1 and K4 is 153 GFLOP per token. At 1,000 tokens/s the whole
step is 8.19 s, i.e. **1,000 us per token**.

Ceilings used (`gh200.md`, cuBLASLt and copy probes): bf16 dense 833 TFLOP/s, tf32 410,
fp32 45, HBM read 3.8 TB/s, copy 3.4 TB/s, launch 5.3 us.

| component | work per token | roofline time | budget at target | what reaches it |
|---|---|---:|---:|---|
| base GEMMs (fwd, recompute, dX) | 146 GFLOP | 175 us at 833 TF | **300 us** (about 490 TF sustained) | cuBLASLt bf16, K1 |
| head GEMMs (fwd and dX, fused CE) | 5.1 GFLOP + CE over 248,320 logits (about 4 MB) | 7 us | 15 us | K1 + K4 |
| NF4 dequant to a bf16 scratch | 24.4 G weights x 3 uses x 2.56 B / 8192 = 23 MB | 7 us | 10 us (0 for bf16) | K2 |
| fp32 to bf16 casts of GEMM operands | about 50 MB | 15 us | 20 us | K1 (removed later by Q6) |
| LoRA (fwd, rebuilds, dA, dB, dX) | 1.5 GFLOP, about 40 MB | 12 us | 30 us | K3 |
| flash attention hd256, 16 layers | 7.25 GFLOP (fwd x2 + bwd 2.5x) | 9 us at 833 TF; 29 us at 250 TF | 40 us (150 us at 32K) | K5 |
| GDN recurrence and chunk math, 48 layers | about 2 GFLOP, about 190 MB | about 60 us (bytes) + about 10 us (serial chunk walk) | 100 us | K6 |
| norms, SwiGLU, rope, gates, residual pack, row windows | about 135 MB | 40 us | 120 us | Q4, K7 |
| optimizer and clip (118 M LoRA params) | 3.3 GB per step | 0.15 us | 1 us | none |
| launch and host overhead (25 K launches, 70 syncs per step) | | 15-30 us | 30 us | Q2, Q3, K6, K8 |
| base streaming | none per step (resident) | 0 | 0 | none |
| **sum** | | | **about 665 us = 1,500 tokens/s** | |

- **1,000 tokens/s has a 33% margin.** 2,000 tokens/s (500 us) additionally needs the base
  GEMMs at 70% of peak (210 us), glue at 60 us (K7 fused epilogues) and the GDN at 70 us.
  It is reachable but is the stretch.
- At T = 32768 attention grows 4x (about 110 us at 250 TF), which gives about 1,300 tokens/s.
- The bf16 base equals NF4 minus the dequant row.
- **The budget is only measurable on an exclusive card.** With four other contexts
  time-slicing the GPU, every number above is divided by the share received. Milestone
  measurements (section 7) must hold the gpu lock AND confirm with `nvidia-smi pmon` that no
  other context is active.

**Which components can reach their budget on today's kernel stack?**

| component | current stack reaches budget? | replacement |
|---|---|---|
| base and head GEMMs | **No** (structural cap about 33 TF, measured 5-18) | cuBLASLt bf16 through a `dlopen`ed library provider (`cuda.md` M8), dequant-to-scratch for NF4 |
| LoRA | **No** (naive kernels) | the same library GEMMs, A matrices concatenated per shared input, B·mid accumulated into y by the GEMM (beta = 1) |
| flash attention hd256 | **No** (4 TF fp32 scalar) | native bf16 `mma.sync` flash forward with LSE plus backward (FlashAttention-2 algorithm, arXiv 2307.08691), grown from `flash_prefill_f16_hd256.cu` |
| GDN | **No** (0.87 M host-walked launches) | native chunked training kernels: the recurrence with state history (extend `gdn_chunk_loop_f32.cu`) and a reverse-sweep backward kernel (chunkwise WY/UT algorithm, arXiv 2406.06484, gated variant arXiv 2412.06464) |
| elementwise and glue | partly (bandwidth OK once launches drop) | fusions (K7) plus an arena (K8) |
| optimizer | yes | none |

---

## 3. Ranked optimizations

Ranked by expected payoff per unit of risk. Each item lists what changes, where, the gain,
the risk, the correctness gates and the benchmark. "W4" is the 4-layer probe and "W8K" the
64-layer probe of section 7.

### Tier Q: quick wins (days, no new kernels)

**Q1. Do not replay the MLP in the backward's layer rebuild.**
- **Change.** Give `layer_forward` (`crates/qwen35/src/model.rs:2521`) a mode used only by
  `backward` (`:2943`) that stops after `xn2` when the MLP is saved as `MlpSaved::Rows`. It
  skips `mlp_fwd_rows` and the final `add2` into `recompute_out`, since `mlp_bwd_saved`
  rebuilds each chunk anyway. With `mlp_rows` unset (`MlpSaved::Full`) nothing changes.
- **Gain.** 34.2 of 189 GFLOP per token (18% of GEMM work), plus 1,024 of the step's
  `poll_wait`s and about 16 x 64 x 20 launches (DERIVED).
- **Risk.** Low.
- **Gates.** Gradients must be bit-identical, because the removed work feeds a buffer
  nothing reads. Keep `crates/qwen35/tests/lora_recompute.rs`, `lora_bf16_residuals.rs`,
  `lora_nf4_base.rs` and `lora_bf16_base.rs` green. Add a fast unit test asserting that
  `backward` with `mlp_rows` produces bit-identical LoRA gradients before and after on a tiny
  config, on CPU and CUDA.

**Q2. Remove the per-chunk `poll_wait` in `mlp_fwd_rows` and `mlp_bwd_saved`.**
- **Change.** In `model.rs:2794` and `:2815`, the wait exists to return chunk scratch before
  the next chunk allocates. Allocate the chunk scratch (take_rows output, gate/up/h, their
  gradients) once per call and reuse it across chunks. Stream order already serializes the
  chunks, so no wait is needed. Keep the per-layer wait at `:2985` until K8.
- **Gain.** About 2,000 host round trips per step, plus re-enabling graph capture of the
  repeated chunk shape (`backend.rs:493-501`). Expected 2-5% at today's speed and far more
  once the GEMMs are fast (a drain is 20-100 us; DERIVED).
- **Risk.** Low. Peak memory must not grow: assert the probe's `step_mib_per_token` stays
  0.55 +/- 0.01.
- **Gates.** As Q1; the trajectory is bit-identical.

**Q3. Size the long-sequence knobs by memory, not by a constant.**
- **Change.** `finetune.rs:130-183` hard-codes GDN segment 128 and MLP rows 512 for every row
  of 2048 or more tokens. Choose the largest segment and MLP chunk that fit the measured
  per-token cost (knowledge 233/234: 0.55 MiB/token at seg 128, 1.04 at seg 2048) in the
  free device memory reported by the backend. Use whole-sequence GDN when it fits. Print the
  choice. Keep the flags as overrides.
- **Gain.** Today, 2.5x on the shared card (VERIFIED-BY-RUN: 29,888 to 11,078 launches,
  104.9 s to 42.1 s). After K6 it matters less; MLP chunks of 2048 instead of 512 cut the
  MLP launch count 4x.
- **Risk.** Low (memory). At 8K, NF4 has room: a 64-layer run at 8K measured 35.9 GiB at
  seg 128. bf16 at 16K and above must stay at small segments.
- **Gates.** `crates/qwen35/tests/lora_gdn_segments.rs`, `lora_seq_len.rs`,
  `finetune_streamed_guard.rs`, plus a unit test of the sizing function (monotone in free
  memory, never above the 4 GiB binding cap, never below `conv_kernel - 1`).

**Q4. Register the tiled rmsnorm rows kernels for training.**
- **Change.** `rmsnorm_rows` / `rmsnorm_dx_rows` are unregistered (`model.rs:527,545`), so every
  norm is one thread per 5120-element row. Register them in `training_kernel_list` and select
  them where `caps.workgroup_reductions` holds, as `matmul_reg3` is.
- **Gain.** Small today (norms are 1% of launches) but needed for the 120 us glue budget.
- **Risk.** Low.
- **Gates.** `crates/qwen35/tests/rmsnorm_variant_agreement.rs`.

**Q5. Make the probe a benchmark instrument.**
- **Change.** In `crates/qwen35/src/bin/qwen35_train_probe.rs`:
  - add `--seed` (the seed is `Rng::random()` today, so runs are not the same workload);
  - add `--profile-replays N` via `gpu_core::profile::set_program_replays`, plus a
    per-layer `program_table` so kernels are ranked by the minimum over replays (§E.0a);
  - print the dispatch count per step (a backend counter) and the number of other compute
    contexts seen by `nvidia-smi` at start and end;
  - print tokens/s excluding step 1.
- **Gain.** No speed-up; it is the precondition for every measurement below.
- **Risk.** None.
- **Gates.** The probe's own flag test.

**Q6 (after K1). Producers write bf16 operand shadows.** rmsnorm, SwiGLU and the mixers write
the bf16 copy of the next GEMM's input directly, which removes the cast kernels of K1 (about
20 us/token, DERIVED). Do it only after K1 measures the casts as a visible row.

### Tier K: kernel work (weeks; tensor-core / CUDA-native)

**K1. bf16 tensor-core GEMMs through cuBLASLt, for base forward, base dX and the head.**
This is the single largest item and gates every milestone past 100.

- **What changes.**
  - New `crates/backend-cuda/src/cublaslt.rs`: `dlopen` `libcublasLt.so.13` exactly as
    `driver.rs` loads `libcuda`, with a per-context handle, a 32-64 MiB workspace from the
    pool, and cached matmul descriptors and heuristics per (m, n, k, transposes, dtypes).
  - A new step kind that enqueues `cublasLtMatmul` on the handle's own stream, in order with
    kernel steps, and stays capturable by the graph path. cuBLASLt is stream-capture safe when
    the workspace is preallocated.
  - Typing: A bf16, B bf16, C/D f32, compute `CUBLAS_COMPUTE_32F`.
  - `crates/gpu-core/src/provider/cublaslt.rs`: a `CudaLibraryProvider`, ranked before
    `CudaProvider` in `ProviderRegistry::for_gpu` (`provider/mod.rs:390`). It declines
    cleanly when the library is absent, recorded as a `Decline` (`cuda.md` "resolution
    principle").
  - A new `ImplSource::Library`.
  - Extend the provider seam with `MatMulDx` and the row-windowed forms. `Ops::matmul`
    (`crates/model/src/ops.rs:1353`), `matmul_dx` (`:1961`), `matmul_rows` (`:2026`) and
    `matmul_dx_rows` (`:2066`) offer bf16-weight requests to the provider first.
  - The fp32 activation operand is converted by a native `cast_f32_bf16` kernel
    (`kernels-cuda/cu/plumb_f32.cu` neighbourhood) into a reusable scratch.
  - dX uses the transposed-weight op flag, so no weight copy is made.
  - LoRA's later B·mid term can accumulate into the same D (beta = 1); see K3.
- **Precision policy.** A new tier choice
  `TrainOpts::gemm_precision = Fp32Reference | Bf16TensorCore`. On CUDA with the library
  present the default is the tensor-core tier; the CPU backend, gradcheck and every existing
  parity test pin `Fp32Reference`. The rounding of activations and output gradients to bf16
  (fp32 accumulation) is the standard mixed-precision recipe for LoRA/QLoRA (QLoRA, arXiv
  2305.14314, computes in bf16). It is a tier, never a silent change.
- **Gain (DERIVED).** 146 GFLOP/token at about 500 TF effective is 0.29 ms/token, against
  about 10 ms/token today, and removes the 175-215 tokens/s structural cap. The skinny
  shapes (b/a projections n = 48, k/v n = 1024) run far below peak but are 4% of the FLOPs.
- **Risk.** Medium. The risks are the library-step plumbing in graph capture, alignment
  (cuBLASLt wants 16-byte pointers and leading dimensions that are multiples of 8 for bf16;
  row-window offsets are already 256-byte aligned, finding 216), and the 4 GiB binding logic
  not applying to raw pointers.
- **Correctness.**
  - Keep `crates/model/tests/ops_nf4_weight.rs`, `matmul_rows.rs`,
    `ops_matmul_step_identity.rs` and `crates/gpu-core/tests/cuda_provider_matmul.rs` green.
  - New `crates/gpu-core/tests/cuda_library_gemm.rs`: bf16 x bf16 to f32 against the fp32
    reference on random shapes covering the 27B set (m in {1, 7, 64, 512, 8192}; (n, k) from
    the projection table), at 1e-2 of output RMS. Also against an f64 oracle at small size,
    and a decline-is-recorded test with the library forced absent.
  - The trajectory gate of section 6.

**K2. NF4 through the same GEMM: dequantize per projection into a bf16 scratch.**
- **What changes.**
  - New native `nf4_dequant_bf16` kernel (`kernels-cuda/cu/nf4_dequant_bf16.cu`), decoding
    with the single `kernels::template::NF4_CODEBOOK`.
  - The provider decodes the requested weight (or row window) into a reusable scratch sized
    to the largest projection: MLP gate/up/down 5120 x 17408 x 2 B = 178 MB. It then issues
    the K1 GEMM.
  - A weight-reuse cache keyed by weight identity lets the forward and recompute share a
    decode when the buffer is still valid. The cache is optional; its gain is small.
- **Gain (DERIVED).** It adds 7 us/token at the target (23 MB/token at 3.4 TB/s) and makes the
  NF4 and bf16 tiers equal-speed.
- **Why not a fused mixed-input tensor-core kernel?** A fused W4A16 `mma.sync` GEMM saves the
  7 us, at the cost of a native, tuned kernel at 60-70% of what cuBLASLt reaches
  with wgmma. Defer it (section 5).
- **Risk.** Low.
- **Gates.** A new unit test that the dequant is bit-exact against
  `model::lut4` host decode on a whole 5120 x 17408 matrix, plus `ops_nf4_weight.rs`, plus
  `crates/qwen35/tests/lora_nf4_base.rs`.

**K3. LoRA on the library GEMMs, with shared-input A concatenation.**
- **What changes.**
  - In `model.rs` `lora_fwd` (`:1772`) and `proj_bwd` (`:1904`), replace the named naive
    `matmul`, `matmul_dw`, `matmul_dx`, `axpy` and `grad_scale` with `Ops` calls that take the
    provider: fp32 or tf32 GEMMs are fine at r = 16, which are bandwidth-bound.
  - Concatenate the A matrices of projections that read the same input (GDN qkv/z/b/a; GQA
    q/k/v; MLP gate/up) into one [T, sum r] GEMM.
  - Fold the alpha/r scale into the GEMM's alpha.
  - Accumulate B·mid directly into the base output with beta = 1, which removes `axpy` and the
    `lora_out` scratch.
  - Compute dA and dB with the library's transposed operands (k = T reductions are what
    tensor-core GEMMs do well, unlike the serial `matmul_dw`).
- **Gain.** It cuts LoRA from 1,787 launches per 4 layers (VERIFIED-BY-RUN count) to about 400,
  and from 10-40 s per 8K step to under 0.3 s (DERIVED).
- **Risk.** Low-medium. The adapter tensor layout must not change: concatenate views by
  offset, not by re-storing, so `lora_adapter_file.rs`, `lora_roundtrip.rs` and
  `lora_resume_gate.rs` stay green.
- **Gates.** `gradcheck::check_qwen35_lora` (CPU, fp32 reference tier) and
  `crates/model/tests/lora_runtime_delta.rs`, plus a new test that LoRA gradients on the
  tensor-core tier agree with the reference tier within 1e-2 relative on a tiny config.

**K4. Fused head CE gradient in the forward.**
- **What changes.** In `HeadCe` (`crates/model/src/head_ce.rs:335-434`), the loss scale (1/count
  and the logit scale) is known before the backward. So compute each chunk's CE gradient and
  apply the head dX while the chunk's logits are live in the forward, storing `d_hidden` for
  the backward. The backward no longer recomputes logits. This is the "fused linear
  cross-entropy" idea (Cut Your Losses in Large-Vocabulary Language Models, arXiv
  2411.09009).
- **Gain.** It removes one head GEMM per token (2.5 GFLOP) and 8 x 1024 x 248,320 fp32 logit
  passes. It is small after K1 (about 1%), but it also removes a backward dependency on
  `self.gpu`-recorded chunks (knowledge 232).
- **Risk.** Medium. A per-token weight or logit scale set after the forward (the
  `TokenObjectives` hooks of `training-27b.md`) must still be honoured. Keep the old path for
  objectives that set weights after the forward.
- **Gates.** `crates/model/tests/head_ce.rs` (chunked equals unchunked) and
  `crates/qwen35/tests/lora_head.rs`.

**K5. Tensor-core flash attention for training at head_dim 256.**
- **What changes.** New native `flash_gqa_train_bf16_hd256.cu` with three entries: forward
  writing O and LSE, backward dQ, and backward dK/dV.
  - The algorithm is FlashAttention-2 (arXiv 2307.08691): causal block skipping, GQA by
    indexing kv head = q head / group, fp32 softmax statistics, bf16 Q/K/V/P/dO operands,
    `mma.sync.m16n8k16` with fp32 accumulators, and `ldmatrix` / `cp.async` staging.
  - It starts from the working fp16 hd256 MMA prefill (`kernels-cuda/cu/flash_prefill_f16_hd256.cu`).
    hd256 needs 64-row q tiles and 64-key k tiles to fit in 227 KiB of shared memory.
  - The drop-in is at the GQA mixer call sites (`crates/model/src/gqa_mixer.rs:127-160,
    540-654`), through a `flash_gqa_train_step`-style provider hook that answers `None` (so the
    portable trio runs) on any decline, as `paged_flash_prefill_step` does.
  - The Hopper-specific wgmma/TMA path (FlashAttention-3, arXiv 2407.08608) is not needed for
    the budget.
- **Gain (DERIVED).** It takes attention from about 15 s to about 0.3-0.5 s per 8K step at
  200-300 TF, and removes the 540 tokens/s cap. At 32K it is the difference between about
  60 s and 1.5 s per step.
- **Risk.** Medium-high. The backward is the hard part (dS = P ∘ (dP - D) with
  D = rowsum(dO ∘ O), atomics-free dK/dV accumulation by iterating q-tiles per k-tile).
- **Gates.** Keep `crates/model/tests/flash_gqa_train.rs` (portable trio as reference) and
  `crates/qwen35/tests/lora_flash.rs` green. Add a new
  `crates/gpu-core/tests/cuda_flash_gqa_train.rs`: forward, dQ, dK and dV against the fp32
  trio and an f64 oracle at T in {64, 257, 1024, 4096}, on causal ragged tails, with the
  tolerance of the existing prefill gate (1.2e-2 of output RMS). Mutation-verify it by
  swapping the kv-head mapping.

**K6. Native chunked GDN training kernels: forward with state history, plus the reverse sweep.**
- **What changes.**
  - Add a training mode to `kernels-cuda/cu/gdn_chunk_loop_f32.cu`, or a sibling
    `gdn_chunk_train_fwd_f32.cu`, that also writes `state_history[ci]` (the state entering
    each chunk) and the `decay_scale` history the backward reads (knowledge 252). It replaces
    the 11-per-chunk host walk in `gdn_chunk_fwd_train` (`crates/model/src/gdn.rs:1073`).
  - New `gdn_chunk_bwd_loop_f32.cu`: one persistent launch for the reverse sweep of
    `gdn_chunk_bwd` (`gdn.rs:1305`). It carries dS across chunks in shared memory (head split
    over 4 blocks of 32 value columns as the forward loop does, 192 blocks). It recomputes the
    four per-chunk working buffers inside the kernel, as the host sweep does today.
  - The whole-tensor phases (UT transform, intra-chunk products) stay batched launches
    (`gdn_ut_fwd`, `bmm_tiled`); they are already parallel over chunks.
  - Fold the five `gdn_layout_permute` calls into strided reads in these kernels.
  - Then set Q3's default segment to whole-sequence where memory allows.
  - The algorithm is the chunkwise-parallel delta rule with WY/UT representation (Parallelizing
    Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484) and its
    gated form (Gated Delta Networks, arXiv 2412.06464).
- **Gain (DERIVED).** It cuts GDN launches from about 830 K to about 15 K per 8K step, and the
  GDN time from about 7 s (exclusive card; minutes on a shared one) to about 0.6-0.8 s.
- **Risk.** Medium. The forward kernel exists and is bit-identical to the walk; the backward
  kernel is new.
- **Gates.** Keep `crates/model/tests/gdn_chunk_bwd.rs` (including `gdn_chunk_bwd_gradcheck`,
  f64 FD), `gdn_chunk_fwd.rs`, `gdn_chunk_fwd_fast.rs`, `gdn_mixer_segmented.rs`,
  `gdn_mixer_equivalence.rs` and `crates/qwen35/tests/lora_gdn_segments.rs` green. Add new
  tests:
  - the native training forward is bit-identical to the walk (state history included) at chunk
    lengths 64..1 and at a ragged last chunk;
  - the native backward is bit-identical to the host sweep (same fp32 summation order, no
    contraction), as `gdn_chunk_loop_f32.cu` is to its walk;
  - mutation-verify by perturbing the decay index.

  `crates/model/tests/gdn_mixer_train_cost.rs` (ignored, CUDA) records launches and peak
  before and after.

**K7. Fused elementwise epilogues (the glue budget).** These are native kernels in
`kernels-cuda` that fuse:
- residual add + rmsnorm + bf16 operand write;
- SwiGLU forward writing the bf16 `h` shadow, and its backward;
- partial rope + q/k norm;
- sigmoid gate + mul in the GQA mixer;
- GDN conv1d + silu + l2norm;
- the bf16 residual pack/unpack inside the norm.

Each one replaces 2-5 launches and 2-4 tensor passes.
- **Gain (DERIVED).** Glue from about 135 MB/token at less than 50% of bandwidth to about
  70 MB/token near it, i.e. 120 us to 40-60 us/token. Needed only for the 2,000 stretch.
- **Risk.** Medium (many small kernels). Rule §E: count integer divisions per element and use
  per-plane blocking (`bn_elem_f32.cu`, `plumb_f32.cu`).
- **Gates.** Each kernel is bit-identical to the composed sequence where the ops are exact.
  Where an fp32 sum is reassociated, use the existing variant-agreement tests and the
  trajectory gate.

**K8. Launch and allocation overhead: arena, parameter ABI, per-layer graphs.**
- **What changes.**
  1. A per-step arena on the training path (`cuda.md` M6). Allocations are taken from a bump
     arena reset per step, so buffer addresses repeat step to step and the 64 per-layer
     `poll_wait`s (`model.rs:2985`) go away.
  2. Pass uniforms as `__grid_constant__` kernel arguments instead of a per-launch
     `cuMemcpyHtoDAsync` (`cuda.md` "Not done": a measured 4.8 us bubble per launch).
  3. Capture each layer's forward, rebuild and backward as frozen programs on step 1 and
     replay them on later steps (`backend.rs:1446,1526` `freeze`/`launch_program`). Fix the
     `Frozen::capture_inner` `launch_raw` defect first: it drops dynamic shared memory and
     clusters (`graph.rs:743`), which the K5 and K6 kernels need.
- **Gain (DERIVED).** After K6 there are about 25 K launches at 5-10 us, which is 0.12-0.25 s
  per step (15-30 us/token). Graphs bring that to a few us/token. It is the last 3-10%.
- **Risk.** High (backend internals, a shared engine used by every model).
- **Gates.** `crates/backend-cuda/tests/block_cache.rs`, `crates/qwen35/tests/cuda_leaks.rs`,
  the CUDA graph tests in `crates/backend-cuda`, and the trajectory gate with graphs on and off
  (`BRAIN_CUDA_GRAPHS=0`), which must be bit-identical.

---

## 4. Milestones (each measured with the section 7 protocol on W8K and the real checkpoint)

| milestone | contains | expected W8K tokens/s (DERIVED, exclusive card) | pass condition |
|---|---|---:|---|
| M0 baseline | Q5, a clean run on an exclusive card | 20-70 | the per-kind table at min of replays, dispatch count and tokens/s are recorded in a knowledge entry |
| **M1 >= 100** | Q1, Q2, Q3, K1 (bf16 tier), K3 | 150-300 (flash, about 15 s, and GDN, about 7 s, now dominate) | 100 tokens/s for both tiers once K2 lands in the same milestone; GEMM rows at 300 TF or more in the replay table |
| **M2 >= 300** | K2 (NF4), K6 (GDN native, whole-sequence segments at 8K) | 400-500 | launches per step at or below 60 K; GDN at or below 1.5 s per step |
| **M3 >= 600** | K5 (flash TC fwd and bwd) | 900-1,200 | flash at or below 0.6 s per step at 8K |
| **M4 >= 1,000** | K4, Q4, Q6, K8 (arena and parameter ABI; graphs if needed) | 1,100-1,500 | 1,000 tokens/s on W8K AND on the real checkpoint through the CLI (bf16 and NF4); trajectory gate green |
| M5 1,500-2,000 (stretch) | K7, GEMM shape tuning (cuBLASLt heuristic sweep, MLP chunk 4096, gate+up as one [2ff, d] GEMM, q/k/v as one), K8 graphs | 1,500-2,000 | recorded, not promised |

Status after lane T1: M1 (>= 100) is met on W8K for the 4-bit base (550-560 tokens/s idle),
M2 (>= 300) on the same workload; M3's pass condition (flash at or below 0.6 s per step at 8K) is met
by the blockwise attention (about 0.75 s DERIVED from 6.7 + 33 ms per layer, 16 layers: not met
strictly); M4 (1,000 tokens/s on W8K and through the CLI) is **not reached**. The bf16 base and the
real-checkpoint CLI were not measured at W8K.

At 32K (W32K) each milestone is expected at about 85% of its W8K figure once K5 lands; before
K5 the attention T² term dominates at 32K.

---

## 5. Dependency order, lanes and file ownership

Dependency order: Q5, then {Q1, Q2, Q3}, then K1, then {K2, K3, K4}. K6 and K5 are
independent of K1 and of each other and can start immediately. K7 follows K1 and Q6. K8 comes
last.

Five lanes, with minimal overlap:

| lane | items | owns (write) | touches lightly (rebase-merge) |
|---|---|---|---|
| **T1 trainer** | Q5, Q1, Q2, Q3, Q4, then K3's call-site half, K4 call sites | `crates/qwen35/src/model.rs`, `finetune.rs`, `bin/qwen35_train_probe.rs`, `crates/qwen35/tests/*` | `crates/model/src/head_ce.rs` (K4) |
| **T2 GEMM library** | K1, K2, K3's GEMM half (Ops entry points) | `crates/backend-cuda/src/cublaslt.rs` (new), `crates/gpu-core/src/provider/{cublaslt.rs (new), mod.rs}`, `crates/model/src/ops.rs`, `kernels-cuda/cu/{cast_f32_bf16, nf4_dequant_bf16}.cu` | `crates/backend-cuda/src/backend.rs` (the library step only), `kernels-cuda/src/lib.rs` (append registry rows) |
| **T3 flash** | K5 | `kernels-cuda/cu/flash_gqa_train_bf16_hd256.cu`, `crates/gpu-core/src/provider/cuda.rs` (one hook), `crates/model/src/gqa_mixer.rs` | `kernels-cuda/src/lib.rs` (append) |
| **T4 GDN** | K6 | `kernels-cuda/cu/gdn_chunk_*`, `crates/model/src/{gdn.rs, gdn_segment.rs, gdn_mixer.rs}` | `kernels-cuda/src/lib.rs` (append) |
| **T5 backend** (starts after T2 merges) | K8, then K7 | `crates/backend-cuda/src/{backend.rs, graph.rs, exec.rs}`, `crates/wgsl-cuda` (parameter ABI), new fused kernels | everything else read-only |

Conflict rules:
- `kernels-cuda/src/lib.rs` `ALL` is append-only per lane. Regenerate
  `make check/generated/kernels-cuda-table` on rebase.
- T1 and T2 meet only at the `Ops` signatures. T2 lands the provider behind the existing
  `Ops` methods, and T1 switches LoRA call sites to them after T2 merges.
- Integrate in the order T1(Q*), T2, T4, T3, T1(K3/K4), T5. Rebase, never merge.

**Not to be attempted:**
- **Tuning the generated WGSL GEMMs or the fp32 native GEMMs for this workload.** Their
  ceiling (about 33 TF without FMA, 45 TF with) caps the step at about 200 tokens/s.
- **Turning `--fmad=false` off globally.** It breaks the cross-backend bit-parity gates and
  buys at most 2x on a path that has to be replaced anyway.
- **Hand-writing a wgmma/TMA GEMM or a fused W4A16 kernel before K1/K2 are measured.**
  cuBLASLt already uses wgmma on this card (833 TF measured). A fused NF4 kernel saves about
  7 us/token of 665.
- **FP8 training GEMMs**, which bring a numerics risk with no budget need, and cuDNN fused
  attention through the backend API, which needs heavy plumbing; K5 is enough.
- **Dropping recompute at 8K.** One GDN layer's whole-sequence activations are about
  1.3 MiB/token (knowledge 250); 64 layers do not fit.
- **Multi-GPU, NCCL, tensor or pipeline parallelism** (one card), and Vulkan/wgpu paths for
  the 27B.
- **CUDA graphs as a substitute for K6.** A graph removes host launch cost, but not the
  microseconds-per-kernel GPU floor of 0.87 M serial tiny kernels, nor the time-slice billing
  on a shared card.
- **Token-budget packing to make GEMMs bigger.** At batch 1, T = 8192 they are already
  m = 8192 (m = 512 for MLP chunks, fixed by Q3 instead).
- **Changing the stored formats** (bf16 residual stream, NF4 base, adapter file layout).

---

## 6. Correctness gates that must stay green

These run in the same commit as each item. Build through the Makefile; never run a bare
`cargo test -p` (each integration binary is about 470 MB).
- **Gradient truth (fp32 reference tier).** `make test/gradcheck`, which includes
  `gradcheck::check_qwen35`, `check_qwen35_lora` and `check_qwen35_mtp`, and
  `crates/model/tests/gdn_chunk_bwd.rs` (`gdn_chunk_bwd_gradcheck`, f64 FD).
- **Tier tolerance.** `make test/conformance` ("each frozen-base tier within tolerance of
  f32"; LoRA B = 0 leaves logits bit-identical; chunked head equals unchunked), plus the qwen35
  tests `lora_bf16_base.rs`, `lora_nf4_base.rs`, `lora_bf16_residuals.rs`, `lora_recompute.rs`,
  `lora_gdn_segments.rs`, `lora_flash.rs`, `lora_head.rs` and `lora_seq_len.rs`.
- **Kernel parity.** The existing `gpu-core/tests/cuda_provider_matmul*.rs`,
  `cuda_flash_prefill.rs`, `model/tests/flash_gqa_train.rs`, `gdn_chunk_fwd_fast.rs`,
  `ops_nf4_weight.rs` and `matmul_rows.rs`, plus each new kernel's test (K1-K7) against both
  the portable kernel and an f64 oracle.
- **New trajectory gate**, the one that catches a drifted reduction (§E.4).
  `crates/qwen35/tests/train_tier_trajectory.rs` runs 30 steps of a tiny real-shape-ratio
  config (2 GDN + 1 GQA layers at d = 256, head_dim 256, vocab 4096, rank 8) on fixed data
  and seed, CUDA only, skip-if-absent. It asserts:
  - the tensor-core tier's loss curve stays within 2% relative of the fp32 reference tier
    at every step;
  - the curve is monotone-decreasing over windows of 5 steps;
  - the final LoRA weights have cosine 0.99 or more to the reference.

  Run it with graphs on and off, and with segments whole and 64.
- **Real-weight sanity at the M4 gate.** The 40-step memorisation run of knowledge 234 (NF4,
  128 x 2) reaches loss 0.01 or less on the tensor-core tier, and the knowledge 280 adapter
  check (the trained adapter changes the greedy answer) still passes.

---

## 7. Benchmark protocol (fixed workloads)

**Workloads** (`qwen35_train_probe` with Q5's flags, release build, `BRAIN_BACKEND=cuda`):
- **W4:** `--layers 4 --tokens 4096 --rank 16 --seed 7 --steps 4 --bf16-residuals on`,
  `--base nf4` and `--base bf16`, with the segment and MLP chunk Q3 picks (also record
  `--gdn-segment 128 --mlp-rows 512` for continuity with section 1.2).
- **W8K (headline):** `--layers 64 --tokens 8192` with the same flags.
- **W32K:** `--layers 64 --tokens 32768 --base nf4` (bf16 does not fit 32K).
- **Real:** `brain qwen35 finetune <8 rows of 8192 tokens> --base-dtype nf4|bf16 --batch 1
  --seq-len 8192 --steps 4 --lora-rank 16`, counting tokens/s over steps 2-4 (excluding the
  161 s load).

**Conditions:**
- Hold `~/project/shared/containers/gpu.lock` for the whole run.
- Record `nvidia-smi pmon -c 5` before and after; a run is valid only if no other compute
  context exceeds 5% SM.
- Warm up (step 1 discarded); report the minimum and the median of steps 2-4 over 3 process
  runs.
- Also run once with `BRAIN_PROFILE=1 --profile-replays 5` for the per-kind table (rank by
  replay minimum, never by eager per-launch time), and record the dispatch count.

**Record:** a knowledge entry per milestone with tokens/s (min and median), the device-time
table, launches per step, peak MiB, step MiB/token and the commit hash. Update the
milestone table here.

---

## 8. Measurement log

| date | commit | workload | tokens/s | notes |
|---|---|---|---:|---|
| 2026-10-16 | c8a33d4e | W4 NF4 seg128 mlp512 (4 layers, 4096) | 38-49 | VERIFIED-BY-RUN, shared card (4 other contexts), 29,888 launches |
| 2026-10-16 | c8a33d4e | W4 bf16 seg128 mlp512 | 49 | VERIFIED-BY-RUN, shared card |
| 2026-10-16 | c8a33d4e | W4 NF4 seg2048 mlp512 | 97 | VERIFIED-BY-RUN, shared card, 11,078 launches, 1.04 MiB/token |
| 2026-10-16 | c8a33d4e | Real NF4 CLI, 8192 x 1 | below 3 | VERIFIED-BY-RUN (other lane's run, first step still running after about 60 min, shared card) |
| 2026-10-16 | 258eaba9 | 64 layers, 8192, NF4 / bf16 (knowledge 234) | 20 / 19 | earlier record |
| 2026-10-16 | t1 series | W4 bf16, old binary vs new, back to back, shared card | 82-186 vs 897-3168 | VERIFIED-BY-RUN, `other_contexts` 1-2; ratio 10-30x; the absolute rate varies 3x with the neighbours |
| 2026-10-16 | t1 series | W8K NF4 (64 layers, 8192), library GEMMs + attention + batched GDN backward | 34 | VERIFIED-BY-RUN, shared card, 121,247 dispatches, 238-243 s/step |
| 2026-10-16 | t1 series | W8K NF4, + freed blocks kept | 53 | VERIFIED-BY-RUN, shared card, 149-158 s/step, peak 51.8 GiB |
| 2026-10-16 | t1 series | W8K NF4, + cooperative grad-norm, CE and decay-gradient kernels | 469 | VERIFIED-BY-RUN, idle at start, gpu lock, 17.5 s/step, 114,207 dispatches |
| 2026-10-16 | t1 series | W8K NF4, + cooperative RMSNorm | 550-560 | VERIFIED-BY-RUN, idle at start (a context arrived in step 3), 14.7-14.9 s/step |
| 2026-10-16 | 6edaafde | W8K NF4 probe, + native GDN backward sweep | 611 | VERIFIED-BY-RUN, idle, 13.42 s/step, 77,151 dispatches, peak 49,973 MiB (knowledge 311) |
| 2026-10-16 | 6edaafde | Real NF4 CLI, 8192 x 1, Qwen3.8-27B-FP8, defaults | 621 | VERIFIED-BY-RUN, idle, 13.2 s/step (steps 2-3), peak 44.4 GiB sampled (knowledge 311) |
| 2026-10-16 | 6edaafde | Real bf16 CLI, 8192 x 1, freed blocks off | 422 | VERIFIED-BY-RUN, idle, 19.4 s/step, peak 65.8 GiB sampled |
| 2026-10-16 | 38141fe5 | Real bf16 CLI, 8192 x 1, freed blocks kept, segment/rows 2048 | 640 | VERIFIED-BY-RUN, idle, 12.8 s/step, peak 79.8 GiB sampled |
| 2026-10-16 | 38141fe5 | W8K bf16 probe, 2048/2048, freed blocks kept | 627 | VERIFIED-BY-RUN, idle, 13.07 s/step, 74,191 dispatches, peak 81.4 GiB |
| 2026-10-16 | 465448ae | W8K NF4 probe (64 layers, 8192, seg/rows 2048, seed 7), before T1, host-loaded node | 385 | VERIFIED-BY-RUN, idle card (`other_contexts` 0) but node load 10-50: the host thread is the bound (`cpu_s` = 21 s of a 21.3 s step, 77,151 dispatches); device time 10.76 s (knowledge 314) |
| 2026-10-16 | 22aa0d99 | W8K NF4 probe, tensor-core GDN forward, same node and flags | 440 | VERIFIED-BY-RUN, same conditions, 18.5 s/step, 64,479 dispatches, device time 8.61 s (GDN chain 5.09 to 3.28 s), losses 13.4607 / 13.4012 / 13.3244 as before |
| 2026-10-16 | 465448ae / 22aa0d99 | W8K bf16 probe, before / after | 407 / 463 | VERIFIED-BY-RUN, same conditions; device time 10.45 s / 8.31 s |
| 2026-10-17 | adf422e8 / 550a0862 / 4920b765 | W8K NF4 probe, step 2 device time: before / tensor-core chunk backward / + layout fold | 405 / 447 / 472 (wall, host-bound) | VERIFIED-BY-RUN, idle card, node load 14-25: device 8.68 s / 7.65 s / 7.22 s, GDN chain 3.29 / 2.59 / 2.22 s, dispatches 64,479 / 56,031 / 54,303, losses equal to the fourth digit (knowledge 316) |
| 2026-10-17 | 14ae29b9 | block cycles of the four tensor-core GDN kernels, before / after tuning | n/a | VERIFIED-BY-RUN (`clock64` inside the kernels, card shared with a decode server): post 311K to 156K, loop 97K to 61K per 2 chunks, sweep 100K to 68K per 2 chunks, prefix 117K to 80K; DERIVED idle step about 6.0-6.2 s, GDN chain about 1.1 s (knowledge 316) |
| 2026-10-16 | t1 series | library GEMM / blockwise attention component rates | 343-819 TFLOP/s; attention 6.7 + 33 ms/layer at 8K | VERIFIED-BY-RUN, `cuda_library_bench.rs`, minimum over runs |
| 2026-10-19 | 8be9f2f96 | 27B Q8_0 GGUF prefill round (4096 deep, 512 rows, int8 KV): before / after flash v2 + fused front end + no layer drain | 2371 / 2969 | VERIFIED-BY-RUN, exclusive card 1, `runs/b27/window6.log` (216.0 ms / 172.5 ms; knowledge 327) |
| 2026-10-19 | 8be9f2f96 | W8K NF4 train probe, 64 layers, after `storage_uninit` (cleared bytes 52 -> 10.8 GiB per step at 4 layers x 4096) | 990-1010 | VERIFIED-BY-RUN, `runs/b27/window6.log`: 8.1-8.3 s/step, device kernels 4.86 s, `library_gemm` 56% of them; the 15 s step 4 is the profiled step |
| 2026-10-29 | 3c498e009 / 5ad040e22 | W8K NF4 train probe (64 layers, 8192, seg/rows 2048, seed 7, `--device gpu1`), before / after the serving host-cost work (parameter block, uninit clears, event reuse, no per-shape uniform; knowledge 381) | 936 / 968 | VERIFIED-BY-RUN, gpu1 idle (one other context counted), warm steps 2-3: 8.58-8.93 / 8.45-8.48 s, same 46,624 dispatches, alloc calls 84.5k / 37.5k, zeroed buffers 19.9k / 11.1k a step, peak 48,148 / 39,988 MiB, losses 13.4071 / 13.3340 both; the gain is smaller than serving's because the step is device bound (wait 5.8 s of 8.5) |
| 2026-10-30 | 241303005 + checkpointed forward | W8K NF4 train probe (64 layers, 8192, seg/rows 2048, seed 7, `--device gpu1`), before / after the bf16 GDN prefill form taking the no-save (checkpointed) forward | 937 / 981 | VERIFIED-BY-RUN, gpu1 idle (one other context counted), warm steps 2-4: forward 2.11 / 1.92 s, same 46,624 dispatches, losses 13.4643 / 13.4072 / 13.3342 / 13.2478 against 13.4644 / 13.4073 / 13.3341 / 13.2479; the bf16-operand GDN backward ran only 5-9 % faster (knowledge 384) and was not kept |
