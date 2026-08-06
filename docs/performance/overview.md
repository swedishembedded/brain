# Performance: how brain's CPU & GPU inference got fast

This documents the optimizations that measurably sped up brain's **from-scratch
WGSL** YOLOv8n inference (the `bench_yolo_inference.py` head-to-head vs
Ultralytics/torch), and *why* each one helped. The throughline: **WGSL stays the
single source of truth** — every speedup is either an execution-only optimization
of how a kernel runs, a fused/better kernel that still runs on both backends, or
a structural fix to the host↔device interaction. No kernel math is forked.

The dominant cost in YOLOv8n@640 is **conv2d** (~95% of CPU kernel time at the
start), so most of the work targets conv and the things that surface once conv is
fast.

Use `BRAIN_PROFILE=1` to see per-kernel (CPU) / op-count (GPU) / per-stage
(`[detect] preprocess|forward|postprocess`) breakdowns; the bench surfaces the
engine's stage split and the real GPU adapter line.

---

## Headline results

| Path | Before | After | Lever |
|---|---|---|---|
| **CPU** (yolov8n@640, 22-core AVX2 box) | 7440 ms | ~115 ms | native AVX2 GEMM conv + fusion + native ops |
| CPU conv microbench | scalar JIT | **107 GFLOP/s** (1×1: 151) | im2col→GEMM, cache-blocking, AVX2 microkernel |
| CPU: WGSL→scalar-JIT vs WGSL→SIMD | 6715 ms | 368 ms | the AVX2 fast path (**18×**, same WGSL) |
| **GPU** (Intel Arc, integrated, Vulkan) | 2749 ms | ~630 ms | kill readback syncs + one pass + register-tiled conv |
| GPU conv `forward` stage | 1218 ms (naive) | 545 ms | register tiling (then coalescing) |
| **FLUX.2 klein-4B DiT forward** (1×P40, 1536 tokens) | 12.87 s (6.7% peak) | **2.76 s (35.7% peak)** | lane-split flash attention (29×) + coalesced QK-norm (19×) + conflict-free GEMM |
| FLUX.2 512² 4-step image (2×P40) | 59.8 s | **18.3 s** | the above |
| **FLUX.2 VAE decode** (64² latent → 512², 1×P40) | 6.47 s | **0.87 s** | parallel GroupNorm stats (159×) + attention as GEMM (80×) + conv as GEMM (3.8×) |
| FLUX.2 Qwen3-4B text encode (512 tok, 28 layers) | 1.23 s | **1.06 s** | coalesced RMSNorm (11.4×) + softmax (4.0×) + `matmul_reg3` |
| FLUX.2 512² 4-step image (2×P40, fp32) | 18.4 s | **12.7 s** | the two above |
| FLUX.2 512² 4-step image (**1**×P40, int8) | 13.4 s | **7.6 s** | the two above |
| **GPT decode** GPU kernel time (`gpt gen`, 6×768, 200 tok, 1×P40) | 1259 ms | **996 ms** | coalesced LayerNorm (10.3× on the kernel; 22.9% → 2.8% of the step) |
| **GPT training step** (`gpt train`, 6×768, 2048 rows, 1×P40) | 6.90 s | **0.84 s** | cooperative optimiser grad-norm (2122× on the kernel; 87.2% → 0.3% of the step) — **every model that uses `optim::Optim`** |

Numbers are noisy on shared boxes (thermal/contention); the stable signals are
the **min-of-N GFLOP/s microbench** (CPU) and the **`forward` stage** (GPU).

---

## CPU backend (native Cranelift-JIT + AVX2)

The CPU backend JIT-compiles the WGSL kernels to native code (scalar, one output
per invocation). The optimizations replace the hot kernels' execution with
hand-vectorized native paths, selected by kernel name, each **validated against
the scalar WGSL reference** (unit tests) and gated by runtime AVX2 detection
(`BRAIN_NO_FASTCONV=1` / non-x86 → scalar fallback).

1. **Native AVX2/FMA GEMM conv2d** *(7.7× end-to-end)*. Route conv through an
   im2col + register-blocked GEMM with an AVX2+FMA `4×16` microkernel, rayon
   tile-parallel over spatial-column bands. 1×1 stride-1 convs skip im2col (B
   aliases x). `7440 → 964 ms`.

2. **Per-panel hot im2col + unified microkernel** *(58.7 → 107 GFLOP/s)*. Never
   materialize the full `[Kg,P]` im2col (a ~29 MB write+reread for a mid layer):
   build only the current L2-sized column panel into a reused scratch and consume
   it hot (implicit-GEMM style). One microkernel, parameterized by B/C row
   strides, serves both 1×1 and KxK.

3. **L2 cache-blocking** of the GEMM so a B-panel stays resident across the whole
   output-channel loop instead of re-streaming `[Kg,P]` once per channel tile.

4. **Fuse conv → BN(eval) → SiLU** into one `conv_act` kernel *(forward −26%)*.
   The BatchNorm-eval transform collapses per channel to a constant `scale|bias`
   computed **once** (no per-frame host stat packing); SiLU is applied in the
   GEMM epilogue. Eliminates two full activation memory passes + two dispatches.

4b. **Fuse the detection-head conv + bias** into one `conv_bias` kernel (the
   bias param is bound directly, added in the conv epilogue). Removes the
   separate `bias_add` pass *and* the `pack_bcast` machinery (a host-built
   `[C*HW]` broadcast buffer + a per-frame bias readback — a former GPU sync
   source) per head branch (×6). Works in train and eval (bias read live).

5. **Native memory-bound ops** (concat2 / concat_split / bn_eval / silu /
   upsample2). These were scalar one-element-per-invocation loops dominated by
   per-element index decode and (silu) a per-element libm `expf`. Replace with
   bulk `memcpy` (concat), per-channel affine FMA (bn_eval), AVX2 Cephes
   `exp256_ps` (silu) — coarse rayon chunking so scheduling cost stays negligible.
   **6.7× forward** vs pure-JIT.

6. **Single-pass C2f channel concat** (`chan_place`). The C2f block concatenated
   `[y0,y1,b1..bn]` with a left-fold of concat2 (re-copying the growing prefix —
   O(n²)); place each chunk once into its slice of the output instead. O(n).

7. **Postprocess argmax on raw logits** *(~2×)*. The detection head computed
   `sigmoid` for every (anchor × class) = ~672K `exp`/frame; sigmoid is
   monotonic, so argmax over raw logits is identical — one `sigmoid` per anchor.

**Result:** CPU went 42× slower than torch → meets the plan's `< 129 ms` gate and
beats torch on boxes where torch is ~127 ms. The 3-scenario bench shows the
WGSL→**scalar**-JIT path at 6715 ms vs the WGSL→**SIMD** path at 368 ms — **18×**
from the same kernels, the clean demonstration that the SIMD lowering is the win.

**Tried, kept opt-in (didn't beat the tuned GEMM):** *Winograd F(2,3)*
(`BRAIN_WINOGRAD=1`). The 2.25× multiply saving was eaten by scalar transforms +
under-parallelized transform-domain GEMMs; the AVX2 GEMM (107 vs 201 GFLOP/s
aggregate... the GEMM wins) is faster. Kept as validated scaffolding.

---

## GPU backend (wgpu / WGSL)

On a real GPU the conv math was never the first problem — the **host↔device
interaction** was. Profiling (`BRAIN_PROFILE` op counters) was the unlock.

1. **`--device gpu` actually uses the GPU.** `Yolo::new` hard-coded
   `Gpu::new_cpu`, so GPU mode silently ran the CPU JIT. Honor an explicit
   `--device` (else default CPU for tooling). The bench passes `--device` and
   prints the engine's real `adapter:` line as ground truth.

2. **Kill per-frame readback syncs** *(2749 → 644 ms, the biggest GPU win)*. Yolo
   inference did **~241 host readbacks per frame** — each a full
   `device.poll(wait)` GPU sync. Two constants were re-read every frame: the
   BN-eval collapse (`pack_sb`, ~200 syncs — the model wasn't pinned in eval mode,
   so detect's eval→train flip invalidated the cache), and the head broadcast bias
   (`pack_bcast`, ~6). Pin eval mode + cache both → **241 → 7 readbacks/frame**
   (the 7 are the necessary output reads).

3. **One `queue.submit` per frame**, not ~104. Each block submitted separately;
   accumulate all dispatches and flush once at the terminal readback.

4. **One compute pass per frame**, not ~130. Each submit was its own
   `begin/end` compute pass — and each pass boundary is a GPU pipeline barrier
   that serializes an integrated GPU. Record all dispatches into a *single* pass
   (wgpu inserts the inter-dispatch storage barriers).

5. **Register-tiled fused conv** (`conv_act_reg`) *(1218 → 545 ms `forward`)*. The
   naive one-output-per-thread conv is memory-bound on the integrated GPU's shared
   RAM (it re-reads the whole input once per output channel and the whole weight
   once per position — hundreds of × the minimum traffic). Each thread now
   computes a **4×4 tile** (4 output channels × 4 positions) holding 16 partial
   sums in **scalar** registers (fully unrolled — arrays *spilled to local memory*
   and gave no win), so each tap loads 4 weights (reused across 4 positions) and 4
   inputs (reused across 4 channels): both global-read traffics drop ~4×. No
   workgroup memory → full occupancy.

6. **Memory coalescing.** The 4 positions per thread are strided (not
   consecutive), so adjacent threads access adjacent addresses and the warp's
   reads/writes coalesce (uncoalesced access is a ~2–4× bandwidth hit).

7. **Hoist boundary checks out of the channel loop** *(545 → 222 ms `forward`,
   low-load)*. The padding/bounds checks + input offsets depend only on
   `(kh,kw,position)`, not the input channel, so reorder to `(kh,kw)` outer / `ci`
   inner and compute them once per tap — ~256× fewer bound evaluations for a
   256-channel layer. This was the biggest single conv win after the tiling:
   per-tap overhead, not just traffic, was limiting it.

**Tried, regressed, made opt-in:** *weight-staged tiled conv* (`conv_act_tiled`,
`BRAIN_TILED_CONV=1`). Staging the full weight tile costs up to 32 KiB of
workgroup memory, which **collapses GPU occupancy** — measured *slower* than naive
(~600 → ~1600 ms). The infrastructure it exercises (below) is the real value.

---

## The single-source work-group JIT (solution B)

To let **one tiled WGSL kernel run on both backends**, the `wgsl-cpu` Cranelift
JIT learned the **GPU work-group execution model**: kernels with `var<workgroup>`
shared memory or `workgroupBarrier()` compile to

```
for wg in start/wgsize .. end/wgsize:
    for lid in 0..wgsize:  <segment before the barrier>   # cooperative load
    for lid in 0..wgsize:  <segment after  the barrier>   # per-invocation compute
```

`var<workgroup>` arrays become per-workgroup scratch; the body splits at the
barrier into two per-invocation segment loops; `local_invocation_id` /
`workgroup_id` are synthesized; pre-barrier `let`s used after the barrier are
re-materialized at each segment top to preserve SSA dominance. The CPU dispatcher
hands these kernels workgroup-aligned chunks. Validated: the tiled conv compiles
and matches the scalar reference, and runs the *full* yolov8n correctly through
the JIT work-group path. This is the foundation for a future input+weight tiled
GEMM that serves GPU and CPU from one kernel.

---

## Cross-model finding: `var<function>` arrays are LOCAL memory, not registers

Found while profiling the FLUX.2 DiT (`docs/models/flux2/status.md` P9), but it
is a rule about the engine, not about FLUX.2 — and it cost **81 % of that
model's whole forward**.

A WGSL `var<function> arr: array<f32, N>` is only a register file if the
compiler can prove every index. When the loop bound is a **runtime uniform**
(`for d in 0..p.head_dim`) it cannot unroll, so the array is placed in *local
memory* — which on every GPU brain targets is backed by global memory. The
kernel then runs at memory bandwidth no matter how good its blocking looks.

`flash_attn_bidir` held `q[128]` + `o[128]` per thread this way: ~3
local-memory accesses per 2 FLOP = 6 bytes/FLOP, a 58 GFLOP/s roof on a P40's
346 GB/s. Measured: **70 GFLOP/s = 0.6 % of the card's fp32 peak**. The header
comment's own "one small spill on Pascal" was the tell — the spill was total.

**The pattern that fixes it** (`flash_attn_bidir_split.wgsl`): split the wide
axis across a *lane group* of threads so each thread's array is small AND its
trip count is a compile-time constant, then recombine through shared memory
once per tile rather than per element. Measured **29× at head_dim 128** and a
win at *every* head_dim down to 32, cosine 1.00000000 against the original.

**How to spot it:** an array whose declared size exceeds ~32 f32, indexed by a
loop whose bound comes from `Params`. Grep the kernel tree for
`var<function>`-scope arrays before writing a new one.

Both diffusion DiTs (`flux2`, `zimage`) now dispatch through
`model::block::flash_bidir_step`, which picks the variant from the device's
**queried** `max_workgroup_size`.

## Cross-model finding: one thread per row is a COALESCING bug, not a decode-regime one

The other half of the same profile. The per-element norm kernels give thread
*t* row *t*, so a warp's 32 loads are one row-width apart and each 32-byte
sector fetched serves **one useful float** — 8× read and write amplification.
The workgroup-per-row variants (`rmsnorm_rows`, `softmax_rows`) walk one row
with 64 threads and are coalesced by construction.

`backend_api::select` gated the cooperative RMSNorm on `m <= 32`
(`DECODE_REGIME_MAX_ROWS`) — the reasoning being that large M saturates the
device anyway. That reasoning is wrong, because the loss is per-access
efficiency, not thread count. Measured on a P40 at a fixed 4.7 M elements:

| rows | width | per-element | workgroup-per-row | speedup |
|---:|---:|---:|---:|---:|
| 36 864 | 128 | 3.85 ms (10 GB/s) | 0.20 ms (190 GB/s) | **19.4×** |
| 18 432 | 256 | 4.64 ms | 0.21 ms | 22.6× |
| 4 608 | 1 024 | 0.73 ms | 0.29 ms | 2.5× |
| 1 536 | 3 072 | 1.27 ms | 0.30 ms | 4.2× |
| 512 | 9 216 | 3.02 ms | 0.27 ms | 11.2× |

The cooperative kernel wins at **every** row count and width, so the policy is
now "prefer it whenever the device has workgroup barriers", with a unit test
that fails if the `m <= 32` gate comes back. Narrow rows (QK-norm at
head_dim 128) are the worst case and the most common one.

### LayerNorm: the same fix, measured (was predicted 4-20×, landed at 2.8-10×)

`layernorm`, `ln_stats` and `layernorm_dx` had the identical bug — and
`layernorm_dx` was the worst offender in the tree, walking its row **four**
times from one thread. `layernorm_rows` / `ln_stats_rows` / `layernorm_dx_rows`
give each row a 64-thread workgroup.

One barrier, not two: the CPU JIT splits a kernel body at exactly one
top-level `workgroupBarrier()`, so the textbook two-pass (mean, then squared
deviations) is unavailable. All three use the **shifted** one-pass form with
`K = x[row, 0]`: `mean = K + S1/d`, `var = S2/d - (S1/d)²`. The shift is what
keeps that subtraction free of the cancellation that makes naive
`E[x²] - E[x]²` unusable. `layernorm_dx` needs four reductions that *look*
sequentially dependent (mean/inv feed `mean(g·x̂)`); in the shifted frame
`mean(g·x̂) = inv·(S4 - moff·S3)/d`, so all four accumulate in one pass behind
the one barrier.

Kernel microbenchmark on a P40 (`cargo test -p brain-gpu-core --test
bench_layernorm -- --ignored`), min-of-8 per dispatch, at the shapes these
models dispatch:

| shape (rows × d) | `layernorm` | `layernorm_rows` | dx: `layernorm_dx` | `layernorm_dx_rows` |
|---:|---:|---:|---:|---:|
| 512 × 768 | 0.270 ms (12 GB/s) | 0.045 ms (70) — **6.0×** | 0.425 ms | 0.075 ms — **5.7×** |
| 2048 × 768 | 0.417 (30) | 0.109 (116) — 3.8× | 0.650 | 0.159 — 4.1× |
| 512 × 2048 | 0.710 (12) | 0.089 (94) — 8.0× | 1.067 | 0.144 — 7.4× |
| 2048 × 2048 | 0.874 (38) | 0.244 (138) — 3.6× | 1.247 | 0.389 — 3.2× |
| 512 × 3072 | 1.077 (12) | 0.148 (85) — 7.3× | 1.648 | 0.182 — **9.1×** |
| 2048 × 3072 | 1.574 (32) | 0.355 (142) — 4.4× | 2.561 | 0.534 — 4.8× |
| 1 × 768 (decode) | 0.142 | 0.043 — 3.3× | 0.262 | 0.071 — 3.7× |

The cooperative kernel wins at **every** shape, small rows included, so the
selector rule is RMSNorm's: prefer it whenever `workgroup_reductions` holds,
with a unit test against a row-count gate creeping back. `ln_stats` tracks the
same 2.4-9.0×. Agreement with the reference is ≤ 4.3e-6 relative. (Per-shape
speedups move ±15 % run to run on a shared box; the band is the signal, not
any single cell.)

Two things this does NOT say. First, the per-element kernel only reached
10-40 GB/s of 346, but the cooperative one tops out around 140-200 GB/s, not
346 — LayerNorm reads `x` twice (reduce, then normalise), and caching the row
is exactly the `var<function>`-array trap above, so the second read stays.
Second, the microbench understates the narrow shapes: at ~0.04 ms the
cooperative kernel is at the single-dispatch launch floor, not its own limit
(in situ, batched, the 1-row case measures 10×, not 3.3×).

**In situ, `gpt` training** (6 layers × d 768, batch 8 × block 256 = 2048 rows,
`BRAIN_PROFILE=1`, 5 steps, P40) — `BRAIN_NO_COOP_LN=1` is the A/B switch:

| kernel | per-element | workgroup-per-row | speedup |
|---|---:|---:|---:|
| `layernorm` (130 calls) | 49.8 ms | 9.3 ms | 5.4× |
| `layernorm_dx` (65) | 38.8 ms | 7.0 ms | 5.5× |
| `ln_stats` (65) | 12.3 ms | 2.2 ms | 5.6× |
| **LayerNorm family** | **100.9 ms** | **18.5 ms** | **5.5×** |

`layernorm_dgamma`/`dbeta` are unchanged by design: they are *column*
reductions (thread `c` walks column `c`), so adjacent threads already read
adjacent addresses — they never had the bug.

**End-to-end, honestly:** that 82 ms is **0.6 % of the GPT training step**, and
wall clock agrees (256.3 s vs 257.3 s over 100 steps). Not because the kernel
win is fake, but because **`gradnorm_sq` is 82.3 % of GPT's GPU training time**
(10 768 ms of 13 092 ms over 5 steps, 385 calls — one serial reduction per
parameter tensor). That is the same class of bug one level up; it was out of
scope there and is **fixed in the next section**, which is what finally makes
the LayerNorm win visible in the training step.

Where LayerNorm is not hidden behind that, the win shows up directly.
**`brain gpt gen`** (KV-cache decode, 6 × 768, 200 tokens):

| | per-element | workgroup-per-row |
|---|---:|---:|
| `layernorm` (2665 calls) | 287.7 ms (**22.9 %** of GPU kernel time) | 28.0 ms (2.8 %) — **10.3×** |
| total GPU kernel time | 1258.6 ms | **996.0 ms** — **1.26×** |
| wall (min of 3) | 6.66 s | 6.33 s |

The wall delta is smaller than the GPU delta because a `gpt gen` process spends
~4-5 s loading a 171 MB checkpoint, initialising the device and compiling
shaders before it decodes anything. `val_perplexity 13.2239` on both sides —
identical, not merely close.

Adopted by `gpt`, `pid`, `seq2seq` (explicit indices) and `model::vit` (which
resolves the variants **by name** through `Gpu::kernel_index`, so a ViT model
opts in by adding three kernels to its PIPELINES and no `VitKernelIds` literal
in any model crate changes). `flux2` can adopt the same
`model::block::layernorm_fwd` seam whenever it wants it; its LayerNorm is
~2.4 % of its forward after the DiT work, so it was deliberately left alone
here.

## Cross-model finding: the OPTIMISER was the training step (`gradnorm_sq`, 2122×)

The previous section ends with the number that made it necessary: after
LayerNorm, **`gradnorm_sq` was 87.2 % of all GPU time in `brain gpt train`**
(30 133 ms of 34 545 ms over 5 steps, 6 layers × d 768, batch 8 × block 256).
It lives in `crates/optim` — the AdamW + global-grad-norm-clip path that
**every** trainable model in the repo drives — so this is not a `gpt` finding.

It is a worse bug than the two before it. `flash_attn_bidir` was a
*local-memory* bug and `layernorm`/`gn_stats` were *coalescing* bugs (8× read
amplification). `gradnorm_sq.wgsl` dispatched **one invocation** per parameter
tensor —

```wgsl
if (gidx != 0u) { return; }
for (var i = 0u; i < p.numel; i = i + 1u) { acc = acc + grad[i]*grad[i]; }
```

— and the host dispatched it with `threads = 1`. A 38.6 M-element embedding
gradient is 38.6 M *dependent scalar loads on one lane* of a 3840-core card:
measured **0.08 GB/s, 0.023 % of the P40's 346 GB/s**. Not a fraction of peak —
three orders of magnitude below it. Grep rule, as a companion to "one thread per
row": **a kernel whose first statement discards every invocation but one is a
reduction that never got written.**

**The fix** is the `gn_part`/`gn_stats2` shape one level up: `gradnorm_part`
gives each tensor `n_wg = clamp(ceil(numel/8192), 1, 512)` workgroups that
grid-stride the buffer (so the whole dispatch reads consecutive words),
reduce through 64 f32 of workgroup memory behind **one** barrier (the CPU JIT's
limit), and write one partial each; `clip_coef_wg` folds *every* tensor's
partials into the clip coefficient in one 64-thread second pass. No atomics, no
subgroups — the cross-workgroup combine **is** the second dispatch.

Kernel microbenchmark on a P40 (`cargo test --release -p brain-optim --test
bench_gradnorm -- --ignored`), at the real per-model size distribution — param
*count* and size *skew* both matter, so the table is weighted by how many
tensors of each size the model has:

| numel | ×count | `gradnorm_sq` | GB/s | `gradnorm_part` | GB/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 768 | 74 | 0.056 ms | 0.1 | 0.025 ms | 0.1 | 2.2× |
| 3 072 | 12 | 0.150 | 0.1 | 0.032 | 0.4 | 4.6× |
| 589 824 | 12 | 22.271 | 0.1 | 0.049 | 48.5 | 457× |
| 1 769 472 | 12 | 89.042 | 0.1 | 0.084 | 83.8 | 1 055× |
| 2 359 296 | 24 | 118.289 | 0.1 | 0.087 | 109.1 | 1 367× |
| 38 597 376 | 2 | 1 931.590 | 0.1 | **0.611** | **252.6** | **3 160×** |
| **whole grad-norm, GPT-2-small** | 148 | **8 080 ms** | 0.1 | **7.6 ms** | 85.5 | **1 059×** |
| **whole grad-norm, Qwen3-0.6B** | 311 | **29 885 ms** | 0.1 | **24.3 ms** | 98.3 | **1 232×** |

The large tensors reach **252–277 GB/s = 73–80 % of the 346 GB/s ceiling** — for
a pure read-and-reduce that is the right answer. The *aggregate* rows sit at
85–98 GB/s only because dozens of 768-element tensors each cost one dispatch;
at 0.025 ms those are at this harness's submit+poll floor, not the kernel's.

**In situ, `gpt` training** (same shape, `BRAIN_PROFILE=1`, 5 steps, P40;
`BRAIN_NO_COOP_GRADNORM=1` is the A/B switch):

| kernel | serial | cooperative | |
|---|---:|---:|---|
| `gradnorm_sq` → `gradnorm_part` (385 calls) | 30 133.0 ms | **14.2 ms** | **2 122×** |
| `clip_coef` → `clip_coef_wg` (5) | 0.0 | 0.1 | — |
| share of GPU training time | **87.2 %** | **0.3 %** | |
| total GPU kernel time | 34 545.4 ms | **4 416.3 ms** | **7.8×** |
| wall, 100 steps | 689.6 s (6.90 s/step) | **83.9 s (0.84 s/step)** | **8.2×** |

479.7 MB of gradient per step in 2.84 ms is **169 GB/s (49 % of peak)** in situ —
below the microbench's 253 GB/s because ~50 of the 77 dispatches are tiny
tensors, and because `BRAIN_PROFILE` puts each dispatch in its own compute pass.
The top kernel is now `emb_bwd` (26.8 %), then `matmul_reg2` (17.7 %) and
`attn_bwd_dscores` (16.1 %) — the step is finally *arithmetic*.

**Trajectory equivalence** (100 steps, seed 1337, same binary, A/B by env):

| step | train (coop / serial) | eval (coop / serial) |
|---:|---|---|
| 20 | 8.7030 / 8.7030 | 8.3850 / 8.3849 |
| 40 | 6.9996 / 6.9995 | 6.8814 / 6.8813 |
| 60 | 6.1074 / 6.1072 | 6.1173 / 6.1173 |
| 80 | 5.4573 / 5.4579 | 5.8493 / 5.8485 |
| 100 | **5.5391 / 5.5387** | 5.7005 / 5.7019 |

Max divergence 1.4e-3 absolute (2.5e-4 relative) at step 100 — fp32 noise, and
**the new side is the more correct one**. Against an f64 oracle, the serial walk
keeps one fp32 accumulator across millions of sequential adds and loses the
small terms: at 4.19 M elements it is **2.34e-3** relative off the exact
sum-of-squares while the tree is **2.40e-7** — four orders of magnitude. The
clip coefficient is not merely cheaper now, it is right.

**Two hypotheses the numbers killed.**

*"385 dispatches for one scalar is the deeper problem; fuse them."* — 385 is
77 tensors × **5 steps**, i.e. 77 per step, not 385. After the fix the entire
grad-norm is 2.84 ms of a 840 ms step; a fused pass over a concatenated view
could recover at most the ~50 tiny-tensor launches, under 0.25 % of the step.
It would also require relayouting `ParamStore` from one buffer per parameter
name into an arena — every model binds `ps.g(name)` directly in its backward and
`DeviceBuffer` carries no offset (the same infrastructure gap the concat-fusion
section describes) — so it is a ~10-site change for <0.25 %. Deliberately not
done; the per-tensor cooperative version is the whole win.

*"`clip_coef` on one thread is fine, it only sums a few numbers."* — true of the
old layout (77 tensors, 0.047 ms), false of the new one: the cooperative pass
produces 11 586 partials for GPT-2-small and 54 385 for Qwen3-0.6B, and the
serial fold over those costs 0.475 ms / 1.97 ms versus `clip_coef_wg`'s
0.054 ms / 0.087 ms. Adding the second cooperative kernel was necessary, not
symmetry.

**The siblings in the step were already fine**, and the profile says so rather
than the code: `grad_scale_buf` (3.9 ms/step, 246 GB/s) and `adamw` (12.2 ms/step,
275 GB/s) are one-element-per-thread kernels — adjacent threads read adjacent
addresses, so they never had the bug. (`max_abs_row` *did* have it — one thread
per row — but it is on the int8 activation-quantisation path, not the optimiser,
so it does not appear in any training profile. Fixed in its own section below,
2.1-13.5×.)

**Who inherits this with no change of their own.** `optim::Optim` resolves the
two kernels **by name** through `Gpu::kernel_index`, and the policy lives in
`backend_api::select` (`Op::GradNorm`), so a model opts in by appending
`gradnorm_part` + `clip_coef_wg` to its PIPELINES — indices do not move and no
`Optim::new` call site changes. Done here for all twelve crates that construct
an `Optim`: **`gpt`, `qwen`, `glm`, `moe`, `pid`, `seq2seq`, `autoencoder`,
`lfm`, `yolo`, `depth`, `kronos`, `wm-diamond`**. The size of each one's win is
set by how many parameters it clips, not by its architecture — `yolo` (3 M
params) gains far less in absolute terms than `qwen` (0.6–4 B), but the *ratio*
is the same 3 orders of magnitude on any tensor over ~1 M elements.
`zimage`/`flux2`/`tts`/`nemotron` run their own host-side optimisers and are
untouched; so is `model::parallel`. Its multi-GPU path computes the grad-norm on
the host, and this section's first draft called that "a workaround for this
kernel, now revisitable". **It was revisited and that claim was wrong** — see
"the host grad-norm that is NOT a workaround" below.

There is no size gate, and `select.rs` carries a unit test that fails if one
creeps back. A `numel <= X` fallback to `gradnorm_sq` would be the exact mistake
`Op::RmsNorm`'s old `m <= 32` gate made: the serial kernel costs the same single
dispatch as the cooperative one, so even a 768-element bias is 2.2× faster with
64 lanes than with 1.

## Cross-model finding: the int8 quant's `max_abs_row` had it too (2.1–13.5×)

The section above found `max_abs_row` while profiling the optimiser and left
it: it is on the **int8 dynamic-activation-quant** path, so it cannot show up in
a training profile. It shows up in an inference one. In the FLUX.2 int8
text-encoder forward it was **43.6 ms of 668 ms (6.5%)**, and it runs once per
quantized activation in `qwen::q8`, `zimage::int8`/`block`, and the FLUX.2 int8
DiT — every int8 linear quantizes through `max_abs_row` → `quant_pack` →
`matmul_i8_dyn`.

The kernel is trap C2 verbatim: `sx[m] = max|x[m,:]| / 127` computed from **one
invocation per row**, which is both an 8×-amplified read (a warp's 32 loads are
`k` floats apart) and a serial chain of `k` dependent loads.

**A fast sibling did NOT already exist** — worth stating, because checklist §A
says to look first. `max_abs_part` + `max_abs_final` *are* a cooperative
two-pass max, but they reduce a whole buffer to **one** scale (per-tensor
quant). Per-token scales are why the deep int8 stacks stay accurate — one
outlier token must not crush every other token's resolution — so they are a
different op, not a faster form of this one. `max_abs_rows.wgsl` is new:
`rmsnorm_rows`' shape, 64 threads per row, one barrier, lane 0 folds the 64
partials.

Kernel microbenchmark on a P40 (`cargo test --release -p brain-gpu-core --test
bench_max_abs_row -- --ignored`), min-of-8, at the shapes the int8 paths
actually quantize:

| rows × k | `max_abs_row` | GB/s | `max_abs_rows` | GB/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 × 1024 | 0.108 ms | 19.4 | 0.030 ms | 70.2 | 3.6× |
| 512 × 3072 | 0.314 | 20.0 | 0.049 | 127.2 | 6.4× |
| 512 × 9216 | 0.887 | 21.3 | 0.109 | 172.9 | **8.1×** |
| 1024 × 1024 | 0.147 | 28.6 | 0.051 | 83.1 | 2.9× |
| 1024 × 3072 | 0.352 | 35.7 | 0.093 | 135.2 | 3.8× |
| 2048 × 1024 | 0.167 | 50.2 | 0.079 | 106.8 | **2.1×** |
| 2048 × 3072 | 0.457 | 55.1 | 0.133 | 189.6 | 3.4× |
| 4096 × 1024 | 0.209 | 80.5 | 0.098 | 172.0 | 2.1× |
| 77 × 3584 | 0.290 | 3.8 | 0.041 | 27.1 | 7.1× |
| 77 × 12288 | 1.138 | 3.3 | 0.084 | 45.1 | **13.5×** |
| 8 × 3072 | 0.226 | 0.4 | 0.054 | 1.8 | 4.2× |
| 1 × 3072 | 0.196 | 0.1 | 0.042 | 0.3 | 4.7× |

It wins at **every** shape (worst 2.1×), so the policy is RMSNorm's again —
prefer it whenever `workgroup_reductions` holds, no row-count gate, with a unit
test in `select.rs` against one creeping back. The shape of the curve is the
tell: the reference kernel's throughput *rises* with row count (19 → 80 GB/s
from 512 to 4096 rows, because more rows means more lanes) and *falls* with row
width — exactly what a per-row serial walk looks like. The cooperative kernel is
flat in rows and rises with width, topping out at 190 GB/s, which is the
coalesced-read answer.

**The numerics do not move at all, and that is provable rather than measured.**
`max` is associative *and exact* on floats, so splitting a row across 64 lanes
and re-folding gives the identical bits — unlike the grad-norm's sum, which
reassociates (and became 4 orders of magnitude *more* accurate as a result). The
benchmark asserts `assert_eq!` on the raw scales at every shape, not a
tolerance. Every downstream int8 activation is therefore bit-unchanged by
construction — the only reason a quant-scale kernel is safe to swap
transparently at all.

### The seam: a fast kernel models inherit without editing a dispatch site

The `gn_stats`/`vae` disaster (§A of the checklist: 2262 ms of a 6.5 s decode
lost to a kernel that had already been fixed) says the fix belongs in
*selection*. But both existing selection seams — `backend_api::select` +
`KernelVariant`, and by-name `Gpu::kernel_index` — still require **editing every
dispatch site**, which is precisely the step the next model forgets. Three
models plus a benchmark dispatch `max_abs_row` today, and `crates/flux2` was
owned by a concurrent agent during this work and could not be touched at all.

So the fix went one level lower, in `gpu_core::upgrade`: a small table of
**drop-in** replacements (same `Params`, same bindings, same result, different
thread count). `Gpu` appends the fast kernel to whatever pipeline set a model
registers — at the end, so no existing index moves — and `Gpu::step` /
`step_sliced` / `step_buf` rewrite `(kind, threads)` when `backend_api::select`
(`Op::MaxAbsRow`) says this device wants it. A model that wrote
`gpu.step(K_MAX_ABS_ROW, .., threads = m)` gets 64 threads per row and `m × 64`
invocations, unchanged.

**It worked**, and the proof is `crates/flux2`, whose whole subtree was owned
by another agent and never touched. `flux2_bench tei8 3` on a P40 (the Qwen3-4B
28-layer 512-token INT8 prefill; `BRAIN_NO_KERNEL_UPGRADE=1` is the A/B):

| | reference | upgraded |
|---|---:|---:|
| `max_abs_row` slot (112 dispatches) | 43.5 ms (**6.4%** of the forward) | **4.6 ms** (0.7%) — **9.4×** |
| whole graph, single submit | 0.676 s | **0.635 s** — 1.065× |
| effective rate | 4371 GFLOP/s (37.2% of peak) | 4648 GFLOP/s (39.5%) |

`crates/qwen`, `crates/zimage` and `crates/zimage/tests/int8_matmul.rs` inherited
it the same way (int8 DP4A parity `cosine=0.999985 rel_l2=0.0056` — the *same
digits* on both sides of the A/B, as bit-identical scales require).
`gpu-core/tests/kernel_upgrade.rs` is the regression test, written from the
consumer's side: register only the slow kernel, dispatch the old thread count,
demand the right answer.

**One thing the seam must not do, learned by breaking it.** The first version
put the *appended* pipeline slot into the recorded `StepMeta.kernel`. That
panicked `flux2_bench` — profilers and cost harnesses index `meta.kernel`
through **their own** kernel list, and slot 14 runs off the end of a 14-entry
array. Transparent means transparent: `meta` records the `kind` and `threads`
the **caller** asked for, and only the dispatch moves. Which kernel physically
ran is the backend's record (`BRAIN_PROFILE=1` names the real pipeline). The
table above reads `max_abs_row: 4.6 ms` for exactly that reason, and it is the
more useful A/B for it.

The bar for adding a row to that table is deliberately high, because it is
invisible machinery: identical contract, *identical* results (not "close"), wins
at every shape, capability-gated through `select`. A sum reduction fails rule 2
and belongs in the explicit seams, where the trajectory gate is visible at the
call site. `BRAIN_NO_KERNEL_UPGRADE=1` is the A/B switch every number above was
taken with.

## Cross-model finding: the host grad-norm that is NOT a workaround

`model::parallel`'s multi-GPU `adamw_step` computes the global grad-norm on the
host, and the optimiser section above assumed that was a workaround for the
serial `gradnorm_sq` — so deleting it looked like free follow-up work once
`gradnorm_part` landed. Reading the code killed that:

1. The clip is over the **summed** gradient, and the sum exists only in host
   RAM. The fused optimiser pulls every replica's gradients to the host anyway
   (that is the design that turns data-parallel from 0.75× into 1.34–1.58×) and
   adds them there. No card holds `Σ_r g_r` at any point.
2. The norm does **not** decompose over replicas: `‖Σ_r g_r‖ ≠ f(‖g_r‖)`.
   "Device norm per rank, reduce the scalars" is a *different number*, and would
   silently change every data-parallel run's clip coefficient. There is no
   per-rank local norm here to replace.
3. Running the device pair would mean **uploading the summed gradient back**
   (2.4 GB for the 0.6B Qwen) onto the PCIe leg that already *is* the cost of a
   step (~5.3 s/step, fixed) — to save a host reduction over buffers still warm
   in cache from the summation.

`model::shard`'s fused optimiser and `model::distributed`'s `Adam` clip a
host-resident gradient for the same reason. The on-device pair is right exactly
where `optim::Optim` uses it: the gradient is on the card and never leaves.

It does not violate the rayon rule either — `parallel.rs` goes through
`backend_cpu::par` (`sum_sq_f64`, `zip_each`), and `backend-cpu` is still the
only crate in the workspace with a `rayon` dependency.

The stale thing was the *comment*, not the code, and it was load-bearing: it is
what made a wrong cleanup look like an obvious one. Both it and this page now
state the structural reason. **A "workaround for X" comment must say what breaks
if X goes away** — otherwise the next reader deletes the code when X is fixed.

## Cross-model finding: shared-memory bank conflicts in `matmul_reg2`

`matmul_reg2` gives each thread 8 **contiguous** columns, so it reads
`Bs[kk*128 + tx*8 + j]`; across a warp `(8·tx) mod 32` takes 4 distinct values,
putting 16 addresses on 4 banks — a **4-way conflict on half of every chunk's
shared loads**. Its staging store is an 8-way conflict for the same reason.

`matmul_reg3.wgsl` fixes both with layout only — *interleaved* (stride-16)
register tiling plus a 128→129 padded tile stride — leaving the K accumulation
order untouched, so its output is **bit-identical by construction** (measured
max_abs 0.0 across 12 shapes). It is 1.05× on the FLUX.2 shape mix, 1.10-1.49×
on shapes with n = 3072 or a narrow K, and never more than 1 % slower.

It is **added alongside**: FLUX.2 uses it, `matmul_reg2` remains the default
for `zimage`/`qwen`/`gpt`/`vision` until each measures its own shapes. The
n = 9216 shapes do not move, which locates the next ceiling: at one shared word
per FMA the kernel is shared-*throughput* bound around 40-50 % of peak, and
getting past that needs vec4 shared loads or a larger register block — not
deeper K-blocking. **Deeper K-blocking cannot help at all**: arithmetic
intensity is `2·BM·BN·BK / ((BM+BN)·BK·4)`, and BK cancels.

## Cross-model finding: a kernel fix does not propagate to models written later

`gn_stats` — one invocation per (n, group), each serially walking its group —
was found and fixed for the DIAMOND UNet in 2025 (it was **77.6 % of a frame**;
see the world-models section below, which adds `gn_part` + `gn_stats2`). The
`vae` crate was written afterwards, against the same kernel set, and reached for
`gn_stats` because that is what the GroupNorm primitive is called. On the FLUX.2
VAE decode it was **2 262 ms of a 6 466 ms decode (35 %)**: 32 threads for
33 M elements.

The lesson is not about GroupNorm. **A "fixed" kernel that is fixed by adding a
faster sibling stays broken for every future caller**, because the obvious name
still points at the slow one. Two things make the fix stick:

* make the **selection**, not the kernel, the shared thing — a `*_step` helper
  that picks the variant from `DeviceCaps` (`model::block::flash_bidir_step`,
  `qwen::Qwen::rms_step`, `vae::Builder::coop`), so a new caller gets the fast
  path by construction; and
* when you profile any model, **grep the profile for the reference kernel
  names** (`gn_stats`, `rmsnorm`, `layernorm`, `attn_softmax*`, `max_abs_row`,
  `ln_stats`) before theorising. Every one of them is a one-thread-per-row
  kernel with a cooperative twin, and finding one in a top-3 line is the cheap
  half of any profile.

FLUX.2's VAE hit `gn_stats`, `attn_scores_bidir` AND `attn_softmax_bidir`; its
text encoder hit `rmsnorm` and `attn_softmax`. Five instances of one pattern in
one model, all of them already solved elsewhere in the tree.

## Cross-model finding: conv as GEMM needs the TRANSPOSED orientation to be chunkable

`docs/performance/p40.md` established im2col + `matmul_reg2` as the conv lowering
for a compute-bound discrete GPU (2.1-2.4× on YOLOv8n@640). The FLUX.2 VAE is
the same trade at a much larger spatial size — `conv_bias_reg` measured a flat
**~700 GFLOP/s (6.0 % of peak) across all 15 of its shapes**, which is its
structural 0.75 byte/FLOP ceiling, not a bug — and it hits a wall the YOLO
shapes never did:

> the im2col operand for a 512×512 conv with Cin=256, K=3 is `[262144, 2304]`
> f32 = **2.4 GB**, past the P40's 2047 MiB `max_storage_buffer_binding_size`.
> The whole-image lowering is not merely expensive, it is **unbindable**.

The fix is an orientation choice. YOLO's lowering computes `y[Cout, HW] =
W · colᵀ` (positions are GEMM *columns*); chunking columns is not expressible,
because a column chunk of the output is a strided region of every row. Compute
`y[HW, Cout] = col · Wᵀ` instead — positions as GEMM **rows** — and a spatial
chunk is a contiguous row range of *both* operands, so both bindings are plain
`step_sliced` sub-ranges and one bounded scratch serves every conv in the graph.
It also *lowers* traffic: with `Cout ≤ 128` the col is read exactly once,
whereas the column-oriented form re-reads it `Cout/128` times.

The cost is that the output lands in NHWC, so a permutation pass is needed
(`nlc_bias_nchw`, which folds the conv bias into it). On the FLUX.2 VAE that
whole path — `im2col_at` + `matmul_reg3` + `nlc_bias_nchw` — is **3 546 → 930 ms
(3.8×)**, with the GEMM itself at **5 126 GFLOP/s = 43.6 % of peak**. Convs with
`Cout < 128` stay on the direct kernel: they would pay for a full 128-wide
column tile (the VAE's `conv_out` at Cout = 3 is 42× wasted).

## Cross-model finding: a workgroup tile pays only where the amplification is large

Both `nlc_bias_nchw` (a transpose) and `im2col_at` (a transpose in disguise) have
the same shape of problem — whichever index the thread follows, the other side
is strided — and the same textbook fix: stage a 64×64 tile in workgroup memory,
pad the row stride to 65 so the column read hits 32 distinct banks. Measured on
the same graph, same card, same session:

| kernel | element-indexed | 64×64 workgroup tile | |
|---|---:|---:|---|
| `nlc_bias_nchw` | 158 ms | **36 ms** | 4.4× |
| `im2col_at` | **275 ms** | 311 ms | 0.88× — *slower* |

The difference is how bad the uncoalesced side actually is. The transpose's
strided side fetches a 32-byte sector per useful float (8×), so removing it is
worth far more than the occupancy the 16.6 KB of workgroup memory costs (5
blocks per SM instead of the shared-memory-free maximum). im2col's gather lands
**three consecutive taps in one sector** (~2.7×), and there the same trade
loses. Estimate the amplification factor before paying occupancy for a tile; if
it is under ~3×, the tile is a pessimisation.

---

## Cross-model finding: a batch dimension is free in the kernels — and pays nothing once a kernel is at its plateau

FLUX.2 Klein's DiT was made batched (`Flux2Model::forward_batch`, B samples in
one device pass) with **no new WGSL at all**, because the shared kernel library
already carries the two hooks any modulated transformer needs:

| hook | kernels | what it gives |
|---|---|---|
| **`bsz` Param** + `x[(b·T + j)·stride]` addressing | `attn_scores_bidir`, `attn_softmax_bidir`, `attn_apply_bidir`, `flash_attn_bidir`, `flash_attn_bidir_split` | a **sample-major** slab batches by raising one Param; one workgroup owns one `(b, head, query-tile)`, so samples cannot attend across each other by construction |
| **`rows_per_cond` condition groups** | `film_row`, `gate_row` | per-sample adaLN/FiLM modulation in ONE dispatch (`NC = B`) — so **different timesteps per batch member cost nothing**, which is what makes continuous batching of a diffusion model possible |

Row-concatenation does the rest: the register-tiled GEMMs accumulate over K
inside a fixed 128×128 output tile, so a larger `M` adds tiles and never
reorders a sum; the per-row norms, SwiGLU and int8 quantizers are row-local.
The consequence worth writing down is that **a batched forward can be
bit-identical to N single forwards** — `crates/flux2/tests/batch_parity.rs`
asserts `max_abs == 0.0` at fp32 *and* int8 — so batching is not a numerical
risk that needs a cosine tolerance, and a future kernel change that breaks the
property fails a test instead of silently drifting served output away from the
single-request path.

The one hook that is **missing**: `layernorm` takes a single `[D]` gamma/beta
pair, with no `rows_per_cond` twin. FLUX.2 works around it with a `(b·D, D)`
binding slice per sample — B dispatches of unchanged size, correct and
bit-identical, but B× the launches. A `layernorm_group` (or teaching the
existing kernel `rows_per_cond`) is the obvious generalisation the next
modulated model will want.

### …and now the negative result, which is the useful half

**Batching bought 4.4 %.** klein-4B int8 at 512², 1×P40, min-of-4:

| B | ms/image | speedup |
|---:|---:|---:|
| 1 | 1295.5 | 1.000× |
| 2 | 1273.6 | 1.014× |
| 4 | 1255.8 | 1.032× |
| 8 | 1241.1 | **1.044×** |

End to end, behind the residency executor at matched GPU temperature, that is
1.02×: 8.28 → 8.46 images/min from concurrency 1 to 2, flat to concurrency 4,
because the un-batched text encoder + VAE are 31 % of a generation and cap the
achievable speedup at 1.44× regardless.

The reason is one measurement, and it is the one to take before implementing
batching anywhere: **GFLOP/s of the dominant GEMM as a function of M.**

| shape (K × N) | M=1536 | 3072 | 4608 | 6144 | 9216 | 12288 |
|---|---:|---:|---:|---:|---:|---:|
| 3072 × 9216 | 3984 | 3598 | 4184 | 4112 | 3981 | 3944 |
| 9216 × 3072 | 5445 | 5497 | 5509 | 5605 | 5713 | 5671 |

Flat. One 512² image is already 1536 rows = 864 output tiles ≈ 29 workgroups
per SM; the card is saturated before a batch dimension exists. Batching helps a
model whose per-request `M` is *small* (a decoder's `m = 1` decode step — where
brain's paged engine gets its real continuous-batching win) and helps a model
whose per-request work is *host*-bound. It does not help a vision/diffusion
forward that already presents thousands of rows: there, the 95 % of the forward
that is device time was already saturated, and only the 5 % of host conditioning
can be amortised.

*Rule:* before building batching for throughput, plot the hot kernel's
achieved GFLOP/s against `M` at the sizes you would batch to. If it is flat at
`M = 1`-request, batching is a **latency-neutral capacity feature** (one
resident weight set serving N requests), not a throughput multiplier — say so
in the ledger rather than reporting the 4 %.

---

## Cross-model finding: a concurrency ladder on a passively-cooled card measures its own cooling curve

Measuring FLUX.2's concurrency ladder as one continuous `brain perf run sweep`
produced a clean, plausible, **completely wrong** result: throughput falling
monotonically with concurrency (0.545 → 0.353 denoise_steps/s from c=1 to c=8),
exactly contradicting the controlled per-forward measurement above.

Running the same ladder **reversed** is the control, and it inverts:

| ladder order | c=1 ial p50 | c=8 ial p50 (per image) |
|---|---:|---:|
| `1,2,4,8` (c=1 first) | 1318 ms | 1647 ms |
| `8,4,2,1` (c=8 first) | **1734 ms** | **1234 ms** |

Whichever level runs *last* is ~1.32× slower. `nvidia-smi` during the sweep:
**83 → 90 °C, SM clock 1531 → 923 MHz (582 MHz at the worst),
`clocks_throttle_reasons.active = 0x20` (SW thermal slowdown)**. The Tesla P40s
in this box are passively cooled datacentre cards in a workstation chassis; a
multi-minute sustained sweep is a thermal ramp, and the ramp is **larger than
every effect the sweep was built to measure**.

*Rule:* any multi-level sweep on this hardware must either start every level
from a matched GPU temperature (poll `temperature.gpu` and wait), or be run in
both directions with the difference reported. A single-direction ladder is not
evidence. `brain perf` has no thermal instrumentation today — recording
`temperature.gpu` / `clocks.sm` / `clocks_throttle_reasons.active` per level in
the artifact's environment block is a real gap, and the reason a "2× throughput
regression" was nearly attributed to a batching change.

---

## Tooling / diagnostics added

- **`BRAIN_PROFILE=1`**: CPU per-kernel timing; GPU op counters (uniforms /
  bind-groups / submits / dispatches / readbacks, on drop); per-frame
  `[detect] preprocess|forward|postprocess` stage timing (with a `poll_wait` so
  the lazy-submit `forward` reflects real GPU compute, not just recording).
- **Conv GFLOP/s microbench** (`cargo test -p brain-gpu-core -- --ignored
  bench_conv_gflops`): min-of-N per representative layer, contention-robust.
- **bench_yolo_inference.py**: one `DEVICE=cpu|gpu` drives both sides; honest
  per-side device labels + the engine adapter line; torch GPU fallback
  (cuda→vulkan→mps→cpu); engine stage timing surfaced under `BRAIN_PROFILE`.
- **bench_qwen_inference.py**: the same head-to-head shape for Qwen3 text
  generation across brain's CPU/GPU/NPU targets vs HuggingFace Transformers on
  CPU/GPU (NPU is brain-only), isolating load cost from per-token latency.

---

## What's next (the path to torch-parity on GPU)

The GPU `forward` is now pure conv throughput (~222 ms low-load vs torch ~50–130 ms,
from ~1218 ms naive — a 5.5× conv speedup, ~12× overall from the 2749 ms start).
The remaining levers were each assessed; most have hit diminishing returns:

- **Wider register tile** (4×8 = 32 sums) — *register-limited*. Arrays spill to
  local memory (the 4×4 array variant measured 927 ms vs the scalar 545 ms), and
  32 scalar accumulators risk an occupancy drop on a memory-bound kernel. 4×4 is
  the practical sweet spot.
- **im2col + tiled GEMM** — *adds traffic, not a win here* (Arc). **[P40 UPDATE: confirmed the opposite on a compute-bound discrete GPU — im2col + `matmul_reg2` is 2-5× the direct conv on deep layers, 2.1-2.4× on the whole YOLOv8n@640 forward. See `docs/P40.md`.]** The GEMM is clean
  (no per-tap overhead), but it materializes the im2col matrix `B`, which a
  register-tiled GEMM re-reads ~`Cout/4` times (×64 for a 256-channel layer). On
  the bandwidth-bound integrated GPU that extra `B` traffic outweighs removing the
  (already-hoisted) per-tap overhead. Worth it on a compute-bound discrete GPU.
- **Concat fusion** (producer→concat slices) — *high blast radius for a modest GPU
  win*. Needs offset sub-buffer views in BOTH backends (the wgpu `DeviceBuffer`
  carries no offset today — ~10 construction/access sites + offset-aware bindings)
  and a C2f forward+backward restructure (gradcheck-critical). The reward is now
  modest because the GPU is conv-bound. Right architectural fusion; deserves its
  own focused, gradcheck-validated pass.
- **Winograd F(2×2,3×3)** — the one remaining *algorithmic* lever: 2.25× fewer
  multiplies for the dominant 3×3 layers. The CPU attempt showed transforms can
  eat the gain (kept opt-in); on the GPU it's more promising but research-grade
  (input/weight/output transforms + 16 transform-domain GEMMs, built on the
  work-group JIT). This is the realistic route to closing the last gap to torch.

> Honest framing: brain's from-scratch WGSL conv now runs ~12× faster than where
> it started, but it competes with a mature vendor library (oneDNN) on the
> vendor's own integrated silicon. The structural wins (sync / launch / fusion /
> tiling / overhead-hoisting) are real and large; the last gap is genuine
> GPU-kernel research (Winograd, or hand-tuned blocked kernels), not a toggle.

### On fusion specifically

The right mental model for fusion: yolov8n must run as a **graph of ~dozens of
kernels separated by unavoidable global barriers** (each layer reads the *whole*
previous output across all workgroups — a kernel boundary *is* the global
barrier; you can't make it one mega-kernel). The win is **vertical /
producer-consumer fusion** *along* each barrier-free chain. Status:

- **conv → BN(eval) → SiLU** — fused (`conv_act` / `conv_act_reg`). The big one.
- **head conv → bias** — fused (`conv_bias`). Done.
- **launch / submit / readback overhead** — minimized (one submit, one pass,
  7 readbacks/frame).
- **concat / slice into the producer** *(not yet — the next fusion)*. The C2f
  split + channel-concats are pure data movement; having the conv that *produces*
  a tensor write directly into its final slice of the concat buffer (and reading
  the split halves as views) drops the `chan_place`/`concat_split` copies. This
  needs **buffer-view infrastructure** (offset sub-buffers in both backends; wgpu
  bindings have a 256-byte offset-alignment constraint) and a careful C2f
  forward+backward restructure (the backward reads those activations, so
  gradcheck must stay green). It's the right architectural fusion but a
  higher-blast-radius change for a now-*modest* GPU win (the GPU is conv-bound),
  so it's scoped as a focused follow-up rather than bundled with kernel tuning.

> Honest framing: the from-scratch WGSL conv competes with a mature vendor
> library (oneDNN/cuDNN) on the vendor's own silicon. The structural wins (sync /
> launch / fusion / tiling) are real and large; closing the last gap is genuine
> GPU-kernel engineering, not a quick toggle.

---

## World models: DIAMOND-Atari denoiser (`brain wm`)

The playable diffusion world model (64x64 RGB, 3 denoising steps per frame,
~4M-param conditional UNet ≈ 6 GFLOP/NFE). Measured with
`brain wm bench --model diamond --profile` (per-kernel wall clock, one submit
per step — ranking only; the production path is a single submit) and
`brain wm bench` (end-to-end frames). Numbers from the Core Ultra 7 155H
(22T AVX2) + its integrated Arc iGPU; this laptop-class part throttles under
sustained load, so frame times vary run-to-run by ~1.5-2x — treat every number
as a warm-run best, and compare only like-for-like.

| Path | Before | After | Lever |
|---|---|---|---|
| CPU end-to-end | ~440 ms/frame | **~166 ms/frame (6 fps)** | native gn fast paths |
| iGPU end-to-end | ~2 390 ms/frame | **~370 ms/frame** | parallel GN + tiled conv + on-device loop |
| iGPU gn_stats (in-profile) | 1 436 ms (77.6% of frame) | ~21 ms (gn_part+gn_stats2) | two-stage parallel reduction |
| iGPU conv (in-profile) | 1 274 ms | ~60 ms | register-tiled conv_bias_reg (conv_act_reg's 8x4 tile) |
| iGPU whole forward (in-profile) | 1 975 ms | **107 ms** | all of the above |
| **Intel NPU end-to-end** | — | **60-75 ms/frame (13-16 fps)** | fp32 ONNX whole-graph via OpenVINO (`brain wm export` + `--device npu`) |

What the profiler taught (in order):

1. **`gn_stats` was the GPU.** The serial per-group reduction dispatches
   2-4 invocations total — one EU lane looping 131k elements. 77.6% of GPU
   frame time for a "free" normalize. Fix: `gn_part` (64 partial sums per
   group) + `gn_stats2` (combine); same stats layout, so `gn_apply`/backward
   consume it unchanged, and native CPU fast paths keep the one-graph rule.
2. **CPU GroupNorm cost 35% of the forward** for the same reason (2-4 rayon
   tasks). First fix attempt nested rayon inside the backend's pool and
   measured *slower* (440 -> 558 ms/frame): de-nested, coarse chunks -> 166 ms.
   Measure after every change; parallelism that looks obviously right can lose.
3. **Naive conv is 20x off on the iGPU.** `conv_bias_reg` re-applies
   conv_act_reg's proven register tile (8 output channels x 4 coalesced
   positions) with a plain bias epilogue: 1 274 -> 60 ms in-profile.
4. **The on-device denoise loop** (per-sigma coefficients as 2-float buffer
   writes into a pre-recorded scale -> UNet -> quantize-wrap -> Euler ring)
   removed per-step readbacks; verified bit-identical rollouts before any
   semantics-adjacent change landed.
5. **Environment lies to benchmarks.** A stale `DISPLAY` without X auth
   SIGSEGVs Vulkan enumeration; 130 orphaned D-state rustc processes from
   killed builds inflate loadavg (harmless) while an actual background build
   poisons every number. `wm bench` runs are only comparable warm + idle.

The NPU is the fastest path on this machine — 23.7 ms per UNet inference on
the NPU silicon (vs 77.9 ms for OpenVINO-CPU on the same graph), parity
2.6e-4 vs brain's engine (fp16 internals). The sampler stays host-side;
`scripts/gates/wm-perf-gate.sh` floors all three paths (hand-set x3 envelopes —
auto-baselining was twice corrupted by background load on this box).

Quality/speed ladder: `--denoise-steps 1..3` (1 step ≈ 3x the fps, still
recognizably Breakout; `--adaptive` walks this automatically). At 1 step the
CPU path reaches ~18 fps — playable-smooth on this machine today.

## The training datapath: the backward is 6.15x the forward, and 84% of it is two kernels

Measured with `vqgan_bench` (`crates/vqgan/src/bin/vqgan_bench.rs`), the first
per-kernel-kind profiler in this tree that covers a **backward**. VQGAN at
256x256 on one P40, because its backward IS the shared block backward set that
`crates/vae`, `crates/unet` and `crates/restore` all train through.

```
FORWARD   348 dispatches    407.72 ms      BACKWARD  1079 dispatches   2508.69 ms
  conv_bias_reg  422.51  88  90.3%           conv2d_dx     1137.93  88  41.2%
  gn_apply        12.02  64   2.6%           conv2d_dw      973.04  88  35.3%
  gn_stats_wg     10.75  64   2.3%           gn_dsum         228.43  64   8.3%
  silu             9.94  56   2.1%           gn_dgamma       100.77  64   3.7%
  add2             6.61  35   1.4%           bias_grad        99.66  88   3.6%
                                             gn_dbeta         72.64  64   2.6%
                                             axpy             52.97 304   1.9%
```

**The headline.** A backward costs about **2x** its forward in theory — one pass
for `dX`, one for `dW`. This one costs **6.15x**. The gap is not spread thin: 84%
of the backward is `conv2d_dx` + `conv2d_dw`, at **12.9 and 11.1 ms/call against
the forward conv's 4.8**.

**The cause is structural, not a tuning miss.** The forward conv was lowered to
`im2col_at` + `matmul_reg3` (register-tiled, ~3-4 TFLOP/s). The backward convs
never were: `conv2d_dx` and `conv2d_dw` are naive gather kernels, one thread per
output element walking the reduction serially. There is no `conv2d_dx_reg`, no
`conv2d_dw_reg` and no `col2im` anywhere in the tree — checked, because the
usual finding is an unwired fast sibling (`docs/lessons.md` #8) and here there
genuinely is none.

If the backward convs matched the forward's efficiency the step would drop from
~2916 ms to roughly **1650 ms — a 1.75x on every conv-net training step in the
workspace**. The lowering is the same one the forward already uses:

* `dW[Cout, Cin*K*K] = dY[HW, Cout]^T . col[HW, Cin*K*K]` — reuses the existing
  `im2col_at` output and `matmul_dw_reg`, which is already the register-tiled
  backward GEMM for linear layers.
* `dX = col2im(dY[HW, Cout] . W[Cout, Cin*K*K])` — the GEMM exists; `col2im`
  (scatter-add, the transpose of `im2col`) is the one genuinely new kernel, and
  it needs its own gradcheck on BOTH backends (`docs/lessons.md` #5).

**Second finding, smaller but free-standing.** `gn_dsum` (3.57 ms/call),
`gn_dgamma` (1.58) and `gn_dbeta` (1.14) are 402 ms — 16% of the backward — and
all three are the per-channel reductions `vae::blocks::BWD_KERNELS` already
documents as having no cooperative twin (the §C.2 gap in
`docs/kernel-checklist.md`). Unlike the convs this is a known, written-down gap;
the profile just says what it costs.

---

## Cross-model finding: the roofline itself was a P40 literal, and it was 10-17% too high

Every "% of peak" this repo printed divided by a hardcoded constant
(`PEAK_TFLOPS = 11.76` in three model benches, `PEAK_GBPS = 346.0` in four
microbenches). `DeviceCaps::peak_bandwidth_gbs` existed for this and was `None`
on all three backends; there was no peak-FLOP field at all.

`gpu_core::roof` now **measures** both roofs once per adapter and `Gpu::caps()`
overlays them, so a utilisation number is a statement about the device that ran:

| roof | spec sheet (P40) | measured | probe |
|---|---:|---:|---|
| fp32 compute | 11 760 GFLOP/s | **10 555** (89.8%) | `roof_fma` — 8 independent FMA chains in registers, no memory traffic in the loop |
| bandwidth | 346 GB/s | **287.6** (83.1%) | `axpy` over 2 × 256 MiB, STREAM-triad shape (12 B/element) |
| ridge | 34.0 FLOP/byte | **36.7** | |

The compute probe deliberately measures the **silicon**, not the best GEMM in
the tree: a roof derived from `matmul_reg3` (which reaches ~43% of peak) would
have hidden exactly the gap the work is meant to close. Both probes
self-calibrate their trip count to clear the launch floor and are
`poll_wait`-bracketed (§E.0). Results persist under `~/.cache/brain/roof-<adapter>-<probe-hash>.txt`,
invalidated by probe-source fingerprint; `BRAIN_NO_ROOF=1` skips probing.

### The VQGAN training step, re-stated against the measured roof

The profiler (`gpu_core::profile`, now shared by `vqgan_bench` / `unet_bench`
instead of a copy in each) classifies every row compute- or memory-bound from
the device's ridge and grades it against **its own class's** roof — reporting a
streaming kernel as a fraction of a FLOP peak is meaningless. It then names the
rows below their class's floor:

```
FORWARD   348 disp  407.98 ms   356.8 GFLOP/s   3.4% of measured peak
  DEFECT  conv_bias_reg          3.2% of its compute roof (floor 30%) — 89.5% of this pass
BACKWARD 1404 disp  457.18 ms   638.1 GFLOP/s   6.0% of measured peak
  DEFECT  col2im                 8.0% of its memory roof  (floor 35%) — 21.5% of this pass
  DEFECT  bias_grad              1.2% of its memory roof  (floor 35%) — 14.9% of this pass
  DEFECT  matmul_dx_reg         18.1% of its compute roof (floor 30%) — 11.3% of this pass
  DEFECT  matmul_dw_reg_splitk  19.1% of its compute roof (floor 30%) — 10.7% of this pass
  DEFECT  axpy                  23.0% of its memory roof  (floor 35%) —  7.7% of this pass
step = 865.2 ms; group sums inflate the passes by 15% (fwd) and 46% (bwd)
```

Two rows that the old table made look like targets are **not**:

* **`im2col_at` is done.** 128.8 GB/s *counted* × its documented ~2.7× sector
  amplification is ~350 GB/s of real traffic — 44.8% of the counted roof and at
  the hardware limit in practice. It can only be *deleted* (fused into the dW
  GEMM's B-operand load), never tuned.
* **`conv_bias_reg` at 3.2%** is classified compute-bound because its *minimum*
  traffic gives it a high arithmetic intensity; the kernel's own 12-loads-per-32-FMAs
  schedule is what puts it at 0.75 byte/FLOP. The classifier is right that this
  is a defect — it needs the algorithmic change (the GEMM lowering, 3.8× where
  it is used), not tuning.

At 512² the ranking shifts and is worth recording, because it is evidence about
*occupancy* rather than about the kernels: `col2im` rises to 30.0% of the
backward while `matmul_dx_reg`/`matmul_dw_reg_splitk` improve to **24.8% / 28.5%**
of the compute roof (from 18.1 / 19.1 at 256²) and `im2col_at` to 59.9% of the
bandwidth roof. The GEMMs are tile-starved at the smaller shape, not badly
written.

---

## Cross-model finding: the LM head was 91% of a Qwen3-0.6B prefill, because a binding budget ignored the device

The first per-kernel profile of the LLM datapath at a **real** shape
(`qwen_bench`, `QwenConfig::qwen3_0_6b()`, T=512, one P40, random weights):

```
=== PREFILL T=512: 548 dispatches, 4375.51 ms ===
kernel              ms      n      %   ms/call   GFLOP/s     GB/s   bound   %roof
matmul_tile     4097.73     7  90.8%  585.389      38.9      0.2 compute    0.4%
gqa_scores       145.67    28   3.2%    5.203     103.8      4.4  memory    1.5%
matmul_reg3      143.79   196   3.2%    0.734    3136.4     21.2 compute   29.7%
WHOLE PASS      4375.51   548                     146.7                     1.4%
  DEFECT  matmul_tile  0.4% of its compute roof (floor 30%) — 90.8% of this pass
```

`matmul_tile` is the **naive** GEMM. The per-layer projections next to it, on the
same graph and the same device, run `matmul_reg3` at 3136 GFLOP/s — **80× the
rate**. The cause is not the kernel; it is which kernel got dispatched:

* `block::TILE_BUDGET_WORDS` was a fixed 24 Mi words (~96 MiB), chosen for the
  smallest binding limit brain had met (128 MB on Mesa-GL). A P40 **reports
  2047 MiB**, and `Backend::max_storage_binding_bytes()` has been a real query
  since the backends stopped defaulting it — nothing consulted it here.
* So `vocab_tiles` split the tied 151936×1024 head into **7 tiles**, 21× finer
  than the device needs.
* And `qwen::model`'s head only routes to `matmul_reg3` when the vocab collapses
  to **one** tile — a tiled head *must* use the sliced-binding `matmul_tile`.

One constant, therefore, silently pinned the largest GEMM in the model to the
naive kernel. Letting it collapse:

| | dispatches | prefill T=512 | GFLOP/s | % of measured roof | tok/s |
|---|---:|---:|---:|---:|---:|
| before | 548 | 4375.51 ms | 146.7 | 1.4% | 117 |
| after | 536 | **362.95 ms** | **1768.6** | **16.8%** | **1411** |

**12.05× on the whole pass**, reproduced twice from a matched idle temperature,
and confirmed by the pre-existing `BRAIN_TILE_BUDGET_WORDS` A/B before any code
changed. `block::vocab_tiles_on` now derives the budget from the device's
queried limit (halved, since the same dispatch also binds logits and the hidden
state); `qwen`, `lfm` and `t5` use it. Models too large for one binding still
tile — Qwen3-8B's 151936×4096 head is 2.49 GB and stays split — and the unit
test asserts every tile fits under the real ceiling.

This is the `docs/lessons.md` #8 defect class (a fast kernel a later model never
learned about) with a twist worth naming: **the unwired thing was not a kernel,
it was a *limit*.** The fast kernel was wired and eager; a capability nobody
queried kept the graph away from it.

### What the prefill profile says next

With the head fixed, the same table ranks very differently and the top rows are
now attention, not GEMMs:

```
matmul_reg3   172.27  197  39.3%   3542.7 GFLOP/s  compute  33.6%
gqa_scores    143.41   28  32.7%    105.4 GFLOP/s   memory   1.6%   <- DEFECT
gqa_apply      38.76   28   8.8%    388.6 GFLOP/s   memory   5.8%   <- DEFECT
ce_value       29.89    1   6.8%                    memory   3.6%   <- DEFECT
```

`gqa_scores` at **1.6% of the bandwidth roof** while consuming a third of the
pass is the next target, and `matmul_reg3` at 33.6% of the compute roof is above
the 30% defect line but well short of the 60% band.

### Qwen3-0.6B decode, first profile

```
=== DECODE @pos 511: 590 dispatches, 20.98 ms ===   -> 47.7 tok/s single-stream
matmul_gemv         23.40  196  38.4%   75.4 GB/s  memory  26.2%  <- DEFECT
rmsnorm_rows         8.93  113  14.6%    0.2 GB/s  memory   0.1%  <- DEFECT
decode_softmax       7.53   28  12.4%    0.2 GB/s  memory   0.1%  <- DEFECT
add2                 5.53   56   9.1%    0.1 GB/s  memory   0.0%  <- DEFECT
attn_decode_apply    3.79   28   6.2%   15.8 GB/s  memory   5.5%  <- DEFECT
attn_decode_scores   3.35   28   5.5%   17.9 GB/s  memory   6.2%  <- DEFECT
sum of groups 60.97 ms vs whole pass 20.98 ms — 191% drain inflation over 422 groups
```

Two things have to be read together here, and neither is visible without the
other:

* **The ceiling.** 595 984 384 params = 2.38 GB fp32, read once per token, so on
  a 287.6 GB/s card single-stream decode cannot exceed **121 tok/s fp32** (483
  int8) however good the kernels are. At 47.7 tok/s the path is at **39% of its
  own achievable ceiling** — a real gap, but a 2.5× one, not the 200× the "0.5%
  of the compute roof" row suggests. Reporting a decode step against the FLOP
  peak is exactly the meaningless comparison the classifier exists to prevent.
* **The dispatch floor.** 590 dispatches at the measured 0.0065 ms/dispatch is
  **~3.8 ms — 18% of a 20.98 ms step.** On the VQGAN backward the same floor was
  2% and fusion was correctly judged not worth it (§E); here it is nine times
  that share, and the four kernels doing almost no work per launch
  (`rmsnorm_rows`, `decode_softmax`, `add2`, `silu_mul` — together 40% of the
  grouped table at ≤0.2 GB/s) are launch-bound, not bandwidth-bound. This is the
  first pass in the tree where the fusion argument is *quantitatively* different.

The 191% drain inflation is also a record for this tree and is the expected
consequence of 422 groups over a 21 ms pass — the group table here ranks and
nothing more.

---

## Cross-model finding: the serving engine had no tiled GEMM registered at all

`qwen::serve` — the paged engine behind HTTP and D-Bus — registers its own
pipeline table, and `matmul_reg3` was **not in it**. `Engine::mm` selected
through `backend_api::select`, whose `KernelVariant` set has no register-tiled
member, so the only two answers available were the decode GEMV
(`m <= DECODE_REGIME_MAX_ROWS = 32`) and the naive one-thread-per-output
reference. Every chunked-prefill chunk above 32 rows therefore ran the naive
kernel — while the batched forward next door dispatched `matmul_reg3` on the
same shapes.

Measured with `qwen_bench serve` at Qwen3-0.6B, 128 rows, one P40:

| | top kernel | whole pass | rows/s |
|---|---|---:|---:|
| before | `matmul` 2400.38 ms, **98.0% of the pass at 0.4% of the compute roof** | 2207.90 ms | 58 |
| after | `matmul_reg3` 78.47 ms, 66.1% at 13.6% | **75.30–82.12 ms** | **1553–1651** |

**≈27–29× on the whole pass**, reproduced across four runs from matched idle
temperatures. The fix is not a new kernel: `Engine::mm` now uses
`block::gemm_variant` with the tier gated on the queried `workgroup_reductions`
— the *same* rule `flux1`, `flux2` and `model::rowemit` already share, so the
serving engine stops having a private policy. `gemm_variant`'s `m <= 32` GEMV
cutoff is the same number as `DECODE_REGIME_MAX_ROWS`, so the decode regime is
bit-for-bit unchanged; only the rows above it move.

Both defects found in this workstream have the same shape and neither is a
kernel-authoring problem:

| where | the fast kernel | why the graph never reached it |
|---|---|---|
| batched forward, LM head | `matmul_reg3` | a binding budget that ignored the device's queried limit |
| serving engine, prefill | `matmul_reg3` | never registered; the selector could not name it |

### …and the negative result from the same commit

`serve.rs`'s QK-norm was the only norm in that tape bypassing `self.rms` — it
called `block::rmsnorm_fwd` (per-element, one thread per row) directly, which is
the coalescing bug measured at 19.4× for exactly this op. Routing it through the
selector like every other norm moved the whole pass **82.43 → 82.12 ms: within
noise.** At 128 rows the q-norm has `128 × 16 = 2048` rows, which already has
enough lanes to interleave with its neighbours — the same discriminator lesson
#21 records for `bias_grad`.

It is kept anyway, because it is a *consistency* fix rather than a perf one (one
policy, selector-driven, capability-gated) and because the row count that would
make it matter — `bsz × n_heads` at `bsz = 1` — is 16 threads on a 3840-core
card. But it is recorded here as a zero, not credited as a win.

**Also fixed en route:** `prefill_submits_scale_with_chunks_not_with_token_count`
compared submit counts through `.unwrap_or(0)`, so on a backend that does not
count submits its first two assertions passed **vacuously** (0 == 0) and only
the third failed. That is brain's own "unmeasured is null, never
0-pretending-complete" rule broken inside a test — and the reason no backend
counted was that **two of the three never implemented `Backend::stats()` at
all**. Making the test skip was the first attempt and was a workaround; the
counters are now real on `backend-cpu` and `backend-vulkan` (per HANDLE, which
is what makes concurrent measurement correct), so the test runs for real
everywhere. See `docs/lessons.md` #30.

### The same guard, twice more: `gpt` and `glm`

`docs/lessons.md` §15 records `m < 128` as the guard that sent an SDXL UNet's
`[77, 2048, 2560]` projection to the naive kernel — 22× on that GEMM, 49% of the
forward. `qwen` and `lfm` were migrated to `block::pick_gemm` (crossover
**measured at m = 8**); `gpt` and `glm` kept private copies of the old constant,
in four functions between them.

A/B'd on a P40 at `k = 768, n = 3072` (`unet_bench gemm`), naive vs
`matmul_reg3`, `max|Δ| = 0.0` at every point:

| m | 8 | 16 | 32 | 64 | 96 | 127 |
|---|---:|---:|---:|---:|---:|---:|
| naive (ms) | 0.46 | 1.11 | 2.86 | 5.92 | 8.28 | 11.26 |
| tiled (ms) | 0.31 | 0.28 | 0.35 | 0.30 | 0.42 | 0.33 |
| **speedup** | 1.5× | 4.0× | 8.2× | 19.7× | 19.8× | **34.1×** |

Worth more here than in the UNet, and note the shape of it: the naive kernel's
cost grows linearly in `m` while the tiled kernel's is flat to 127 — a 128×128
tile is not full until `m = 128`, so everything in the band was paying for
masked lanes *and* getting one thread per output. Every GPT/GLM shape with
`8 ≤ m < 128` — short prompts, small eval batches, the `m = T` generate path —
was on the wrong side of it.

Both now call `pick_gemm`, so the rule has one home and the next model inherits
it rather than copying the constant a fifth time.

### Correcting the "next target" call — and what `gqa_scores` actually costs

The entry above called `gqa_scores` "the next target" off the `qwen_bench prefill`
profile. That profile drives `Qwen::new`'s **batched forward**. The *served*
prefill does not use that kernel at all: `qwen::serve` shares one paged tape
between prefill and decode, so it goes through `paged_decode_scores_batched`
(4.8% of a 128-row served step). `gqa_scores` is dispatched by the batched
forward — training, eval, `codec`, `tts`, `kronos` — which makes it a **training**
target, not a serving one. Both matter here; conflating them would have aimed the
next fix at the wrong datapath.

What it is, measured (T=512, one P40): **143.41 ms over 28 calls, 105.4 GFLOP/s,
1.6% of the bandwidth roof, 32.7% of the batched forward.**

The structure explains the number without any further profiling:

* One thread per `(b, h, i, j)`, each walking a serial 128-element dot product —
  a 2/5 shape by the catalogue's rubric, and it is *below* the ridge (≈23
  FLOP/byte of minimum traffic against a measured ridge of 36.7), so it is
  memory-bound and no amount of arithmetic tuning helps.
* The reads are the problem. Adjacent threads vary `j`, and
  `k_base = (b*T + j)*k_row + hkv*hd`, so consecutive lanes read addresses
  `k_row = 1024` floats — **4 KB** — apart. That is a fully scattered load, the
  same defect class as `col2im`, and the amplification is far past the ~3×
  threshold where a workgroup-staged tile starts paying
  (see "a workgroup tile pays only where the amplification is large").

Two candidate fixes, neither yet built, in the order the evidence favours:

1. **Stage `k` transposed once per layer** — `[b, hkv, d, T]` makes consecutive
   `j` contiguous, turning the scattered read into a coalesced one for the cost
   of transposing 2.1 MB. `nchw_nlc` already does this shape of work at 4.4×.
2. **Lower it to a GEMM** — `scores[b,h] = q[b,h] · k[b,h]ᵀ` *is* a matmul, and
   `matmul_reg3` measures 3542 GFLOP/s on this card against this kernel's 105.
   Same trade the conv path already took (im2col + GEMM, 3.8×), and the reason
   to prefer (1) first is that (2) needs a batched/strided GEMM over the
   head-major layout, which is a larger change.

`flash_attn_bidir*` is NOT the sibling to reach for: it is bidirectional, and
this path is causal. Checked first per §F.3.
