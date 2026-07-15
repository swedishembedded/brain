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

---

## What's next (the path to torch-parity on GPU)

The GPU `forward` is now pure conv throughput (~222 ms low-load vs torch ~50–130 ms,
from ~1218 ms naive — a 5.5× conv speedup, ~12× overall from the 2749 ms start).
The remaining levers were each assessed; most have hit diminishing returns:

- **Wider register tile** (4×8 = 32 sums) — *register-limited*. Arrays spill to
  local memory (the 4×4 array variant measured 927 ms vs the scalar 545 ms), and
  32 scalar accumulators risk an occupancy drop on a memory-bound kernel. 4×4 is
  the practical sweet spot.
- **im2col + tiled GEMM** — *adds traffic, not a win here*. The GEMM is clean
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
`scripts/wm-perf-gate.sh` floors all three paths (hand-set x3 envelopes —
auto-baselining was twice corrupted by background load on this box).

Quality/speed ladder: `--denoise-steps 1..3` (1 step ≈ 3x the fps, still
recognizably Breakout; `--adaptive` walks this automatically). At 1 step the
CPU path reaches ~18 fps — playable-smooth on this machine today.
