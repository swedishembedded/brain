# Performance overview

brain is built so performance claims are measured, not guessed. This page
explains the three pieces of that: per-kernel profiling with a hardware
roofline, a runtime kernel selector, and INT8 inference support — plus one
honest example of what a real optimization pass has produced.

## Profiling and the roofline probe

Set `BRAIN_PROFILE=1` on any `brain` run to get a per-kernel breakdown: time,
call count, bytes moved, and — where the device supports it — the percentage
of that device's own measured compute or memory-bandwidth ceiling the kernel
is hitting.

That ceiling is not a datasheet number. On first use, brain runs a small
roofline probe against the actual device present (compute throughput, memory
bandwidth, and the "ridge point" where a kernel stops being bandwidth-bound
and starts being compute-bound), and grades every subsequent kernel against
*that* measurement. A laptop iGPU and a datacenter GPU get different roofs,
because they have different roofs — the profiler never assumes a vendor
spec is what you actually have.

This is also why a kernel report never silently claims 0% of peak when the
answer is unknown: an unmeasured or unmodeled cost is reported as such, not
folded into the totals as if it were free.

## The runtime kernel selector

Most operations in brain (attention, normalization, matrix multiply, an
optimizer's gradient-norm reduction, and others) have more than one
implementation — different tilings, different thread-cooperation strategies,
different fits for a given shape and device. brain does not pick one at
compile time and hope. A selector inspects the device's capabilities (and,
for some operations, the shape of the work) and dispatches to whichever
registered variant is expected to be fastest there, falling back to a safe
default when a device lacks what a faster variant needs. You do not choose
kernels by hand; the engine does, and `BRAIN_PROFILE=1` shows you which one
actually ran.

## INT8 inference

Several served models support INT8 inference for lower memory use and higher
throughput than full-precision — check the specific model's own page under
`docs/models/` (for example `docs/models/qwen3.md` or `docs/models/flux2.md`)
to see whether it's supported and how to enable it, since support and the
exact accuracy trade-off are model-specific.

## Cost accounting: `brain flops`

`brain flops <model> <shape>` reports how many floating-point (or, for INT8
paths, integer) operations a forward or backward pass actually costs, without
running it. It's a coverage-honest cost registry: every kernel brain dispatches
has either a registered cost formula or is explicitly listed as **not
measured** — the tool never rounds an unmodeled cost down to zero and folds
it into a total that looks complete. See `docs/performance/flops.md` for the
full accounting model and CLI flags.

## A real example — not a promise

The following are real, reproducible numbers from brain's own optimization
work. They are reported here as illustrations of what a properly-tuned kernel
path can look like on the hardware it was measured on. They are **not** a
prediction for your hardware, your model, or your shapes — always measure
your own setup (see `docs/performance/benchmarking.md`).

- **CPU YOLOv8n@640 inference**, on a 22-core AVX2 workstation: replacing a
  scalar kernel-JIT execution path with a hand-vectorized AVX2/FMA GEMM
  convolution path took a from-scratch WGSL inference pipeline from ~7440 ms
  to ~115 ms per frame — the same WGSL kernels, a faster execution path
  underneath.
- **GPT-2-small-shaped training step**, 6 layers × d768, batch 8 × block 256,
  on one NVIDIA Tesla P40: replacing a single-threaded gradient-norm
  reduction kernel with a cooperative, workgroup-parallel one took a training
  step from 6.90 s to 0.84 s (roughly 8×), because that one reduction had
  been costing over 85% of the step's GPU time.

Your mileage will vary by device, driver, model shape, and batch size — that
is exactly what `brain perf run` and `brain flops` are for.

### DeepSeek-OCR decode: O(T²) recompute to O(T) KV-cache

The DeepSeek-OCR decoder (`crates/deepseekv2`, `deepseekocr::caps` for the
served path) had no KV cache: every generated token re-ran the WHOLE sequence
so far through all 12 MoE layers, `O(T²)` over a generation. Following the
loop in `.agents/rules/kernels.md` §F - profile per kernel kind, check for an
already-faster sibling before writing anything, put the fix in the shared
selector, re-measure the whole pass rather than trust a per-kernel table -
this is what that loop found, in order.

**Baseline (before, this repo's own prior measurement, 22 CPU cores, release
build, a 283-token prompt):** `--max_new 2` 1 min 54 s, `--max_new 10` 4 min
54 s - roughly 22 s per additional token, the signature of a quadratic decode
loop. `--max_new 32` extrapolates to roughly twelve minutes.

**The fix.** `DeepseekV2::generate_greedy_kv`/`step` adds the `O(T)` KV-cache
decode tier this decoder was missing - the same two-tier shape
`crates/gpt`/`crates/glm`/`crates/qwen3` already keep (`generate` vs
`generate_kv`). The prompt still pays one batched forward, which also seeds a
persistent per-layer K/V cache in bulk; every generated token after that is
one incremental attention step (`model::block::gqa_decode_step`, already used
by four other decoders in this tree) plus a single-row MoE/dense FFN pass -
no per-token re-run of the whole sequence. It needed exactly one new kernel
wiring, `kernels::ROPE_AT` (rotation at an explicit absolute position, since
`rope_base.wgsl`'s `row % tcols` convention has no way to express a single new
row past position 0) - no new WGSL source.

**Measured, same machine, same real `DeepSeek-OCR-Q8_0` weights, a real
document page, `--max_new 32`:**

| decode strategy | total wall | vs baseline |
|---|---|---|
| `O(T²)` recompute (previous) | ~12 min (extrapolated) | 1x |
| `O(T)` KV-cache (this change) | ~123-147 s | ~5-6x |

`BRAIN_PROFILE`'s per-kernel table for the KV-cached run (one 283-token
prompt-prefill forward plus 32 decode steps, CPU backend):

```
=== BRAIN_PROFILE (CPU backend, total 34889.0 ms) ===
  moe_linear_gated  23347.0 ms  67584 calls  (66.9%)
  matmul             3697.7 ms   3020 calls  (10.6%)
  matmul_reg3        3384.2 ms     52 calls  ( 9.7%)
  silu_mul           1577.3 ms  22912 calls  ( 4.5%)
  scale_add          1481.5 ms  22528 calls  ( 4.2%)
  gqa_apply           435.6 ms     12 calls  ( 1.2%)
  gqa_scores          377.2 ms     12 calls  ( 1.1%)
  ... (attention decode-step kernels, rope_at, kv_append, router_gate,
       decode_softmax, embed - each well under 1%)
```

**A follow-up hypothesis tried and killed, per the numbers, not a guess.**
`moe_linear_gated` at 67% looked like an obvious next target: 64 routed
experts are dispatched every layer, but only `top_k = 6` are ever selected
per row, and `crates/glm` already has a proven ~7x-faster row-compacted MoE
path (`model::moe::expert_fwd_compact`) for exactly this shape. Wiring the
same trade into the decode step (read the router gate back to the host once
per MoE layer, skip dispatching the ~58 non-selected experts) dropped
`moe_linear_gated`'s call count 67584 -> 8250 (~8.2x) but left its OWN total
time UNCHANGED (~23.3 s either way), and the WHOLE decode's profiled total
went UP (34.9 s -> 39.3 s): the per-row gate check inside `moe_linear_gated`
already makes a non-selected expert's dispatch cheap at this shape, so the
67% was real compute in the selected experts all along, and the 352 extra
host round-trips this needed (11 MoE layers x 32 tokens) cost about as much
as the skipped dispatches saved. Reverted - a per-kernel call-count win that
does not move the whole-pass number is not a win, exactly the case
`.agents/rules/kernels.md` §F.1 warns the per-kernel table is an upper bound
for.

**Where the remaining cost goes, same machine, same real weights, a real
document page, `--max_new 32`:** the fix closed the loop that was
structurally quadratic, but with decode no longer dominant, the remaining
~123-147 s splits roughly as ~18 s model construction, ~50 s decode (prompt
prefill + 32 `O(1)` steps), and ~75-80 s vision encoding (SAM ViT-B at
1024x1024 → CLIP-L/24 → compressor → projector) - and that last number is
not yet broken down per kernel. Re-profiling after a fix is supposed to
promote the next bottleneck (§F.9) - this is that promotion, landing
squarely on the vision encoder, not the decode loop, as the next place to
look.

### CPU AVX2/AVX-512 fast paths for the decode loop's dominant kernels

The `moe_linear_gated` 66.9% number in the table above is the scalar,
one-invocation-per-element Cranelift-JIT path - `crates/backend-cpu` had a
native AVX2 fast path (`fast_conv.rs`/`fast_ops.rs`) for the base GEMM, the
conv2d family and cross-attention only; `moe_linear_gated{,_dx,_dw}` and the
plain causal self-attention family (`gqa_scores`/`attn_softmax`/`gqa_apply` +
the `gqa_bwd_{dscores,dv,dq,dk}` backward quartet - what `deepseekv2`'s own
decoder dispatches for its attention) ran the slow path the whole time. This
pass added native paths for both, plus a third AVX-512 tier gated behind
runtime `avx512_available()` detection.

**What was added** (`crates/backend-cpu/src/fast_ops.rs`, wired from
`lib.rs`'s dispatch table exactly like every existing fast path):

- `moe_linear_gated_fwd` reuses the SAME `row_abt_avx2`/`row_abt_avx512`
  microkernel `matmul_abt` already uses, adding only the WGSL kernel's own
  per-row gate early-exit (a non-routed row is never reduced, not
  computed-then-discarded - the same contract the WGSL source documents).
- `moe_linear_gated_dx`/`moe_linear_gated_dw` reuse a new shared `axpy`
  SAXPY primitive (`dst += scale*src`, AVX2+FMA) for their accumulation loops.
- `gqa_scores`/`gqa_apply` pack each head's q/k or probs/v slice contiguous
  and reuse `matmul_abt`, mirroring the existing `attn_scores_cross`/
  `attn_apply_cross` pattern exactly, with the causal mask applied after
  (`gqa_scores`) or falling out for free because the softmax already zeroed
  the invalid region (`gqa_apply`).
- `gqa_bwd_dscores`/`gqa_bwd_dv`/`gqa_bwd_dq`/`gqa_bwd_dk` reuse `matmul_abt`
  (`dscores`, whose causal masking also falls out for free the same way) or
  the `axpy` primitive threaded over their own output rows (`dv`/`dq`/`dk`),
  the same `par_chunks_mut`-over-output-rows shape `matmul_dx`/`matmul_dw`
  already use.
- A third tier, `avx512_available()` (F+VL+DQ, mirroring `avx2_available()`'s
  exact runtime-detection shape) gates a `row_abt_avx512` microkernel wired
  ahead of the AVX2 check in `matmul_abt` and `moe_linear_gated_fwd`.
  **This machine (Intel Core Ultra 7 155H) does not implement AVX-512 at
  all** - the tier compiles and its own unit test explicitly detects that and
  prints "AVX-512 not available, skipping" rather than silently passing as if
  it had verified the vector logic. It has not executed on real hardware.

**Correctness**: every new kernel is gated against a scalar reference that
mirrors its own `.wgsl` kernel's exact formula (not `matmul_abt` with a
post-hoc mask - the row-gating logic itself is under test), the same
bit-approximate/fp-reassociation-tolerance style this crate's existing tests
already use. Full regression green after landing: `cargo test -p
brain-backend-cpu --release`, `-p brain-deepseekv2 --release` (including the
real finite-difference gradchecks and the KV-cache-vs-recompute parity test),
`-p brain-model --release`, `-p brain-sam1 --release`, and `make gradcheck`.

**Measured** (microbenchmarks in `fast_ops.rs`'s own `mod tests`, `cargo test
-p brain-backend-cpu --release <name> -- --ignored --nocapture`), at
`deepseekv2`'s real decoder shape (`d_model=1280, moe_ff=896`, 64 experts
top_k=6, 10 heads, head_dim=128) and its real 283-token prompt-prefill length:

| kernel | comparator | speedup | note |
|---|---|---|---|
| `moe_linear_gated` fwd, `m=283` | scalar, same rayon threading | 2.4x-6.1x across 5 runs | apples-to-apples baseline; range is real measurement variance under a contended machine (see below), not code changing between runs |
| `gqa_scores`, `T=283` | scalar, single-threaded | 6.4x | not apples-to-apples: `matmul_abt`'s own internal threading is folded into this number too - see caveat below |
| `gqa_apply`, `T=283` | scalar, single-threaded | 15.5x | same caveat |

Both benchmarks were run on a machine with a sibling agent's own concurrent
`cargo test -p brain-model` (LTO, 10+ rustc processes at times, swap fully
saturated) running throughout measurement - exactly what the `moe_linear_gated`
range above reflects; a quieter box would read tighter. The `gqa_scores`/
`gqa_apply` numbers compare against a single-threaded scalar reference (not a
threaded one), so part of that multiplier is parallelism the old scalar-JIT
fallback path *also* gets from `backend-cpu`'s own always-threaded dispatch,
not AVX2 alone - the wall-clock difference is still real (that's the path
being replaced either way), just don't read "15.5x" as "AVX2 gave 15.5x."

A single-decode-row (`m=1`) variant of the `moe_linear_gated` bench was tried
and dropped: at that shape the whole call is a few KFLOPs, small enough that
this workspace's release build (LTO) could prove the repeated near-identical
call loop-invariant and hoist it out of the timing loop even past `black_box`
guards on both the arguments and a per-iteration input perturbation, reading
a physically impossible ~10-30 TFLOP/s. Shipping a number that could not be
trusted was judged worse than shipping none.

**No real-weight end-to-end re-run.** Getting an updated `--max_new 32`
wall-clock/per-kernel table needs the real `DeepSeek-OCR-Q8_0`/`mmproj` GGUF
pair (~22 GiB resident); this machine had only an empty HF-cache ref stub
this session, no downloaded weights, so that run could not be reproduced.
The kernel-level microbenchmarks above are what could be measured honestly
here. Whoever has the real weights should re-run
`BRAIN_DEEPSEEK_OCR_DIR=<dir> BRAIN_PROFILE=1 brain do deepseek-ai/DeepSeek-OCR
generate --max_new 32 --in image=page.ppm --json` and update this section and
`.agents/roadmap/deepseek-ocr.md`'s Phase 8 entry with the real before/after
numbers.

### DeepSeek-OCR vision encoder: the CPU-vs-GPU gap, and the real per-kernel cost

Follow-up pass, same repo, same loop. Two constraints shaped it before any
profiling: `crates/sam1`'s wgpu backend is known to corrupt its per-block
buffers at this tower's production shape (3+ blocks, 1024x1024) - a separate,
still-open, actively-tracked defect that is NOT this pass's to fix - so
`crates/cli/src/resident_deepseekocr.rs` pins the whole composite (SAM, CLIP,
the decoder) to the CPU backend. The first job was therefore to measure how
much of the ~75-80 s vision-encode estimate is attributable to that CPU pin at
all, before touching a single kernel.

**CPU-vs-GPU, isolated.** `crates/sam1/src/bin/sam1_bench.rs` (new) drives
`SamEncoder` alone at the real `SamViTConfig::deepseek_ocr()` geometry with
random weights (`sam1::init_dense` - the tower's cost depends on shape, not
values, and the wgpu corruption is a values bug, not a hang, so a wgpu timing
number is still meaningful even though its output cannot be trusted and is
never shipped):

| backend | full 12-block forward (best of 3) | one windowed block (tower-isolated) | one global block (tower-isolated) |
|---|---|---|---|
| CPU (shipped) | 72.2 s | 4.04 s | 6.81 s (1.7x windowed) |
| wgpu (NOT shipped - known output corruption) | 20.0 s | 1.25 s | 2.63 s (2.1x windowed) |

So the CPU pin costs roughly **3.6x** on this tower alone - a real, measured
number for whoever eventually closes the wgpu correctness gap, not a guess.
Global (T=4096, full-grid) blocks cost 1.7-2.1x a windowed (T≈196/window)
block on either backend, consistent with the O(T²) attention-score/apply cost
the decomposed relative-position design pays at full grid extent.

**Per-kernel breakdown of the CPU forward** (`sam1_bench profile`,
`BRAIN_PROFILE`'s own per-kernel wall-time accumulator - the only source of
a per-kernel table on CPU, since this backend has no device-timestamp path):

```
attn_apply_cross   70-71%   (264 calls, ~192 ms/call average)
matmul (+matmul_reg3)  15-19%
attn_scores_cross   2-4%
attn_softmax_cross  4%
attn_relpos_qr/add  ~1% each
gelu_erf, bias_add, layernorm, conv2d, embed  each < 1%
```

`attn_scores_cross` and `attn_apply_cross` run the IDENTICAL total FLOP count
per call (both are `m x k x n = 256 x 4096 x 64` GEMMs, just with `k` and `n`
swapped between the two), yet `attn_apply_cross` cost 20-30x more wall time.
The only structural difference is the packing loop each runs before its
shared `matmul_abt` (AVX2+rayon) call: `attn_scores_cross` packs its K
operand with one contiguous copy per row (the same shape `attn_scores_cross`'s
own row-copy already is); `attn_apply_cross` TRANSPOSES V into `vt[hd,tk]`
with `for j in 0..tk { for d in 0..hd { vt[d*tk+j] = kv[..] } }` - `hd=64`
elements per `j`, each landing in a DIFFERENT row of `vt`, `tk` floats
(16 KiB) apart, so 64 distinct cache lines are touched per `j` and each is
not revisited until 63 unrelated lines have been touched in between, for all
`tk=4096` values of `j` - a scatter pattern with essentially no reuse.

**Two optimizations applied, both re-verified against the full test matrix.**

1. **`crates/sam1` now dispatches its forward GEMMs through
   `model::block::pick_gemm`** (the same selector `crates/clip` and
   `crates/deepseekv2` already use), instead of hardcoding the naive `matmul`
   kernel index for every QKV/proj/fc1/fc2 linear - per kernels.md's "is
   there already a faster sibling" check, this tower had simply never been
   wired to the answer its neighbours already found. **Measured effect on
   the CPU backend: none.** `backend-cpu`'s dispatcher already routes
   `matmul`/`matmul_reg`/`matmul_reg2`/`matmul_reg3` to the SAME native
   AVX2+rayon `fast_ops::matmul_abt` regardless of which kernel name is
   requested, so this tower's forward was never taking the slow scalar-JIT
   path the naive-vs-tiled selector exists to route around. Kept anyway as a
   correctness-neutral consistency fix (same selector, same convention every
   other GEMM-dispatching model in this tree already follows) that becomes a
   real win the moment SAM runs on a backend where that kernel-family
   collapse does not hold (wgpu's naive and 128x128-tiled GEMMs are genuinely
   different code paths).

2. **`backend-cpu::fast_ops::attn_apply_cross`'s V-transpose is now tiled**
   (`transpose_rows_tiled`, `JT=16` rows): buffer the tile's source rows
   first (a plain contiguous read, same shape the scores kernel already
   does), THEN write `vt` `d`-major within the tile, so each 64-byte
   destination cache line is written whole, once, instead of revisited 16
   times with 63 unrelated lines touched in between. Same math - unit-tested
   against the direct definition over tile-aligned and non-aligned shapes and
   non-zero stride/offset (a real caller slices one head out of a fused qkv
   buffer). **Isolated A/B, same process, back to back** (so both sides see
   the same external load - `crates/backend-cpu`'s own
   `attn_apply_cross_bench`) at the exact SAM global-block shape: 1.0x-2.0x
   across repeated runs, never measured worse. **Whole-tower effect: not
   cleanly resolved this session.** This machine ran under sustained, heavy,
   variable CPU load from a second agent's concurrent builds/tests in a
   separate worktree for most of this pass (`uptime` load average swung
   between ~1 and ~32 on a 22-core box over the course of measurement, and
   `free -h` showed available memory swing between 3 GiB and 25 GiB) - a
   `sam1_bench` full-tower "before"/"after" pair taken minutes apart under
   those conditions moved from 68.8 s to 72.2 s, i.e. within the noise this
   machine was producing, not a clean regression or a clean win. The
   isolated, same-process A/B is the only measurement in this pass immune to
   that noise, and it is real; the full-pass number needs a quiet machine to
   confirm, which this session did not have on demand. Per kernels.md's own
   "use the whole-pass number to decide, not the per-kernel table" rule,
   this is reported as an unresolved-magnitude win, not a proven one -
   exactly the honesty standard the earlier MoE row-compaction entry above
   set, except that entry had clear evidence of being a net LOSS and this one
   has no such evidence either way.

**Per-stage `BRAIN_PROFILE` numbers for the vision tower, real weights, real
document image** (new instrumentation this pass -
`DeepEncoder::run_forward` now brackets SAM/bridge/CLIP/projector separately,
where the composite's stage line previously lumped the whole encode+decode
loop into one number):

| stage | run 1 | run 2 |
|---|---|---|
| SAM forward | 33.0 s | 23.8 s |
| compressor NCHW→NLC + bridge | 1.7 ms | 1.7 ms |
| CLIP forward | 1.42 s | 1.47 s |
| concat + projector | 8.3 ms | 12.1 ms |
| **vision encode total** | **34.4 s** | **25.3 s** |

Both runs are real: same checkpoint, same document image, same machine, a few
minutes apart, with the pick_gemm and tiled-transpose fixes both applied.
Whatever the run-to-run variance is attributable to (the same external load
noted above), the vision encoder's real cost is now DIRECTLY measured at
25-34 s rather than an undifferentiated "~75-80 s" estimate the prior pass
could only bound in aggregate - the per-stage instrumentation alone is a
genuine improvement in what is knowable here, independent of whether the
kernel fixes above moved the number.

**Model construction, reconfirmed, still not optimized this pass:** 23.5 s
and 28.1 s across the same two runs (mmproj import 2.3-3.1 s + weight
upload/tape build 20.1-25.5 s). `crates/deepseekocr/src/import.rs` was
inspected for redundant work: the fp32 expansion is cached and skipped when
present (both runs hit that path), `WeightReader` mmaps and streams one
tensor at a time, and `ParamStore::new_with_roles_src` already uploads via
`raw_words`/`with_tensor_chunks` with a peak-one-chunk host allocation and a
periodic `poll_wait` flush - no redundant copy or non-streaming read was
found. This remains a real but unexamined-for-kernel-level-wins cost, not
touched this pass.

### DeepSeek-OCR model construction: profiled below the crate boundary, JIT compile ruled out

Follow-up pass, same loop. The paragraph above left model construction's
20-28 s "real but unexamined" - `BRAIN_PROFILE` only ever timed the whole
`caps::Session::load` call as one bracket. This pass added fine-grained
`std::time::Instant` brackets (gated on the same `BRAIN_PROFILE` env var,
zero cost when unset) at every candidate named in the investigation brief -
GGUF header parse, per-`Gpu::new_cpu` Cranelift JIT compile time
(`crates/wgsl-cpu::Jit::new`), and the streaming upload loop's own
alloc/read+write/flush phases (`crates/paramstore::ParamStore::
new_with_roles_src`, `crates/deepseekv2::DeepseekV2::new_on`) - to find out
which of them the 20+ seconds actually belongs to, rather than guess.

**The Cranelift JIT hypothesis (§F.4, "profile the branch your hardware does
not take") is KILLED, cleanly, by real numbers.** `deepseekocr::caps::
Session::load` builds FIVE separate `gpu_core::Gpu` instances (one per
kernel-set: SAM, CLIP, the glue stage, the decoder, and the preprocessor),
each compiling its whole `PIPELINES` list up front. Measured on the real
checkpoint:

| `Gpu::new_cpu` call | kernels compiled | JIT time |
|---|---|---|
| SAM (`sam1::model::PIPELINES`) | 42 | 50.5 ms |
| CLIP (`CLIP_VISION_PIPELINES`) | 37 | 58.7 ms |
| glue (`GLUE_PIPELINES`) | 9 | 8.6 ms |
| decoder (`deepseekv2::PIPELINES`, 42 slots after this session's KV-cache/LoRA additions) | 47 (incl. duplicates the CPU fast path also serves) | 43.6-51.3 ms |
| preprocessor | 5 | 12.3-24.6 ms |
| **total, all five** | | **~174-186 ms, i.e. under 0.5% of a 40.7-48.2 s load** |

Compiling 40+ WGSL kernels per `Gpu` five times over was a reasonable thing
to suspect - it is real, measurable work, just not enough of it. The
dominant cost is squarely the weight stream/upload:

```
stage build: decoder Gpu::new_cpu (Cranelift JIT compile): 44.1 ms
stage build: decoder new_on (weight stream/upload + scratch alloc): 36698.4 ms
  deepseekv2: new_on: ParamStore::new_with_roles_src (2234 tensors): 36324.7 ms
    paramstore: alloc 863.2 ms, read+write 35444.5 ms, flush/readback 1.9 ms
  deepseekv2: new_on: scratch buffer allocation: 366.1 ms
  deepseekv2: new_on: tape build (fwd): 7.0 ms
stage load: TOTAL: 40692.1 ms
```

`read+write` - fetching each of the decoder's 2234 tensors via
`WeightReader::raw_words` (zero-copy mmap slice, the fp32 expansion already
matches dtype) and copying it into a freshly `gpu.storage`-allocated
destination buffer via `write_at` - is **97.6% of the upload bracket and
87% of the whole load**, on both runs measured (43.3 s of 48.2 s; 35.4 s of
40.7 s). `alloc` (the `gpu.storage` call that reserves each destination
buffer) is 2.4%; the scratch buffers `DeepseekV2::new_on` allocates after
the weights (res/dres, per-layer attention/MoE activations, at this
model's `SEQ_LEN=512`) are 1%; tape build is noise. So the "reduce
avoidable buffer allocation churn" candidate from this investigation's own
brief is also ruled out - the allocation calls themselves are cheap; the
cost is in the first real touch of the memory they return.

**Splitting `read+write` further: source read vs destination write.** A raw
`dd if=<fp32 file> of=/dev/null bs=4M` on the exact same 11.7 GB decoder
file read it in **7.9 s (1.5 GB/s)** - so a pure sequential disk scan is not
the bottleneck either. A throwaway diagnostic
(`crates/deepseekv2/tests/weight_read_order_bench.rs`, `#[ignore]`d, kept as
a real tool rather than deleted) reads every decoder tensor via the exact
same `raw_words` call the real upload loop uses, in the real upload's
construction order, touching one word per 4 KiB page (enough to force the
same page faults `write_at` would) but allocating **no destination
buffer**: **13.3 s for the same 10.93 GiB (881 MB/s)**. That leaves roughly
**35.4 - 13.3 ≈ 22 s** unaccounted for by source reads at all - which,
since `write_at` is a plain `copy_from_slice` into an already-`gpu.storage`-
reserved buffer, can only be the cost of physically backing that
destination memory the first time it is touched (a fresh `vec![0u32; n]`
on Linux is a lazy zero-page mapping - `alloc`'s own 863 ms confirms nothing
is actually written at allocation time - so the real first-touch cost is
deferred into `write_at`, indistinguishable there from the source fetch
until this diagnostic separated them).

**This machine's swap was observed pinned at 97-100% full (7.8-8.0 GiB of
8.0 GiB) continuously for the ENTIRE investigation** - independent of these
runs specifically, checked repeatedly via `free -h`/`/proc/swaps` across
more than half an hour and multiple sibling agents' concurrent builds on
this shared 30 GiB box (one of this session's own real-weight runs was
SIGKILLed by the OOM killer during vision-encode, after model construction
had already completed and printed every stage above it). Allocating ~12 GiB
of brand-new destination pages for the decoder's weights, on a box already
that memory-starved system-wide, is exactly the condition under which
first-touch page faults get expensive (the kernel must reclaim or evict to
free a physical frame for each new page, competing with everything else
resident). The two orderings above were also cross-checked for a
locality effect (construction order - the real upload's own layer/expert
order - vs file/physical order) but the comparison was confounded by page
cache warmth between back-to-back runs in one process (both produced the
identical `0x5ecf400` checksum over the same 2234 tensors, at least
confirming `param_list()` and the file's own tensor set agree byte-for-byte);
getting a clean cold/cold pair needs a quiet machine this session did not
have on demand.

**No fix landed this pass.** All three candidates this investigation's own
brief named going in - JIT caching, per-tensor dequant parallelism, and
buffer-allocation churn - were checked against real measurement and ruled
out: there is no dequant on this path at all (the fp32 expansion already
matches dtype, so `raw_words` is zero-copy), JIT is under 0.5% of the load,
and allocation is 2.4% of the upload bracket. What is actually expensive -
first-touch physical memory allocation for ~12 GiB of destination weights -
is not a `ParamStore`-level inefficiency to tune away; it is the real cost
of making that much memory resident, currently inflated by this shared
machine's own memory pressure rather than by anything `crates/deepseekocr`,
`crates/deepseekv2`, `crates/paramstore` or `crates/wgsl-cpu` does wrong.
The one structural lever big enough to matter further - a genuine zero-copy
`CpuBuffer` that borrows straight from the source mmap for the exact-dtype
`raw_words` case, instead of duplicating into a freshly allocated `Vec<u32>`
- is a `backend-cpu` buffer-representation change, larger and riskier than
this pass's scope (and `backend-cpu`'s fast-path kernels are explicitly out
of bounds for this investigation). A quiet, uncontended machine would also
answer how much of the 22 s destination-side cost is inherent vs this
session's specific contention; this box did not offer one.

**What this pass DID leave behind, real and load-bearing:** the fine-grained
`BRAIN_PROFILE` brackets above are now permanent (zero cost when the env var
is unset - a single `std::env::var` check per `Jit::new`/
`new_with_roles_src` call, the same pattern `deepseekocr::stage_time`
already uses), so the next investigation of this cost starts with a real
per-stage table instead of one lumped 20-28 s number. Verified against the
full test matrix after landing (`cargo test -p brain-deepseekocr --release`,
`-p brain-deepseekv2 --release`, both 100% green including the
`grads_match_finite_differences_*` gradchecks and
`generate_greedy_kv_matches_recompute`) plus two real end-to-end real-weight
CLI runs through every touched code path (`BRAIN_DEEPSEEK_OCR_DIR=<dir>
BRAIN_PROFILE=1 brain do deepseek-ai/DeepSeek-OCR generate --max_new 2..8
--in image=page.ppm --json`), both producing coherent generated output.

**Brain's own wall-clock, same machine, same real weights, same image, same
prompt, `--max_new 32`, greedy:** the absolute number has moved across the
passes documented in this page:

| pass | total wall | condition |
|---|---|---|
| `O(T²)` decode, pre-KV-cache | ~12 min (extrapolated) | quieter machine |
| `O(T)` KV-cache decode | 123-147 s | quieter machine |
| + vision-encoder tiled-transpose fix, contended machine | 95.6-97.1 s | sibling agent load on the box |
| + same, clean single-tenant re-run | 83.1 s | machine verified idle: `free -h` 24 GiB available, `ps aux` clean, no sibling worktrees active |

The 83.1 s clean, single-tenant number (283 prompt tokens, 32 completion
tokens, `prompt_tokens`/`completion_tokens` both correctly populated) is
faster than either contended run above (95.6 s, 97.1 s), consistent with
shared-machine load inflating absolute numbers during this session's
concurrent-agent passes - use 83.1 s as the number to cite going forward,
not the contended pairs.

**What is still open:** the wgpu correctness bug itself (out of scope here,
now with a real 3.6x cost-of-CPU-pin number attached for whoever prioritizes
it); model construction's 20-25 s is unexamined at the kernel level; and the
tiled-transpose fix's whole-pass magnitude needs a quiet-machine re-measure
isolated from the other changes landed in the same pass.

### DeepSeek-OCR: a real 50-page document, and what concurrent requests actually do

Every DeepSeek-OCR number above is one synthetic image, one request. This pass
asks the two questions that matters for an operator, not a microbenchmark: what
does a real, multi-page document cost end to end, and what happens when more
than one request arrives at once against the one resident instance this model
serves from.

**Workload.** 50 real pages (not synthetic) - pages 60-109 of a public STM32F4
reference manual PDF, 150 DPI PNG, chosen to land inside a real peripheral-
register chapter (dense text, tables, register-bit diagrams), rendered once and
reused unchanged. Served path: **one resident `brain serve --openai` process**
(the model built ONCE, `BRAIN_DEEPSEEK_OCR_DIR` pointing at the real
`Q8_0`/`mmproj` pair plus the cached fp32 decoder expansion), driven over the
real OpenAI-compatible `/v1/chat/completions` HTTP route with the default
instruction (`<|grounding|>Convert the document to markdown.`) and a real
document image attached per request - not the `brain do` single-shot CLI, which
would pay the ~22 GiB weight-upload cost fresh on every one of the 50 pages.
Machine: the same 22-core CPU-backend box every other number on this page was
measured on.

**Token budget: `max_new = 128`, not the CLI default of 32.** Two real,
measured calibration requests decided this (not a guess): at `max_new = 64` a
warm page cost 59.9 s; at `max_new = 128` a warm page cost 81.2 s and produced
414 characters of real markdown-ish output (vs. 203-220 characters at 64) -
enough to capture a heading plus real body text, not a two-token fragment.
`max_new` is capped at 229 by this model's own context budget (`SEQ_LEN = 512`
minus the ~283-token image+instruction prompt), so "a few hundred tokens" is
not reachable at all on this model without exceeding its context; 128 is the
largest budget this pass's own calibration run showed would keep the full
50-page sequential sweep inside the session's real time budget (measured, see
below - not assumed).

**Part 1 result: 50/50 pages succeeded, zero errors.**

| | |
|---|---|
| total wall (50 pages, warm instance) | 3947.6 s = **65.8 min** |
| pages/min | 0.76 |
| per-page min / median / max / mean | 66.2 s / 78.5 s / 99.7 s / 79.0 s (stdev 5.5 s) |
| one-time model activation (separate, not in the total above) | 27.0 s |
| `finish_reason` | 49/50 `length` (used the full 128-token budget), 1/50 `stop` (hit EOS at 72 tokens - wall time was still ~76 s, confirming this decoder's documented "no early stop": the token budget is honored but the *wall clock* is not shortened) |

Per-stage breakdown (`BRAIN_PROFILE`'s `stage_time` brackets, already
instrumented by an earlier pass in this session, read straight off the resident
server's stderr for all 50 real requests):

| stage | min | median | max | mean |
|---|---|---|---|---|
| vision encode (SAM+CLIP+glue+projector) | 11.2 s | 14.0 s | 51.1 s | 19.0 s |
| decode (prefill + up to 128 KV-cached steps) | 40.1 s | 61.9 s | 69.7 s | 59.8 s |

The spread is real and worth being honest about: vision-encode cost should be
content-independent (every page is resized to the same 1024x1024 grid before
SAM ever sees it), yet it ranged 11.2-51.1 s and CLIP alone ranged 1.2-8.9 s
across identically-shaped work. This machine ran the entire 50-page sweep with
available memory pinned at 2.6-3.6 GiB and swap 95-100% full for the whole
run (see the memory section below) - the most likely explanation for
same-shape work taking up to 4-5x longer on some pages is host-level
contention (page reclaim, swap activity) from that pressure, not the document
content. This pass did not isolate the two causes further; a quiet machine
would.

**Memory, watched continuously, not just checked once.** `free -h` before the
run, before every concurrency level in Part 2, and every 20 s throughout both
parts via a background sampler. The resident model's own steady-state
footprint on this 30 GiB box is ~26-27 GiB used / **2.6-3.6 GiB available**,
confirmed STABLE (not still declining) across three calibration requests
before the 50-page loop was allowed to start, and stable for the entire 66
minutes after that (280 samples, minimum available 2.6 GiB, no downward
trend). This is tighter than the ~4-6 GiB floor named going into this pass,
but that floor was set before knowing this specific model's own documented
footprint (`docs/models/deepseek-ocr.md`: "a box with less than ~24 GiB free
will not activate it" - this box had exactly ~24 GiB free at the start) - since
the number was measured stable rather than declining, the run was allowed to
continue with tight monitoring rather than aborted outright, and would have
been killed immediately on any sign of continued decline. It never showed one:
50 real pages, then the full N=2/4/8 concurrency sweep below, zero OOM
kills, zero crashes.

### Part 2: concurrent requests against ONE resident instance

**Method.** One resident server, N concurrent HTTP clients (backgrounded shell
processes, `wait`ed together), each a distinct real page from the same 50,
non-streaming `/v1/chat/completions`, `max_new = 128`. Wall time is measured
for the WHOLE batch (dispatch to last reply), not any individual request.

**First finding, at the default settings: concurrent requests are not queued,
they are dropped.** N=2 at the default `BRAIN_ADMIT_DEADLINE_MS` (10 s) sent
both requests at once: one ran and finished normally (82.6 s); the other was
rejected with **HTTP 429** after 10.1 s - the standard admission-shedding path
(`crates/apiserve/src/bridge.rs`'s `wait_for_admission`), which gives a request
10 s to be *claimed onto a lane* before shedding it, a deadline sized for fast
models. DeepSeek-OCR's own single-request latency (66-100 s) vastly exceeds
that window, and its instance's only lane was occupied by the first request
the entire time, so the second request could not be admitted before the
deadline fired - it was shed, not queued. **This is real, current, out-of-the-box
behaviour of this model behind `brain serve`**: two clients hitting it at once
today get one real answer and one 429, not two real answers. Fixing the
generic admission policy (e.g. distinguishing "queued behind a busy peer" from
"cold build", the way `cold_build_admit_deadline` already does for the
build-time case) is a `crates/apiserve`/`crates/residency` change and out of
scope here; it is named so whoever puts this model behind a shared frontend
knows to raise `BRAIN_ADMIT_DEADLINE_MS` or expect silent 429s under any real
concurrent load.

**Second finding, with `BRAIN_ADMIT_DEADLINE_MS=1800000` (30 min, so the
HTTP-admission artifact above is out of the way and the actual compute-level
concurrency behaviour is what gets measured):**

| N | batch wall time | implied per-request | vs. Part 1's steady-state median (78.5 s) |
|---|---|---|---|
| 2 | 181.5 s | 90.7 s | 1.16x |
| 4 | 315.8 s | 79.0 s | 1.01x |
| 8 | 649.7 s | 81.2 s | 1.03x |

**Zero speedup at any N tested - purely additive, exactly as documented.**
`docs/models/deepseek-ocr.md` already claimed this ("`run_batch` is therefore
the serial default: two concurrent requests share no work") and
`crates/cli/src/resident_deepseekocr.rs`'s own header explains why (no KV
cache to share, no batched vision-encoder forward); this pass confirms it with
real measurement rather than trusting the doc's claim on faith. The per-request
average at every N lands within a few percent of Part 1's single-stream
median, which is exactly what "no shared work, no overlap" predicts.

**A more specific finding underneath "no speedup": a real tail-latency
artifact from HOW the serial default replies, not just THAT it is serial.**
Reading `crates/residency/src/executor.rs` (`assign`/`group_rows`,
`Policy::max_batch = 8`) and `crates/residency/src/model.rs` (`Instance::
run_batch`'s default: `invs.iter().enumerate().map(|(i, inv)| self.run(...))
.collect()`) together with `executor.rs::run_group` (which calls `run_batch`
ONCE per dispatched group and only invokes each job's own `reply` callback
AFTER the WHOLE call returns, `results.len() == replies.len()` then
zip-dispatch) explains an asymmetry the N=8 run's own numbers show directly:
one of the 8 concurrent requests was dispatched alone and got its answer in
84.1 s; the other 7 landed in the dispatcher's next scheduling window together,
were coalesced into ONE `run_batch` call, and **all seven received their
answer at the identical wall-clock instant, 649.6 s** - even though internally
they were computed one at a time, ~81 s apart. A non-streaming caller whose
request happens to be grouped with others therefore pays the FULL group's
total latency, not just its own place in a fair FIFO queue - a request
computed FIRST inside a 7-request group still waits for the other 6 before its
own answer is returned. This is a consequence of pairing the default
all-or-nothing `run_batch` reply contract with the executor's own
up-to-`max_batch` coalescing, not a DeepSeek-OCR-specific bug, and is worth
naming precisely because "no speedup" alone understates the effect on
individual request latency under concurrency. *(Caveat: measured on the
non-streaming endpoint only - a streaming client's per-token deltas ARE routed
via `progress`/`sinks` during each job's own turn inside the loop, so its
perceived latency may differ; this pass did not measure the streaming path
under concurrency.)*

**No batching change was attempted this pass, on purpose.** A real vision-
encoder batching improvement (the only place shared work could exist, since
the decoder's MoE forward has no batch axis across different prompts either)
would need `crates/sam1`'s ViT tower to accept batch > 1 - and that tower's own
module documentation already states it cannot: *"Batch is 1. `sam1` is a
single-image tower (its windowed attention spans' storage-binding offsets are
not 256 B aligned across a batch stride)"* (`crates/deepseekocr/src/lib.rs`).
Making that true would be a `crates/sam1`/`crates/backend-cpu` architecture
change, explicitly larger than a "measure it" pass and outside this session's
own file-avoidance list for `crates/checkpoint`/`crates/wgsl-cpu`/
`crates/paramstore` neighbours. Given the machine's own memory headroom was
already down to 2.6-3.6 GiB for a SINGLE resident instance, attempting a
structural change to how images are batched through that instance - without
being able to verify its peak-memory behaviour on a quieter box - was judged
the wrong trade for this pass, exactly the "lean conservative" guidance this
investigation started from. **No `crates/deepseekocr`, `crates/deepseekv2`,
`crates/sam1`, `crates/residency` or `crates/apiserve` source file was changed
this pass** - only this page and `.agents/roadmap/deepseek-ocr.md`'s tracking.
