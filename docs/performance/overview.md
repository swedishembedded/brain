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

**Head-to-head vs `llama-mtmd-cli`, same machine, same real weights, same
image, same prompt, `--max_new 32`, greedy decoding:** `llama-mtmd-cli`
completes in ~20.6 s; brain's KV-cached run takes ~123-147 s. **Not a win
yet.** The fix closed the loop that was structurally quadratic, but with
decode no longer dominant, the remaining cost split roughly as ~18 s model
construction, ~50 s decode (prompt prefill + 32 `O(1)` steps), and ~75-80 s
vision encoding (SAM ViT-B at 1024x1024 → CLIP-L/24 → compressor →
projector) - and that last number is not yet broken down per kernel.
`llama-mtmd-cli` encodes the same-shaped image in roughly 15 s. Re-profiling
after a fix is supposed to promote the next bottleneck (§F.9) - this is that
promotion, landing squarely on the vision encoder, not the decode loop, as
the next place to look.

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

**No real-weight end-to-end re-run.** The `--max_new 32` head-to-head above
needs the real `DeepSeek-OCR-Q8_0`/`mmproj` GGUF pair (~22 GiB resident); this
machine had only an empty HF-cache ref stub this session, no downloaded
weights, so that run could not be reproduced to get a real updated wall-clock
or per-kernel table. The kernel-level microbenchmarks above are what could
be measured honestly here. Whoever has the real weights should re-run
`BRAIN_DEEPSEEK_OCR_DIR=<dir> BRAIN_PROFILE=1 brain do deepseek-ai/DeepSeek-OCR
generate --max_new 32 --in image=page.ppm --json` and update this section and
`.agents/roadmap/deepseek-ocr.md`'s Phase 8 entry with the real before/after
numbers.
