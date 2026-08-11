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
