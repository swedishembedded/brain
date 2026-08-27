# Performance overview

brain is built so performance claims are measured, not guessed. This page
explains the pieces of that: per-kernel profiling with a hardware roofline, a
runtime kernel selector, and INT8 inference support. It deliberately carries
no specific timing/throughput numbers of its own - a measured number
describes one specific machine, driver, and code revision at one point in
time, and goes stale the moment any of those change. See the bottom of this
page for where real, current numbers live.

## Profiling and the roofline probe

Set `BRAIN_PROFILE=1` on any `brain` run to get a per-kernel breakdown: time,
call count, bytes moved, and - where the device supports it - the percentage
of that device's own measured compute or memory-bandwidth ceiling the kernel
is hitting.

That ceiling is not a datasheet number. On first use, brain runs a small
roofline probe against the actual device present (compute throughput, memory
bandwidth, and the "ridge point" where a kernel stops being bandwidth-bound
and starts being compute-bound), and grades every subsequent kernel against
*that* measurement. A laptop iGPU and a datacenter GPU get different roofs,
because they have different roofs - the profiler never assumes a vendor
spec is what you actually have.

This is also why a kernel report never silently reports zero utilization when
the answer is unknown: an unmeasured or unmodeled cost is reported as such, not
folded into the totals as if it were free.

## The runtime kernel selector

Most operations in brain (attention, normalization, matrix multiply, an
optimizer's gradient-norm reduction, and others) have more than one
implementation - different tilings, different thread-cooperation strategies,
different fits for a given shape and device. brain does not pick one at
compile time and hope. A selector inspects the device's capabilities (and,
for some operations, the shape of the work) and dispatches to whichever
registered variant is expected to be fastest there, falling back to a safe
default when a device lacks what a faster variant needs. You do not choose
kernels by hand; the engine does, and `BRAIN_PROFILE=1` shows you which one
actually ran.

## INT8 inference

Several served models support INT8 inference for lower memory use and higher
throughput than full-precision - check the specific model's own page under
`docs/models/` (for example `docs/models/qwen3.md` or `docs/models/flux2.md`)
to see whether it's supported and how to enable it, since support and the
exact accuracy trade-off are model-specific.

## Cost accounting: `brain flops`

`brain flops <model> <shape>` reports how many floating-point (or, for INT8
paths, integer) operations a forward or backward pass actually costs, without
running it. It's a coverage-honest cost registry: every kernel brain dispatches
has either a registered cost formula or is explicitly listed as **not
measured** - the tool never rounds an unmodeled cost down to zero and folds
it into a total that looks complete. Run `brain flops --help` for the full
set of flags and the accounting model.

### A whole generation, by stage

`--model flux2` and `--model ltxv` price an image or a video the way one
actually happens: text encode, then N denoise evaluations, then a VAE decode,
reported separately and then totalled - because *which stage dominates* is the
question the number exists to answer. A video is reported per second of output
as well as per clip.

```
brain flops --model flux2 --variant klein-4b --width 1024 --height 1024 [--i8] [--per-kernel]
brain flops --model ltxv --variant ltx25-22b --width 768 --height 512 --frames 121 --fps 24
```

Nothing runs, and no checkpoint is needed. The denoise cost is derived from
probe builds of the same config at one and zero blocks, which is exact because
the graph is affine in the block count - and every run re-checks that at a
point outside the basis rather than assuming it, refusing to print a total it
could not verify. That is what lets a 22B video model be priced on a card that
cannot hold it.

Each stage also reports its arithmetic intensity against the device's own
measured ridge point, so the output says whether a stage is compute- or
memory-bound, and a roofline lower bound in seconds. That bound is a bound:
what it is for is the *ratio* a real run achieves against it, which `--run`
measures on a build small enough to execute. `--vae <dir>` additionally checks
the weight-free VAE graph against the one the real checkpoint builds.

## A number measured on one machine is not a number for yours

Different clocks, different memory bandwidth, different driver, different
model shape, and different concurrent load on a shared machine all change the
answer. A result from someone else's GPU, CPU, or NPU - however carefully it
was measured - is an illustration of what the profiling and selection tools
above can find, never a prediction for your own hardware. The way to know
your own numbers is to run them yourself.

## Where the numbers are

This page intentionally stops short of citing specific timings or
throughputs. Real, current, reproducible numbers live in places that stay
honest as the code and hardware change under them, not frozen in prose here:

- **`make perf`** / **`make perf/<scenario>`** - run brain's own performance
  harness (latency, throughput, serving-under-load, and a concurrency sweep)
  against your own device. See [Benchmarking your setup](benchmarking.md).
- **`make perf/compare`** - a leaderboard over every saved run under
  `results/*.json` on your machine, refusing to rank across incompatible
  units or past a failed correctness gate.
- **`results/*.json`** - the raw artifacts `brain perf` writes, one per run,
  each carrying the hardware it ran on so it's never silently compared
  against a different machine.
- **The committed perf gates under `scripts/gates/`** (e.g.
  `qwen-serving-perf-gate.sh`, `forecast-perf-gate.sh`, `wm-perf-gate.sh`) -
  regression floors checked in CI against a saved baseline, the closest thing
  in this repo to a durable performance number, because they're re-verified
  on every run rather than quoted once and left to rot.
- **Session-specific investigation logs** (a model's own performance pass,
  with real measured numbers, caveats, and what was tried and killed) live in
  `.agents/roadmap/<model>.md`, not here - see that model's roadmap entry if
  you want the detailed history behind a specific optimization.
