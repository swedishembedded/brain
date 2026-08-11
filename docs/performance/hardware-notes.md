# Hardware notes

brain has been measured and tuned on a couple of real GPU/CPU configurations
during development. This page is an illustration of the range you might see —
**not a promise for your hardware.** Different device, different driver,
different model shape, different result.

## What one older datacenter GPU showed

On an NVIDIA Tesla P40 (an older, non-tensor-core datacenter card), a few
things were durable enough to be worth writing down as illustrations of what
"finding the ceiling" looks like in practice:

- The card's raw fp32/fp64/int8 throughput, measured directly, reached
  93–100% of its own datasheet peak — confirming that when a kernel is
  written to actually saturate the hardware, brain's measurement and the
  vendor's own numbers agree.
- A naive, one-thread-per-output matrix-multiply kernel reached well under
  1% of that peak — it was starved on memory bandwidth, not compute. A
  tiled, software-pipelined GEMM kernel targeting the same math reached
  roughly a third of the card's fp32 peak. Same math, same card — the
  difference was entirely how the kernel was written to use the hardware.
- An INT8 matrix-multiply path, using the card's integer dot-product
  hardware, measured 2.5–3.6× the throughput of the tuned fp32 path on
  identical shapes — the expected win from the lower-precision datapath,
  actually realized rather than assumed.

## What one CPU workstation showed

On a multi-core AVX2 workstation, replacing scalar kernel execution with a
hand-vectorized native path for the dominant operation in a real inference
workload cut per-frame latency by well over an order of magnitude — see
`docs/performance/overview.md` for the specific example.

## The point

These numbers describe two specific machines at a specific point in
development. They are not a spec sheet, and they will not reproduce on your
hardware — different clocks, different memory bandwidth, different driver,
different model shape all change the answer. The way to know your own
numbers is `brain perf run` and `brain flops`, not this page — see
`docs/performance/benchmarking.md` and `docs/performance/overview.md`.
