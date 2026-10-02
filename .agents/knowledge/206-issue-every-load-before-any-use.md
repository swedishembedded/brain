<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 206. In a small latency-bound kernel, issue every load before any use, and every store after

The first fused decode kernel (residual add + RMSNorm + per-row int8 quantiser,
one 64-thread block per 5120-wide row) ran at 26 us - barely better than the
generated chain it replaced (52 us), and a latency, not a bandwidth, problem:
20 KB of data.

The SASS showed why: load, dependent add, load, dependent add. Two causes, both
the compiler being right:

- brain's buffers alias by design (a sliced step binds ranges of one
  allocation) and these kernels deliberately carry no `__restrict__`, so a load
  cannot be moved above an earlier store through another pointer. A loop that
  stored `sum[c]` and then loaded the next element's operand made every
  iteration a serialised L2 round trip (~0.3 us x 80).
- `if (add) v += load(b)` puts the dependent add right behind the load it needs.

Writing the body as three phases - every global load of the row into registers
first, with nothing between them; the arithmetic in registers and shared memory;
then every store - took it from 26 us to 7.8 us (of which ~2.75 us is the
event-pair and launch floor of the harness). The same restructure took the
SwiGLU quantiser from 16 us to 11.5 us. `__ldg` alone did not help (it made one
variant worse): the structure has to change, not the load instruction.

The register cost is the price: 80 elements x (value, addend, gain) is 240
registers, which is why the kernel serves rows up to 5120 and the host keeps the
WGSL chain for anything wider.

Rule: when a small kernel is slower than a few L2 round trips, read its SASS for
a load immediately followed by its consumer, and separate load / compute / store
phases explicitly. A standalone `nvcc` harness with cudaEvent timing (kernel
source `#include`d, same flags: `--fmad=false`) iterates in seconds instead of
through the crate build.
