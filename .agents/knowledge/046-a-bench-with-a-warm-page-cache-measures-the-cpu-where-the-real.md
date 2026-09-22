<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 46. A bench with a warm page cache measures the CPU where the real run measures the disk

The isolated harness for one model's streamed forward predicted ~275s of
denoise for a real generation. The real generation measured 440s, and the
per-stage split disagreed in BOTH directions - the harness under-predicted
the first (cold) step by 2.1x and over-predicted every later step by 1.4x.

The harness was not wrong about anything it measured. It had simply been run
several times in a row over the same four transformer blocks, so those blocks
were resident in page cache and its "checkpoint read + dequantize" stage was
timing the dequantize. The real run reads ~50 GB of checkpoint that nothing
has touched, on storage measured at **58-68 MB/s cold** (against 4.3 GB/s for
the same bytes once cached - a ~70x cliff), and its same-named stage is
timing the disk.

Consequences worth keeping:

* **Check whether the resource you are measuring is the one the real run
  hits.** `free`, the process state (`D` vs `R`), and CPU-time-versus-wall
  -clock all answer it cheaply: the run that looked like 964s of computation
  spent 7m12s of CPU across 16m14s of wall clock, which says "waiting" before
  any profiler is opened.
* **Measure the device, do not assume it.** A `dd` from an uncached region
  took under three minutes and turned a kernel-optimization brief into an
  I/O finding. Sixteen parallel readers measured the SAME throughput as one,
  which additionally killed the natural follow-up hypothesis (deepen the
  queue) before any code was written for it.
* **A warm-cache number is still the right number for the CPU work inside
  the stage** - it is how the parallel dequantize and quantize wins were
  isolated from I/O noise at all. Keep both, and label which is which; the
  mistake is letting one stand in for the other in an extrapolation.
