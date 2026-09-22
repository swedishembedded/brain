<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 37. A roofline probe calibrated for GPU throughput reads as a hang on the CPU backend

`gpu_core::roof::measure_compute`'s self-calibrating loop starts at a small
iteration count and doubles (or jumps straight to a computed target) until a
dispatch's wall-clock time clears `MIN_PROBE_SECONDS` (50 ms), capping at
`1 << 20` iterations over `FMA_THREADS = 1 << 20` parallel lanes. Every
existing caller of `gpu_core::roof::ensure` (`qwen_bench` and its siblings)
had only ever been run against a real GPU backend, where that calibration
converges in a handful of doublings. The first CPU-backend run of it sat
alive for hours of wall-clock time - negligible RSS, negligible CPU, tens of
seconds of total CPU time accumulated - before being killed. That is not a
slow computation (which would show high utilization across the whole thread
pool the whole time); it is a blocked/deadlocked one. Bisected by killing the
process and checking where its stdout had stopped: it never got past the
banner print, i.e. it was stuck inside the probe itself, before any real
per-layer work began.

The root cause (somewhere in `measure_compute`'s calibration loop,
`crates/backend-cpu`'s rayon dispatch, or their interaction under this
kernel set / thread count) was not tracked down at the time - see #38 for the
follow-up. `gpu_core::roof` already ships the escape hatch, `BRAIN_NO_ROOF=1`,
which skips the probe outright and makes callers report "roofline unmeasured"
instead of guessing. **Any new bench or profiler that calls
`gpu_core::roof::ensure` (directly, or via a shared banner-style helper) and
might run on the CPU backend should default to skipping the probe there, or
at minimum document the escape hatch loudly** - the existing benches never
hit this because they only ever ran on GPU.
