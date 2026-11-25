<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 595. A training build drew a random value for every weight its base then overwrote

**Lesson:** When a build starts from a checkpoint, draw initial values only for the parameters the checkpoint does
not supply. A full random init followed by an overwrite looks free in a small test and costs tens of seconds of host
time per build at real size, and a tool that builds a model several times pays it every time.

## What happened
`SpeechSftModel::build` called `qwen3::init_weights` (a Box-Muller Gaussian for every decoder parameter, single
threaded) and then replaced every parameter but the never-read token table with the base's tensors. For
Sten-TTS-1.7B (1.9 B parameters) that was 20-30 s per build. Weight surgery builds a model for calibration, for the
held-out measurement, for the curvature estimate, for each measured edit and for a greedy layer choice, so the draws
dominated a run: an nsys CPU-sampling profile of a small surgery (2 calibration batches, a Fisher channel selection
and a one-layer removal, n460, GH200) put `qwen3::init::init_weights` and `__log_finite` on the main thread in every
build window, while the GPU kernels summed to about 4 s of a 198 s run.

## What to do
`qwen3::init_weights_where(cfg, seed, drawn)` draws only the selected parameters; the speech build selects the ones
its base does not supply. The same small surgery then took 56 s with bit-identical measurements (held-out KL 0.3170
and 0.3535 before and after). The trainer's and the frozen teacher's builds take the same path.

When profiling a host-bound run with nsys `--sample=cpu`, look at the main thread: idle rayon workers spinning in
`crossbeam_epoch::try_advance` and `Stealer::steal` fill most samples of the whole process and hide where the time
goes.
