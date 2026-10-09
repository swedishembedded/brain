<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 212. A shared device bills another context's time slice to your next kernel

Profiling a YOLOv8n training step on the CUDA backend (`BRAIN_PROFILE=1`, batch
8 at 512) on a card another process kept at 100% utilisation, the per-kernel
table showed almost every kind at about the same cost per call: `bn_running`
(a one-thread-per-channel update of a few hundred floats) 2.35 ms, `silu`
2.40 ms, `add2` 2.2 ms, `native:brain_conv2d_dw_reduce` (a 4096-element sum)
2.35 ms - while a native conv forward timed alone took 0.05 ms. The step's
kernel table summed to seconds of "device time" that no kernel spent.

The floor is not the kernels. Every step that is not the first of its
submission is preceded ON THE STREAM by the upload of its uniform words, a
copy-engine operation. While the compute engine waits on that copy the device
is free to switch to another context, and on a part without concurrent
contexts that context then runs its whole time slice before the next kernel of
ours starts - between our start and end timestamps. A kernel that is first in
its submission (or a submission of one) has no copy in front of it and reads
true. Measured, same shape, same binary: the weight-gradient reduction read
2.31-2.36 ms in seven of seven runs as the second step of a two-step
submission, and under 0.05 ms submitted on its own.

What it means in practice:

1. **A per-kernel table taken on a busy shared card is not a profile.** The
   inflation is per LAUNCH, not per microsecond of work, so it buries every
   small kernel at the same value and makes the launch count look like the
   cost. Check `nvidia-smi --query-compute-apps` before trusting one, and
   compare a known-trivial kernel's per-call time against its size: if
   `bn_running` costs milliseconds the table is measuring the neighbour.
2. **Per-kernel numbers on a shared card come from one step per submission,
   fastest of several.** `gpu-core/tests/conv2d_native_bench.rs` times each
   step alone and keeps the minimum of seven; that reproduces the quiet-card
   numbers for the kernel itself, though not the cost of the uploads between
   steps that a real submission pays.
3. **The uniform upload is itself the reason the slice lands there.** A
   kernel whose parameters reach it without a copy on the stream (a captured
   graph's single parameter block, or by-value kernel arguments) gives the
   scheduler no idle compute engine to switch on between kernels of one
   submission.
