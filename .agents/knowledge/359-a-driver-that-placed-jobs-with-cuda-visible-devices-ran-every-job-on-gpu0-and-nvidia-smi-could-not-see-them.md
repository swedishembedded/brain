<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 359. A driver that placed jobs with `CUDA_VISIBLE_DEVICES` ran every job on gpu0, and nvidia-smi could not see them

Symptom: a measurement driver ran two training jobs at once, one per card,
by setting `CUDA_VISIBLE_DEVICES=0` and `=1` on the child processes. Both jobs
ran on gpu0 at half speed while gpu1 sat idle, and every job's recorded peak
device memory was 0 MiB.

Two independent causes:

1. **`CUDA_VISIBLE_DEVICES` filters the CUDA driver's enumeration, nothing
   else.** brain's default backend is wgpu over Vulkan, which enumerates
   adapters itself and never reads the variable, so every child built its
   `Gpu` on the first adapter (see #108 for what the variable does even on the
   CUDA backend). A brain process is placed by brain's own selection: the
   `--device` grammar in the CLI, or `gpu_core::devices::set_ambient_gpu`
   (what `--device` sets) in a program that embeds the engine, called before
   any `Gpu` is built.
2. **Inside a container, nvidia-smi reports PIDs of the host's PID
   namespace.** `--query-compute-apps=pid,used_memory` listed processes, but
   never under the PID the driver's `Popen` returned, so a per-process lookup
   matched nothing and the "peak" stayed at its initial 0. A job that has its
   card to itself is measured instead as the card's `memory.used` above the
   idle value read just before the job starts.

The second failure produced a plausible-looking number. A value nothing
measured must be recorded as absent, never as 0; the records written that way
were rewritten with the field null.
