<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 514. A fine-tune gate that passed at one seed failed when the GEMM changed its rounding order, because its recipe learned from some seeds only

**Lesson:** A learning gate with a fixed seed certifies that seed's trajectory, not the recipe. Before a threshold
is set on what a training run reaches, sweep the training seed (and the arithmetic route, where there are two) and
pick a recipe every seed clears with margin; otherwise any change that only reorders a sum (a new GEMM, a different
thread split) flips the gate and reads as a regression in whatever changed.

## What happened

`toy_training::the_toy_model_learns_a_cipher_it_is_tested_on_with_prompts_it_never_saw` (`crates/sdk/tests/it`)
fine-tunes the toy chat checkpoint (LoRA rank 16, 500 steps at lr 5e-3, seed 1) and requires 90% held-out cipher
accuracy. It started failing on AArch64 at `b81ff25bb`, which routed every f32 GEMM and the GQA attention natives to
a NEON GEMM (#486): seed 1 reached 0.69. Excluding any one of the three native groups (the matmul forward, its two
backward GEMMs, the GQA family) made it pass again, which no single wrong kernel would do. Sweeping the training seed
settled it: at lr 5e-3 and 500 steps the held-out accuracy across seeds 2 to 6 was 1, 0.88, 0, 0.76 and 0.01 on the
NEON route and 1, 0.03, 1 and 0.09 (seeds 3 to 6) on the JIT route. The recipe diverged or stalled for some seeds on
either route; seed 1 had happened to clear it on the old one.

## Fix

lr 2e-3 for 800 steps: seeds 1 to 8 reach 0.96 to 1.0 on the NEON route, and the seeds that collapsed on the JIT
route (4, 6) reach 1.0 there. The NEON GEMM was not changed; its backend parity tests pass.
