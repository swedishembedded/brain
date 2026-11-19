<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 533. A warm Qwen3-4B LoRA SFT step runs at 14% of bf16 peak, and attention is three quarters of it

**Lesson:** profile a training step before choosing what to optimise, and count FLOPs from what actually ran. A
whole-run estimate put packed SFT at about 2% MFU and blamed padding; the profile put a warm step at 14% of the bf16
peak with 2% padded tokens, and found one kernel family, causal flash attention, taking 74% of the step at about 8% of
peak. Everything else is either at the roof or too small to matter.

## What happened

2026-11-19, Nsight Systems on a GH200 (989 TFLOP/s bf16 peak, 673 TFLOP/s measured GEMM roof), Qwen3-4B bf16 base,
LoRA rank 16 on all seven projections, batch 1, `SavedPolicy::Recompute`, a warm `brain finetune` step on a
13,514-token row (4.03 s):

- **Attention: 74% of device time at about 83 TFLOP/s** (forward about 100, backward dq+dkv about 71). The flash
  kernels are FA2-shaped `mma.sync.m16n8k16` with 64x32 tiles and no wgmma or TMA, an Ampere-style design on a
  Hopper card. Its share falls to 46% at 4k tokens and grows with the row, so tokens/s halves from 4k to 16k.
- **Base GEMMs: 10.5% at 660-720 TFLOP/s**, at or above the measured roof; nothing to gain there.
- **LoRA GEMMs in fp32/TF32 plus their casts: about 7%** for under 1% of the FLOPs, the second target.
- **Whole step: 139 TFLOP/s hardware (14% of peak), 101 TFLOP/s model FLOPs (10%).**
- **Padding: 2.0%** of computed tokens, because SFT already trims each row to its last supervised position
  (`Qwen::needed_len`). Builds that cannot trim compute the full block (a 9.5k record in a 16k block: 42% padded
  tokens, 66% of attention FLOPs).
- **Host gaps: under 2%** of a warm step; the card is 98-99.7% busy. Only the first step of a process loses 12-18% to
  module loads and allocation.

The earlier 2% MFU figure divided useful tokens of a whole run (start-up, scoring and evaluation included) by peak,
and predated SFT row trimming; it described the run, not the step.

## What to do

- A Hopper-native causal flash attention (wgmma/TMA), forward and backward, is the first training-speed item: at
  GEMM-class attention throughput the 13.5k step would go from about 4.0 s to about 1.5 s.
- Do not spend effort on host overhead, base GEMM selection or SFT padding for this shape; they are measured as small.
- Quote utilisation per warm step with the FLOP count of what ran (recompute included) beside model FLOPs, and keep
  run-level throughput as a separate number.

## Evidence

Kernel-family table, FLOP accounting and host-gap analysis in `.agents/roadmap/training-efficiency.md` (diagnosis rows
2 and 10, deferred item R6), from an nsys capture on brain `1bdd5400c`.
