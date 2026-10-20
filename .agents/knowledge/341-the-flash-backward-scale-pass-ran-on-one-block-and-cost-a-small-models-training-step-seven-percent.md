<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 341. The flash backward's scale pass ran on one block and cost a small model's training step 7 percent

Symptom: a Qwen3-0.6B LoRA step (3731 tokens, rank 16) was at about 7 percent of the fp16 tensor-core peak with the card
fully busy. `nsys profile -t cuda` (the kernel-kind table, not brain's own per-kernel table, which cannot time the
library GEMMs) showed `brain_flash_bwd_scale` at 645 us per call, 28 calls per step: 18 ms of a 275 ms kernel sum.

Cause: the kernel that derives the call's power-of-two dO scale was one block of 1024 threads striding over the whole
dO (7.6 M floats, 30 MB): 47 GB/s. It is the "one block does all the work" trap in `.agents/rules/kernels.md` C3, hidden
because the three-kernel flash backward looks like attention work and the table's share (6.6 percent) never
reached the top of a per-op ranking.

Fix: `brain_flash_bwd_absmax` (256 blocks, float4 loads, one partial each) then the one-block `brain_flash_bwd_scale` over
the 256 partials: 7 us + 1.2 us per call, the same scale (same max, same rounding). `cuda_flash_attn_bwd` gained a case
larger than one block's share with the largest gradient in the last element.

Also measured (same profile): the forward flash kernel at head_dim 128 is scheduler-bound (12.5 percent occupancy at 159
registers, 0.52 eligible warps per scheduler, stalls on short scoreboard and fixed latency); `__launch_bounds__(256, 2)` at
head_dim 128 takes it from a median 787 us to 698 us without spills. The same bound on the backward dq kernel gains 3
percent and on dkv spills (1227 us to 3004 us), so both stay at one block per SM. Profile and ranking of what is left:
runs/b27/profile-small-lora.md.
