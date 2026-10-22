<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 240. The flash trio trains at head_dim 256; around it, three elementwise kernels cost as much

The causal GQA flash trio held head_dim <= 128 (`CH = 32` channels per lane
times 4 lanes). Qwen3.5/3.8 full-attention layers are 256 wide, so they used the
materialised chain: `24 * T^2 * 4` bytes per slab, 1.6 GiB at T = 4096 and
over the 4 GiB binding at T = 13377.

Fix: keep `CH = 32` (the compile-time trip count the register argument needs)
and let the lane count follow the head: 4 lanes x 64 rows up to 128, 8 lanes x
32 rows up to 256. Same 256-thread workgroup, same registers per thread, tile
rows `lanes * CH` wide, 32 KiB shared in the backward. The dispatch's workgroup
count follows (`flash_gqa_rows_per_workgroup`); the dispatch refuses a shape
whose `b*t*heads*head_dim` exceeds u32.

Measured on one GH200 (CUDA backend, kernels generated from the WGSL, shared
card so +-40% run to run), qwen3.8 attention shape 24 q heads / 4 kv heads /
hd 256, batch 1, mixer forward+backward for one layer:

| T | flash trio device time | whole mixer wall | materialised mixer |
|---|---|---|---|
| 4096 | 187 ms (fwd 49, dq 65, dkv 73) | 310-520 ms | 1.1-1.9 s |
| 16384 | 2.8 s (fwd 0.70, dq 0.99, dkv 1.14) | 3.3-5.5 s | not run (24 GiB slabs) |
| 32768 | not split out | 12.6-20.8 s | not run |

Card memory growth at T = 16384 for the layer, inputs included: 6.5 GiB.
Time scales as T^2, as expected; the trio runs at roughly 4 TFLOPS fp32, a few
percent of the card, so a native tiled (or tensor-core) kernel is the
next lever and is not done.

At T = 4096 the mixer's non-attention kernels cost 118 ms against the trio's
187 ms: `concat2` 46 ms, `concat_split` 46 ms (two calls), `rmsnorm_dw` 26 ms.
They move a few hundred MB each, so they are far below bandwidth. They are
shared ops outside this lane.

The CPU backend cannot run the trio (barrier kernels), so
`FlashGqaIds::supported` is false there and the mixer takes the materialised
chain; its gradcheck is run on that fallback.

Rule: when a long-sequence path becomes memory-bound-free, profile the layer by
kernel before tuning the headline kernel; here 40% of the layer sat elsewhere.
