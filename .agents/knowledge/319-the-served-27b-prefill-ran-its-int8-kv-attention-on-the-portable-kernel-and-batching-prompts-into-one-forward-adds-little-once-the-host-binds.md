<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 319. The served 27B prefill ran its int8-KV attention on the portable kernel; batching prompts into one forward then adds little

Question: agent workloads are prefill-bound (each turn adds 2-20k prompt tokens and generates 30-150), and
the served Qwen3.8-27B (Q8_0 GGUF served INT8, `Qwen35GgufEngine` under `model::serve::Scheduler`, int8 KV,
ctx 32768, slice 512) prefilled the 16-request agent mix (110,967 prompt tokens, one runaway of 400 tokens)
at about 1,000 prompt tok/s, 2.7 times under the 2,649 tok/s a single-sequence profile reached. Why, and how
much can be taken back? Every figure is **VERIFIED-BY-RUN** on one GH200 shared with other lanes (host load
average 11-50, which stretches every host-bound part; differences under about 5 percent are noise), greedy,
through `qwen35_spec_sweep --mix` (scheduler, no HTTP) unless a row says "server" (`brain serve` and
`replay.py`, concurrency 16).

## The gap: the int8 KV cache had no native attention kernel

The fp16 tensor-core flash prefill (`flash_prefill_f16_hd256`) read only an f32 cache; a bf16 or int8 cache
fell back to the portable `paged_flash_prefill_hd256` WGSL kernel (3.7 TFLOPS against 56 on the same shape).
`qwen35_gguf_prefill_profile --kv` (new) shows it. One round, per-kernel device time:

| round | KV | tok/s | attention ms (16 layers) | int8 GEMM ms | whole round device ms |
|---|---|---:|---:|---:|---:|
| depth 2048, 1024 rows | f32 | 2,866 | 29 | 183 | 307 |
| depth 2048, 1024 rows | int8, portable attention | 1,752 | 511 | 183 | 789 |
| depth 8192, 512 rows | f32 | 2,302 | 33 | 101 | 186 |
| depth 8192, 512 rows | int8, portable attention | 702 | 619 | 101 | 775 |
| depth 4096, 1024 rows | int8, native attention | 2,633 | 68 | 182 | 345 |
| depth 8192, 1024 rows | int8, native attention | 2,411 | 105 | 182 | 382 |

The served mix is at depth 0-12k, so the portable kernel cost it 60-80 percent of every round, and grew
with depth (host stalls, the 512-row slices' smaller GEMMs and the decode steps between slices are the rest).
The kernel is now one source for three tiers (`BRAIN_KV_TIER`): bf16 planes are widened and int8 planes
dequantised (value times the per-row scale) while they are staged. Attention for 256 rows at depth 1792, int8
cache: 3.281 ms to 0.307 ms (10.7x); agreement with the portable kernel and an f64 oracle over the stored
values within 6.7e-3 of the output rms (bar 1.2e-2), bf16 within 3.9e-3. A bf16 cache now prefills with fp16
attention, as the f32 one already did.

## What the changes bought

| scheduler harness, 16-request mix (load average 14-21) | prompt tok/s | wall s |
|---|---:|---:|
| portable attention, slice 512 (the evidence) | 944 | 117.6 |
| native int8 attention, one sequence per forward, slice 512 | 1,753 / 1,759 | 63.3 / 63.1 |
| ... slice 256 | 1,453 | 76.4 |
| ... slice 1024 | 1,888 | 58.8 |
| ... slice 2048 | 1,914 | 58.0 |
| slices of several sequences in one forward: 1024 rows, slice 512 | 1,900 | 58.4 |
| ... 2048 rows, slice 1024 | 1,981 | 56.0 |
| server (HTTP, concurrency 16), portable attention (earlier evidence) | 1,008 | 110.1 |
| server, native attention, slice 512, 1024-row batches | 1,753 | 63.3 |
| server, native attention, slice 512, one sequence per forward | 1,796 | 61.8 |
| server, native attention, one sequence per forward, slice 512, again (load average 41-49) | 1,679 / 1,674 | 66.1 / 66.3 |
| server, native attention, slice 1024 (load average 33-45) | 1,773 / 1,702 | 62.6 / 65.2 |

The attention kernel is worth about 1.75x; a slice of 1024 rows instead of 512 another 7 percent in the harness and 3 percent through the server (within
its run-to-run noise of about 8 percent at this host load), with the first requests finishing 8-10 s later, so
the default stays 512 (`--qwen35-prefill-slice`). Prefilling several sequences' slices in one
forward (`PagedDecoder::prefill_chunks`, `Qwen35::run_prefill_multi_stage`) gains 8 percent over the same
slice of one sequence at 512 rows and 5 percent at 2048 rows, and nothing through the server at slice 512: GEMMs of 512 rows run at 276 TOPS and of 1024 at 305, and what a slice also pays (host
recording and a drain per layer, the embedding and the readback of the block, the decode steps in between)
does not shrink with M. It also makes sequences advance together: the first request finishes at 21 s instead of
11 s, so it is off by default (`--qwen35-prefill-batch`). Greedy output of 8 prompts of 1800-2717 tokens
(48 new): batched, one at a time and whole-prompt are 8/8 identical. A tiny-model test gates the multi-sequence
forward against each sequence alone to 1e-5 (a wrong slot fails it at 8.9e-3).

## The int8 GEMM: the bound is the MMA pipeline, not the scale fold

`matmul_i8_mma` is 53 percent of a round, at 288-305 TOPS. Skipping the fold altogether (accumulate the raw
sums, wrong results) at the same 64x64 tile reaches 368 TOPS: the fold costs at most 22 percent. Removing the
bias inside the scale FMA and adding a k64 tile's two terms before the running total (`p = fma(d, s, -bias*s)`,
exact for f16 scales) measured +3.5 percent at 64x64, so it was not kept. The 128x64 tile is level with 64x64
(291 against 288 TOPS) and 256-row tiles are slower (267). Going past this means `wgmma` (sm_90), where the
fold, at 2-4 FP32 instructions per output per k32, bounds the kernel at about 470-930 TOPS; not attempted.

A trap worth keeping: the first sweeps of larger tiles read 700-970 TOPS and were wrong. The harness
prepends `#define`s to the kernel source; a shell `$(printf ...)` stripped the trailing newline of the first
one, which swallowed `#define BRAIN_I8_BM`, so the kernel ran its default 64-row tile under a grid built for
256 rows and computed a quarter of the output. A TOPS figure from a sweep needs the output checked.

## What bounds the rest

At depth 4096 a 1024-row round is 389 ms of wall for 345 ms of device: the int8 GEMM 182 (53 percent),
attention 68 (20), Gated DeltaNet 36 (10), element-wise glue 54 (16: `silu_mul` 13, `add2` 11, `quant_pack` 9,
norms 14, the rest), host 44. The served mix runs at 2,130-2,240 prompt tok/s inside `prefill_chunk` and
about 1,850 end to end, 15 percent of the wall being the decode steps between slices; the target of 5,000 is
not reached. The levers left, in order of size: an int8 GEMM on `wgmma`; an attention kernel with more than one
block per SM (it runs at 27 TFLOPS of an mma.sync ceiling several times that); the quantisation epilogue the
decode tape fuses (`silu_mul`, `max_abs`, `quant_pack`) in the prefill rounds; fewer drains per layer.
