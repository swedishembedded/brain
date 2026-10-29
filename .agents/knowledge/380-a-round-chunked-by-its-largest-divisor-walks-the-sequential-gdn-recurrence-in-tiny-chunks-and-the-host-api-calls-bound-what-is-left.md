<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 380. A round chunked by its largest divisor walks the sequential GDN recurrence in tiny chunks, and the host's CUDA API calls bound what is left

Question: the mixed iteration (knowledge 379) ran at 3300 prompt tok/s against 5000 standalone and left the card idle between
launches. What does `nsys` say the card does inside `serve.mixed`, and what is the first thing to fix?

Measured (gpu1, FP8 Qwen3.8-27B, kv int8, `--max-batch 64 --speculative off`, `vllm bench serve` chat 1K/128, c=16, nsys
cuda + nvtx, kernel time summed over the `serve.mixed` ranges):

1. **`brain_gdn_chunk_loop_tc` was 13.3 of the 25.6 s the card was busy** (52 %, 1136 us a call, 11712 calls) - and 81 % of that
   in calls longer than 1 ms. The across-chunk recurrence is one launch per layer that walks the chunks in sequence at about 100 us
   each (192 blocks, 8 warps an SM, latency bound). `gdn_chunk_size(rows)` picked the largest of 64, 32, ... 1 that divides the
   round, so a 600-row slice ran 75 chunks of 8 and a 195-row one 195 chunks of 1. The serving rule now runs the whole multiple of
   64 as chunks of 64 and the rest as one chunk of its own size (`qwen35::gdn_stream_pieces`, used by the mixed stage and by the
   single-sequence prefill round): the loop is 1.7 s, and the iterations that carry a slice are no longer card bound.
2. **What is left is the host.** After (1) the card is busy 39 % of a mixed iteration's wall time (~5000 launches, ~60 us each under
   the profiler). The CUDA API calls of a median iteration: `cuLaunchKernel(Ex)` ~5300 (14-16 us), `cuMemsetD8Async` 1230 (12-16 us),
   `cuMemcpyHtoDAsync` 5500 (2.4 us, the step's parameter upload), alloc/free from the pool ~6000 each (1.5 us). nsys CPU sampling puts
   52 % of the serving thread's samples in `libcuda`, 15 % in the kernel, 14 % in libc (malloc), 10 % in brain itself. The zero-fresh
   allocations that every kernel then overwrote (`take_rows`, the layer outputs, the fused front ends) cost one memset each: they are
   `storage_uninit` now (1230 -> 858 an iteration; ~12 more are still cleared in `model::gdn_mixer`, e.g. the chunk-major query/key).
3. Client results (output tok/s at c = 1 / 4 / 16 / 64; before = knowledge 379): 50 / 187 / 208 / 237 -> 47 / 190 / 294 / 366; ITL p99 at 16
   589 -> 293 ms; TTFT p50 at 16 2.5 s -> 1.3 s; 31K-prompt TTFT 7.3 s. (c = 1 and 4 are decode-bound and unchanged within the noise.)

Tried and not kept:

- A 64-row-tile build of `matmul_fp8_wgmma` (one warpgroup, two blocks an SM, a tuner candidate): the tuner took it for 28 of 96 classes
  (the small row counts, 27 us a call) and the sum of GEMM time per mixed iteration did not move (55.0 -> 57.3 ms, within the spread of
  the slice mix). The 500-1100 row slices stay on the 128-row tile: their cost is wave quantisation (200 blocks on 132 SMs for n = 5120), which
  a smaller tile does not change. A stream-K / persistent kernel is what that needs (roadmap item 17).
- The minimum prompt slice (`MIN_MIXED_PROMPT_ROWS`): 256 / 512 / 1024 rows give c=16 / c=64 output 224 / 343, 294 / 366, 275 / 408 tok/s and
  ITL p99 184 / 293 / 362 ms at c=16. 512 stays: the best c=16 throughput and the middle latency.

Left, in order of measured weight: the launch and API-call count of a mixed forward (a graph replay of the layer round with the row count
bucketed; the 1008 `concat_split` row copies that a row-view binding would remove; an uploaded-once parameter block instead of one
`cuMemcpyHtoDAsync` per step), the rms-norm -> e4m3 front end (the int8 packing of `add_rms_quant`/`quant_epilogue` is computed for FP8 weights and read by
nothing, then `quant_e4m3_128` runs again: ~10 % of the kernel time), and the GEMM fill. Pre-existing: `gguf_conformance`
`prompts_sharing_a_document_resume_from_where_they_part` fails on the commit before this work too.

Quality: `qwen35_fp8_parity` (CHANGELOG.md, 2 x ~1000 tokens): agreement 100 %, no flip above margin 0.5; mean KL 1.5e-3 -> 3.1e-3 because a
975-token prompt moves from 975 chunks of 1 (the exact recurrence) to chunks of 64 (the production algorithm).
