<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 327. The 27B prefill round went from 216 to 172 ms with a two-block flash kernel, fused norms and no layer drain

Measurement (window of 2026-10-19 13:06-13:12, GH200 card 1 exclusive, `qwen35_gguf_prefill_profile`
Q8_0 4096 512 4 --kv int8, fastest of four rounds; log `runs/b27/window6.log`):

| build | best round | tok/s |
|---|---:|---:|
| window5 binary (before the changes below) | 216.0 ms | 2371 |
| origin/main: flash v2 + fused front end + no layer drain (the default) | 172.5 ms | 2969 |
| same, per-layer drain forced (`BRAIN_PREFILL_ARENA=drain`) | 177.8 ms | 2880 |
| same, no scratch arena | 185.7 ms | 2757 |
| flash kernel v1 only (`--flash-prefill v1`) | 188.6 ms | 2715 |

Where it came from (device time of the layer stack, 196.3 ms before, 150.1 ms now):

- Flash prefill for head_dim 256 over an int8/bf16 cache, version 2: 40.8 ms to 16.2 ms. One warp owns 16
  query rows over the whole head (version 1 had each warp pair compute the same scores), Q lives in shared
  memory, a block is 128 threads sized for two resident blocks per SM, 64-key tiles, and a compact cache is
  widened only when written to shared memory. About 16 ms of the round.
- The decode tape's fused kernels in the prefill layer stack (`add_rms_quant`, `quant_epilogue`, with the next
  layer's normed activation carried across the layer boundary): `silu_mul`, `rmsnorm_rows`, `max_abs_rows`,
  `quant_pack` and `add2` (about 28 ms of device time) became 6.2 ms. About 8 ms of the round wall time.
- The pooled round's per-layer `poll_wait` is skipped where `Backend::submissions_ordered` (CUDA: one handle's
  work runs in stream order, which already keeps a recycled slot's writer behind its readers): 5 ms.
  The no-arena row is worse than the arena without a drain, so the arena stays.

Not changed: `matmul_i8_mma` is 99.7 ms, 66% of the device time and now the bound; the Hopper wgmma
variant measured no gain (`runs/b27/window5.log`).

Training probe (64 layers, NF4, 8192 tokens, same window): steady 990-1010 tokens/s, 8.1-8.3 s per step.
The 15 s step 4 of a `--steps 4 --profile` run is the profiled step: kernel timing serialises every launch.
The bytes cleared by memset per step fall with `Backend::storage_uninit` (debug builds poison those
blocks): at 4 layers x 4096 tokens 52 GiB to 10.8 GiB.

Lesson: the single-process kernel table (`device time`) understates what a host-side drain costs only when
the host is slow; here the drain was 3% of the round, the flash kernel's duplicated scores were 20%.
