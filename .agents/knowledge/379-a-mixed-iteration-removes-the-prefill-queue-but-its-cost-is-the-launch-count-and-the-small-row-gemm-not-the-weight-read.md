<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 379. A mixed iteration removes the prefill queue, but its cost is the launch count and the small-row GEMM, not the weight read

Question: with decode rows and prompt slices in one forward (serving-runtime.md "Mixed iterations"), what does the FP8 27B serve
on one GH200 through `brain serve` and `vllm bench serve` (1024 in / 128 out), and what bounds it now?

Measured (gpu1, kv int8, `--max-batch 64`, warm, N = max(16, 4C), seed 1), output tok/s at concurrency 1 / 4 / 16 / 64:

| | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| before (separate passes, knowledge 378) | 46.3 | 117.3 | 165.6 | 172.9 |
| mixed, 128-row minimum slice | 49.1 | 174.7 | 136.9 | 166.2 |
| mixed, 512-row minimum slice, 25 % stretch | 50.2 | 186.6 | 208.0 | 237.0 |
| vLLM 0.32 / SGLang | 79 / 86 | 299 / 214 | 759 / 781 | 1241 / 1044 |

TTFT p50 at 16 / 64: 6.6 / 41.8 s before, 2.5 / 2.0 s now (p99 5.2 / 25.9 s). ITL p50 18.9 / 19.2 / 24.1 / 231 ms; ITL p99 at 16 is
589 ms: the iterations that carry a 1K-row slice are 300-700 ms long. `brain perf run sweep chat` out tok/s at 1 / 4 / 16:
52.9 / 128.5 / 181.7 before, 54.5 / 192.3 / 202.3 now.

What the timeline says (nsys, NVTX `serve.mixed`, c=16):

1. **A 128-row slice beside 14 decode rows took 82-97 ms** against 25 ms for the decode step alone and about 25 ms for the
   slice alone: 2970 kernel launches, the card busy 48 ms of 97. The forward is host-bound at that size (about 32 us a
   launch), so a small slice pays the whole fixed cost for 128 rows. At equilibrium (one 1K prompt every 8 decode iterations at
   c=16) it is better to carry the prompt in few large slices and leave the other iterations to the replayed decode graph:
   hence the 512-row minimum and, when the latency target is out of reach even for the minimum slice, 25 % more than that
   slice's predicted time (`StepCost::prompt_rows`).
2. **With 512-1100 prompt rows the iteration is card-bound** (324-693 ms wall, 237-604 ms busy; 4000-6150 kernels) and the
   prompt part runs at about 3300 tok/s, below the 5000 tok/s of a standalone prefill: the leading decode rows' copies
   (`concat_split`, 587 launches an iteration) and one gather/scatter plus chunk-kernel chain per prompt segment.
3. **ncu, `matmul_fp8_wgmma` at 142-600 rows:** grids of 80-160 blocks on 132 SMs, 23-28 % SM throughput, 30 % memory
   throughput, 39-74 us each (395 launches = 23.7 of 48 ms of the c=16 iteration). Small-row fp8 GEMMs underfill the card; this is
   the open persistent-GEMM item (17), not the scheduler.

Consequences: the scheduler now reads weights once per iteration and removes prefill queueing (TTFT p50 -75 % at 16, -95 % at 64),
but throughput stays at 28 % (c=16) and 19 % (c=64) of vLLM. What is left is kernel work: the small-row fp8 GEMM, prefill at 3300 against
12800 tok/s, decode at 16 rows 25 ms against 20, and the launch count of a mixed forward (a recorded/graph replay of the chunk
round, or fewer copies). The engine falls back to the background lane (kept, with `PrefillPace`) where the mixed step is
unavailable: MTP builds, adapter banks, exact decode.

Tests: `model/tests/serve_mixed_step.rs` (mock decoder: tokens equal the separate path, rows bounded, latency target met once
the cost is learnt, a prompt always advances, a filled forward with nothing decoding), `serve::budget` unit tests,
`qwen35/tests/gguf_mixed_step.rs` (card: scheduler tokens agree for the first 12 tokens of every request and one forward's
decode logits and prompt rows are within 5e-2 of the two passes; in-flight prefix sharing is kept). A mixed forward's decode
rows run as rows of a wide GEMM, not the decode GEMV, so a late near-tie can flip against the separate path (one token of 24
did in the tiny fixture); `questions_arriving_together_prefill_their_document_once` therefore pins the separate path.
