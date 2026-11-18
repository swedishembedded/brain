<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 447. `--exact-decode` makes 27B-FP8 decode repeatable alone but not independent of its co-batch, and costs three times the throughput

**Lesson:** The batch-invariant mode that exists (`--exact-decode`, #358) was built and gated for the int8 GGUF path. On the
FP8 checkpoint it removes the run-to-run variance of a single request (speculation off) and leaves the co-batch variance,
at a third of the throughput; Qwen3 serving does not take the flag. True on-policy training and exams on these models need
the invariance built for their own kernels before they can lean on it.

## Measured

One identical request at temperature 1.0 with a fixed seed (so any difference is arithmetic, not sampling), 128-200 tokens,
sampler log-probabilities and ids compared; then the same request beside 7 different concurrent requests; then 8 identical
requests at once. GH200, one card, `brain serve` default settings except as named. Divergence = first token that differs from
the request alone; delta = largest |log-probability difference| over the tokens before it.

| model and mode | 5 identical requests alone: distinct outputs | beside 7 others: diverges at token, largest delta | 8 identical at once: distinct | 8-request aggregate |
|---|---|---|---|---|
| Qwen3.8-27B-FP8, default (MTP speculation on at <= 2 running) | 5 | 9-11, 0.16 | 2 | 558-608 tok/s |
| Qwen3.8-27B-FP8, `--exact-decode` | 1 | 25-126, 0.017-0.042 | 3-4 | 191-197 tok/s |
| Qwen3-4B, int8 weights and KV (default) | 1 | 29, 0.070 | 2 | 672-812 tok/s |
| Qwen3-4B, bf16 weights, int8 KV | 2 | 76, 0.177 | 1 | 538 tok/s |
| Qwen3-4B, bf16 weights, f32 KV | 2 | 170, 0.044 | 1 | 644 tok/s |

- Qwen3 refuses the flag by name (`qwen3: --exact-decode is not supported`), so no Qwen3 row has an exact mode.
- On Qwen3-4B at bf16 the first of five identical requests differs from the other four (token 76 and 190): the warm
  prefix cache serves a different prefill arithmetic than the cold prompt did, which is a second source of variance beside
  the co-batch.
- On the 27B FP8 the cost of exact decode is 0.32-0.35 of the default's 8-request throughput, in line with the grouping cost
  #358 reasoned (a pass over the weights per group), and the property it was built for is not obtained: the co-batch still
  changes the tokens, so the FP8 projections are not in the bit-identical regime #358 pins for int8.

## What this changes

Exams that need paired same-load comparison should keep the same-session base replicate (the bench rule) and not rely on the
flag. A trainer that wants zero serve/train mismatch cannot get it from decode-mode choice today; the importance correction
(`--importance truncate:CAP`) and the numerics gate (#446) are the instruments that exist. Making the FP8 projections and the
Qwen3 decode batch-invariant is a kernel task (`kernels.md`: selection of a fixed-reduction candidate by capability, not per-model
code) and is open in `.agents/roadmap/serving-runtime.md`.

## The FP8 resident's fixture on the CPU (2026-11-18)

`crates/qwen35/tests/hf_fp8_score.rs` (`-- --device cpu --backend cpu --ignored measure`): four seeded requests at
temperature 1.0, each alone and then beside three others, default and `--exact-decode`, on the CPU backend (21-64 tokens
each). No request diverged in either mode and every sampler log-probability before the end was bit-identical: the CPU
backend's kernels take no row-count-dependent choice on this fixture, so the co-batch dependence measured above belongs
to the card's kernel selection (row-class tuned FP8 candidates), and a CPU run cannot reproduce or rule it out. On a GH200
(CUDA, the same fixture) no request diverged in either mode either, but the log-probabilities of the same tokens
differed by up to 1.5-2.5e-2 (default) and 0.8-1.3e-2 (exact) beside the co-batch: on a card the fixture's rows are
not bit-identical across batch shapes, and the random fixture's decisive logits keep the tokens.

## Qwen3.8-27B-FP8 again (2026-11-18, brain 98e3a8024, one GH200, `--max-batch 16`)

The table's protocol through `brain serve` on chat completions (160 tokens, `ignore_eos`, seed 4242, thinking off; the
first alone request is cold, the rest hit its cached prompt):

| mode | 5 alone: distinct, diverge at | beside 7 others: token, largest delta | 8 identical at once: diverge at, distinct | 8-request aggregate |
|---|---|---|---|---|
| default (MTP speculation at <= 2 running) | 5; tokens 1-70 | 1, 0 | 82 (delta 0.087) and 1, 2 | 591 tok/s |
| `--exact-decode` | 1; never | 82, 0.023 | 82-142 (delta 0.023-0.051), 2 | 201 tok/s |

Unchanged from the 2026-11-13 table: exact decode makes a request alone repeatable and leaves the co-batch dependence, at
0.34 of the throughput.
