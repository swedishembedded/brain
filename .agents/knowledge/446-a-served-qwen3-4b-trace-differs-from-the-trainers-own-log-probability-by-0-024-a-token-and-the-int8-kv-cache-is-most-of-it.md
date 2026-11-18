<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 446. A served Qwen3-4B trace differs from the trainer's own log-probability by 0.024 a token, and the int8 KV cache is most of it

**Lesson:** Before an RL arm, measure the serve/train gap on the rollouts the engine really produced, with the numerics
gate; do not assume bf16 weights make it small. On Qwen3-4B the engine's default int8 KV cache is the larger part of the
gap, and which tier the trainer uses (f32 or bf16 base) is not.

## Measured

`brain finetune Qwen/Qwen3-4B --lora 16 --policy-gradient --numerics-gate --steps N` (no update is taken) over rollouts
collected through `brain serve` with `policy_trace: true` (knowledge #442): 20 prompts x 4 samples, temperature 1.0, no
top-k or top-p, 256 tokens at most, thinking off, 51-53 finished records (about 5,000 tokens) per row, one GH200.
Gap = trainer log-probability minus recorded one, per sampled token; ratio = exp(gap).

| served as | trainer base | mean abs gap | largest gap | mismatch KL (k3) | ratio outside 0.8-1.2 | ESS token / sequence | grad norm max / median |
|---|---|---|---|---|---|---|---|
| int8 weights, int8 KV (default) | bf16 | 0.0229 | 1.79 | 0.0036 | 3.07% | 0.992 / 0.52 | 49.5 / 3.07 |
| bf16 weights, int8 KV | bf16 | 0.0245 | 1.66 | 0.0041 | 3.33% | 0.990 / 0.58 | 31.1 / 3.63 |
| bf16 weights, int8 KV | f32 | 0.0244 | 1.34 | 0.0041 | 3.30% | 0.990 / - | - |
| bf16 weights, f32 KV | f32 | 0.0099 | 28.96 (one token) | 0.0060 (that token) | 0.27% | 0.999 / 0.96 | 88.0 / 3.89 |

- Quantising the weights to int8 changes nothing measurable next to bf16 (0.0229 against 0.0245); an f32 trainer base
  changes nothing next to a bf16 one (0.0244 against 0.0245). The int8 KV cache is what separates the rows: the mean gap
  falls 2.5 times, the share of tokens outside a 0.2 clip band 12 times and the sequence ESS from 0.58 to 0.96 when it is f32.
- At the default serving the sequence ESS is 0.5: half the sequence-level importance weight sits on a few sequences, so a
  `clip_eps` of 0.2 spends its band on the gap before any update. A trainer should take the recorded log-probabilities as
  `old` only with the importance correction (`--importance truncate:2`) or a KV tier the gap allows.
- The f32-KV row's single outlier is a token the engine scored at log-probability 0.0 (probability 1) and the trainer at
  -28.96, sequence 46 offset 61 of the run's draw order. Not explained: one candidate is that the record's text did not
  re-tokenize to the ids the engine scored (the gate sees records, not ids), which a token-in trace would rule out; it was
  not checked. The gate's `--gate-max-gap` bound exists for exactly this token.
- The gradient norm of a batch at step 0 is 3-4 at the median and 30-90 at the largest in every row: a 9-23 times spike on
  a 53-batch sample, the shape the step guard (`--max-step-grad-norm`) was written for (#440), seen here with no update taken.

## Method notes

- The recorded log-probabilities are the serving model's own teacher-forced pass over prompt and generation, not the
  decode kernels' (`PagedDecoder::score_continuation`). The decode path's own figure is the sampler's `logprobs` and was not
  compared here.
- The template's text after the end-of-turn marker (Qwen's newline after `<|im_end|>`) is context in a rollout record: the
  sampler never chose it, and supervising it needs a log-probability nobody sampled. Before that change the gate read a
  recorded 0.0 there against the trainer's own value and reported a largest gap of 15.5 and a mean of 0.060.
- Qwen3.8-27B-FP8 was not gated here. Its served engine (the released-checkpoint resident, `Qwen35GgufEngine`) scores
  continuations since 63de9c881, an hour after this measurement, so `policy_trace` is served for it; it did not declare
  the `score` action until 2026-11-18. Its rows are below.

Evidence: the gate reports and the rollout records (one JSONL per run) are in the private Hugging Face dataset
`swedishembedded/brain-numerics-rollouts`, with a card naming each file (2026-11-13).

## The FP8 resident's fixture on the CPU (2026-11-18)

`crates/qwen35/tests/hf_fp8_score.rs` (`-- --device cpu --backend cpu --ignored measure`): the engine that serves an FP8
release, on a random 4-layer FP8 directory (d_model 128, vocabulary 64), 8 requests at temperature 1.0 untruncated, 181
generated tokens per row. "Decode" is the sampler's log-probability at the step that drew the token; "trace" is the
engine's `policy_trace` scoring of the same ids at the same K/V tier; the f32-K/V trace stands for a trainer that rounds
no K/V.

| K/V | decode vs own trace: mean / largest | trace vs f32-K/V trace: mean / largest |
|---|---|---|
| f32 | 5.0e-7 / 2.4e-6 | 0 / 0 |
| bf16 | 7.3e-7 / 8.7e-6 | 2.4e-4 / 1.6e-3 |
| int8 | 5.6e-7 / 3.7e-6 | 7.8e-4 / 3.3e-3 |
| fp8 | 4.8e-7 / 2.4e-6 | 3.7e-3 / 2.7e-2 |

- The trace reads the K/V at the tier the decode wrote, so it reproduces the decode-time distribution to 1e-5 at every
  tier: `old` from a trace is the behaviour policy, and the whole serve/train gap is the trainer's own K/V (none)
  against the served tier. The parity is gated (`a_trace_reproduces_the_decode_time_log_probabilities_of_an_untruncated_draw`,
  tolerance 2e-3, at temperatures 1.0 and 0.7).
- The tier ordering of the 4B table holds on this fixture (f32 < bf16 < int8 < fp8), at a much smaller size: a 128-wide
  random model has little to lose. The numbers are the method's floor on the CPU, not an estimate for the 27B.

## Qwen3.8-27B-FP8 on a GH200 (2026-11-18, brain 98e3a8024)

Rollouts through `brain serve --device gpu0 --backend cuda --max-batch 16` (the server's defaults otherwise: FP8 weights
as stored, bf16 GDN state, speculation `auto`), `policy_trace: true`, 20 prompts x 4 samples, temperature 1.0 with no
top-k or top-p, 512 tokens at most, thinking off, 8 requests in flight. The gate reads the finished rollouts of groups
with at least two (a truncated answer has no end-of-turn token for the template to render, so it is not a record):
`brain finetune Qwen/Qwen3.8-27B-FP8 --lora 16 --policy-gradient --numerics-gate --steps 45 --no-thinking`, trainer base
bf16 (chosen for the card).

| served as | trainer | records / tokens | mean abs gap | largest gap | mismatch KL | ratio outside 0.8-1.2 | ESS token / sequence | grad norm max / median |
|---|---|---|---|---|---|---|---|---|
| int8 KV (default) | bf16 base | 45 / 8412 | 0.0194 | 0.52 | 0.0011 | 1.02% | 0.998 / 0.54 | 1.77 / 0.21 |
| int8 KV (default) | bf16, `--kv-roundtrip int8` | 45 / 8412 | 0.0194 | 0.51 | 0.0011 | 1.05% | 0.998 / 0.41 | 1.82 / 0.21 |
| f32 KV | bf16 base | 44 / 8050 | 0.0184 | 0.42 | 0.0009 | 0.94% | 0.998 / 0.61 | 1.90 / 0.19 |

- On the 27B the K/V tier is not most of the gap: f32 K/V moves the mean gap from 0.0194 to 0.0184 (4B: 0.0245 to 0.0099),
  and reading the trainer's K/V through the int8 encoding changes nothing. The gap is the size of the serving engine's own
  decode-against-prefill gap: the sampler's log-probability of the same tokens differs from the trace by 0.018 mean
  (p50 0.0008, p99 0.19-0.20, largest 1.4-1.6 over 26,000 tokens) at either tier. A trace (a prefill) is therefore not the
  decode-time distribution to better than about 0.02 a token on a card; the fixture shows the same on a GH200 (mean 1.3e-2)
  whatever the activation stream or the GDN prefill form, and none of it on the CPU. Which product rounds differently
  (the FP8 prefill GEMM against the decode GEMV, the bf16 recurrent state between decode steps) is not isolated yet.
- The sequence ESS is 0.4-0.6 at every tier, as on the 4B: with `clip_eps` 0.2 a step-0 batch of whole sequences is
  dominated by a few, so a 27B run that takes the recorded log-probabilities as `old` should use `--importance truncate:2`.
- The step-0 gradient norms are 0.19-0.21 at the median and 1.8-1.9 at the largest (a 9 times spike over 45 batches).
- Records and gate reports: `~/project/shared/tmp/learn-score13` on the node of the run (not copied to the evidence
  dataset).

