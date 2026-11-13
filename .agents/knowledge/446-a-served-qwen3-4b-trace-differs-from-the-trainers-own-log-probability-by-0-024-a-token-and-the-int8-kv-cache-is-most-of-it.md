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
- Qwen3.8-27B-FP8 was not gated: its served engine (the released-checkpoint resident) does not implement
  `score_continuation`, so `policy_trace` is refused for it by name.

Evidence: the gate reports and the rollout records (one JSONL per run) are in the private Hugging Face dataset
`swedishembedded/brain-numerics-rollouts`, with a card naming each file (2026-11-13).
