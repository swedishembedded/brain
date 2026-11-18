<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 520. A policy trace scored by a prefill is not the decode that drew the tokens on a card, because each row count has its own rounding

**Lesson:** On a GPU, a teacher-forced pass over the tokens a decode drew does not reproduce the decode's
distribution. The kernel autotuner picks a candidate per row count, and the candidates round their operands
differently. The behaviour policy of a rollout is the decode step's own log-probability. Record that. Do not
recompute it.

## Measured

`crates/qwen35/tests/hf_fp8_score.rs` ran on one GH200 with `--device gpu0 --backend cuda`. The fixture is the FP8
resident's random 4-layer directory with f32 K/V, 8 requests at temperature 1.0, untruncated, 177-191 tokens. "Sampler"
is the decode-time log-probability of the drawn token. "Decode tape" is the instance's teacher-forced one-row decode
steps. "Trace" is the scoring prefill (`Qwen35GgufEngine::score_tokens`). Every figure is |difference| per token, mean
and then largest.

| pair | before | after |
|---|---|---|
| sampler vs trace | 1.29e-2 / 8.71e-2 | 5.93e-3 / 3.34e-2 |
| decode tape vs trace | 1.26e-2 / 7.06e-2 | 4.30e-3 / 3.28e-2 |
| sampler vs decode tape (the batch alone) | 3.62e-3 / 2.27e-2 | unchanged |
| sampler vs trace, every FP8 product forced to one candidate (`matmul_fp8_dq_stream_q4`) | - | 2.55e-3 / 1.30e-2 |
| CPU backend, sampler vs trace | 5e-7 | unchanged |

What does not move it:

- The activation stream between products (`auto` and `f32` give identical rows).
- The Gated DeltaNet prefill form (pinned `exact`: 1.37e-2 against 1.29e-2).
- The prefill attention (the portable fp32 kernel in place of the fp16 flash prefill: identical).
- Pinning the fp32 kernels (`model::ops::fp32_kernels`): identical.

What moves it:

1. **The FP8 GEMM of a prefill quantises its activations to E4M3.** The tensor-core candidates (`matmul_fp8_wgmma*`,
   `Input::FP8_NATIVE`) read E4M3 rows with a scale per 128. The decode's streaming GEMVs read bf16 rows. Scoring
   with the FP8 products restricted to bf16-reading candidates (`model::ops::fp8_bf16_activations`) halves the gap. On
   the real 27B tables the tuned choice is E4M3 for every class from 17 rows up, both prefill and decode, and for two
   classes at 9-16 rows. It is bf16 at 1-8 rows.
2. **The bf16-reading candidates do not share one rounding either.** `matmul_fp8_dq_stream_*` and `dq_wgmma` round
   `code * scale` to bf16 before the MMA. `matmul_fp8_gemv_f16*` keeps the code exact and applies the block scale after
   accumulation. The tuned table mixes both families within one row class (the 27B at 1 row: 3 classes `dq_stream_r4`,
   7 `gemv_f16*`). Forcing one candidate everywhere halves the gap again.
3. **The rest (2.6e-3 mean) is outside the FP8 products.** It is not isolated. The candidate is the dense bf16
   products: the one-row GEMV reads f32 activations, while 2-16 rows (split-K) and the library GEMM round them to bf16.

Because the decode's own arithmetic changes with its batch (the third row, and #447's co-batch dependence), no single
scoring pass can be the behaviour policy of every decode.

## Fix

- `llmserve` records the policy trace of a whole-distribution draw from the decode (a temperature, no top-k, top-p or
  min-p, no processor or thinking budget, on a decoder that draws over its whole row:
  `Scheduler::reports_whole_distribution_logprobs`). The trace's log-probabilities are the sampler's, stop token
  included, with `log_prob_source: "decode"`, and no scoring prefill runs. This is gated by `llmserve`
  `a_policy_trace_of_a_whole_distribution_draw_is_the_decode_time_log_probabilities`.
- Checked on Qwen3.8-27B-FP8 through `brain serve` on one GH200 (defaults, `max-batch` 64, 4 concurrent requests):
  `max |sampler - trace|` was 0.0 on all 4 replies (75-96 tokens each). Before this change the same protocol measured
  mean 0.018 and p99 0.2 (#446).
- A truncated or greedy draw has no whole-distribution figure from the sampler, so its trace is still scored
  (`log_prob_source: "score"`). That scoring pass reads the FP8 activations as bf16, the form of the 1-8 row decode.
  The card gate is tightened to the achieved bound: `a_trace_reproduces_the_decode_time_log_probabilities_of_an_untruncated_draw`,
  largest 0.05 and mean 0.012 (was 0.15 / 0.03).

## Not covered

The decode itself is not one rounding across batch sizes. Making it one is a selection change: one input contract per
FP8 candidate, and E4M3 against bf16 activations declared as different `Input`s that a row-count-invariant regime can
hold fixed. That work is open in `.agents/roadmap/serving-runtime.md`.
