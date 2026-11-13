<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 442. A served token log-probability is the sampler's truncated figure, not the policy's, so training needs a scored trace

**Lesson:** The `logprobs` an endpoint returns say how likely the sampler was to draw each token, not how likely the model
is to say it. A policy-gradient step that takes them as the behaviour policy's `old` log-probability is comparing
two different distributions at every sampled token, and every 27B RL job so far dodged the question with
`--behaviour start`. Behaviour log-probabilities for training are taken from the model's full distribution, by the
serving model, in the same reply.

## What happened

`logprobs: true` on `/v1/chat/completions` returns, per generated token, the log-probability of `sample_from_topk_impl`
(`crates/model/src/serve.rs`): the decoder's top-K candidate list (K = 64 for qwen3, 32 for qwen35: the on-device top-k
extraction width, a ceiling no request can raise), then temperature, top-k, top-p and min-p, renormalised over what
survived. Two consequences were never written down:

- Greedy decoding (`temperature: 0`) reports `0.0` for every token, whatever the model thought.
- A request with `top_k` 40 and `top_p` 0.95 reports the probability inside a nucleus: always at or above the model's
  own log-probability for that token, and by an amount that depends on how peaked each step was.

The trainer (`ExternalPolicyGradient`) wants `old` = the log-probability of `softmax(logits / T)` over the whole
vocabulary, the distribution it trains. With the sampler's figure the step-0 ratio `exp(new - old)` is below one at
every token the truncation renormalised, by a token-dependent amount, and the clip band is spent before the first
update. That is why every served 27B job ran with `--behaviour start` (the start-of-run policy's log-probability as a
proxy), which is exact only when the rollouts were sampled from the start adapter at T = 1.0 untruncated.

## Fix

`policy_trace: true` (`docs/reference/api/http.md`) returns prompt ids, generated ids (the stop token included) and
`generation_log_probs` under the full distribution at the request temperature, scored by the serving decoder under the
request's adapter (`PagedDecoder::score_continuation`: qwen3 and qwen35 engines). It is refused where the request
reshapes the distribution (penalties, DRY, no-repeat, thinking budget). Gated: the log-probabilities equal the
trainer's teacher-forced ones on a tiny qwen35 on the CPU (bit for bit there) and on a tiny qwen3 engine within 1e-3
(`crates/catalog/tests/it/scheduled_batch.rs`, `crates/qwen3/tests/it/engine_score_continuation.rs`).

## Not covered

The trace is scored by a prefill pass, not by the decode kernels that drew the token, so it hides any decode-versus-
prefill numeric gap in the serving engine. The numerics gate (`brain/numerics`) measures the serve-versus-train gap
of exactly these figures on real weights; the decode path's own figure remains the sampler's `logprobs`.
