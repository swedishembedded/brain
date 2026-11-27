<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 619. An agent episode sent as one record per call weighs by its calls and re-forwards its context every call

**Lesson:** the unit a loss normalises over is the record, not the episode. When an episode arrives as one record per
model call, every aggregation weighs it by its calls (or their tokens) and every call re-encodes the context the
earlier calls built. Merge an append-only session into one sample with a mask over the model's own tokens, prove it
equal to the per-call reference, and only then choose the normalisation on purpose.

## What happened

An expert review of online agent RL asked whether long (often failed) episodes dominate the policy gradient. Each of
an episode's records carries the episode's advantage; a stream step divides the sum over its policy tokens by their
count (`token-sum`), or each record's sum by a constant averaged over records (`constant-length`); the offline default
divides each record by its own count and averages over records (`seq-mean`). Pinned on a toy decoder: with 60 calls
against 2 of the same length, the long episode's share of the step's gradient is exactly 30 times the short one's,
under every aggregation. A 12-call session re-forwarded 6.9 times its own length across its call records.

## Why

- A record per call is one row: aggregations count rows or the tokens in them, so an episode's weight is its call
  count (or its policy-token count), never one.
- Each call's record holds the request as sent, so call `k` encodes the context and all `k` earlier turns again.
- A per-sequence mean could not be chosen for a stream at all: it needed one sequence per micro-batch.

## What to do

- Send an append-only session as one packed rollout record (every sampled answer `train` with `token_ids`,
  `prompt_token_ids`, `behavior_logprobs`; `pack_answers: true`). A record that names `prompt_token_ids` is refused by
  rule `session_not_append_only` when a later request is not the session so far (an earlier answer re-rendered, its
  reasoning dropped, a compaction): never scored in a context the sampler did not see. The producer falls back to
  per-call records.
- Under `token-sum` a merged sample trains exactly its per-call records' gradient; under `seq-mean` (now trained by
  the stream through a per-position scale) every episode of a step weighs alike. Pick by intent and say which.
- Prove a merge against the unmerged reference on every architecture: the A49 conformance check fails if a forward
  is not causal or padding leaks.

## Evidence

`crates/rl/tests/it/session_samples.rs` (the 30:1 pin; merged vs per-call gradient, relative gap 3.4e-7; seq-mean of
two merged sessions is the mean of each alone; 4 sessions x 12 calls: 11328 sequence tokens and 0.43 s forward+backward
per call against 1648 and 0.05 s merged on the CPU, same 576 policy tokens), `crates/rl/src/conformance_async.rs`
(`merged_sessions`, every registered architecture), `crates/data/tests/it/chat_session_records.rs` (layout and refusal).
