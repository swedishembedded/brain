<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 321. A drawn seed above i64::MAX was dropped from the reply and refused on the way back

**Lesson:** A value that is echoed so it can be sent back must use the full range of its source
type at both ends of the round trip, or about half the values silently fail to reproduce.

## What happened

Question: the server draws a fresh seed for an unseeded request and echoes it in the reply's `brain`
object so the decode can be reproduced by sending it back. Does that hold for every seed it can
draw? It did not. **VERIFIED-BY-TEST**
(`crates/apiserve/src/sampling.rs`, `a_seed_the_model_drew_is_accepted_back_over_the_whole_u64_range`;
`crates/catalog/tests/it/scheduled_batch.rs`).

A trajectory recorder that trusted the echo therefore recorded the seed for half of the requests
and nothing for the other half, with no error anywhere.

## Why

`data::rng::random_seed` returns the whole `u64` range, and the generation actions carry the seed
as a `u64` (`Invocation::seed` refuses a negative one). The HTTP layer read the echoed value with
`as_i64` and parsed an incoming `seed` with `as_i64`, so about half of all drawn seeds were:

- absent from the reply (the `brain` object omitted `seed` when `as_i64` failed), so an unseeded
  decode could not be reproduced at all; and
- refused with 400 when a caller sent the seed back, because it was above `i64::MAX`.

## What to do

The reply echo and the request parser both use `u64` (`bridge::ChatOutcome::seed`,
`sampling::apply`); a negative seed is still refused.

### What the sampled-token echo means (decided in the same change)

`logprobs: true` and `return_token_ids: true` return, per generated token, its id and its
log-probability under the distribution it was decoded from. That is the temperature-scaled, top-k
and top-p truncated softmax a sampled token was drawn from, and `0.0` for greedy decoding where the
choice is deterministic.

That is the quantity an on-policy trainer needs for the behaviour policy. It is not the untruncated
model distribution, and a consumer that needs the latter must score the tokens with a forward pass.
