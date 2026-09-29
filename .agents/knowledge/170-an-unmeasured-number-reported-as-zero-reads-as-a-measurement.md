<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 170. An unmeasured number reported as zero reads as a measurement

Four results carried a `0.0` for something nobody measured:

- the promotion gate's `worst_block_delta` (and `worst_block`) were `0.0`
  (and `0`) when the gate had no anchor blocks at all - identical to "every
  block held";
- `ImproveOutcome.anchor_delta` was documented "always `0.0`", while a
  single improve cycle has no anchor suite and its gate's pooled check ran
  on the held-out mean instead;
- `ChatScore.token_accuracy` was `0.0` when no position was scored, beside
  a loss that was already NaN for the same case;
- `AtifVerifier` scored a task without a reward stamp as `0.0` - a failure
  the recorded trajectory never showed.

A downstream gate or report cannot tell any of these from a real zero, and
a zero is the most plausible number there is: "no regression", "no
accuracy", "failed". Each is now `None` (or, for the verifier, whose trait
cannot say "unmeasured", a debug-build refusal and a NaN that the sign test
counts as a tie, never a win).

**Rule:** a value that was not measured is `Option::None` (or NaN where the
type cannot change), never `0`. A doc comment that says "always `0.0`" is
describing an unmeasured field.
