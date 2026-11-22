<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 569. A greedy token behind a routed near tie is not a precision metric

**Lesson:** through a top-k router a rounding is not small: an operand rounded one way or the other turns a near tie
between two experts into a different expert, and every later row of that token then differs by a whole expert's
output, not by an ulp. A gate that asserts a reduced-precision tier keeps the reference's greedy token wherever the
reference leads by more than a margin therefore measures where the route flips landed, and any equally accurate
change of an f32 reduction order anywhere upstream can turn it red or green. Hold such a tier by statistics that a
systematic defect moves and a single flip does not (the KL on full logit rows against the reference's own reduced-
precision run; flips counted by distinct CONTEXT, since prompts that share a chat-template prefix share its rows in a
causal model and one flip there shows in every prompt) and pin the per-position greedy check to the binary32 arm, where
it is stable.

## Evidence (2026-11-22)

- Inkling 2 Mini on a GH200 (CUDA), the default arithmetic against the f32 golden, prompt 1 (27 tokens), position 5
  (f32 lead 0.65 logits). Positions 0-5 are the same six chat-template tokens in all three golden prompts; whether the
  flip shows depends also on the prompt's length (22 or 27 rows pick different library product algorithms), so one
  arm showed it in prompts 1 and 2 and not in prompt 0. Four arms, each equally accurate at binary32 (`real_parity`'s worst visit distance 1.1e-4
  to 3.2e-4 of the bf16 spread in all of them):
  - reference depth attention, TF32 message gates: greedy kept;
  - cooperative depth attention (`depth_attn_fwd_rows`), TF32 message gates: greedy changed (12.60 vs 12.48), after a
    route flip at visit 27, row 5 that the other arm does not take;
  - reference depth attention, binary32 message gates: greedy changed (12.57 vs 12.48);
  - cooperative depth attention, binary32 message gates: greedy kept.
- Every arm already differs from the f32 run's routing at row 5 from visit 10 on (rows `[5, 10, 12]`), as the bf16
  reference run differs on up to 4% of its choices.
- The binary32-router defect (#557) changed position 5 in all three prompts - that same one context - and raised the
  worst KL from 4.4e-2 to 1.1e-1; the reworked gate (`inkling2` `tests/it/real_parity.rs`, `MAX_ROUTED_FLIPS` counts
  distinct contexts) still fails on it through the KL.
