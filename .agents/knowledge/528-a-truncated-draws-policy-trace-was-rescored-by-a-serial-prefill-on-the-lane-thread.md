<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 528. A truncated draw's policy trace was rescored by a serial prefill on the lane thread

**Lesson:** A figure the decode step already holds the inputs for (the row's best logit and its mass, next to the drawn
token's logit) is read from the step, not recomputed afterwards. A second pass on the one thread that also drives the
batch makes every reply wait for every other reply's pass.

## What happened

Qwen3-4B bf16 with a LoRA adapter on one GH200, `--max-batch 64`, about 48 concurrent agent sequences, every request
with `policy_trace: true` and a cut sampler (top_k 20, top_p 0.95, temperature 1.0). The lane was host-bound: GPU
utilisation 28%, about 8 tok/s a sequence against 70 in a plain 64-wide run, a plain 400-token request 54 s and a traced
one 90 s. Stack samples of the lane thread sat in `Scheduler::score_continuation` (a host full-vocabulary softmax in its
closure, then a whole-prompt-plus-reply prefill per reply) and in `cuStreamSynchronize`.

Knowledge 520 had made an untruncated draw's trace the decode's own log-probabilities; a cut draw still reported only
the survivors' figure, so its trace was scored by a prefill, serially, at the end of every reply.

## What fixed it

The keyed draw (`model::serve::Drawn::Token`) now carries a third figure, the token's log-probability under the whole
row at the request temperature. The row drawer already read back each row's best `logit / temp` and mass; the drawn
token's logit is in the candidate list (or one single-element read for a nucleus or min-p row whose token is not), so
the figure is `logit / temp - best - ln(mass)` with no new device work and no extra launch. A row drawn on the host gets
`row_logprob_at`. The scheduler keeps the figures per sequence (`Running::whole`, handed to the report as
`completed_traces`, and `Scheduler::trace_logprobs_of` for a request a caller cancels), and a sequence any of whose
tokens came from a path that does not report one (greedy, speculated, a row sampled on the host) simply has no trace
and is scored as before. `llmserve` then answers with `log_prob_source: "decode"` and runs no scoring pass. The first
token of a keyed request reads its one logits row instead of the candidate list so that it carries the figure too.

Gated by `llmserve` `a_policy_trace_of_a_sampled_draw_is_the_whole_row_log_probability_of_the_decode_that_drew_it` (top-k,
top-p and min-p draws against the toy row's log-softmax, zero scoring passes run), and by the figure assertions in
`readout_draw` (the device draw against the host row, per cut) and the conformance suite's keyed-draw check.

A first GPU check found every top-k 20 reply still scored. A step reads whole rows (and so gets the figure) only when
some sequence needs them, and a top-k inside the candidate list does not, so a pure top-k batch took the candidate-list
path, which holds the best logits but not the row's mass. A mixed test batch had hidden it. Every keyed draw on a decoder
that reads rows back now takes the drawn path (`Scheduler::wants_full_rows`); the test runs each cut alone.

Not measured on a card: the GPU gain (the 28% utilisation and 90 s figures above are the reporter's).
