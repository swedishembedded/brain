<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 491. The looped decoder served a batch one sequence at a time through the portable attention triad, so sixty-four agents shared 3,700 tokens a second

**Lesson:** "Continuous batching" in the scheduler is only as batched as the engine's step. Check that a decode
step of `b` rows is one graph of `b`-row products, that its attention takes the native launch for the model's head
width, and that a sequence's addressable context is its own, not the pool's; then profile with
`--cuda-graph-trace=node`, or replayed graphs hide the kernels that cost the step.

## What happened

`ouro::serve::Engine::forward_batched_*` looped over the batch, one token graph, one submit and one 16 KB logits
readback per sequence; each sequence's attention addressed the whole pool (`cap = pool tokens`); the decode
attention was the scores/softmax/apply triad (the native batched kernel took head width 128 only, and never one
sequence); prefills allocated forty buffers each. The default from-scratch size (5.3M parameters, four heads of
64, four passes) on gpu3, agent workload (600-token prompts, 128 out): 231 tokens/s at one stream, 3,713 at 64.

## What changed

- One decode step carries every running sequence: each pass's products over all rows (through `Ops`), RoPE at
  each row's own position (`rope_paged`), each row's own block table, every pass's gates in one product, each row's
  exit row gathered from the per-pass slab and the head over the gathered rows. A step graph per batch size is
  recorded over one set of buffers sized for the largest batch.
- A sequence addresses the model's positions (or the pool, if smaller); the pool is shared. A server that knows
  its card plans the pool from the card's usable bytes and the batch from the pool (`OuroResident::
  with_card_budget`, up to 256 sequences); `brain perf --target ouro:<checkpoint>` measures that served sizing.
- `decode_attn_batched.cu` takes a 64-wide head over an fp32 pool, one sequence included (no split-key kernel
  serves that width), and `block::gqa_decode_batched_step` takes the native launch wherever the device has one.
- A prefill or a teacher-forced scoring pass (`score_continuation`, now served) runs in one prompt arena.
- Same card, same workload: 1 stream 231 -> 715 tokens/s, 8: 1,012 -> 3,351, 32: 2,494 -> 8,208, 64: 3,713 ->
  11,085; time to the first token at one stream p99 139 -> 4.3 ms (p50 3.7 ms for 600 prompt tokens, about 160k prefill
  tokens/s).
  The served tokens equal the model's own greedy forward and the serving conformance suite passes on the card.
