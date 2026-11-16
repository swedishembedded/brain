<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 488. A sampled decode drawn on the host held one core while the card idled, and a keyed Gumbel-max draw moves it to the device exactly

**Lesson:** A draw that needs every token's mass (a temperature without `top_k`) must not read the `[rows, vocab]`
logits back and walk them serially on the serving thread: at 64 streams that is the step's bound. The Gumbel-max rule
(`argmax(logit / temp + g)`, `g` keyed noise) draws from the exact softmax in one reduction per row, so it runs where
the logits are and returns `(token, logprob)` per row; keying the noise by the request's seed and position (never the
batch slot) makes the draw the same on the device, on the host and in any batch.

## What happened

Qwen3-4B at 64 concurrent temperature-1 requests (no `top_k`, `top_p` 1): one `brain serve` core at ~100 %, the card
busy 12-23 %. Every such request takes the whole-row path (`SampleParams::needs_full_row`), so each step read 64 x 151936
floats back and `sample_logits_row` built a `(token, logit)` vector per row and summed its exponentials in f64, one row
after another.

## Why

The candidate-list path (#443) is exact only for a `top_k` within the list, so the exact path for a plain temperature
was the host row sampler, and nothing parallelised it. The cost grows with rows x vocabulary on one core while the
device work per step barely changes.

## What to do

- `model::serve::draw_row` is the reference: survivors are the tokens of positive weight (or the min-p set), the noise
  is `token_gumbel(draw_key(seed, position, 0), token)`, the log-probability is the survivor softmax's.
- `model::readout::RowDrawer` runs it on the device: the WGSL pair `draw_part` + `draw_final` (strided slices per row,
  no barriers, runs on the CPU JIT too; one more pair for min-p) and the native `draw_rows_f32` (a block of 1024 per
  row, one launch) through the provider seam, gated against the host reference row for row (`readout_draw.rs`) and by
  `conformance::keyed_draws_are_the_rows_draws` for every engine.
- A decoder implements `PagedDecoder::forward_batched_drawn` by appending the drawer's steps after its head; the default
  reads the rows and draws on the CPU pool. Rows the keyed draw does not cover (a processor, a nucleus, a `top_k`
  inside the vocabulary) come back as rows and are sampled on the host in parallel (`backend_cpu::par`).

## Evidence

`brain perf run serve --target qwen:Qwen3-4B:kvf32 --max-batch 64 --output 300 --temperature 1` (one GH200, gpu3):
64 streams 688 -> 2484 tokens/s against 2548 greedy (GPU busy 23 -> 68 %), 16 streams 396 -> 1502 against 1515;
Qwen3.8-27B-FP8 at 64 streams 368 -> 1385 against 1482. The full table, the adapter runs and what stays open are in
`.agents/roadmap/serving-runtime.md` "Sampled decode on the device".

A recorded decode step must take the keyed head too: the 27B's sampled steps first ran eagerly (only greedy and top-k
heads were recorded) and stayed about 10 % behind greedy with the draw itself at 0.2 ms a step.

A nucleus answered from a fixed candidate list is not a device draw on high-entropy rows: at temperature 1 most rows'
nuclei were wider than 256 candidates, fell back to the host, and the 27B at 16 streams got slower (266 -> 246 tok/s).
The nucleus is a threshold over the row, found by reductions (bisection of an ordered-key interval on the mass at or
above it, ties at the threshold by column), so it needs no list at all: 268 -> 597 there.
