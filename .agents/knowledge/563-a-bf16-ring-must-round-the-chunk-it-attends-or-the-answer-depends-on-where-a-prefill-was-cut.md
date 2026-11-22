<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 563. A bf16 ring must round the chunk it attends, or the answer depends on where a prefill was cut

**Lesson:** a cache tier narrower than the activations has to be what EVERY read sees, the reading chunk's own rows
included. `gqa_mixer::ring_chunk_attend` gathered the earlier positions out of the ring (rounded to bf16 when the ring
is bf16) but copied the chunk's own keys and values in at f32, and only rounded them when it wrote them back. A key
was therefore read at f32 by the queries of its own chunk and at bf16 by every later one, so the logits of a position
depended on whether a chunk boundary fell before it: a prompt prefilled whole, in slices, or interleaved with another
request by the scheduler decoded differently. The paged path never had this (it appends, then reads the planes). The
chunk's rows now go through the same bf16 pack and unpack before they are attended.

## Evidence (2026-11-22)

- `inkling2` `kv_precision::the_bf16_tier_passes_the_serving_conformance_suite` failed on the tiny checkpoint with
  bf16 K/V: "a batched embedded request must decode exactly as it does alone" (`[27, 0, 33, 12, 18, 58]` against
  `[27, 0, 47, 0, 26, 32]`); the f32 tier passed. With the rounding it passes, and the measured bf16-vs-f32 spread on
  the tiny checkpoint fell from mean KL 7.2e-4 (top-1 92%) to 1.2e-4 (top-1 95%).
- The f32 ring is unchanged (a plain copy, as before); EdgeRock's ring tests and `model` `flash_ext` pass unchanged.
