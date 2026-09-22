<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 32. A profiler that drives the graph wrong measures a graph that does nothing

With device timing correct, `paged_decode_scores_batched` still reported
**7060 GB/s**. The timing was right and the kernel really was that fast - because
it was attending to nothing.

`qwen_bench serve` drove the served tape with `Input::Resident`. That is the
on-device decode-window mode: it deliberately performs **no host writes**,
because `decode_feed`/`decode_advance` are supposed to have produced the token
ids *and* the paged metadata on the device already. Driven from a profiler,
nothing had, so `seq_lens` stayed zero, every attention thread early-returned,
and **61% of the pass did no work**.

The damage was not confined to the attention rows. The measured "≈27–29× serve
prefill speedup" from registering the tiled GEMM was published from that harness
and is wrong; corrected, it is **11.8×** (2413.18 → 204.4 ms, 53 → 626 rows/s).
The *ratio* was stable across four runs precisely because both arms ran the same
no-op attention - a like-for-like comparison against a pass missing most of its
work. **Reproducibility is not validity**, and a stable ratio is exactly the
shape of evidence that makes this kind of error survive review.

Three defences, in order of how much they would have caught:

1. **Make the profiler drive the model the way production does.** `Resident`
   exists for a mode with a device-side producer; a profiler has none. This is
   the whole bug.
2. **Cross-check the pass against its own roofline.** The impossible-rate guard
   (#31) is what refused to publish 7060 GB/s and forced the question. Without
   it the number would have been printed as a percentage and believed.
3. **Sanity-check the shape of the answer.** A suspiciously cheap rows/s figure
   for a small model should prompt "against what ceiling?" - the weight-bandwidth
   budget the same tool already prints says a served step cannot be that cheap.
