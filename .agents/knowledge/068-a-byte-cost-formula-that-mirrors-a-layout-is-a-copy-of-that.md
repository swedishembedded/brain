<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 68. A byte-cost formula that mirrors a layout is a copy of that layout, and it went stale silently

`Qwen35Config::layer_i8_bytes` - the per-layer INT8 byte cost that
`model::shard`'s placement planner and `crates/perf`'s `weights` scenario both
build on - charged `n*4` for a leaf's quantization scale (one f32 per output
row, the WHOLE-CHANNEL convention) long after `model::ops::Weight::I8`
itself became group-wise (`GROUP=32`, one f32 per 32-element block along `k`,
`n*k.div_ceil(32)*4` bytes). Nothing broke: the formula still returned a
plausible-looking number, `Weight::upload` still worked, every existing test
still passed - a hand-transcribed cost model has no way to notice that the
thing it transcribes changed shape out from under it. The gap was 12.5% per
layer (the packed-word term, `n*k`, dominates at real `k`, so a scale term
too small by `k/32/4` is easy to miss until it is added up: 3.4 GB across the
real 64-layer model), and it fed directly into a placement DECISION - a
multi-GPU shard plan built on an undercount can genuinely not fit the card it
was told would hold it.

The fix is not "update the constant" - it is "stop transcribing the layout at
all". `layer_i8_bytes` now runs the real `model::int8::quantize_weight` over
each leaf's real shape and sums what it actually returns, gated by a test
(`layer_i8_bytes_equals_what_weight_upload_really_places_on_the_card`) that
would go red the next time `Weight::I8`'s layout changes, instead of staying
green while quietly disagreeing with it.

The general shape: any formula whose whole job is to predict a real
function's output size, kept as a SEPARATE hand-written expression instead of
calling that function (or a cheap stand-in for it) and measuring, is a copy
with no way to notice the original moved. If the real cost is affordable to
compute directly, compute it; if it must stay a formula (too expensive to
call per-plan), gate the formula against one real measurement, not against
its own previously-hand-verified numbers.
