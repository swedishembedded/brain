<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 65. A weight scale that spans the whole reduction axis is one outlier away from destroying the row - group it at the block size of the format you interoperate with

`model::int8::quantize_weight` and `model::int4::quantize_weight_q4` computed
ONE scale per output row, over a `k` that reaches 17408 in the models this tier
serves. `AGENTS.md` had already recorded the measurement that condemns that
shape - whole-channel INT8 on the Qwen3 KV-cache decode graph at **cosine
0.994 / max_abs 11.2** against fp32's **1.000000 / 0.0002** - and the NPU-side
quantizer (`npu::topo::linear_quant`, `QUANT_GROUP = 32`) had been fixed for
it. The GPU-serving path had not; the invariant's own text ended "if/when it is
audited for the same granularity", which is what a known defect looks like
while it is still shipping.

Measured on the fixture the fix added (`int8::tests::group_wise_confines_a_row_
outlier_that_whole_channel_spreads_over_the_row` - one 100.0 outlier in a row
of `|w| <= 0.31`, `k = 512`): outside the outlier's own 32-element block, worst
absolute error is **0.001189** group-wise against **0.312000** whole-channel
(**262x**), and the count of elements quantized to worse than 0.1 is **39**
against **687** of 1022. The group-wise figure is bounded by the BLOCK, not by
`k` - that boundedness is the property, and it is what the test asserts rather
than the ratio alone.

Four things the fix turned out to depend on, none obvious from the quantizer:

* **32 is not a tuning parameter.** It is `Q8_0`/`Q4_0`'s block size, so one
  brain group IS one Q8_0 block and the requantization becomes the IDENTITY:
  Q8_0 stores `d = max|x|/127` with `max|q| = 127`, so the group's absmax is
  `127*d`, brain's scale comes out as exactly `d`, and every `q` re-quantizes
  to itself. Pick 16 or 64 and that door closes.
* **The GEMM's k-chunk decides what the group costs.** `matmul_i8_dyn`'s `BKG`
  is 8 packed words = 32 int8 = exactly one group, so the integer accumulators
  still sum a whole chunk exactly and fold once per chunk (64 FMAs per 512
  DP4A, at the price of doubling the accumulator register block). The two GEMV
  kernels stride 64 WORDS, which spans eight groups, so they have no same-group
  run to accumulate over and convert per word instead. Same layout, two
  different right answers - written into each kernel's header, because the
  wrong one costs 3x the inner loop or 2x the registers.
* **The stale scale shape is `[n]`, and it is a valid length.** An old
  checkpoint's whole-channel scale would be read as the first row's worth of
  group scales and produce plausible garbage. `model::int8::check_scale_len`
  recognises exactly that length and names the format change; the on-disk
  convention is versioned (`checkpoint::weightio::PACKED_INT8_LAYOUT`). A
  format change whose old form still parses needs a check that fires on the
  OLD shape by name, not merely a new writer.
* **It is not free, and the storage gate had to be re-derived, not widened.**
  One f32 per 32 weights is 0.125 B/param on top of the packed byte, so the
  int8 storage ceiling is `4 / 1.125 = 3.5556x`, not 4x. The real 22B LTX-2.5
  AV DiT measures **3.3640x** where it used to measure ~3.6x
  (`ltxv/tests/device_bytes_real.rs`). The right response to that gate going
  red was to compute the new ceiling from the layout and assert against it,
  not to loosen the band until the number fit.

The general shape: an invariant documented as "applies equally to X, if/when X
is audited" is not an invariant, it is a TODO with a citation. Either the audit
lands in the same change as the rule, or the rule carries a named exception
with the measurement it is exempting itself from.
