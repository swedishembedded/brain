<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 72. W8A8 error compounds along the SEQUENCE through a recurrent state, so a depth-only quantization gate is blind to what serving actually does

`crates/qwen35`'s `gguf_i8_vs_fp32_real.rs` builds 8 real layers twice from
the same GGUF and reports a worst cosine of 0.9862 over 8 tokens. Its own
doc argued "if eight real layers already diverge, sixty-four cannot
converge" - true, but the contrapositive was then used as a rule-out, and it
does not hold. Measured against a reference forward at 32 real layers (half
the model), the SAME tier on the SAME tokens gives cosine 0.9888 at position
0, 0.9098 at position 1, 0.7988 at position 2. At 8 and 16 layers those same
positions are all 0.988-0.999, so the depth-8 gate cannot see it at all.

The mechanism is the hybrid decoder's own serving path: the resident decodes
one token at a time through a PERSISTENT Gated-DeltaNet recurrent state, so
every step's activation-quantization error is written into the state and
read back by every later step. A path that re-runs the whole sequence
through chunked prefill each step (this crate's `stream::generate`) never
carries error in a state and does not show this - which is exactly why a
`" Paris."` obtained through that path was mistakenly treated as proof that
the same int8 tier works at 64 layers.

**Rules that follow.** Quantization gates for a recurrent or stateful model
must sweep POSITION as well as depth, and must drive the same tape serving
drives (decode-with-state, not re-prefill). Report the worst cell of the
`depth x position` grid, not the worst of one row. And when a truncated-depth
measurement is used to rule something out, say what it measured: "0.986 at 8
layers" is not "0.986 at 64 layers", and for a stateful model it is not even
"0.986 at 8 layers, token 20".
