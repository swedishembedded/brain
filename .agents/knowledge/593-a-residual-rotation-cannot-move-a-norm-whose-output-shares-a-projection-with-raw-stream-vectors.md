<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 593. A residual rotation cannot move a norm whose output shares a projection with raw stream vectors

**Lesson:** rotating a residual stream (`modelopt::surgery::rotate_stream`, the SliceGPT construction) is exact only
because every norm's gain is folded into the linears that read its output and the gain-free norm commutes with an
orthogonal rotation. In Qwen3-TTS the Talker's final norm is not foldable: its output (the hidden state handed to the
code predictor) and the raw codec embeddings both enter `small_to_mtp_projection`, one matrix for both. With a general
rotation `Q`, `diag(g) Q != Q diag(g')` for any per-channel `g'`, so one of the two inputs of that projection is off
even at full width. Measured on the 1.917 B checkpoint (64 validation records, exact fp32): a full-width PCA rotation
of the Talker stream left the Talker's own logits exact (KL 3e-11) and the code predictor at KL 1.476 nats per
position, loss 2.596 -> 3.041. The gain spans -1.05 .. 13.7 and that run already fitted `g'` by least squares on the calibration states.

A rotation inside groups of channels whose shared gain is equal does commute with it. The released gains are BF16, so
the 2048 channels hold only 190 distinct values (56 of them on 8 or more channels, 1734 channels in all): PCA within
equal-gain groups (`basis_within_groups`, `gain_groups`, the plan's default for a PCA basis on a stream with a shared
norm) keeps the rotation exact at full width (measured: KL 1.5e-11 Talker, 4.6e-11 code predictor) and still mixes
most channels. `gain_tolerance` widens the groups and trades exactness of the code predictor's input for more
mixing: 0.25 costs KL 0.001 at full width and, cut to 1536, beats the exact groups (KL 0.84 against 1.78).

Where: `crates/modelopt/src/surgery/stream.rs`, `crates/modelopt/src/surgery/plan.rs`, the declaration in
`crates/qwen3tts/src/surgery.rs` (`talker.model.norm.weight` is the stream's shared norm).
