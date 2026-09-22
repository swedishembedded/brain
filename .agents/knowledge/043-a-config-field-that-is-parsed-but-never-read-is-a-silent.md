<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 43. A config field that is parsed but never read is a silent architecture gap, and it can affect more than one caller at once

`crates/codec/src/config.rs`'s `CodecConfig::sliding_window` (real value 72)
has existed since the Code2Wav port landed - parsed straight from the real
`decoder_config.sliding_window` JSON key, correctly, every time. Nothing in
`crates/codec/src/model.rs::transformer` ever read it: the pre-transformer's
attention dispatched plain `model::block::gqa_fwd` (full causal, unbounded),
not a windowed variant. The real reference
(`Qwen3OmniMoeCode2WavAttention`/`MimiAttention`, both built on
`create_sliding_window_causal_mask(sliding_window=config.sliding_window)`)
applies the window on **every** forward call, one-shot included - not only a
chunked/streaming path - so this was a plain correctness bug, not a missing
optimization: `Codec::decode_omni`'s output has been wrong for any T > 72
since the day it shipped. It was invisible because the only Omni Code2Wav
test uses T=8, and the standalone-codec unit test uses T=24 - both comfortably
under the window, so both passed at cosine ~1.0 while silently exercising the
plain-causal fallback that happens to equal the windowed answer at small T.

**Two compounding lessons, not one:**

1. A config field with no reader is not evidence the field is unused - it is
   evidence nobody checked whether it's used *yet*. `grep`ping for a config
   field's read sites (not just its parse site) before trusting "this model's
   forward pass is complete" would have caught this immediately; nothing in
   this codebase's own review process does that grep automatically.
2. **The bug was shared by two callers who look independent.** `Codec::transformer`
   is dispatched by both `Codec::decode` (the standalone Qwen3-TTS codec) and
   `Codec::decode_omni` (Qwen3-Omni's Code2Wav) - one function, one config
   field, two model families. Finding it while working on Omni's chunking
   prerequisites fixed the standalone TTS codec's identical, previously
   unknown bug for free; conversely, `crates/codec`'s ENCODE-side transformer
   (`enc_transformer`) has the *exact same* unapplied-`sliding_window` shape
   (`EncoderConfig::sliding_window`, 250) and was deliberately left unfixed
   here (filed as `.todo/codec-encoder-sliding-window.md`) - a shared-code
   bug rarely has exactly one call site, and finding one instance is a reason
   to grep siblings, not a reason to stop at the first one found.

Fixed with a new kernel (`gqa_scores_win.wgsl`, `crates/kernels/wgsl/`) and a
new shared primitive (`model::block::gqa_fwd_win`,
`crates/model/src/block.rs`) that degenerates to `gqa_fwd`'s plain causal mask
exactly when `window >= t` - so `Codec::transformer` now dispatches it
unconditionally, with no `Option<u32>` branch and no risk of the old
plain-causal call site quietly coming back at the next edit. Proven two ways
in `crates/model/tests/gqa_fwd_win.rs` (no real checkpoint available in this
environment - see that file's own doc for why): window-covers-sequence
matches plain causal exactly, and window-narrower-than-sequence matches an
independent host-computed masked-attention oracle AND provably diverges from
the old unwindowed output at the row where it must (a mutation-style check
that the window is load-bearing, not merely wired and ignored).
