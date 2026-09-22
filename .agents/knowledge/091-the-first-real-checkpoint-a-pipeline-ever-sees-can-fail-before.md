<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 91. The first real checkpoint a pipeline ever sees can fail before reaching the code you just wrote - and that is still useful information, not a wasted run

FLUX.1-dev's real weights landed in this workspace for the first time this
session. Before `pulid::caps::Bundle::face_embeds` (this session's own new
`bisenet::align`/`BiSeNet::forward`/`bisenet::mask` chain) ever got to run
against them, TWO independent, pre-existing defects surfaced:

1. `data::unigram::UnigramTokenizer` errors on SentencePiece's `Precompiled`
   normalizer - already a KNOWN, DOCUMENTED gap (`unigram.rs`'s own module
   docs and a dedicated test asserting the clean error), never actually
   exercised against a real T5-XXL `tokenizer.json` before because no such
   file existed in this workspace until now. Blocks tokenization for BOTH
   `brain flux1 text2image` and `brain pulid text2image` - upstream of
   identity conditioning entirely.
2. `pulid::caps::Bundle::load` panics inside `clip::model::EvaVision::new_on`
   (a wgpu bind-group validation error) when built with real weights, on the
   FIRST attempt to run PuLID end to end ever. Diagnostic `eprintln!`s
   proved the new BiSeNet alignment code had not even been reached yet -
   `EvaVision::new_on` only uploads weights, so this is most likely wgpu's
   asynchronous error reporting surfacing a fault from an earlier dispatch
   at the next GPU operation. `BRAIN_NO_KERNEL_UPGRADE=1` and swapping
   BiSeNet's `Gpu::new_like` for a fresh `Gpu::new` both left it unchanged -
   two real hypotheses tested and ruled out, not guessed away.

**The thing to carry forward:** a component's own parity tests (BiSeNet's:
cosine 1.0000000000 against real `facexlib` output) prove that component is
correct in ISOLATION - they say nothing about whether the surrounding
pipeline can reach it with real data. This repo's own "treat a first real
generation as the actual test of this file" note (`flux1/src/pipeline.rs`)
is not a formality: this session hit it exactly as described, twice, in
code neither defect's fix belongs inside. Record what a first real run
found and where it stopped, even when the root cause is not yet fixed -
that is what makes the NEXT session's first hour investigation instead of
rediscovery.
