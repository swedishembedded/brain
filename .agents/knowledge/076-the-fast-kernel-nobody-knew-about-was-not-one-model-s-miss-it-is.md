<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 76. The "fast kernel nobody knew about" was not one model's miss - it is 14 crates, and two SHARED mixer helpers keep re-seeding it

Checklist section A opens with "the most expensive mistake in this repo is not
a slow kernel - it is a fast kernel nobody knew about", and cites one model
that inherited a slow reduction by name. Finding it again in qwen35 (M22: the
naive `rmsnorm` was 48% of decode device time, and the coalesced
`rmsnorm_rows` had existed all along with `backend_api::select` already
preferring it and `block::rms_variant` already the seam three models selected
it through) prompted an audit of every crate that registers an `rmsnorm` or
`rmsnorm_eps` pipeline. The result is that the archetype is the RULE here, not
the exception:

* **Selecting correctly** (`rms_variant` or an equivalent local seam): flux1,
  flux2, minimaxmusic3, wan, t5encoder, ltxv's `block.rs`, qwen3.
* **Dispatching the naive kernel by hardcoded index**: qwen35moe, glmdsa,
  deepseek2, lfm2, qwen3omnimoe (thinker and talker), qwen3tts, gemma4,
  toymoe, chronos2, fincast, mimi, kronos, s3dit, ltxv's `na_decoder.rs`.

The ones that matter most are the per-token decoders, where the dispatch is
`rows = 1` and the "kernel" is therefore a 5120-element serial reduction on
ONE thread of a 3840-core card: qwen35moe (the direct sibling, same
`run_decode_step` shape), qwen3omnimoe, qwen3tts, glmdsa, deepseek2, lfm2.

Two things generalise past RMSNorm.

**A selection seam only helps the callers that opt in, and most did not.**
`rms_variant` has existed and been correct the whole time. Seven crates found
it; fourteen did not, and several of the fourteen are NEWER than the seam. A
seam that every call site must be individually edited to adopt is a seam the
next model will miss - which is precisely the argument checklist section A4
makes for `gpu_core::upgrade`, and RMSNorm cannot go there, because folding 64
partial sums in a different order is not bit-identical. So for a *sum*
reduction there is no mechanism that reaches callers automatically, and the
only defence is to audit the whole tree periodically rather than per model.

**Fix the SHARED helper, not the N call sites.** `model::block::rmsnorm_fwd`
is called from inside `model::gqa_mixer` and `model::gdn_mixer` for the q/k
and out norms, so every crate composing those mixers inherits the naive kernel
at those points no matter what it does at its own call sites. Two edits there
cover several crates at once, and they are also the edits that make the next
mixer-composing model correct by default. Grep for the shared *builder* that
dispatches a kernel, not only for the kernel name at model level - a per-model
audit reports "this crate is fine" about call sites while the mixer underneath
it is not.

The audit itself is the deliverable to keep: the qwen35 fix was worth 1.82x on
one model's whole decode pass, and the list above says where the same order of
win is still sitting.
