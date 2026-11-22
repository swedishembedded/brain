// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive sampling from a [`Qwen35`] over its single-sequence
//! incremental decode path ([`Qwen35::step`]). Mirrors
//! `qwen35moe::sample` exactly (same temperature/top-k/top-p contract, same
//! reasoning for why the sampling helpers
//! below are a small, deliberate duplication of `qwen3::sample`'s private
//! `argmax`/`sample_logits` rather than a shared crate - every model crate in
//! this repo owns its own tiny sampling tail, not a cross-model dependency
//! for ~60 lines of elementwise math).
//!
//! Feeds the WHOLE prompt through [`Qwen35::step`] one token at a time.
//! The batched alternative now exists - [`Qwen35::prefill_chunked`] (M25),
//! which leaves byte-for-byte the same decode state and is what
//! `crate::serve::Engine::prefill` runs - so this is a straightforward swap
//! whenever this helper's own callers are worth re-baselining; it is left on
//! the simple path here only because nothing that uses it is throughput-
//! sensitive.

use data::rng::Rng;

use crate::model::Qwen35;

pub(crate) fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

/// Temperature + top-k + nucleus (top-p) sampling: `temperature <= 0.0` is
/// greedy argmax, `top_k == 0` disables top-k filtering, `top_p` outside
/// `(0,1)` disables nucleus filtering. The one implementation is
/// `qwen3vl::sample::sample_logits` (NaN-safe), re-exported here for this
/// crate's generation paths and the wrappers built on its decoder.
pub use qwen3vl::sample::sample_logits;

/// Generate `max_new` tokens continuing `prompt` via [`Qwen35::step`].
/// `temperature <= 0` is greedy; `top_k = 0` disables top-k; `top_p` in
/// `(0,1)` enables nucleus filtering. Stops early at any id in `eos`.
///
/// Each step's logits are projected on the device through the model's
/// resident head ([`Qwen35::head_logits`]): only the `[vocab]` row comes back,
/// and no host copy of the head is ever made.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv(model: &Qwen35, prompt: &[u32], max_new: usize, temperature: f32, top_k: usize, top_p: f32, eos: &[u32], rng: &mut Rng) -> Vec<u32> {
    generate_kv_stream(model, prompt, max_new, temperature, top_k, top_p, eos, rng, &mut |_, _| true)
}

/// [`generate_kv`] with a per-token callback: `on_token(index, token)` fires
/// as each token is accepted (before the next decode step), giving callers a
/// true streaming timeline (TTFT/ITL) without re-implementing the decode
/// loop. Returning `false` from `on_token` stops generation early (the token
/// is kept), letting callers honour cancellation or stop-strings.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream(
    model: &Qwen35,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    model.reset_decode_cache();
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    let mut hidden = Vec::new();
    for &tok in seed_prompt {
        hidden = model.step(tok);
    }

    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let next = sample_logits(&model.head_logits(&hidden), temperature, top_k, top_p, rng);
        if eos.contains(&next) {
            break;
        }
        out.push(next);
        if !on_token(out.len() - 1, next) {
            break; // caller asked to stop (cancellation / stop-string)
        }
        hidden = model.step(next);
    }
    out
}
