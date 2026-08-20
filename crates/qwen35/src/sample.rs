// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive sampling from a [`Qwen35`] over its single-sequence
//! incremental decode path ([`Qwen35::step`]). Mirrors
//! `qwen35moe::sample` exactly (same temperature/top-k/top-p contract, same
//! host-side head application, same reasoning for why the sampling helpers
//! below are a small, deliberate duplication of `qwen3::sample`'s private
//! `argmax`/`sample_logits` rather than a shared crate - every model crate in
//! this repo owns its own tiny sampling tail, not a cross-model dependency
//! for ~60 lines of elementwise math).
//!
//! Feeds the WHOLE prompt through [`Qwen35::step`] one token at a time
//! (rather than a fast batched prefill-into-cache path): correctness-first
//! for this pass, matching every other qwen35 decode primitive landed so
//! far - a batched prefill fast path is deferred performance work, same as
//! `qwen35moe::serve::Engine`'s own per-token prefill loop names.

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

/// Temperature + top-k + nucleus (top-p) sampling - identical contract to
/// `qwen3::sample::sample_logits`: `temperature <= 0.0` is greedy argmax,
/// `top_k == 0` disables top-k filtering, `top_p` outside `(0,1)` disables
/// nucleus filtering.
///
/// `pub(crate)` (not just this module's own private helper): `crate::stream`'s
/// real-generation entry point (real prompt -> real streaming forward -> real
/// lm_head logits) needs the exact same temperature/top-k/top-p/greedy
/// contract over a bare `&[f32]` logits slice this already implements -
/// reusing it here, rather than re-implementing the same ~30 lines a second
/// time in `stream.rs`, is what this visibility bump is for. Still not `pub`:
/// no caller outside this crate needs it.
pub(crate) fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits) as u32;
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    if top_k > 0 && top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let threshold = scaled[idx[top_k - 1]];
        for v in scaled.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in scaled.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if top_p > 0.0 && top_p < 1.0 && sum > 0.0 {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let mut kept = 0.0f32;
        let mut cut = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            kept += scaled[i];
            if kept / sum >= top_p {
                cut = rank + 1;
                break;
            }
        }
        for &i in &idx[cut..] {
            scaled[i] = 0.0;
        }
        sum = kept;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

/// Generate `max_new` tokens continuing `prompt` via [`Qwen35::step`].
/// `temperature <= 0` is greedy; `top_k = 0` disables top-k; `top_p` in
/// `(0,1)` enables nucleus filtering. Stops early at any id in `eos`.
///
/// Reads the LM head weight fresh from `model` - a `[vocab, d_model]`
/// device-to-host transfer once per call, not once per token. A caller
/// serving many requests against one resident model should read it once and
/// reuse the buffer - not done here since this is not that caller (that's
/// `crates/qwen35/src/serve.rs`'s own, separate concern).
#[allow(clippy::too_many_arguments)]
pub fn generate_kv(model: &Qwen35, prompt: &[u32], max_new: usize, temperature: f32, top_k: usize, top_p: f32, eos: &[u32], rng: &mut Rng) -> Vec<u32> {
    let head = model.read_weight(model.cfg.head_weight());
    generate_kv_with_head(model, prompt, max_new, temperature, top_k, top_p, eos, rng, &head)
}

/// [`generate_kv`] with the LM head weight supplied by the caller instead of
/// read fresh from `model` - see [`generate_kv`]'s own doc for why a
/// multi-request caller wants this instead.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_with_head(
    model: &Qwen35,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    head: &[f32],
) -> Vec<u32> {
    let vocab = model.cfg.vocab as usize;
    let d = model.cfg.d_model as usize;
    let logits_of = |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(head, hidden, vocab, d) };

    model.reset_decode_cache();
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    let mut hidden = Vec::new();
    for &tok in seed_prompt {
        hidden = model.step(tok);
    }

    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let next = sample_logits(&logits_of(&hidden), temperature, top_k, top_p, rng);
        if eos.contains(&next) {
            break;
        }
        out.push(next);
        hidden = model.step(next);
    }
    out
}

/// [`generate_kv`] with a per-token callback: `on_token(index, token)` fires
/// as each token is accepted (before the next decode step), giving callers a
/// true streaming timeline (TTFT/ITL) without re-implementing the decode
/// loop. Returning `false` from `on_token` stops generation early (the token
/// is kept), letting callers honour cancellation or stop-strings. Reads the
/// head fresh from `model` on every call; a caller serving many requests
/// against one resident model should read it once and call
/// [`generate_kv_stream_with_head`] directly instead.
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
    let head = model.read_weight(model.cfg.head_weight());
    generate_kv_stream_with_head(model, prompt, max_new, temperature, top_k, top_p, eos, rng, &head, on_token)
}

/// [`generate_kv_stream`] with the LM head weight supplied by the caller
/// instead of read fresh from `model` on every call - same reasoning as
/// [`generate_kv_with_head`] vs [`generate_kv`], now with the per-token
/// callback. This IS the streaming implementation; [`generate_kv_stream`]
/// delegates here with a freshly-read head.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream_with_head(
    model: &Qwen35,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    head: &[f32],
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    let vocab = model.cfg.vocab as usize;
    let d = model.cfg.d_model as usize;
    let logits_of = |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(head, hidden, vocab, d) };

    model.reset_decode_cache();
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    let mut hidden = Vec::new();
    for &tok in seed_prompt {
        hidden = model.step(tok);
    }

    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let next = sample_logits(&logits_of(&hidden), temperature, top_k, top_p, rng);
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
