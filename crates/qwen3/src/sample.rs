// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive sampling from a Qwen model (temperature + top-k). Cache-free:
//! re-runs the forward over the (cropped) context each step. Correct and simple;
//! a KV-cache fast path is a separate inference optimisation.

use data::rng::Rng;

use crate::model::{PrefillInput, Qwen};

/// Generate `max_new` tokens continuing `prompt`. The context is cropped to the
/// model's sized length (`ctx_len`). `temperature <= 0` selects greedy argmax;
/// `top_k = 0` disables top-k filtering; `top_p` in (0,1) enables nucleus
/// filtering (>= 1 disables). Stops early at `eos` if provided.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: Option<u32>,
    rng: &mut Rng,
) -> Vec<u32> {
    let cap = model.ctx_len();
    let vocab = model.cfg.vocab as usize;
    let mut ctx: Vec<u32> = prompt.to_vec();
    let mut out = Vec::with_capacity(max_new);

    for _ in 0..max_new {
        let window: Vec<u32> = if ctx.len() > cap { ctx[ctx.len() - cap..].to_vec() } else { ctx.clone() };
        let logits = model.logits_all(&window);
        let last = &logits[logits.len() - vocab..];
        let next = sample_logits(last, temperature, top_k, top_p, rng);
        if Some(next) == eos {
            break;
        }
        ctx.push(next);
        out.push(next);
    }
    out
}

/// KV-cache generation: the O(T) fast path. Feeds the prompt through the
/// incremental `step` (filling the cache), then samples one token per `step`
/// instead of re-running the whole context each time. Produces the same tokens
/// as [`generate`] for greedy decoding (the cache is algebraically exact). The
/// tied/untied head is applied on the host to the final-norm hidden state.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: Option<u32>,
    rng: &mut Rng,
) -> Vec<u32> {
    let eos_arr: [u32; 1];
    let eos_slice: &[u32] = match eos {
        Some(e) => {
            eos_arr = [e];
            &eos_arr
        }
        None => &[],
    };
    generate_kv_stream(model, prompt, max_new, temperature, top_k, top_p, eos_slice, rng, &mut |_, _| true)
}

/// [`generate_kv`] with a per-token callback: `on_token(index, token)` fires as
/// each token is accepted (before the next decode step), giving callers a true
/// streaming timeline (TTFT/ITL) without re-implementing the decode loop.
/// Returning `false` from `on_token` stops generation early (the token is kept),
/// letting callers honour cancellation or stop-strings. This IS the
/// implementation; [`generate_kv`] delegates here with a keep-going callback.
///
/// `eos` is a set of stop ids (Qwen3 has two: `<|im_end|>` 151645 and
/// `<|endoftext|>` 151643) — generation stops as soon as the sampled token
/// matches ANY of them; an empty slice disables the stop check.
///
/// Reads the LM head weight fresh from `model` on every call — for tied
/// embeddings that is a `[vocab, d_model]` device→host transfer (hundreds of
/// MB at real vocab sizes) repeated per request. A caller serving many
/// requests against one resident model should read the head once and call
/// [`generate_kv_stream_with_head`] directly instead.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    let head = model.read_weight(model.cfg.head_weight()); // [vocab, d]
    generate_kv_stream_with_head(model, prompt, max_new, temperature, top_k, top_p, eos, rng, &head, on_token)
}

/// [`generate_kv_stream`] with the LM head weight supplied by the caller
/// instead of read fresh from `model` on every call — the fix for the 594 MiB
/// (vocab 151936 × d_model 1024, f32) tied-embedding re-download `generate_kv_stream`
/// otherwise pays per request. `head` is `[vocab, d_model]` row-major, exactly
/// [`Qwen::read_weight`]`(cfg.head_weight())`'s shape; the caller reads it once
/// (e.g. at model load) and reuses the buffer across calls.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream_with_head(
    model: &Qwen,
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
    // Row-parallel: the single-threaded head was measured at hundreds of ms
    // PER TOKEN at real vocabularies (one implementation: model::hostmath).
    let logits_of = |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(head, hidden, vocab, d) };
    model.reset_cache();
    let mut out = Vec::with_capacity(max_new);
    // Feed the prompt in ONE prefill call (single readback at the end), not a
    // step()-per-token loop: step() does a full submit+fence+map round trip
    // per call, and Qwen::prefill's own doc calls that "pure waste" during
    // prefill, where every intermediate hidden is discarded anyway. Proven
    // identical to the old step()-per-token behavior by
    // model::tests::prefill_matches_step_by_step. For a real agentic prompt
    // (a system prompt plus a full tool-schema block, easily 1000+ tokens)
    // this was measured to dominate turn latency by orders of magnitude --
    // multiple real end-to-end runs against Qwen3-0.6B took 600+ seconds
    // before this fix. (Empty prompt → seed a single newline-like id 0.)
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    let prefill_inputs: Vec<PrefillInput<'_>> = seed_prompt.iter().map(|&t| PrefillInput::Token(t)).collect();
    let mut hidden = model.prefill(&prefill_inputs);
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

/// Total order over `f32` for sampling comparisons, with every NaN treated as
/// strictly least-preferred (sorting below `-inf`). Two problems rule out the
/// obvious alternatives: `f32::partial_cmp` returns `None` for a NaN operand,
/// which is why a bare `.unwrap()` on it panics the instant a NaN logit shows
/// up (the crash this function used to hit); and `f32::total_cmp` alone is
/// NOT a safe drop-in either, since IEEE 754 places positive-payload NaNs
/// ABOVE `+inf` in its total order, which would make a NaN logit WIN
/// `argmax`/top-k selection outright instead of losing it. A text-generation
/// sampler must never crash on a NaN logit, and if one shows up it must never
/// be preferred over a real (finite) alternative, so NaN is defined here as
/// the least-preferred value, full stop, regardless of its sign or payload.
#[inline]
fn cmp_nan_last(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.total_cmp(&b),
    }
}

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        // Plain `s[i] > s[bi]` is NaN-unsound: comparisons against NaN are
        // always false, so a NaN at `s[bi]` (e.g. `s[0]`) could never be
        // displaced by a later, genuinely larger finite value - argmax would
        // silently and permanently lock onto the NaN position instead of
        // picking the best real logit. `cmp_nan_last` treats NaN as
        // least-preferred so a finite value always beats it.
        if cmp_nan_last(s[i], s[bi]) == std::cmp::Ordering::Greater {
            bi = i;
        }
    }
    bi
}

/// Temperature + top-k + nucleus (top-p) sampling. `top_p` in (0,1) keeps the
/// smallest set of highest-probability tokens whose cumulative mass reaches
/// `top_p` (at least one) and zeroes the rest; `top_p >= 1` (or `<= 0`) is a
/// no-op, so the top-k-only path is bit-for-bit unchanged.
///
/// Robust to a NaN logit (which should never happen, but a serving process
/// must not crash if a forward pass produces one anyway): any NaN is treated
/// as least-preferred, exactly like `-inf` - it can only be picked if every
/// other logit is NaN too. `scaled` is sanitized right after the
/// temperature divide (a NaN input logit stays NaN through that division, so
/// there is nothing else `temperature` alone can do to filter it) so no NaN
/// ever reaches the softmax accumulation below and poisons the whole
/// distribution (`sum` becoming NaN would make every later comparison
/// against it silently false, degrading sampling instead of just correctly
/// disfavoring the one bad logit).
fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits) as u32;
    }
    let mut scaled: Vec<f32> =
        logits.iter().map(|&l| if l.is_nan() { f32::NEG_INFINITY } else { l / temperature }).collect();
    if top_k > 0 && top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| cmp_nan_last(scaled[b], scaled[a]));
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
    // Nucleus (top-p): keep the highest-probability prefix reaching `top_p` mass.
    if top_p > 0.0 && top_p < 1.0 && sum > 0.0 {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| cmp_nan_last(scaled[b], scaled[a]));
        let mut kept = 0.0f32;
        let mut cut = idx.len(); // first rank NOT kept
        for (rank, &i) in idx.iter().enumerate() {
            kept += scaled[i];
            if kept / sum >= top_p {
                cut = rank + 1; // keep through this rank (always >= 1)
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

#[cfg(test)]
mod kv_gen_tests {
    use super::*;
    use crate::config::QwenConfig;
    use crate::model::Qwen;
    use std::collections::HashMap;

    /// KV-cache generation must produce the SAME greedy tokens as the O(T²)
    /// recompute path (the cache is algebraically exact; logits agree to ~1e-7).
    #[test]
    fn generate_kv_matches_recompute_greedy() {
        let cfg = QwenConfig::tiny();
        let mut rng = data::rng::Rng::new(1);
        let mut map = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.contains("norm") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.05).collect()
            };
            map.insert(name, v);
        }
        let model = Qwen::new(cfg.clone(), 1, 32, &map);
        let prompt = vec![1u32, 5, 3];
        let mut r1 = data::rng::Rng::new(0);
        let recompute = generate(&model, &prompt, 16, 0.0, 0, 1.0, None, &mut r1);
        let mut r2 = data::rng::Rng::new(0);
        let kv = generate_kv(&model, &prompt, 16, 0.0, 0, 1.0, None, &mut r2);
        assert_eq!(recompute, kv, "KV greedy generation must equal recompute generation");
    }

    fn tiny_model(seed: u64) -> Qwen {
        let cfg = QwenConfig::tiny();
        let mut rng = data::rng::Rng::new(seed);
        let mut map = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.contains("norm") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.05).collect()
            };
            map.insert(name, v);
        }
        Qwen::new(cfg, 1, 32, &map)
    }

    /// `generate_kv_stream`'s `eos` is a SET of stop ids: generation must stop
    /// as soon as the sampled token matches ANY of them (Qwen3 has two —
    /// `<|im_end|>` and `<|endoftext|>` — this proves the multi-id membership
    /// check generically, not tied to those specific ids).
    #[test]
    fn eos_stops_on_any_id_in_the_slice() {
        // Stochastic sampling (not greedy): a freshly (randomly) initialized
        // tiny model's argmax collapses to a repeating fixed-point token under
        // greedy decoding (verified: every seed 0..16 did), which would make
        // "the first sampled token" and "a later token" coincide and defeat
        // the point of this test. `temperature > 0` with a fixed rng seed is
        // still fully reproducible (same draws, same prefix) for comparing
        // truncated vs. unconstrained generation below.
        let prompt = vec![1u32, 5, 3];
        let mut found = None;
        for seed in 0..16u64 {
            let model = tiny_model(seed);
            let mut r0 = data::rng::Rng::new(0);
            let full = generate_kv(&model, &prompt, 16, 1.0, 0, 1.0, None, &mut r0);
            if full.iter().any(|&t| t != full[0]) {
                found = Some((model, full));
                break;
            }
        }
        let (model, full) = found.expect("at least one seed must give a non-degenerate continuation");

        // A token that occurs partway through (not the very first sampled
        // token) as one of two "eos" ids; the other id never occurs at all.
        // Generation must still stop at the real one, at its first occurrence.
        let stop_at = *full.iter().find(|&&t| t != full[0]).unwrap();
        let first_idx = full.iter().position(|&t| t == stop_at).unwrap();
        assert!(first_idx > 0, "stop id must not be the very first sampled token");
        let never_occurs = 999_999u32;

        let mut r1 = data::rng::Rng::new(0);
        let truncated = generate_kv_stream(&model, &prompt, 16, 1.0, 0, 1.0, &[never_occurs, stop_at], &mut r1, &mut |_, _| true);
        assert_eq!(truncated, &full[..first_idx], "must stop as soon as ANY eos id in the slice is sampled");

        // Order in the slice must not matter.
        let mut r2 = data::rng::Rng::new(0);
        let truncated2 = generate_kv_stream(&model, &prompt, 16, 1.0, 0, 1.0, &[stop_at, never_occurs], &mut r2, &mut |_, _| true);
        assert_eq!(truncated2, &full[..first_idx]);

        // An empty eos slice never stops early.
        let mut r3 = data::rng::Rng::new(0);
        let no_stop = generate_kv_stream(&model, &prompt, 16, 1.0, 0, 1.0, &[], &mut r3, &mut |_, _| true);
        assert_eq!(no_stop, full, "empty eos slice must not stop generation");
    }

    /// [`generate_kv_stream_with_head`] with a caller-supplied head must be
    /// bit-for-bit identical to [`generate_kv_stream`]'s self-reading wrapper —
    /// the hoist is a pure caching optimisation, not a behaviour change.
    #[test]
    fn with_head_matches_the_self_reading_wrapper() {
        let model = tiny_model(2);
        let prompt = vec![2u32, 4, 6];
        let head = model.read_weight(model.cfg.head_weight());

        let mut r1 = data::rng::Rng::new(7);
        let a = generate_kv_stream(&model, &prompt, 10, 0.0, 0, 1.0, &[], &mut r1, &mut |_, _| true);
        let mut r2 = data::rng::Rng::new(7);
        let b = generate_kv_stream_with_head(&model, &prompt, 10, 0.0, 0, 1.0, &[], &mut r2, &head, &mut |_, _| true);
        assert_eq!(a, b, "generate_kv_stream_with_head must match generate_kv_stream given the same head");
    }

    /// A tight nucleus collapses the distribution onto the single dominant token
    /// regardless of the RNG draw; disabling it (`top_p >= 1`) can pick others.
    #[test]
    fn top_p_restricts_to_the_nucleus() {
        // Token 0 carries ~0.9997 of the softmax mass at temperature 1.
        let logits = [8.0f32, 0.0, 0.0, 0.0];
        for seed in 0..8u64 {
            let mut rng = data::rng::Rng::new(seed);
            // top_p below the dominant mass keeps only token 0.
            assert_eq!(sample_logits(&logits, 1.0, 0, 0.9, &mut rng), 0);
        }
        // Flat logits + tiny top_p keeps exactly one token (the first by rank).
        let flat = [1.0f32, 1.0, 1.0, 1.0];
        let mut rng = data::rng::Rng::new(3);
        let only = sample_logits(&flat, 1.0, 0, 0.01, &mut rng);
        for seed in 0..8u64 {
            let mut rng = data::rng::Rng::new(seed);
            assert_eq!(sample_logits(&flat, 1.0, 0, 0.01, &mut rng), only, "nucleus keeps a single token");
        }
        // Disabled nucleus over a flat distribution reaches non-zero tokens.
        let mut seen_other = false;
        for seed in 0..32u64 {
            let mut rng = data::rng::Rng::new(seed);
            if sample_logits(&flat, 1.0, 0, 1.0, &mut rng) != only {
                seen_other = true;
                break;
            }
        }
        assert!(seen_other, "top_p>=1 must not restrict sampling");
    }

    /// Regression: a NaN logit used to panic `sample_logits` via
    /// `partial_cmp(...).unwrap()` inside the top-k sort (`f32::partial_cmp`
    /// returns `None` for a NaN operand). A NaN logit must never crash
    /// sampling, and - since it is treated as least-preferred - must never be
    /// the token picked while a finite alternative exists, at any
    /// temperature/top-k/top-p combination that exercises every code path
    /// (greedy argmax, top-k only, top-p only, both together, and both
    /// disabled).
    #[test]
    fn nan_logit_does_not_panic_and_is_never_preferred() {
        let logits = [1.0f32, f32::NAN, 3.0, 2.0];
        // Greedy (temperature <= 0): argmax must skip the NaN, not lock onto it.
        let mut rng = data::rng::Rng::new(0);
        assert_eq!(sample_logits(&logits, 0.0, 0, 1.0, &mut rng), 2, "greedy argmax must pick the real max, not the NaN");

        // Every other sampling configuration must at least not panic, and must
        // never draw the NaN token index (1) while finite alternatives exist.
        let configs: &[(f32, usize, f32)] = &[
            (1.0, 0, 1.0),  // top-k and top-p both disabled
            (1.0, 2, 1.0),  // top-k only
            (1.0, 0, 0.9),  // top-p only
            (1.0, 2, 0.9),  // top-k and top-p together
        ];
        for &(temp, top_k, top_p) in configs {
            for seed in 0..16u64 {
                let mut rng = data::rng::Rng::new(seed);
                let picked = sample_logits(&logits, temp, top_k, top_p, &mut rng);
                assert_ne!(picked, 1, "NaN logit (index 1) must never be picked over a finite alternative");
            }
        }
    }

    /// Regression: `argmax`'s naive `s[i] > s[bi]` is NaN-unsound - comparisons
    /// against NaN are always false, so a NaN sitting at the initial best index
    /// (`s[0]`) could never be displaced by a later, genuinely larger finite
    /// value. Covers a NaN at the front, in the middle, and an all-NaN input
    /// (which must return some in-bounds index, not panic).
    #[test]
    fn argmax_skips_nan_regardless_of_position() {
        assert_eq!(argmax(&[f32::NAN, 1.0, 5.0, 2.0]), 2, "NaN at the front must not stick as the answer");
        assert_eq!(argmax(&[1.0, 5.0, f32::NAN, 2.0]), 1, "NaN in the middle must not beat a real max before it");
        assert_eq!(argmax(&[1.0, 2.0, 3.0, f32::NAN]), 2, "NaN at the end must not beat a real max before it");
        let all_nan = argmax(&[f32::NAN, f32::NAN, f32::NAN]);
        assert!(all_nan < 3, "an all-NaN input must return an in-bounds index, not panic");
    }
}
