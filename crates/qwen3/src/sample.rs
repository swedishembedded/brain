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
/// filtering (>= 1 disables). Stops early at any id in `stop`.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    stop: &[u32],
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
        if stop.contains(&next) {
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
/// Any id in `stop` ends generation; an empty slice runs to `max_new`.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    stop: &[u32],
    rng: &mut Rng,
) -> Vec<u32> {
    generate_kv_stream(model, prompt, max_new, temperature, top_k, top_p, stop, rng, &mut |_, _| true)
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
    // An unarmed token never fires, and one whole-prompt chunk is exactly the
    // single prefill call this function has always been: existing callers are
    // unchanged, byte for byte on the device.
    generate_kv_stream_cancellable(
        model,
        prompt,
        max_new,
        temperature,
        top_k,
        top_p,
        eos,
        rng,
        head,
        &capability::CancelToken::default(),
        prompt.len(),
        on_token,
    )
}

/// [`generate_kv_stream_with_head`] with brain's cooperative cancellation:
/// the prompt is prefilled in chunks of `prefill_chunk` tokens with `cancel`
/// polled between chunks, and the decode loop stops when the caller's
/// `on_token` says so (the same place a cancel surfaces through
/// `qwen3::chat`'s [`SeqState`](crate::chat::SeqState)).
///
/// Why chunk at all: `Qwen::prefill` submits the whole prompt and fences
/// ONCE at the end, so a whole-prompt prefill is one uninterruptible device
/// wait - for a real agentic prompt (a system prompt plus a full tool
/// schema, thousands of tokens) that is work a cancellation requested
/// mid-flight cannot reach, and a process that must not wait for it cannot
/// exit cleanly under it. Chunking costs one extra readback per chunk -
/// noise against the per-chunk compute - and bounds the cancellation latency
/// to one chunk. Numerically it is the single-call prefill up to fp32
/// summation order: how the prompt is split moves only the batched
/// forward's chunk boundaries (proved by
/// `chunked_prefill_generation_matches_the_single_call_path` and
/// `tests/batched_prefill.rs`).
///
/// A token already cancelled when the call starts submits nothing and
/// returns the empty sequence; one armed mid-prefill stops the generation
/// with whatever tokens were produced so far.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream_cancellable(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    head: &[f32],
    cancel: &capability::CancelToken,
    prefill_chunk: usize,
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    let vocab = model.cfg.vocab as usize;
    let d = model.cfg.d_model as usize;
    // Row-parallel: the single-threaded head was measured at hundreds of ms
    // PER TOKEN at real vocabularies (one implementation: model::hostmath).
    let logits_of = |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(head, hidden, vocab, d) };
    generate_kv_core(model, prompt, None, max_new, temperature, top_k, top_p, eos, rng, cancel, prefill_chunk, &logits_of, on_token)
}

/// [`generate_kv_stream_cancellable`] with the LM head applied **on the
/// device** the weights already live on ([`Qwen::decode_logits`], tiled over
/// the vocabulary for the binding limit), so a generation needs no host copy
/// of the head and spends no host compute per token: at a 152k-token
/// vocabulary and `d_model` 3584 that is a 2.2 GB table held in RAM and a
/// half-gigaflop matvec per token on the CPU. The tokens are the host head's:
/// both are the same GEMV, and the logits agree to rounding.
///
/// `decode_logits` reads the hidden state the last prefill or step left on the
/// device, so the core calls it directly after each of those and ignores the
/// hidden state it is handed.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream_on_device(
    model: &Qwen,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    cancel: &capability::CancelToken,
    prefill_chunk: usize,
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    let logits_of = |_hidden: &[f32]| -> Vec<f32> { model.decode_logits() };
    generate_kv_core(model, prompt, None, max_new, temperature, top_k, top_p, eos, rng, cancel, prefill_chunk, &logits_of, on_token)
}

/// A run of prompt positions whose input is a given embedding row rather than a
/// token's: speech (or any other modality) projected into the model's input
/// space, standing where a user's words would be.
#[derive(Clone, Copy, Debug)]
pub struct RowRun<'a> {
    /// The first prompt position the rows replace.
    pub at: usize,
    /// The rows, row-major, `width` values each.
    pub rows: &'a [f32],
    /// The model's embedding width.
    pub width: usize,
}

/// [`generate_kv_stream_on_device`] for a prompt in which a run of positions is
/// embedding rows: the token ids at those positions are ignored.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv_stream_on_device_rows(
    model: &Qwen,
    prompt: &[u32],
    run: &RowRun<'_>,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    cancel: &capability::CancelToken,
    prefill_chunk: usize,
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    let logits_of = |_hidden: &[f32]| -> Vec<f32> { model.decode_logits() };
    generate_kv_core(model, prompt, Some(run), max_new, temperature, top_k, top_p, eos, rng, cancel, prefill_chunk, &logits_of, on_token)
}

/// The chunked-prefill, KV-cached decode loop both heads share; `logits_of`
/// turns the hidden state of the position just computed into `[vocab]` logits.
#[allow(clippy::too_many_arguments)]
fn generate_kv_core(
    model: &Qwen,
    prompt: &[u32],
    run: Option<&RowRun<'_>>,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos: &[u32],
    rng: &mut Rng,
    cancel: &capability::CancelToken,
    prefill_chunk: usize,
    logits_of: &dyn Fn(&[f32]) -> Vec<f32>,
    on_token: &mut dyn FnMut(usize, u32) -> bool,
) -> Vec<u32> {
    model.reset_cache();
    let mut out = Vec::with_capacity(max_new);
    // Feed the prompt in as few prefill calls as the cancellation policy
    // needs, never a step()-per-token loop: Qwen::prefill runs each call as
    // batched chunks through the layer forward, where step() would pay a
    // whole decode tape and a submit+fence+map round trip per token.
    // tests/batched_prefill.rs holds it to the step()-per-token reference.
    // (Empty prompt → seed a single newline-like id 0.)
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    let mut hidden = Vec::new();
    for (n, chunk) in seed_prompt.chunks(prefill_chunk.max(1)).enumerate() {
        if cancel.is_cancelled() {
            return out;
        }
        let first = n * prefill_chunk.max(1);
        let prefill_inputs: Vec<PrefillInput<'_>> = chunk
            .iter()
            .enumerate()
            .map(|(i, &t)| match run {
                Some(r) if (r.at..r.at + r.rows.len() / r.width).contains(&(first + i)) => {
                    let row = first + i - r.at;
                    PrefillInput::Embed(&r.rows[row * r.width..(row + 1) * r.width])
                }
                _ => PrefillInput::Token(t),
            })
            .collect();
        hidden = model.prefill(&prefill_inputs);
    }
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
        let recompute = generate(&model, &prompt, 16, 0.0, 0, 1.0, &[], &mut r1);
        let mut r2 = data::rng::Rng::new(0);
        let kv = generate_kv(&model, &prompt, 16, 0.0, 0, 1.0, &[], &mut r2);
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

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// Chunked, cancel-aware prefill must be numerically the single-call
    /// prefill: the KV cache and the final hidden do not depend on how the
    /// prompt is split across device round trips, so the same seed must give
    /// the same tokens at every chunk size - including the whole-prompt chunk
    /// that reproduces the old single-call behavior exactly.
    #[test]
    fn chunked_prefill_generation_matches_the_single_call_path() {
        if gpu_disabled() {
            return;
        }
        let model = tiny_model(3);
        let prompt = vec![1u32, 5, 3, 9, 2, 7];
        let cancel = capability::CancelToken::default();
        let baseline = {
            let mut rng = data::rng::Rng::new(4);
            generate_kv_stream_with_head(
                &model,
                &prompt,
                8,
                1.0,
                0,
                1.0,
                &[],
                &mut rng,
                &model.read_weight(model.cfg.head_weight()),
                &mut |_, _| true,
            )
        };
        for chunk in [1usize, 2, 4] {
            let mut rng = data::rng::Rng::new(4);
            let chunked = generate_kv_stream_cancellable(
                &model,
                &prompt,
                8,
                1.0,
                0,
                1.0,
                &[],
                &mut rng,
                &model.read_weight(model.cfg.head_weight()),
                &cancel,
                chunk,
                &mut |_, _| true,
            );
            assert_eq!(chunked, baseline, "chunk size {chunk} must not change the sampled tokens");
        }
    }

    /// The device head is the same GEMV as the host head, applied where the
    /// weights already are: greedy generation through it must pick exactly the
    /// tokens the host-head sampler picks, at every prefill chunk size, and
    /// must honour an armed cancel token the same way.
    #[test]
    fn device_head_generation_matches_the_host_head_greedy() {
        if gpu_disabled() {
            return;
        }
        let model = tiny_model(3);
        let prompt = vec![1u32, 5, 3, 9, 2, 7];
        let head = model.read_weight(model.cfg.head_weight());
        let cancel = capability::CancelToken::default();
        let mut r = data::rng::Rng::new(4);
        let host = generate_kv_stream_with_head(&model, &prompt, 12, 0.0, 0, 1.0, &[], &mut r, &head, &mut |_, _| true);
        assert!(!host.is_empty());
        for chunk in [1usize, 3, prompt.len()] {
            let mut r = data::rng::Rng::new(4);
            let device = generate_kv_stream_on_device(&model, &prompt, 12, 0.0, 0, 1.0, &[], &mut r, &cancel, chunk, &mut |_, _| true);
            assert_eq!(device, host, "chunk size {chunk}: the device head must pick the host head's tokens");
        }
        let armed = capability::CancelToken::armed();
        armed.cancel();
        let mut r = data::rng::Rng::new(4);
        let none = generate_kv_stream_on_device(&model, &prompt, 12, 0.0, 0, 1.0, &[], &mut r, &armed, 2, &mut |_, _| true);
        assert!(none.is_empty(), "an armed cancel stops before any token");
    }

    /// A cancel token armed before the call must stop the generation at the
    /// first chunk boundary: nothing is submitted, and the empty token
    /// sequence is returned. (Mid-prefill cancellation is covered by the
    /// between-chunk poll itself; this pins the boundary case.)
    #[test]
    fn a_cancel_armed_before_the_call_submits_nothing() {
        if gpu_disabled() {
            return;
        }
        let model = tiny_model(3);
        let prompt = vec![1u32, 5, 3];
        let cancel = capability::CancelToken::armed();
        cancel.cancel();
        let mut rng = data::rng::Rng::new(4);
        let out = generate_kv_stream_cancellable(
            &model,
            &prompt,
            8,
            1.0,
            0,
            1.0,
            &[],
            &mut rng,
            &model.read_weight(model.cfg.head_weight()),
            &cancel,
            1,
            &mut |_, _| true,
        );
        assert!(out.is_empty(), "a cancelled prefill must produce no tokens");
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
            let full = generate_kv(&model, &prompt, 16, 1.0, 0, 1.0, &[], &mut r0);
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
