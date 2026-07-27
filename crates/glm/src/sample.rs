// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive sampling from a GLM model (temperature + top-k). Cache-free:
//! re-runs the forward over the (cropped) context each step. Correct and simple;
//! a KV-cache fast path is a separate inference optimisation.

use data::rng::Rng;

use crate::model::Glm;

/// Generate `max_new` tokens continuing `prompt`. The context is cropped to the
/// model's sized length (`ctx_len`). `temperature <= 0` selects greedy argmax;
/// `top_k = 0` disables top-k filtering. Stops early at `eos` if provided.
pub fn generate(
    model: &Glm,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
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
        let next = sample_logits(last, temperature, top_k, rng);
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
/// as [`generate`] for greedy decoding (the cache is algebraically exact). GLM's
/// untied `lm_head` is applied on the host to the final-norm hidden state.
pub fn generate_kv(
    model: &Glm,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    eos: Option<u32>,
    rng: &mut Rng,
) -> Vec<u32> {
    let vocab = model.cfg.vocab as usize;
    let d = model.cfg.d_model as usize;
    let head = model.read_weight(model.cfg.head_weight()); // [vocab, d]
    let logits_of = |hidden: &[f32]| -> Vec<f32> {
        (0..vocab)
            .map(|o| head[o * d..o * d + d].iter().zip(hidden).map(|(a, b)| a * b).sum())
            .collect()
    };
    model.reset_cache();
    let mut out = Vec::with_capacity(max_new);
    // Feed the prompt; the hidden after the last prompt token gives the first
    // next-token distribution. (Empty prompt → seed a single id 0.)
    let mut hidden = Vec::new();
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    for &t in seed_prompt {
        hidden = model.step(t);
    }
    for _ in 0..max_new {
        let next = sample_logits(&logits_of(&hidden), temperature, top_k, rng);
        if Some(next) == eos {
            break;
        }
        out.push(next);
        hidden = model.step(next);
    }
    out
}

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits) as u32;
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    if top_k > 0 && top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let threshold = scaled[idx[top_k - 1]];
        for val in scaled.iter_mut() {
            if *val < threshold {
                *val = f32::NEG_INFINITY;
            }
        }
    }
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for val in scaled.iter_mut() {
        *val = (*val - max).exp();
        sum += *val;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &pr) in scaled.iter().enumerate() {
        acc += pr;
        if acc >= r {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

#[cfg(test)]
mod kv_gen_tests {
    use super::*;
    use crate::config::GlmConfig;
    use crate::model::Glm;

    /// KV-cache generation must produce the SAME greedy tokens as the O(T²)
    /// recompute path (the cache is algebraically exact; logits agree to ~1e-3).
    #[test]
    fn generate_kv_matches_recompute_greedy() {
        let cfg = GlmConfig::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let model = Glm::new(cfg.clone(), 1, 8, &init);
        let prompt = vec![1u32, 5, 3];
        let mut r1 = data::rng::Rng::new(0);
        let recompute = generate(&model, &prompt, 4, 0.0, 0, None, &mut r1);
        let mut r2 = data::rng::Rng::new(0);
        let kv = generate_kv(&model, &prompt, 4, 0.0, 0, None, &mut r2);
        assert_eq!(recompute, kv, "KV greedy generation must equal recompute generation");
    }
}
