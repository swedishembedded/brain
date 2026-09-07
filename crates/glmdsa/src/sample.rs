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
        let logits = model.logits_all_compact(&window);
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

// Note on what is NOT claimed here: `generate_kv`'s sampled path (temperature
// > 0, the served default of 0.8/top_k=40) is never asserted bit-identical to
// plain `generate`. `generate_kv` applies GLM's untied `lm_head` on the host
// in a scalar loop (see its own doc comment), which agrees with the device
// path's logits to only ~1e-3 - enough to flip which of two close candidates
// a sampler draws, even though both walk the exact same RNG draw sequence
// (one `rng.next_f32()` per emitted token, identically ordered in both
// functions - see `sample_logits`). Bit-identity between the two functions is
// only claimable at `temperature=0` (greedy, `generate_kv_matches_recompute_
// greedy` below); the sampled-path test below instead pins `generate_kv`'s
// own contract (valid token ids, and self-reproducible given the same seed).
#[cfg(test)]
mod kv_gen_tests {
    use super::*;
    use crate::config::GlmConfig;
    use crate::model::Glm;

    /// KV-cache generation must produce the SAME greedy tokens as the O(T²)
    /// recompute path (the cache is algebraically exact; logits agree to
    /// ~1e-3, never enough to flip an argmax). Two prompts, >=32 tokens each,
    /// so this is a real regression net rather than the original 4-token/
    /// one-prompt smoke.
    #[test]
    fn generate_kv_matches_recompute_greedy() {
        let cfg = GlmConfig::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let model = Glm::new_on(gpu_core::testgpu::dev(crate::model::PIPELINES), cfg.clone(), 1, 64, &init);
        let mut compared = 0usize;
        for prompt in [vec![1u32, 5, 3], vec![2u32, 6, 4, 1, 7, 3, 5]] {
            let mut r1 = data::rng::Rng::new(0);
            let recompute = generate(&model, &prompt, 32, 0.0, 0, None, &mut r1);
            let mut r2 = data::rng::Rng::new(0);
            let kv = generate_kv(&model, &prompt, 32, 0.0, 0, None, &mut r2);
            assert_eq!(recompute, kv, "KV greedy generation must equal recompute generation for prompt {prompt:?}");
            compared += recompute.len();
        }
        println!("generate_kv_matches_recompute_greedy: compared {compared} tokens across 2 prompts, 0 differing token ids");
        assert!(compared >= 64, "expected >=32 tokens compared per prompt across 2 prompts, got {compared}");
    }

    /// The SAMPLED path's own contract (temp=0.8, top_k=40, the served
    /// default) - not bit-identity vs `generate` (see the module note above
    /// for why that is never claimed). Every emitted id must be a valid
    /// vocab index, and `generate_kv` must be byte-identical to ITSELF across
    /// two runs seeded identically (the RNG is the only source of
    /// nondeterminism in this path - no other hidden state should leak in).
    #[test]
    fn generate_kv_sampled_path_is_well_formed_and_self_reproducible() {
        let cfg = GlmConfig::tiny();
        let init = crate::init::init_weights(&cfg, 11);
        let model = Glm::new_on(gpu_core::testgpu::dev(crate::model::PIPELINES), cfg.clone(), 1, 64, &init);
        let vocab = cfg.vocab as u32;
        let prompt = vec![2u32, 6, 4, 1, 7, 3, 5];
        let max_new = 32;

        let mut r1 = data::rng::Rng::new(42);
        let run1 = generate_kv(&model, &prompt, max_new, 0.8, 40, None, &mut r1);
        let mut r2 = data::rng::Rng::new(42);
        let run2 = generate_kv(&model, &prompt, max_new, 0.8, 40, None, &mut r2);

        println!("generate_kv_sampled_path_is_well_formed_and_self_reproducible: {} tokens/run, vocab={vocab}", run1.len());
        assert_eq!(run1.len(), max_new, "sampled generate_kv must emit max_new tokens with no eos set");
        for &id in &run1 {
            assert!(id < vocab, "sampled token id {id} must be < vocab ({vocab})");
        }
        assert_eq!(run1, run2, "generate_kv must be byte-identical to itself given the same seed");
    }
}
