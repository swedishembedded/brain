// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive sampling from a trained GPT (temperature + top-k), the
//! generation half of nanogpt's sampler.

use data::rng::Rng;

use crate::model::Gpt;

/// Generate `max_new` tokens continuing `prompt`. Context is cropped to the
/// model's block size. `temperature <= 0` selects greedy argmax; `top_k = 0`
/// disables top-k filtering.
pub fn generate(
    model: &Gpt,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut Rng,
) -> Vec<u32> {
    let block = model.cfg.block_size as usize;
    let vocab = model.cfg.vocab as usize;
    let mut ctx: Vec<u32> = prompt.to_vec();
    let mut out = Vec::with_capacity(max_new);

    for _ in 0..max_new {
        let window: Vec<u32> = if ctx.len() > block {
            ctx[ctx.len() - block..].to_vec()
        } else {
            ctx.clone()
        };
        let logits = model.logits_all(&window);
        // last position's vocab logits
        let last = &logits[logits.len() - vocab..];
        let next = sample_logits(last, temperature, top_k, rng);
        ctx.push(next);
        out.push(next);
    }
    out
}

/// KV-cache generation: the O(T) fast path. Feeds the prompt through the
/// incremental `step` (filling the cache), then samples one token per `step`
/// instead of re-running the whole context each time. Produces the same tokens
/// as [`generate`] for greedy decoding (the cache is algebraically exact). GPT's
/// head is the **untied** `lm_head.weight` (`[vocab, d_model]`, no bias); it is
/// applied on the host to the final-LayerNorm hidden state returned by `step`.
pub fn generate_kv(
    model: &Gpt,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut Rng,
) -> Vec<u32> {
    let vocab = model.cfg.vocab as usize;
    let d = model.cfg.d_model as usize;
    let head = model.read_weight("lm_head.weight"); // [vocab, d], untied, no bias
    let logits_of = |hidden: &[f32]| -> Vec<f32> {
        (0..vocab)
            .map(|o| head[o * d..o * d + d].iter().zip(hidden).map(|(a, b)| a * b).sum())
            .collect()
    };
    model.reset_cache();
    let mut out = Vec::with_capacity(max_new);
    // Feed the prompt; the hidden after the last prompt token gives the first
    // next-token distribution. (Empty prompt -> seed a single id 0.)
    let mut hidden = Vec::new();
    let seed_prompt: &[u32] = if prompt.is_empty() { &[0] } else { prompt };
    for &t in seed_prompt {
        hidden = model.step(t);
    }
    for _ in 0..max_new {
        let next = sample_logits(&logits_of(&hidden), temperature, top_k, rng);
        out.push(next);
        hidden = model.step(next);
    }
    out
}

fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits) as u32;
    }
    // temperature scale
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // top-k: keep only the k largest logits, rest -> -inf
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

    // softmax (numerically stable) then inverse-CDF sample
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in scaled.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut rng = Rng::new(0);
        let logits = [0.1, 5.0, 0.2, -1.0];
        // temperature <= 0 => greedy
        assert_eq!(sample_logits(&logits, 0.0, 0, &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut rng = Rng::new(1);
        let logits = [0.1, 5.0, 0.2, -1.0];
        for _ in 0..20 {
            assert_eq!(sample_logits(&logits, 1.0, 1, &mut rng), 1);
        }
    }
}

#[cfg(test)]
mod kv_gen_tests {
    use super::*;
    use crate::model::{Gpt, GptConfig};
    use std::collections::HashMap;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// KV-cache generation must produce the SAME greedy tokens as the O(T²)
    /// recompute path (the cache is algebraically exact; logits agree to ~1e-3).
    #[test]
    fn generate_kv_matches_recompute_greedy() {
        if gpu_disabled() {
            return;
        }
        // d_model=32 keeps the fused-qkv weight-slice offsets 256B-aligned (see
        // model::kv_step_matches_full_recompute).
        let cfg = GptConfig::tiny();
        let mut rng = Rng::new(1);
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "ln.weight" {
                vec![1.0f32; count] // LayerNorm gain = 1
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.05).collect()
            };
            map.insert(name, v);
        }
        // t sized to block_size so full-window recompute never exceeds the model.
        let model = Gpt::new(cfg.clone(), 1, cfg.block_size, &map);
        let prompt = vec![1u32, 5, 3];
        let mut r1 = Rng::new(0);
        let recompute = generate(&model, &prompt, 16, 0.0, 0, &mut r1);
        let mut r2 = Rng::new(0);
        let kv = generate_kv(&model, &prompt, 16, 0.0, 0, &mut r2);
        assert_eq!(recompute, kv, "KV greedy generation must equal recompute generation");
    }
}
