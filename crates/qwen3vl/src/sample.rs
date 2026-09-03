// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Temperature/top-k/top-p sampling over a `[vocab]` logit row, for
//! [`crate::model::Qwen3Vl::generate_timed`]'s incremental decode loop.
//!
//! A small, deliberate duplication of `qwen3::sample`'s private
//! `argmax`/`sample_logits` (identical algorithm and contract) rather than a
//! cross-model dependency for ~40 lines of elementwise math - the same
//! convention every model crate in this repo already follows for its own
//! sampling tail (`qwen3::sample`, `qwen35moe::sample`, `qwen35::sample`,
//! `gpt2::sample`, `glmdsa::sample`).
//!
//! Swedish Embedded AB implements autoregressive decoding for its clients. If
//! your team needs expertise in LLM/VLM inference and serving then you can
//! procure our services by sending an email to info@swedishembedded.com.

use data::rng::Rng;

/// Total order over `f32` for sampling comparisons, with every NaN treated as
/// strictly least-preferred. Mirrors `qwen3::sample::cmp_nan_last` - a bare
/// `partial_cmp` panics on a NaN operand, and `total_cmp` alone would rank a
/// positive-payload NaN ABOVE `+inf`, letting a NaN logit win argmax/top-k
/// outright instead of losing it.
#[inline]
fn cmp_nan_last(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.total_cmp(&b),
    }
}

/// Greedy pick over one `[vocab]` logit row; ties (and an all-NaN row) go to
/// the lowest index.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if cmp_nan_last(logits[i], logits[best]) == std::cmp::Ordering::Greater {
            best = i;
        }
    }
    best as u32
}

/// Temperature + top-k + nucleus (top-p) sampling - identical contract to
/// `qwen3::sample::sample_logits`: `temperature <= 0.0` is greedy argmax,
/// `top_k == 0` disables top-k filtering, `top_p` outside `(0,1)` disables
/// nucleus filtering. Robust to a NaN logit (treated as least-preferred, like
/// `-inf`), so a serving process never panics on one.
pub fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| if l.is_nan() { f32::NEG_INFINITY } else { l / temperature }).collect();
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
    if top_p > 0.0 && top_p < 1.0 && sum > 0.0 {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| cmp_nan_last(scaled[b], scaled[a]));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_at_zero_temperature_picks_the_max() {
        let mut rng = Rng::new(0);
        assert_eq!(sample_logits(&[0.1, 5.0, 0.2, -1.0], 0.0, 0, 1.0, &mut rng), 1);
    }

    #[test]
    fn ties_break_to_the_lowest_index() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 0.0]), 1);
    }

    #[test]
    fn top_k_one_is_deterministic_like_greedy() {
        let mut rng = Rng::new(42);
        for _ in 0..8 {
            assert_eq!(sample_logits(&[0.1, 5.0, 0.2, -1.0], 1.0, 1, 1.0, &mut rng), 1);
        }
    }

    #[test]
    fn nucleus_top_p_can_collapse_to_a_single_token() {
        let mut rng = Rng::new(7);
        let flat = [0.0f32; 6];
        let only = sample_logits(&flat, 1.0, 0, 0.01, &mut rng);
        for _ in 0..10 {
            assert_eq!(sample_logits(&flat, 1.0, 0, 0.01, &mut rng), only, "nucleus keeps a single token");
        }
    }

    #[test]
    fn a_nan_logit_never_wins_over_a_real_value() {
        let logits = [1.0f32, f32::NAN, 2.0, 0.5];
        let mut rng = Rng::new(1);
        assert_eq!(sample_logits(&logits, 0.0, 0, 1.0, &mut rng), 2, "greedy argmax must pick the real max, not the NaN");
    }

    #[test]
    fn temperature_changes_the_sampled_distribution_across_seeds() {
        // A non-trivial, non-degenerate logit row: different seeds at a real
        // temperature must not all collapse onto the same token, or sampling
        // would be indistinguishable from greedy argmax.
        let logits = [1.0f32, 1.05, 0.95, 1.02, 0.98, 1.03];
        let mut seen = std::collections::HashSet::new();
        for seed in 0..32u64 {
            let mut rng = Rng::new(seed);
            seen.insert(sample_logits(&logits, 1.0, 0, 1.0, &mut rng));
        }
        assert!(seen.len() > 1, "temperature=1.0 sampling over near-uniform logits must vary by seed");
    }
}
