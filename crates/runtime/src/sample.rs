// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Token sampling from a vocab logit row — temperature + top-k, the same scheme
//! as `gpt::sample`, lifted here so [`crate::StreamPump`] can run against any
//! [`crate::InferModel`] (real GPT or a fake) without depending on gpt internals.

/// A tiny deterministic PRNG (SplitMix64), so sampling is reproducible from a
/// seed without pulling in the `data` crate.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// Sample a token index from `logits`. `temperature <= 0` → greedy argmax;
/// `top_k == 0` disables top-k filtering. Mirrors `gpt::sample::sample_logits`.
pub fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
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
        assert_eq!(sample_logits(&[0.1, 5.0, 0.2, -1.0], 0.0, 0, &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            assert_eq!(sample_logits(&[0.1, 5.0, 0.2, -1.0], 1.0, 1, &mut rng), 1);
        }
    }
}
