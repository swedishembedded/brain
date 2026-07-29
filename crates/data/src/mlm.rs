// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Masked-language-model batch corruption (BERT/LFM2.5 recipe).
//!
//! MLM supervision differs from the causal loader in two ways:
//! - **No shift**: `y[t]` is the ORIGINAL token at position `t` (not `t+1`).
//! - **Corruption**: a fraction of positions is selected for prediction; each
//!   selected position's input becomes `<mask>` (80%), a random token (10%),
//!   or stays itself (10%). Unselected positions get [`crate::loader::IGNORE`]
//!   targets and never enter the loss.
//!
//! Windows are packed fixed-length (no padding — bidirectional attention makes
//! unmasked pads unsound), sampled uniformly like the causal `get_batch`.
//! Deterministic via [`crate::rng::Rng`] so runs reproduce exactly.

use crate::loader::IGNORE;
use crate::rng::Rng;

/// MLM corruption parameters. LFM2.5 pre-training used `mask_prob = 0.30`;
/// the 80/10/10 split is the standard BERT recipe.
#[derive(Clone, Debug)]
pub struct MlmConfig {
    /// Fraction of (non-special) positions selected for prediction.
    pub mask_prob: f64,
    /// The `<mask>` token id (LFM2.5: `<|mask|>` = 16).
    pub mask_token: u32,
    /// Of selected positions: probability the input becomes `<mask>`.
    pub p_mask: f64,
    /// Of selected positions: probability the input becomes a random token.
    pub p_random: f64,
    /// Vocabulary size for random replacement.
    pub vocab: u32,
    /// Token ids never selected for prediction (BOS/EOS/pad and friends).
    pub special_ids: Vec<u32>,
}

impl MlmConfig {
    pub fn new(mask_token: u32, vocab: u32) -> MlmConfig {
        MlmConfig {
            mask_prob: 0.30,
            mask_token,
            p_mask: 0.8,
            p_random: 0.1,
            vocab,
            special_ids: Vec::new(),
        }
    }
}

/// Corrupt one window in place: returns `(x, y)` with `x` the corrupted input
/// and `y` the unshifted targets (`IGNORE` at unselected positions).
pub fn corrupt(window: &[u32], cfg: &MlmConfig, rng: &mut Rng) -> (Vec<u32>, Vec<i32>) {
    let mut x = window.to_vec();
    let mut y = vec![IGNORE; window.len()];
    for (i, &tok) in window.iter().enumerate() {
        if cfg.special_ids.contains(&tok) {
            continue;
        }
        if rng.next_f64() >= cfg.mask_prob {
            continue;
        }
        y[i] = tok as i32;
        let r = rng.next_f64();
        if r < cfg.p_mask {
            x[i] = cfg.mask_token;
        } else if r < cfg.p_mask + cfg.p_random {
            x[i] = (rng.next_u64() % cfg.vocab as u64) as u32;
        } // else: keep the original token (still supervised)
    }
    (x, y)
}

/// Sample a `[batch, block]` MLM batch from a token stream: uniform windows
/// (like the causal loader) but corrupted + unshifted. Flattened row-major.
pub fn get_mlm_batch(
    data: &[u32],
    batch: usize,
    block: usize,
    cfg: &MlmConfig,
    rng: &mut Rng,
) -> (Vec<u32>, Vec<i32>) {
    assert!(data.len() > block, "dataset ({}) smaller than block {block}", data.len());
    let mut xs = Vec::with_capacity(batch * block);
    let mut ys = Vec::with_capacity(batch * block);
    for _ in 0..batch {
        let start = (rng.next_u64() % (data.len() - block) as u64) as usize;
        let (x, y) = corrupt(&data[start..start + block], cfg, rng);
        xs.extend(x);
        ys.extend(y);
    }
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MlmConfig {
        MlmConfig { special_ids: vec![0, 1], ..MlmConfig::new(16, 1000) }
    }

    #[test]
    fn corruption_is_deterministic_and_unshifted() {
        let window: Vec<u32> = (2..514).collect();
        let (x1, y1) = corrupt(&window, &cfg(), &mut Rng::new(7));
        let (x2, y2) = corrupt(&window, &cfg(), &mut Rng::new(7));
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
        // Unshifted: every supervised target is the window's ORIGINAL token.
        for (i, &t) in y1.iter().enumerate() {
            if t != IGNORE {
                assert_eq!(t as u32, window[i]);
            }
        }
    }

    #[test]
    fn corruption_rates_match_config() {
        let window: Vec<u32> = (2..8194).map(|i| i % 900 + 2).collect();
        let c = cfg();
        let (x, y) = corrupt(&window, &c, &mut Rng::new(42));
        let n = window.len() as f64;
        let selected = y.iter().filter(|&&t| t != IGNORE).count() as f64;
        assert!((selected / n - 0.30).abs() < 0.03, "selection rate {}", selected / n);
        let masked = x.iter().filter(|&&t| t == c.mask_token).count() as f64;
        assert!((masked / selected - 0.8).abs() < 0.06, "mask share {}", masked / selected);
        // ~10% of selected keep their original input yet stay supervised.
        let kept = window
            .iter()
            .zip(&x)
            .zip(&y)
            .filter(|((&o, &xi), &yi)| yi != IGNORE && xi == o)
            .count() as f64;
        assert!((kept / selected - 0.1).abs() < 0.05, "keep share {}", kept / selected);
    }

    #[test]
    fn specials_are_never_selected() {
        let mut window = vec![5u32; 256];
        window[0] = 1; // BOS
        window[100] = 0; // pad
        let (x, y) = corrupt(&window, &cfg(), &mut Rng::new(3));
        assert_eq!(x[0], 1);
        assert_eq!(y[0], IGNORE);
        assert_eq!(x[100], 0);
        assert_eq!(y[100], IGNORE);
    }

    #[test]
    fn batch_shape_and_range() {
        let data: Vec<u32> = (0..4096).map(|i| i % 997).collect();
        let (x, y) = get_mlm_batch(&data, 4, 128, &cfg(), &mut Rng::new(9));
        assert_eq!(x.len(), 4 * 128);
        assert_eq!(y.len(), 4 * 128);
        assert!(x.iter().all(|&t| t < 1000));
    }
}
