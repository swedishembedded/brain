// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One generic training/eval/sample loop over any [`Model`](crate::Model)
//! (ADR §3). [`fit`] is `gpt::train::train` lifted to `M: Model` — same control
//! flow (cosine-with-warmup LR, grad accumulation with averaging, periodic eval,
//! resumable atomic checkpointing); [`generate`] is `gpt::sample::generate`
//! lifted to any token-head model.
//!
//! PR-1 lands [`FitOpts`] + [`cosine_lr`] (moved here from `gpt::train`); the
//! generic `fit`/`generate` bodies arrive in PR-2.

/// Training-loop options (the CLI-facing hyperparameters), independent of any
/// particular architecture. This is `gpt::train::TrainOpts` lifted to the model
/// crate.
#[derive(Clone, Debug)]
pub struct FitOpts {
    pub steps: u32,
    pub batch_size: u32,
    pub block_size: u32,
    pub lr: f32,
    pub min_lr: f32,
    pub warmup: u32,
    pub decay_iters: u32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub grad_accum: u32,
    pub eval_interval: u32,
    pub eval_batches: u32,
    pub seed: u64,
    /// Mask loss up to & including this char (e.g. `'='` for calculator).
    pub mask_before: Option<char>,
    pub mask_per_line: bool,
    pub align_to_lines: bool,
}

impl Default for FitOpts {
    fn default() -> Self {
        FitOpts {
            steps: 2000,
            batch_size: 32,
            block_size: 64,
            lr: 3e-4,
            min_lr: 3e-5,
            warmup: 100,
            decay_iters: 2000,
            weight_decay: 0.1,
            grad_clip: 1.0,
            grad_accum: 1,
            eval_interval: 250,
            eval_batches: 20,
            seed: 1337,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: false,
        }
    }
}

/// Cosine LR schedule with linear warmup (nanogpt's `get_lr`). Moved verbatim
/// from `gpt::train::cosine_lr`.
pub fn cosine_lr(it: u32, opts: &FitOpts) -> f32 {
    if it < opts.warmup {
        return opts.lr * (it + 1) as f32 / opts.warmup.max(1) as f32;
    }
    if it >= opts.decay_iters {
        return opts.min_lr;
    }
    let ratio = (it - opts.warmup) as f32 / (opts.decay_iters - opts.warmup).max(1) as f32;
    let coeff = 0.5 * (1.0 + (std::f32::consts::PI * ratio).cos());
    opts.min_lr + coeff * (opts.lr - opts.min_lr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_lr_warmup_peak_and_floor() {
        let o = FitOpts { lr: 1.0, min_lr: 0.1, warmup: 10, decay_iters: 100, ..Default::default() };
        assert!(cosine_lr(0, &o) < cosine_lr(5, &o)); // ramping up
        assert!((cosine_lr(9, &o) - 1.0).abs() < 0.11); // near peak at end of warmup
        assert!((cosine_lr(200, &o) - 0.1).abs() < 1e-6); // floor after decay
    }
}
