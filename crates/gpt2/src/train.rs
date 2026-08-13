// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPT training entry point. The training loop itself now lives in the
//! architecture-agnostic `model` crate (ADR 0001 §3); this is a thin wrapper
//! that delegates to [`model::train::fit`] over [`crate::model::Gpt`], keeping
//! `gpt2::train::train`'s signature and behavior identical for existing callers
//! (the CLI and `crates/bench`).

use std::path::Path;

use crate::model::GptConfig;

/// Training hyperparameters (CLI-facing). This is now `model::FitOpts` — the
/// architecture-agnostic training-loop options moved to the `model` crate
/// (ADR §3); kept as `TrainOpts` here for source compatibility.
pub type TrainOpts = model::FitOpts;

/// Cosine LR schedule with linear warmup (nanogpt's `get_lr`). Moved to
/// `model::train::cosine_lr` and re-exported here for source compatibility.
pub use model::cosine_lr;

/// Train a GPT on the dataset in `dir`, writing the final checkpoint to `out`.
/// `cfg` carries the architecture; its `vocab`/`block_size` are overridden from
/// the dataset and `opts`. Returns `(initial_train_loss, final_train_loss)`.
///
/// This delegates to the generic [`model::train::fit`] over [`Gpt`] — the loop
/// body (cosine LR, warmup, grad-accum, grad-clip, periodic eval, resumable
/// atomic checkpointing) is shared with every other `Model` and is no longer
/// duplicated here.
pub fn train(dir: &Path, cfg: GptConfig, opts: &TrainOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    model::train::fit::<crate::model::Gpt>(dir, cfg, opts, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_lr_warmup_peak_and_floor() {
        let o = TrainOpts {
            lr: 1.0,
            min_lr: 0.1,
            warmup: 10,
            decay_iters: 100,
            ..Default::default()
        };
        assert!(cosine_lr(0, &o) < cosine_lr(5, &o)); // ramping up
        assert!((cosine_lr(9, &o) - 1.0).abs() < 0.11); // near peak at end of warmup
        assert!((cosine_lr(200, &o) - 0.1).abs() < 1e-6); // floor after decay
    }

    #[test]
    fn trains_calculator_and_reduces_loss() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        // Prepare a tiny calculator dataset.
        let dir = std::env::temp_dir().join(format!("brain_gpt_train_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        data::prepare::prepare(data::prepare::Dataset::Calculator, &dir, 4000, 1).unwrap();

        let cfg = GptConfig { vocab: 0, block_size: 32, n_layers: 2, d_model: 64, n_heads: 4, d_ff: 256 };
        let opts = TrainOpts {
            steps: 200,
            batch_size: 16,
            block_size: 32,
            lr: 3e-3,
            warmup: 20,
            decay_iters: 200,
            eval_interval: 0,
            mask_before: Some('='),
            mask_per_line: true,
            ..Default::default()
        };
        let (initial, final_loss) = train(&dir, cfg, &opts, None).unwrap();
        assert!(final_loss.is_finite() && initial.is_finite());
        assert!(final_loss < initial * 0.9, "loss did not drop: {initial} -> {final_loss}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
