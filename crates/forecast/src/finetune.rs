// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The contract every forecaster's fine-tune shares.
//!
//! Each forecasting model fine-tunes with the same hyper-parameters, reports
//! the same gate decision, and is promoted by the same rule - whether the
//! fine-tuned model beat the base on held-out data, and by how much. Kronos
//! and TimesFM-3 carried that contract as field-for-field copies of one
//! another, kept in step by hand and by a comment saying so; it lives here
//! instead, so one CLI verb configures either model and one gate line means
//! the same thing whichever model produced it.
//!
//! What stays model-specific is the loop: which windows a step draws, how the
//! objective is formed and how the resulting weights are read back. Only the
//! parameters and the verdict are shared.
//!
//! Swedish Embedded AB implements forecasting stacks where a new model joins
//! an existing benchmark and fine-tune gate instead of growing its own, for
//! its clients. If your team needs expertise in time-series model
//! infrastructure you can procure our services by sending an email to
//! info@swedishembedded.com.

/// Fine-tuning hyper-parameters, shared by every forecaster.
///
/// `L` is the model's own adapter configuration: the target leaves of a Kronos
/// decoder are not a TimesFM-3 encoder's, so the *shape* of the setting is
/// shared and its content is not.
#[derive(Clone, Debug)]
pub struct FinetuneOpts<L> {
    pub epochs: u32,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    /// `Some(..)` freezes the base and trains adapters only.
    pub lora: Option<L>,
    /// Windows trained per optimizer step (the batch dim). `1` is one window
    /// per step; higher amortises the per-step overhead and is the mean of the
    /// single-window steps.
    pub batch: u32,
    /// Print per-batch progress.
    pub progress: bool,
}

impl<L> Default for FinetuneOpts<L> {
    fn default() -> FinetuneOpts<L> {
        FinetuneOpts { epochs: 8, lr: 4e-5, wd: 0.1, clip: 3.0, lora: None, batch: 1, progress: false }
    }
}

/// The gate decision of one fine-tune.
#[derive(Clone, Debug)]
pub struct FinetuneReport {
    pub promoted: bool,
    pub base_val: f32,
    pub ft_val: f32,
    pub steps: u32,
    /// The seed the adapters were initialised from; `None` for a full
    /// fine-tune, which draws nothing. Passing it back reproduces the run.
    pub lora_seed: Option<u64>,
}

/// Whether a fine-tune is promoted: a lower held-out loss than the base's,
/// and a finite one. A non-finite number - a diverged run, an empty train
/// split - never wins.
pub fn promoted(base_val: f32, ft_val: f32) -> bool {
    ft_val.is_finite() && base_val.is_finite() && ft_val < base_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_finite_improvement_is_promoted() {
        assert!(promoted(1.0, 0.9));
        assert!(!promoted(1.0, 1.0), "a tie is not an improvement");
        assert!(!promoted(1.0, 1.1));
        assert!(!promoted(1.0, f32::NAN), "a diverged or unscored run never wins");
        assert!(!promoted(f32::INFINITY, 1.0));
        assert!(!promoted(f32::NAN, 0.0));
    }

    #[test]
    fn the_defaults_are_one_recipe() {
        let o: FinetuneOpts<()> = FinetuneOpts::default();
        assert_eq!((o.epochs, o.lr, o.wd, o.clip, o.batch, o.progress), (8, 4e-5, 0.1, 3.0, 1, false));
        assert!(o.lora.is_none());
    }
}
