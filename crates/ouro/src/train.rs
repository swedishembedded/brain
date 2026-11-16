// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A short causal-LM training run on an Ouro checkpoint itself: the released
//! objective (every pass's cross-entropy weighted by its exit probability,
//! summed, averaged over supervised rows) through the analytic backward, run by
//! the one training loop ([`model::fit_controlled`]) over windows of the given
//! token stream - a falling loss is measurable evidence that the shared looped
//! stack trains.
//!
//! Swedish Embedded AB implements training for looped and recurrent-depth
//! language models for its clients. If your team needs expertise in
//! adapting such models to its own text then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::Path;

use crate::model::Ouro;
use crate::OuroConfig;

/// One run.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainRun {
    pub steps: u32,
    /// Tokens per row and rows per step.
    pub seq: u32,
    pub batch: u32,
    /// The learning rate, held constant over the run.
    pub lr: f32,
    /// Global-norm clip on the averaged gradient; 0 disables it. The released
    /// checkpoint's logits are large, so an unclipped first step can blow up.
    pub clip: f32,
    pub activations: Option<model::saved::SavedPolicy>,
    /// The seed of the window draws.
    pub seed: u64,
}

/// What a run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Trained {
    /// The loss estimated before the first step.
    pub first_loss: f32,
    /// The last step's training loss.
    pub final_loss: f32,
    pub steps: u32,
}

/// The token stream a run of `seq`-token rows draws its windows from: `corpus`, repeated until a window fits.
pub fn stream(corpus: &[u32], seq: u32) -> Result<Vec<u32>, String> {
    if corpus.len() < 2 {
        return Err(format!("the training data is {} token(s); it needs at least 2", corpus.len()));
    }
    let need = seq.max(2) as usize + 2;
    let mut text = corpus.to_vec();
    while text.len() < need {
        text.extend_from_slice(corpus);
    }
    Ok(text)
}

/// The configuration of the checkpoint at `path`: a Hugging Face directory
/// or a brain checkpoint.
pub fn config_of(path: &Path) -> Result<OuroConfig, String> {
    crate::finetune::open(path).map(|(cfg, _)| cfg)
}

/// Train the checkpoint at `path` on `corpus` per `run`, writing it to
/// `save` when given; `on_step(step, loss)` sees every step and stops the
/// run by returning false.
pub fn train(path: &Path, corpus: &[u32], run: &TrainRun, save: Option<&Path>, on_step: &mut dyn FnMut(u32, f32) -> bool) -> Result<Trained, String> {
    let (cfg, src) = crate::finetune::open(path)?;
    let (batch, seq) = (run.batch.max(1), run.seq.max(2));
    let batch_cfg = data::loader::BatchConfig { batch_size: batch as usize, block_size: seq as usize, mask_before_token: None, mask_per_line: false, align_to_lines: false, newline_token: None };
    let train = data::loader::TokenDataset::new(stream(corpus, seq)?, &batch_cfg);
    let val = data::loader::TokenDataset::new(Vec::new(), &batch_cfg);
    let opts = model::FitOpts {
        steps: run.steps,
        batch_size: batch,
        block_size: seq,
        lr: run.lr,
        min_lr: run.lr,
        warmup: 0,
        decay_iters: 0,
        weight_decay: 0.0,
        grad_clip: run.clip,
        eval_interval: 0,
        eval_batches: 0,
        seed: run.seed,
        checkpoint_secs: 0,
        ..model::FitOpts::default()
    };
    let mut m = Ouro::new_train_with(cfg, batch, seq, &*src, opts.gemm_precision);
    if let Some(policy) = run.activations {
        model::Model::set_saved_policy(&mut m, policy)?;
    }
    let mut watch = |s: &model::StepReport| on_step(s.step, s.loss);
    let control = model::FitControl { on_step: Some(&mut watch), ..Default::default() };
    let (report, _) = model::fit_controlled(m, model::causal_lm::<Ouro>(train, val, batch_cfg, None), &opts, save, control).map_err(|e| e.to_string())?;
    Ok(Trained { first_loss: report.initial_loss, final_loss: report.final_loss.unwrap_or(report.initial_loss), steps: report.steps_completed })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corpus repeats until a window of `seq + 1` tokens fits, and a corpus of one token cannot train.
    #[test]
    fn the_stream_holds_a_window_and_a_single_token_is_refused() {
        assert_eq!(stream(&[1, 2, 3], 3).unwrap(), [1, 2, 3, 1, 2, 3]);
        assert!(stream(&[7], 4).unwrap_err().contains("at least 2"));
    }
}
