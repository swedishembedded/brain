// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A short causal-LM training run on an Ouro checkpoint itself: the released
//! objective (every pass's cross-entropy weighted by its exit probability,
//! summed, averaged over supervised rows) through the analytic backward, one
//! AdamW step per batch over the same rows - a falling loss is measurable
//! evidence that the shared looped stack trains.
//!
//! Swedish Embedded AB implements training for looped and recurrent-depth
//! language models for its clients. If your team needs expertise in
//! adapting such models to its own text then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::Path;

use crate::model::{Ouro, IGNORE};
use crate::OuroConfig;

/// One run.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainRun {
    pub steps: u32,
    /// Tokens per row and rows per step.
    pub seq: u32,
    pub batch: u32,
    pub lr: f32,
    /// Global-norm clip on the averaged gradient; 0 disables it. The released
    /// checkpoint's logits are large, so an unclipped first step can blow up.
    pub clip: f32,
    pub activations: Option<model::saved::SavedPolicy>,
}

/// What a run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Trained {
    pub first_loss: f32,
    /// The loss of the rows after the last step.
    pub final_loss: f32,
    pub steps: u32,
}

/// Causal-LM rows from `corpus` (repeated until it fills them): every row
/// predicts the next token, its last position unsupervised.
pub fn rows(corpus: &[u32], batch: u32, seq: u32) -> Result<(Vec<u32>, Vec<u32>), String> {
    if corpus.len() < 2 {
        return Err(format!("the training data is {} token(s); it needs at least 2", corpus.len()));
    }
    let (b, t) = (batch.max(1) as usize, seq.max(2) as usize);
    let need = b * t + 1;
    let mut text = corpus.to_vec();
    while text.len() < need {
        text.extend_from_slice(corpus);
    }
    let (mut tokens, mut targets) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t));
    for row in 0..b {
        let start = row * t;
        tokens.extend_from_slice(&text[start..start + t]);
        targets.extend_from_slice(&text[start + 1..start + 1 + t]);
        let last = targets.len() - 1;
        targets[last] = IGNORE;
    }
    Ok((tokens, targets))
}

/// The configuration of the checkpoint at `path`: a Hugging Face directory
/// or a brain checkpoint.
pub fn config_of(path: &Path) -> Result<OuroConfig, String> {
    if path.is_dir() {
        crate::import::config_from_hf(&std::fs::read_to_string(path.join("config.json")).map_err(|e| format!("{}: {e}", path.display()))?)
    } else {
        let reader = checkpoint::weightio::WeightReader::open(&path.to_string_lossy()).map_err(|e| format!("{}: {e}", path.display()))?;
        OuroConfig::from_json(&reader.config())
    }
}

/// Train the checkpoint at `path` on `corpus` per `run`, writing it to
/// `save` when given; `on_step(step, loss)` sees every step and stops the
/// run by returning false.
pub fn train(path: &Path, corpus: &[u32], run: &TrainRun, save: Option<&Path>, on_step: &mut dyn FnMut(u32, f32) -> bool) -> Result<Trained, String> {
    let (tokens, targets) = rows(corpus, run.batch, run.seq)?;
    let cfg = config_of(path)?;
    let src: Box<dyn checkpoint::TensorSource> = if path.is_dir() {
        Box::new(crate::import::open_dir(path)?.1)
    } else {
        Box::new(checkpoint::weightio::WeightReader::open(&path.to_string_lossy()).map_err(|e| format!("{}: {e}", path.display()))?)
    };
    let mut m = Ouro::new_train(cfg, run.batch.max(1), run.seq.max(2), src.as_ref());
    if let Some(policy) = run.activations {
        model::Model::set_saved_policy(&mut m, policy)?;
    }
    model::Model::set_batch(&m, model::Batch::Lm { tokens: &tokens, targets: &targets });
    let mut first = None;
    let mut steps = 0;
    for step in 0..run.steps {
        model::Model::zero_grads(&m);
        let loss = model::Model::forward(&m);
        model::Model::backward(&m);
        // AdamW's bias correction is 1-based: step 0 would divide by zero.
        model::Model::adamw_step(&m, step + 1, run.lr, 0.0, model::Adam::default(), (run.clip > 0.0).then_some(run.clip), 1.0);
        first.get_or_insert(loss);
        steps = step + 1;
        if !on_step(steps, loss) {
            break;
        }
    }
    if let Some(out) = save {
        model::Model::save(&m, &out.to_string_lossy());
    }
    let final_loss = model::Model::forward(&m);
    Ok(Trained { first_loss: first.unwrap_or(final_loss), final_loss, steps })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows are shifted by one token, the corpus repeats to fill them, and
    /// each row's last target is unsupervised.
    #[test]
    fn rows_predict_the_next_token_and_leave_the_last_unsupervised() {
        let (tokens, targets) = rows(&[1, 2, 3], 2, 3).unwrap();
        assert_eq!(tokens, [1, 2, 3, 1, 2, 3]);
        assert_eq!(targets, [2, 3, IGNORE, 2, 3, IGNORE]);
        assert!(rows(&[7], 1, 4).unwrap_err().contains("at least 2"));
    }
}
