// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-token objectives for the Ripple trainer: the weighted loss, the log-probability readout and the logit
//! scale that preference optimisation, reward-weighted training and distillation run on.
//!
//! Swedish Embedded AB implements training stacks where preference optimisation, reward-driven training and
//! distillation work on every model the stack supports. If your team needs expertise in post-training of large
//! language models then you can procure our services by sending an email to info@swedishembedded.com.

use model::lossw::WeightedCe;
use model::training::TokenObjectives;

use super::Ripple;

impl TokenObjectives for Ripple {
    fn enable_weighted_loss(&mut self) {
        assert!(self.train, "ricursive: a weighted loss needs the training build");
        if self.weighted.is_none() {
            let (n, v) = (u64::from(self.rows()), self.cfg.model.vocab_size as u64);
            // The chunked head scales one chunk of the logit gradient at a time; an inference build scales all of it.
            let w = match &self.head_ce {
                Some(head) => WeightedCe::new_chunked(&self.gpu, n, v, u64::from(head.rows_per_chunk())),
                None => WeightedCe::new(&self.gpu, n, v),
            };
            w.write(&self.gpu, &vec![1.0; n as usize]);
            self.weighted = Some(w);
            self.record();
        }
    }

    fn batch_token_logprobs(&self) -> Vec<f32> {
        self.gpu.read(&self.ce_rows, self.rows() as usize).iter().map(|&nll| -nll).collect()
    }

    fn set_loss_weights(&self, weights: &[f32]) {
        self.weighted.as_ref().expect("ricursive: set_loss_weights needs enable_weighted_loss() first").write(&self.gpu, weights);
    }

    fn set_loss_denominator(&self, tokens: f32) {
        assert!(tokens.is_finite() && tokens > 0.0, "ricursive: the loss denominator must be positive, got {tokens}");
        self.count.set(tokens);
    }

    fn enable_sparse_targets(&mut self, k: u32) -> Result<(), String> {
        let had = self.sparse.is_some();
        let (weighted, rows) = (self.weighted.is_some(), self.rows());
        model::head_ce::SparseTargets::enable(&mut self.sparse, &self.gpu, self.head_ce.as_ref(), weighted, rows, k, "ricursive")?;
        if !had {
            // The recorded passes read the extra targets' buffers.
            self.record();
        }
        Ok(())
    }

    fn set_sparse_target_ids(&self, ids: &[u32]) -> Result<(), String> {
        let t = self.sparse.as_ref().ok_or("ricursive: set_sparse_target_ids needs enable_sparse_targets first")?;
        t.write_ids_checked(&self.gpu, ids, self.cfg.model.vocab_size as u32)
    }

    fn set_sparse_target_weights(&self, coefs: &[f32]) -> Result<(), String> {
        self.sparse.as_ref().ok_or("ricursive: set_sparse_target_weights needs enable_sparse_targets first")?.write_coefs_checked(&self.gpu, coefs)
    }

    fn batch_sparse_target_logprobs(&self) -> Option<Vec<f32>> {
        self.sparse.as_ref().map(|t| t.read_logprobs(&self.gpu, self.rows()))
    }

    fn set_logit_scale(&mut self, scale: f32) {
        assert!(scale.is_finite() && scale > 0.0, "ricursive: the logit scale must be finite and positive, got {scale}");
        if scale != self.logit_scale {
            self.logit_scale = scale;
            self.record();
        }
    }
}
