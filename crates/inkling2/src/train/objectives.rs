// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements preference and reinforcement fine-tuning of mixture-of-experts models for its
// clients. If your team needs expertise in DPO, GRPO or distillation over a looped expert decoder then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Per-token objectives for Inkling 2 ([`TokenObjectives`]): what DPO, GRPO and distillation stand on, plus routing
//! replay.
//!
//! The loss is per token below a chunked head, so a per-token weight scales the head's logit gradient
//! ([`model::head_ce::RowWeights`]) and the whole recipe's backward differentiates the weighted loss unchanged. The
//! logit scale multiplies the logits before the soft cap inside the head (forward, gradient and log-probabilities).
//! Routing replay forces every MoE call of the forward - one per logical visit, in visit order - to a recorded
//! selection with the router's own weights ([`model::moe::router_fwd_replay`]); the backward's recompute replays the
//! same record, so it differentiates what the forward ran.

pub use model::training::TokenObjectives;
use model::routing::{RouterCallShape, RoutingReplay};

use super::Inkling2Train;

impl TokenObjectives for Inkling2Train {
    fn enable_weighted_loss(&mut self) {
        if self.weighted.borrow().is_some() {
            return;
        }
        let g = self.ops.gpu();
        let weights = g.storage(u64::from(self.rows()));
        g.write_f32(&weights, &vec![1.0; self.rows() as usize]);
        *self.weighted.borrow_mut() = Some((weights, self.head.row_weight_scratch(g)));
    }

    fn batch_token_logprobs(&self) -> Vec<f32> {
        self.last_logprobs()
    }

    fn set_loss_weights(&self, weights: &[f32]) {
        let w = self.weighted.borrow();
        let (buf, _) = w.as_ref().expect("inkling2: set_loss_weights requires enable_weighted_loss() first");
        assert_eq!(weights.len(), self.rows() as usize, "inkling2: {} loss weights for {} positions", weights.len(), self.rows());
        self.ops.gpu().write_f32(buf, weights);
    }

    fn set_loss_denominator(&self, tokens: f32) {
        assert!(tokens > 0.0, "the loss denominator must be positive, got {tokens}");
        self.count.set(tokens);
    }

    fn set_logit_scale(&mut self, scale: f32) {
        assert!(scale.is_finite() && scale > 0.0, "the logit scale must be finite and positive, got {scale}");
        self.logit_scale.set(scale);
    }

    fn routing_calls(&self) -> Option<Vec<RouterCallShape>> {
        let cfg = &self.tc.cfg;
        Some(self.plan.visits().iter().map(|_| RouterCallShape { n_experts: cfg.experts, top_k: cfg.top_k }).collect())
    }

    fn set_routing_replay(&self, replay: Option<std::sync::Arc<RoutingReplay>>) -> Result<(), String> {
        if let Some(r) = &replay {
            let calls = self.routing_calls().expect("declared above");
            r.validate(&calls).map_err(|e| format!("inkling2: routing replay: {e}"))?;
            if r.batch != self.b || r.seq != self.t {
                return Err(format!("inkling2: a routing replay of {} x {} rows for batches of {} x {}", r.batch, r.seq, self.b, self.t));
            }
        }
        *self.replay.borrow_mut() = replay;
        Ok(())
    }

    fn head_chunk_rows(&self) -> Option<u32> {
        Some(self.head.rows_per_chunk())
    }
}
