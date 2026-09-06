// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `Objective` seam (self-improve roadmap P8): the one piece of the
//! training step loop that genuinely differs between plain causal-LM SFT
//! (`train::fit`), weighted/reward-driven training (`crates/rl`'s
//! `fit_weighted`), and future regimes (DPO, GRPO, distillation, replay).
//!
//! Everything else - LR schedule, grad accumulation and its averaging scale,
//! global-norm clipping, AdamW, resume, wall-clock checkpointing, eval
//! cadence, the final save - stays in one loop, owned by [`crate::train::fit_with`],
//! never by the objective, so it cannot drift between objectives the way the
//! five copy-pasted step loops this phase replaces already had (one of them,
//! `rl::fit_weighted`, had silently dropped the char-vocab save call).
//!
//! Swedish Embedded AB implements training-loop infrastructure that adds a
//! new training regime without copy-pasting the step loop, for teams whose
//! model roadmap outgrows a single fixed training recipe. If your team needs
//! expertise in composing training regimes (SFT, weighted/reward training,
//! preference optimization) over a shared execution engine, you can procure
//! our services by sending an email to info@swedishembedded.com.

use data::rng::Rng;

use crate::Model;

/// One training objective: the batch-drawing, forward/backward, and
/// (optional) eval/logging behavior that varies per training regime.
///
/// The seam is deliberately at the level of "one micro-step" (set batch,
/// forward, decide weights, backward), not "produce a `Batch`" - a
/// pairwise/grouped objective's per-token weights are a function of that
/// forward's own output (e.g. DPO/GRPO reading back per-token logprobs), so
/// a `Batch`-producing seam could not express them; a micro-step-producing
/// one can.
pub trait Objective<M: Model> {
    /// Short, stable identifier for this training regime (e.g. `"causal_lm"`,
    /// `"weighted_lm"`) - for logging and future run-provenance records.
    fn regime(&self) -> &'static str;

    /// One-time setup after the model is constructed, before the first
    /// [`Objective::micro_step`] - e.g. opting the model into weighted-loss
    /// support via [`Model::enable_weighted_loss`].
    fn prepare(&mut self, _model: &mut M) {}

    /// Run exactly one micro-step: draw a batch, upload it via
    /// [`Model::set_batch`], forward, decide any per-position loss weights,
    /// and backward - returning the scalar loss [`Model::forward`] produced.
    /// Called once per grad-accumulation slot, and (with a throwaway `rng`)
    /// a few times before training starts to estimate the initial loss.
    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32;

    /// Forward-only evaluation loss averaged over `batches` held-out
    /// batches, or `None` if this objective has no eval split to report
    /// against (default: unsupported, so the training loop simply skips the
    /// periodic eval line).
    fn eval(&mut self, _model: &M, _rng: &mut Rng, _batches: u32) -> Option<f32> {
        None
    }

    /// Extra scalar metrics to print alongside the periodic eval line
    /// (default: none).
    fn metrics(&self) -> Vec<(&'static str, f32)> {
        Vec::new()
    }

    /// Char-tokenizer vocab to embed in the checkpoint via
    /// [`Model::save_with_itos`], if this objective's dataset carries one
    /// (default: none). The training loop always asks for this and always
    /// calls `save_with_itos` - never `save` - so no objective can silently
    /// forget to carry it, the way `rl::fit_weighted` used to.
    fn itos(&self) -> Option<&[char]> {
        None
    }

    /// The `(rows, block)` shape this objective's [`Model::set_batch`] calls
    /// will actually upload, for a caller that CONSTRUCTS the model on the
    /// objective's behalf rather than being handed one already sized (today:
    /// `rl::improve::cycle`). `None` (the default) means "one row at the
    /// checkpoint's own block size" - what every rollout-driven objective
    /// here uploads, and what those callers hardcoded before this existed.
    ///
    /// This is not a preference the caller may override: a model built for
    /// `b` rows has a token buffer of exactly `b * t` elements, and uploading
    /// a differently-sized batch into it either overruns the buffer or leaves
    /// stale rows in the forward - a silent wrong-loss, not a crash. So the
    /// objective, which is the only thing that knows what it uploads, is what
    /// declares it.
    fn batch_shape(&self) -> Option<(u32, u32)> {
        None
    }
}

/// Forwarding impl so a caller that must pick between two structurally
/// DIFFERENT concrete objectives at runtime can still feed the one seam every
/// consumer offers: [`crate::train::fit_with`] (and everything layered on it)
/// takes `O: Objective<M>` BY VALUE and is monomorphized per call site, so a
/// runtime choice has to be erased into a single type first. `Box<dyn
/// Objective<M> + '_>` is that type, and this impl is what makes it usable.
///
/// `?Sized` so it covers both the trait object and an ordinary boxed concrete
/// objective. The non-`'static` case is load-bearing rather than incidental:
/// an objective is free to BORROW its environment (`rl`'s GRPO objective
/// borrows the curriculum it draws tasks from), so pinning this to `'static`
/// would exclude exactly the objectives a runtime choice is usually between.
///
/// It lives here, next to the trait, because it has to: `Box` is
/// `#[fundamental]`, so the same impl written in a downstream crate would
/// unwrap to a blanket impl of a foreign trait for a foreign type and violate
/// the orphan rule.
impl<M: Model, O: Objective<M> + ?Sized> Objective<M> for Box<O> {
    fn regime(&self) -> &'static str {
        (**self).regime()
    }
    fn prepare(&mut self, model: &mut M) {
        (**self).prepare(model)
    }
    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        (**self).micro_step(model, rng)
    }
    fn eval(&mut self, model: &M, rng: &mut Rng, batches: u32) -> Option<f32> {
        (**self).eval(model, rng, batches)
    }
    fn metrics(&self) -> Vec<(&'static str, f32)> {
        (**self).metrics()
    }
    fn itos(&self) -> Option<&[char]> {
        (**self).itos()
    }
    fn batch_shape(&self) -> Option<(u32, u32)> {
        (**self).batch_shape()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use super::*;
    use crate::{Batch, ModelConfig};

    #[derive(Clone)]
    struct NullCfg;
    impl ModelConfig for NullCfg {
        fn param_list(&self) -> Vec<(String, usize)> {
            Vec::new()
        }
        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn from_json(_v: &serde_json::Value) -> Self {
            NullCfg
        }
        fn vocab(&self) -> u32 {
            0
        }
        fn block_size(&self) -> u32 {
            0
        }
        fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
            self
        }
    }

    /// A `Model` that computes nothing: this test is about the OBJECTIVE
    /// seam's type plumbing, and a real model would only add GPU cost to a
    /// question no arithmetic is involved in.
    struct NullModel;
    impl Model for NullModel {
        type Config = NullCfg;
        fn new(_cfg: NullCfg, _b: u32, _t: u32, _init: &HashMap<String, Vec<f32>>) -> Self {
            NullModel
        }
        fn init_weights(_cfg: &NullCfg, _seed: u64) -> HashMap<String, Vec<f32>> {
            HashMap::new()
        }
        fn config(&self) -> &NullCfg {
            &NullCfg
        }
        fn set_batch(&self, _b: Batch) {}
        fn forward(&self) -> f32 {
            0.0
        }
        fn backward(&self) {}
        fn zero_grads(&self) {}
        fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {}
        fn poll_wait(&self) {}
        fn param_names(&self) -> Vec<String> {
            Vec::new()
        }
        fn read_weight(&self, _name: &str) -> Vec<f32> {
            Vec::new()
        }
        fn write_weight(&self, _name: &str, _data: &[f32]) {}
        fn read_grad(&self, _name: &str) -> Vec<f32> {
            Vec::new()
        }
        fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
            None
        }
        fn save(&self, _path: &str) {}
        fn config_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    /// An objective that BORROWS - i.e. is not `'static` - the way a
    /// GRPO objective borrows the curriculum it draws its tasks from.
    struct Borrowing<'a> {
        calls: &'a Cell<u32>,
        itos: Vec<char>,
    }
    impl<M: Model> Objective<M> for Borrowing<'_> {
        fn regime(&self) -> &'static str {
            "borrowing"
        }
        fn prepare(&mut self, _model: &mut M) {
            self.calls.set(self.calls.get() + 100);
        }
        fn micro_step(&mut self, _model: &M, _rng: &mut Rng) -> f32 {
            self.calls.set(self.calls.get() + 1);
            self.calls.get() as f32
        }
        fn eval(&mut self, _model: &M, _rng: &mut Rng, _batches: u32) -> Option<f32> {
            Some(-1.0)
        }
        fn metrics(&self) -> Vec<(&'static str, f32)> {
            vec![("calls", self.calls.get() as f32)]
        }
        fn itos(&self) -> Option<&[char]> {
            Some(&self.itos)
        }
        fn batch_shape(&self) -> Option<(u32, u32)> {
            Some((7, 11))
        }
    }

    /// An objective that overrides nothing - the shape every rollout-driven
    /// objective in this repo has, and the one whose defaults a boxed choice
    /// must not change.
    struct Plain;
    impl<M: Model> Objective<M> for Plain {
        fn regime(&self) -> &'static str {
            "plain"
        }
        fn micro_step(&mut self, _model: &M, _rng: &mut Rng) -> f32 {
            0.5
        }
    }

    struct Driven {
        regime: &'static str,
        loss: f32,
        eval: Option<f32>,
        metrics: Vec<(&'static str, f32)>,
        itos: Option<Vec<char>>,
    }

    /// The shape every consumer of this seam has: generic over `O`, taking it
    /// BY VALUE. If `Box<dyn Objective<M> + '_>` does not itself implement
    /// `Objective<M>`, a runtime choice between two objectives cannot reach
    /// this function at all.
    fn drive<M: Model, O: Objective<M>>(model: &mut M, mut obj: O) -> Driven {
        obj.prepare(model);
        let mut rng = Rng::new(0);
        let loss = obj.micro_step(model, &mut rng);
        let eval = obj.eval(model, &mut rng, 1);
        Driven { regime: obj.regime(), loss, eval, metrics: obj.metrics(), itos: obj.itos().map(|s| s.to_vec()) }
    }

    #[test]
    fn a_boxed_non_static_trait_object_is_itself_an_objective_and_forwards_every_method() {
        let calls = Cell::new(0u32);
        let mut model = NullModel;
        // The erasure a runtime regime choice needs: a borrowing objective,
        // boxed as `dyn Objective<M> + '_` (NOT `'static`), handed to a
        // by-value generic consumer.
        let boxed: Box<dyn Objective<NullModel> + '_> = Box::new(Borrowing { calls: &calls, itos: vec!['x', 'y'] });
        let d = drive(&mut model, boxed);
        assert_eq!(d.regime, "borrowing", "regime() must forward to the inner objective, not report the box");
        assert_eq!(calls.get(), 101, "prepare() and micro_step() must both reach the inner objective exactly once");
        assert_eq!(d.loss, 101.0);
        assert_eq!(d.eval, Some(-1.0));
        assert_eq!(d.metrics, vec![("calls", 101.0)]);
        assert_eq!(d.itos, Some(vec!['x', 'y']), "itos() must forward, or a boxed objective would silently drop the checkpoint's char vocab");
        let shape: Box<dyn Objective<NullModel> + '_> = Box::new(Borrowing { calls: &calls, itos: Vec::new() });
        assert_eq!(shape.batch_shape(), Some((7, 11)), "batch_shape() must forward, or a boxed objective would be handed a model sized for the wrong batch");
    }

    /// The same impl must also cover an ordinary boxed CONCRETE objective
    /// (`Box<Anchor>`, not `Box<dyn _>`), which is what the `?Sized` bound
    /// buys - without it a caller boxing a concrete arm would not compile.
    #[test]
    fn a_boxed_concrete_objective_is_also_an_objective() {
        let calls = Cell::new(0u32);
        let mut model = NullModel;
        let boxed: Box<Borrowing> = Box::new(Borrowing { calls: &calls, itos: Vec::new() });
        let d = drive(&mut model, boxed);
        assert_eq!(d.regime, "borrowing");
        assert_eq!(d.itos, Some(Vec::new()));
    }

    /// Boxing must not change an objective's DEFAULTS either: the callers that
    /// build a model from `batch_shape` reproduce their old hardcoded one-row
    /// shape only as long as an unoverridden objective still answers `None`,
    /// boxed or not.
    #[test]
    fn boxing_preserves_the_defaults_an_objective_did_not_override() {
        let plain = Plain;
        assert_eq!(Objective::<NullModel>::batch_shape(&plain), None);
        assert_eq!(Objective::<NullModel>::itos(&plain), None);
        let boxed: Box<dyn Objective<NullModel>> = Box::new(Plain);
        assert_eq!(boxed.batch_shape(), None, "a boxed default must stay None, or every GRPO-shaped call site silently changes model shape");
        assert_eq!(boxed.itos(), None);
        assert_eq!(boxed.metrics(), Vec::new());
        assert_eq!(boxed.regime(), "plain");
    }
}
