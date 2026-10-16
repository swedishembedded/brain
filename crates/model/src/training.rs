// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training capabilities a [`crate::Model`] may offer beyond plain
//! next-token training, and the typed refusal for a model that lacks one.
//!
//! The per-token objectives (weighted SFT, DPO, GRPO, distillation) all reduce
//! to one forward of [`crate::Batch::LmWeighted`], a readback of the batch's
//! per-token log-probabilities, host-side per-token coefficients, and one
//! backward. [`TokenObjectives`] is that seam. A model that implements it gets
//! every objective in `rl` and the conformance suite without per-model glue; a
//! model that does not is refused with [`Unsupported`] when the run is
//! configured, instead of panicking inside the training loop.
//!
//! Adding an architecture: implement [`TokenObjectives`] on the model type and
//! return it from [`crate::Model::token_objectives`] and
//! [`crate::Model::token_objectives_mut`]. Nothing else in the objectives or
//! the SDK names the architecture.
//!
//! Swedish Embedded AB implements training stacks where preference
//! optimisation, reward-driven training and distillation work on every model
//! the stack supports and refuse cleanly on the ones it does not. If your team
//! needs expertise in generic fine-tuning infrastructure you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::fmt;

use crate::Model;

/// A capability a model can lack, named in the refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// [`TokenObjectives`]: per-token weighted loss and log-probability readout.
    TokenObjectives,
    /// An implementation of the chat fine-tuners (supervised, preference and
    /// full fine-tuning) for the architecture.
    ChatFineTune,
}

impl Capability {
    /// The capability as a noun phrase, for messages ("does not provide token objectives yet").
    pub fn describe(self) -> &'static str {
        match self {
            Capability::TokenObjectives => "token objectives",
            Capability::ChatFineTune => "chat fine-tuning",
        }
    }
}

/// A model was asked for a capability it does not provide. Returned at
/// configuration time (an objective's `prepare`, the SDK's builders) so the
/// refusal names the model and the missing capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    /// The model or architecture that was asked.
    pub model: String,
    pub capability: Capability,
}

impl Unsupported {
    pub fn new(model: impl Into<String>, capability: Capability) -> Unsupported {
        Unsupported { model: model.into(), capability }
    }

    /// The refusal for model type `M`, named by its last path segments
    /// (`qwen35::Qwen35`) since a [`Model`] carries no display name.
    pub fn of<M>(capability: Capability) -> Unsupported {
        let full = std::any::type_name::<M>();
        let short = full.split('<').next().unwrap_or(full);
        let mut parts: Vec<&str> = short.rsplit("::").take(2).collect();
        parts.reverse();
        Unsupported::new(parts.join("::"), capability)
    }
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} does not provide {} yet", self.model, self.capability.describe())
    }
}

impl std::error::Error for Unsupported {}

impl From<Unsupported> for std::io::Error {
    /// `ErrorKind::Unsupported` with the typed error as the source, so a
    /// caller of an `io::Result` API can still `downcast` it.
    fn from(e: Unsupported) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::Unsupported, e)
    }
}

/// Per-token objectives over a causal-LM model: everything DPO, GRPO and
/// distillation need beyond ordinary [`crate::Batch::Lm`] training.
///
/// The calls compose in one micro-step: [`Self::enable_weighted_loss`] once
/// after construction; per step [`crate::Model::set_batch`],
/// [`crate::Model::forward`], [`Self::batch_token_logprobs`], the objective's
/// per-token coefficients on the host, [`Self::set_loss_weights`], then
/// [`crate::Model::backward`].
pub trait TokenObjectives {
    /// Opt into per-position weighted-loss training ([`crate::Batch::LmWeighted`])
    /// - call once after construction, before the first backward. Ordinary
    /// (unweighted) training pays nothing for it until called.
    fn enable_weighted_loss(&mut self);

    /// Per-position `log p_theta(target_i)` for the batch most recently
    /// forwarded, length `b*t`, `0.0` at IGNORE positions. Valid only
    /// immediately after a completed forward, for the batch that forward ran
    /// on. When a logit scale is set it is the log-probability under the
    /// scaled logits (the distribution the loss differentiates).
    fn batch_token_logprobs(&self) -> Vec<f32>;

    /// Replace only the per-position CE-gradient weights for the batch already
    /// set, without disturbing the activations a completed forward left behind
    /// (what lets DPO/GRPO run on one forward and one backward per micro-step).
    /// Requires [`Self::enable_weighted_loss`].
    fn set_loss_weights(&self, weights: &[f32]);

    /// Divide the loss and gradient of the batch just set by `tokens` instead
    /// of its own count of supervised positions, so batches accumulated into
    /// one optimiser step share one denominator. Call after `set_batch`.
    fn set_loss_denominator(&self, tokens: f32);

    /// Multiply the logits by `scale` before the softmax inside the loss, the
    /// gradient and [`Self::batch_token_logprobs`]: the sampler-temperature
    /// correction (`scale = 1/T`), so the policy that is trained is the one
    /// that sampled. `1.0` is the identity; the value must be finite and
    /// positive. `Model::logits_all` returns the scaled logits while a scale
    /// is set, so reset it to `1.0` before scoring raw logits.
    fn set_logit_scale(&mut self, scale: f32);
}

/// The model's [`TokenObjectives`], or the typed refusal naming `M`.
pub fn require_token_objectives<M: Model>(model: &mut M) -> Result<&mut dyn TokenObjectives, Unsupported> {
    match model.token_objectives_mut() {
        Some(objectives) => Ok(objectives),
        None => Err(Unsupported::of::<M>(Capability::TokenObjectives)),
    }
}

/// The model's [`TokenObjectives`] for a step that runs after the objective's
/// `prepare` already verified the capability with [`require_token_objectives`].
/// Reaching it on a model without the capability is a caller bug, reported
/// with the same typed message.
pub fn objectives<M: Model>(model: &M) -> &dyn TokenObjectives {
    model.token_objectives().unwrap_or_else(|| panic!("{}", Unsupported::of::<M>(Capability::TokenObjectives)))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;

    #[test]
    fn refusal_names_the_model_and_the_capability() {
        let e = Unsupported::new("qwen35", Capability::TokenObjectives);
        assert_eq!(e.to_string(), "qwen35 does not provide token objectives yet");
    }

    #[test]
    fn refusal_for_a_type_uses_its_trailing_path() {
        let e = Unsupported::of::<Fake>(Capability::TokenObjectives);
        assert_eq!(e.model, "tests::Fake");
        assert!(e.to_string().ends_with("does not provide token objectives yet"));
    }

    #[test]
    fn refusal_survives_an_io_error_round_trip() {
        let io: std::io::Error = Unsupported::new("m", Capability::TokenObjectives).into();
        assert_eq!(io.kind(), std::io::ErrorKind::Unsupported);
        let back = io.get_ref().and_then(|e| e.downcast_ref::<Unsupported>()).expect("typed source");
        assert_eq!(back.capability, Capability::TokenObjectives);
    }
}
