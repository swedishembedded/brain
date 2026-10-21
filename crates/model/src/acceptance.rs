// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning models of any
// architecture over quantized frozen bases through one training stack, for its
// clients. If your team needs expertise in wiring a new model family into
// quantized LoRA training then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The acceptance test of the training architecture: one LoRA training
//! workflow, run unchanged over every architecture and every frozen-base
//! encoding it supports.
//!
//! An architecture joins by implementing [`LoraArch`] (which encodings it
//! trains over, and how to build a tiny random instance of it) and returning a
//! [`LoraStep`] (one optimiser step over a fixed batch, plus the digests the
//! workflow checks). [`run`] holds the workflow and names no architecture, so
//! wiring a model in means implementing the trait, never editing the check.
//! Every [`Model`] gets its [`LoraStep`] from [`ModelStep`]; a trainer that is
//! not a [`Model`] (an image-diffusion LoRA trainer) implements the trait
//! directly.
//!
//! The workflow asserts, per encoding:
//!
//! * the base is held in the encoding that was asked for (no implicit
//!   expansion to a wider one),
//! * the loss and the adapter stay finite and the training loss falls,
//! * the adapter changes and the frozen base does not.

use gpu_core::select::Dtype;

use crate::adapter::device::is_adapter_param;
use crate::{Adam, Batch, Model, ModelConfig};

/// One architecture under the LoRA acceptance workflow.
pub trait LoraArch {
    const NAME: &'static str;

    /// The frozen-base encodings the architecture trains LoRA over.
    fn encodings() -> Vec<Dtype>;

    /// The learning rate of the overfit; tiny random models differ in how fast
    /// their adapters can move.
    fn learning_rate() -> f32 {
        3e-2
    }

    /// Optimiser steps of the overfit.
    fn steps() -> u32 {
        40
    }

    /// The share of the starting loss the trained loss must be below. An
    /// architecture whose adapters reach only part of the network (the
    /// mixers, not the experts) says how much it can fit.
    fn overfit_ratio() -> f64 {
        0.9
    }

    /// A tiny random instance with its base held in `encoding`, drawn from `seed`.
    fn build(encoding: Dtype, seed: u64) -> Box<dyn LoraStep>;
}

/// One architecture instance on a fixed batch.
pub trait LoraStep {
    /// The encoding the frozen base is actually held in.
    fn held_encoding(&self) -> Dtype;

    /// The training loss of the batch, without updating anything.
    fn loss(&mut self) -> f64;

    /// One optimiser step at learning rate `lr`; the loss it started from.
    fn step(&mut self, lr: f32) -> f64;

    /// A digest of every trainable tensor.
    fn adapter_digest(&self) -> u64;

    /// A digest of every frozen tensor the instance can read back; `None`
    /// when it keeps the base where the workflow cannot see it.
    fn frozen_digest(&self) -> Option<u64>;
}


/// The workflow over every encoding `A` trains over. `seed` draws the weights.
pub fn run<A: LoraArch>(seed: u64) -> Result<(), String> {
    let encodings = A::encodings();
    if encodings.is_empty() {
        return Err(format!("{}: offers no frozen-base encoding", A::NAME));
    }
    for encoding in encodings {
        let at = format!("{} over a {encoding:?} base", A::NAME);
        let mut step = A::build(encoding, seed);
        if step.held_encoding() != encoding {
            return Err(format!("{at}: the base is held as {:?}", step.held_encoding()));
        }
        let (adapter, frozen) = (step.adapter_digest(), step.frozen_digest());
        let start = step.loss();
        if !start.is_finite() {
            return Err(format!("{at}: the starting loss is {start}"));
        }
        for _ in 0..A::steps() {
            step.step(A::learning_rate());
        }
        let end = step.loss();
        if !end.is_finite() || end >= A::overfit_ratio() * start {
            return Err(format!("{at}: loss {start} -> {end} after {} steps, wanted below {} of the start", A::steps(), A::overfit_ratio()));
        }
        println!("acceptance: {at}: loss {start:.4} -> {end:.4} in {} steps", A::steps());
        if step.adapter_digest() == adapter {
            return Err(format!("{at}: no adapter tensor changed"));
        }
        if step.frozen_digest() != frozen {
            return Err(format!("{at}: the frozen base changed"));
        }
    }
    Ok(())
}

/// A [`Model`] on one fixed language-model batch as a [`LoraStep`].
pub struct ModelStep<M: Model> {
    model: M,
    held: Dtype,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    steps: u32,
}

impl<M: Model> ModelStep<M> {
    /// `model` must be built for `crate::conformance::BATCH` rows of
    /// `crate::conformance::SEQ` tokens; `held` is the encoding its base was
    /// built in, as the model itself reports it.
    pub fn new(model: M, held: Dtype) -> ModelStep<M> {
        let (tokens, targets) = crate::conformance::batch(model.config().vocab());
        ModelStep { model, held, tokens, targets, steps: 0 }
    }

    fn digest(&self, adapter: bool) -> u64 {
        let mut names = self.model.param_names();
        names.retain(|n| is_adapter_param(n) == adapter);
        names.iter().flat_map(|n| self.model.read_weight(n)).fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3))
    }
}

impl<M: Model> LoraStep for ModelStep<M> {
    fn held_encoding(&self) -> Dtype {
        self.held
    }

    fn loss(&mut self) -> f64 {
        self.model.set_batch(Batch::Lm { tokens: &self.tokens, targets: &self.targets });
        f64::from(self.model.forward())
    }

    fn step(&mut self, lr: f32) -> f64 {
        self.steps += 1;
        self.model.zero_grads();
        let loss = self.loss();
        self.model.backward();
        self.model.adamw_step(self.steps, lr, 0.0, Adam::default(), Some(1.0), 1.0);
        self.model.poll_wait();
        loss
    }

    fn adapter_digest(&self) -> u64 {
        self.digest(true)
    }

    fn frozen_digest(&self) -> Option<u64> {
        Some(self.digest(false))
    }
}
