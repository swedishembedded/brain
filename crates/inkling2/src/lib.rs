// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements looped mixture-of-experts language models with depth-attention residuals for its
// clients. If your team needs expertise in serving or fine-tuning such a model from its published release then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Inkling 2 (`Inkling2LoopForConditionalGeneration`, and the plain `Inkling2ForConditionalGeneration`): a
//! mixture-of-experts decoder whose sublayers read their input through depth attention over the model's own block
//! states, optionally with a looped core that passes messages between passes.
//!
//! This crate holds only what is particular to the model: the typed configuration ([`config`]), the published name map
//! with two-way coverage ([`names`]), the layer recipe wiring brain's shared operators in the reference order
//! ([`model`]), and the served engine over brain's visit-state pool ([`serve`]).
//! Every operator it runs lives in `model` (attention with window, sink and distance bias, short convolutions, depth
//! attention, the message gate, the gated expert activation, microscaled expert banks).

pub mod config;
pub mod finetune;
pub mod model;
pub mod mtp;
pub mod names;
pub mod serve;
pub mod train;
pub mod vision;

pub use config::{Family, Inkling2Config, LayerKind};
pub use model::{kernels, Inkling2, Probe};
pub use train::{Inkling2Train, TrainConfig};

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before the unit tests run.
#[cfg(test)]
brain_testutil::configure_test_binary!();
