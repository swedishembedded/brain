// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Full-weight supervised fine-tuning of a Qwen3-TTS checkpoint (Talker and
//! code predictor) from recorded speech: the objective the reference trains
//! with, teacher forced, every gradient routed.
//!
//! * [`model`]: the Talker and code predictor as one trainable model, and
//!   where each parameter lives.
//! * [`objective`]: the speech objective over a [`crate::shard::Shard`] for
//!   the shared fit loop (checkpoints, exact resume, step reports and
//!   cancellation come from [`model::train::fit_controlled`]).
//!
//! The text embedding and the text projection are frozen in this release: a
//! row's gradient stops at the codec rows it summed. The speaker x-vector is a
//! frozen input.

pub mod model;
pub mod objective;
mod tables;
#[cfg(test)]
mod tests;

pub use model::{Example, ForwardStats, LrMultipliers, SftShape, SpeechSftModel};
pub use objective::{RowMap, SpeechObjective};
