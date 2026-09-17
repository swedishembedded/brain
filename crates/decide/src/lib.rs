// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! brain's decision model: `(state, question, allowed answers)` in, a
//! calibrated probability distribution over those answers out.
//!
//! The options are supplied per request, so this is not a classifier with a
//! fixed head. A state is encoded once and shared by every question in the
//! request; each question's options are scored independently against it by a
//! cross-attention head. Nothing is generated, and an answer outside the
//! supplied set cannot be produced.
//!
//! This module tree is the encoder half.
//!
//! The shape is forced by the contract rather than chosen: because one state
//! encode is shared by every question in a request, the state and the
//! question+option text must be encoded SEPARATELY and combined late. An
//! encoder that reads them together scores better and would have to re-encode
//! the state per question set, which is the whole cost this model exists to
//! avoid. That is also why a question's instructions travel with its options
//! rather than with the state: it is what makes question independence
//! structural instead of a promise.

pub mod banking77;
pub mod config;
pub mod decide;
pub mod head;
pub mod kern;
pub mod import;
pub mod init;
pub mod loss;
pub mod model;
pub mod pack;
pub mod primitives;

pub use config::EncoderConfig;
pub use decide::{Decide, Example, Limits};
pub use model::Encoder;
