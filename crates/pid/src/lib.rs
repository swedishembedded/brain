// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PID event/effect control Transformer and its data pipeline.
//!
//! - [`model`] — the transformer (forward + backprop), checkpoint I/O.
//! - [`data`] — plant physics, the PID oracle, the CBOR-style tokenizer, and
//!   trajectory/rollout generation.
//!
//! The model's types/constants are re-exported at the crate root so callers can
//! write `pid::Pid`, `pid::PidConfig`, `pid::BOS`, …; the data pipeline stays
//! under `pid::data`.

pub mod data;
pub mod model;

pub use model::*;
