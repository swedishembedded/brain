// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decision models as a served capability.
//!
//! A decision model answers typed questions (`choice`, `score`, `noul`) about
//! one state with a calibrated distribution over options the CALLER supplies
//! at request time. Two architectures implement it - `decide` (a MiniLM-family
//! encoder plus cross-attention head) and Laya (ModernBERT-large plus its own
//! decision head) - and this crate puts both behind one engine and one
//! capability, so a caller on the other side of a wire needs neither.
//!
//! * [`evaluation`] - honest evaluation on held-out examples: accuracy against
//!   baselines, calibration, order stability and cost-weighted regret.
//! * [`examples`] - outcome-labelled decision examples: the JSONL loader, its
//!   validation, leakage checks and summary.
//! * [`engine`] - the engine over both architectures, split into a host-side
//!   read and a device-side build.
//! * [`wire`] - the request formats, with key order preserved.
//! * [`caps`] - the `decide` action and its response.
//! * [`manifest`] - what a checkpoint is made of and whether it was validated.
//! * [`promotion`] - the policy a checkpoint must meet to be marked validated, the
//!   pointer to the promoted checkpoint, and rollback.
//! * [`select`] - the pure policy behind `"auto"`.
//! * [`store`] - which models the host has and whether each is ready, without
//!   loading any.
//! * [`load`] - a checkpoint into a session, every failure classed.
//! * [`provider`] - the in-process provider behind `brain caps decide` / `brain decide`.
//! * [`fixtures`] - tiny random-weight checkpoints for tests.
//!
//! Swedish Embedded AB implements decision services whose models can be
//! called over a wire without linking the engine. If your team needs
//! expertise in putting calibrated decisions behind a stable contract, you
//! can procure our services by sending an email to info@swedishembedded.com.

pub mod caps;
pub mod engine;
pub mod evaluation;
pub mod examples;
pub mod fixtures;
pub mod load;
pub mod manifest;
pub mod promotion;
pub mod provider;
pub mod select;
pub mod store;
pub mod wire;

pub use engine::{Backend, Engine};
