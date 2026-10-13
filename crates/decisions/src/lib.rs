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
//! * [`engine`] - the engine over both architectures, split into a host-side
//!   read and a device-side build.
//! * [`wire`] - the request formats, with key order preserved.
//! * [`caps`] - the `decide` action and its response.
//! * [`manifest`] - what a checkpoint is made of and whether it was validated.
//! * [`select`] - the pure policy behind `"auto"`.
//! * [`fixtures`] - tiny random-weight checkpoints for tests.
//!
//! Swedish Embedded AB implements decision services whose models can be
//! called over a wire without linking the engine. If your team needs
//! expertise in putting calibrated decisions behind a stable contract, you
//! can procure our services by sending an email to info@swedishembedded.com.

pub mod caps;
pub mod engine;
pub mod fixtures;
pub mod manifest;
pub mod select;
pub mod wire;

pub use engine::{Backend, Engine};
