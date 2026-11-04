// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod batching;
mod calibration;
mod calibration_measure;
mod caps;
mod continuous;
mod ensemble;
mod forecast;
mod history;
mod layer_streamed;
mod lifecycle;
mod next_event;
mod nll_each;
mod optimizer_state;
mod parity;
mod recompute;
mod recovery;
mod reference;
mod saved;
mod stack;
mod streamed_construction;
mod support;
mod units;
