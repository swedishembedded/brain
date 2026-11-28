// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bottleneck autoencoder (ADR 0001 §6), a first-class [`model::Model`] with a
//! [`model::Head::Regression`] (MSE) objective.
//!
//! - [`model`] — forward + backprop as WGSL dispatches ([`model::Autoencoder`]).
//! - [`init`] — weight initialization (small-normal linears, zero biases).

pub mod init;
pub mod model;

pub use init::init_weights;
pub use model::{Autoencoder, AutoencoderConfig};

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before the unit tests run.
#[cfg(test)]
brain_testutil::configure_test_binary!();
