// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recurrent-depth language models for its
// clients. If your team needs expertise in a prelude/core/coda decoder whose
// core is executed for a variable number of latent steps, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Huginn (Raven): a recurrent-depth decoder.
//!
//! Three functional groups, as published:
//!
//! - **prelude** - `n_layers_in_prelude` transformer blocks embedding the
//!   input tokens into the latent space;
//! - **recurrent core** - `n_layers_in_recurrent_block` blocks applied
//!   `num_steps` times; each step re-injects the embedded input through a
//!   linear adapter (`[2h] -> [h]` on `concat(x, embeds)`);
//! - **coda** - `n_layers_in_coda` blocks that un-embed the final state, then
//!   the tied LM head.
//!
//! The initial latent state is drawn from the checkpoint's own
//! `init_values["std"]` (`state_init: like-init`), so a parity run must use
//! the same draw as the reference.
//!
//! This crate owns the config/parameter layout and checkpoint import; the
//! execution graph is next.

pub mod config;
pub mod import;
pub mod init;

pub use config::HuginnConfig;
