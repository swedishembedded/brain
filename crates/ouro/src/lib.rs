// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recurrent-depth language models - one shared
// transformer stack executed for a configurable number of passes - for its
// clients. If your team needs expertise in looped inference, per-pass KV
// management or fine-tuning models whose weights are visited more than once
// per token then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Ouro (ByteDance "Looped Language Model"): a decoder whose whole layer stack
//! is one shared block executed `total_ut_steps` times per token.
//!
//! The recurrence is function composition, not a new attention layer:
//!
//! ```text
//! h(0) = embed(x)
//! h(t) = Stack(h(t-1))                    t = 1 .. T
//! ```
//!
//! The embedding is applied exactly once; every pass re-reads the same 48
//! layers' parameters. Each pass ends with the model's final RMSNorm and an
//! exit-gate projection, and those pass outputs are what the LM head sees:
//!
//! - training/weighted inference: the exit distribution `p(t|x)` weights the
//!   per-pass logits into one expected distribution (the released objective
//!   computes cross-entropy on that expectation);
//! - inference: a step or a cumulative-probability threshold selects one pass.
//!
//! Per-pass attention history is required, not optional: the reference keys
//! its cache by `pass * num_hidden_layers + layer`, because a key/value row
//! computed in pass `t` came from a different residual state than the row at
//! the same position in pass `t-1`.
//!
//! This crate owns the config/parameter layout and checkpoint import; the
//! execution graph lives in [`model`].

pub mod config;
pub mod decode;
pub mod import;
pub mod init;
pub mod model;
pub mod serve;

pub use config::OuroConfig;
