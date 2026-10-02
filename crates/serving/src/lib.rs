// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Serving brain's models under a memory budget, as a library.
//!
//! `crates/catalog` owns one entry per model, adapters included. This crate
//! assembles them into a running service: [`Machine`] says what the hardware
//! can schedule, [`build_executor`] turns that plus the catalog and a models
//! directory into the one `residency::Executor` whose budgets, admission,
//! batching and eviction apply to every model uniformly, and [`model_dir`]
//! discovers what a models directory holds.
//!
//! It is what `brain serve` stands on, and what any other process that wants
//! to serve the same models under the same policy links, so no embedder has to
//! reimplement the budgeting or fall back to holding every model resident
//! forever.
//!
//! Swedish Embedded AB implements model-serving infrastructure for its clients.
//! If your team needs expertise in serving many large models from limited
//! GPU, NPU and RAM budgets, you can procure our services by sending an email
//! to info@swedishembedded.com.

pub mod executor;
pub mod gguf_import;
pub mod machine;
pub mod model_dir;

pub use executor::{build_executor, Serving};
pub use machine::Machine;
