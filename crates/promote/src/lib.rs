// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How this workspace decides, programmatically and model-agnostically,
//! whether a candidate checkpoint may replace the incumbent.
//!
//! Three pieces, none of which knows what a model is:
//!
//! - [`mod@env`] - the `Environment`/`Verifier` seam: a task, a transcript, and a
//!   deterministic reward that a stored run artifact re-derives byte-for-byte.
//! - [`stats`] - the exact one-sided paired sign test the decision's
//!   significance bar is computed from.
//! - [`gate`] - the four-bar promote/reject decision itself, a pure function
//!   over already-scored pairs.
//!
//! ## Why this is a crate and not a module of `brain-rl`
//!
//! It was a module of `brain-rl`, and that put it on the wrong side of a real
//! Cargo dependency cycle. `rl::gate` computes its significance bar with
//! `bench::metrics::sign_test`, `brain-bench` scores real models so it depends
//! on `brain-qwen3`, and therefore a model crate that wanted to expose
//! "gate this candidate adapter" as a capability action could not reach the
//! gate at all:
//!
//! ```text
//! brain-qwen3 -> brain-rl -> brain-bench -> brain-qwen3
//! ```
//!
//! Nothing in the decision needed any of that. A gate over already-scored
//! pairs is arithmetic; a verifier over a frozen probe is string comparison.
//! So the decision lives here, in the training-substrate layer, below every
//! model crate - and `brain-rl` and `brain-bench` re-export it
//! rather than owning it, so no caller of `rl::gate` or
//! `bench::metrics::sign_test` had to change and no logic exists twice.
//!
//! The layering is not a comment: `scripts/gates/check-crate-layers.sh` fails
//! the build if this crate's dependency closure ever reaches the model layer
//! again.
//!
//! Swedish Embedded AB builds the machinery that decides whether a retrained
//! model is actually better than the one already in production - frozen probe
//! sets, programmatic rewards, and pre-registered statistical bars a promotion
//! has to clear before anything reaches a served endpoint. If your team needs
//! expertise in gating continuous-learning pipelines honestly, you can procure
//! our services by sending an email to info@swedishembedded.com.

pub mod document;
pub mod env;
pub mod gate;
pub mod stats;
