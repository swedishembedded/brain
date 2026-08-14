// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `atif` - an independent Rust implementation of the **Agent Trajectory
//! Interchange Format (ATIF) v1.7** wire schema, a validator, and a small
//! set of persistence helpers (atomic whole-document writes, cheap
//! header-only reads, and NDJSON step streaming).
//!
//! **Manual mirror**: this crate is byte-for-byte copied from
//! `applications/sven/crates/atif` (sven's own copy of the same crate,
//! briefly named `crates/trace` before sven's internal rename) so that
//! brain - a separate Cargo workspace - can parse ATIF trajectories sven
//! writes without a cross-repo path/git dependency (brain stays a
//! self-contained workspace). Kept in sync **manually** for now: re-sync by
//! diffing this crate's `src/`/`tests/` against sven's. Any brain-side
//! consumer of ATIF trajectories (see `crates/rl`) reads only real, sven-
//! written trajectory files - this crate contributes no brain-specific
//! behavior of its own.
//!
//! This crate is a standalone, spec-complete building block with zero
//! dependencies on other sven crates (true here too - brain adds none). In
//! sven, `sven-session-store::trace_session` consumes it as the session
//! store backing the TUI/GUI/CI surfaces. See the ATIF RFC (v1.7) for the
//! normative schema this crate mirrors byte-for-byte on the wire, even
//! though the Rust-side type and module names here are an independent
//! design.
//!
//! # Module map
//!
//! - [`model`] - the ATIF schema: [`Trajectory`], steps, tool calls,
//!   observations, multimodal content, subagent references, and the
//!   Section VII context-management convention.
//! - [`validate`] - [`validate::validate_trajectory`], which walks a
//!   [`Trajectory`] (recursively, through embedded subagents) and collects
//!   every rule violation rather than stopping at the first one.
//! - [`persist`] - atomic whole-document JSON writes with concurrent
//!   modification detection, a fast header-only reader, and an NDJSON
//!   step stream reader/writer.

pub mod model;
pub mod persist;
pub mod validate;

pub use model::*;
pub use validate::{validate_trajectory, ValidationError};
