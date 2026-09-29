// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `atif` - an independent Rust implementation of the **Agent Trajectory
//! Interchange Format (ATIF) v1.7** wire schema, a validator, and a small
//! set of persistence helpers (atomic whole-document writes, cheap
//! header-only reads, and NDJSON step streaming).
//!
//! It is how brain reads the trajectories an agent runtime records as ATIF:
//! any producer that writes spec-conformant ATIF v1.7 JSON can hand its
//! trajectories to brain's trajectory consumers (see `crates/rl`) without
//! brain depending on that producer. brain never depends on any agent
//! runtime; the ATIF wire format is the whole contract.
//!
//! What it validates: [`validate::validate_trajectory`] checks the ATIF
//! v1.7 rules the type system cannot enforce on a parsed [`Trajectory`] -
//! the schema version, sequential `step_id`s, agent-only fields appearing
//! only on agent steps, observation `source_call_id`s resolving to a tool
//! call, and subagent references resolving to uniquely identified embedded
//! trajectories (validated recursively) - and reports every violation
//! rather than stopping at the first one.
//!
//! This crate is a standalone, spec-complete building block with no
//! dependencies on other brain crates. See the ATIF RFC (v1.7) for the
//! normative schema this crate follows byte-for-byte on the wire; the
//! Rust-side type and module names are an independent design.
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
