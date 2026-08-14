// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! The ATIF v1.7 wire schema, split by concern rather than kept in one file.
//!
//! Every type here is an independent Rust design: names, module boundaries,
//! and field grouping are our own, but every `#[derive(Serialize,
//! Deserialize)]` is written (with `#[serde(rename = "...")]` where our Rust
//! name differs) so the JSON on the wire matches the ATIF v1.7 RFC exactly.

mod agent;
mod content;
mod context_mgmt;
mod metrics;
mod observation;
mod step;
mod subagent;
mod tool;
mod trajectory;

pub use agent::AgentProfile;
pub use content::{ContentSegment, ImageMediaType, ImageRef, MessageBody};
pub use context_mgmt::ContextManagement;
pub use metrics::StepMetrics;
pub use observation::{ObservationEntry, StepObservation};
pub use step::{ReasoningEffort, StepOrigin, TraceStep};
pub use subagent::SubagentRef;
pub use tool::ToolInvocation;
pub use trajectory::{FinalMetrics, Trajectory};
