// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Root `Trajectory` object and `FinalMetricsSchema`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent::AgentProfile;
use super::step::TraceStep;

/// Aggregate statistics for an entire trajectory. Every field is optional.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FinalMetrics {
    /// Sum of all `prompt_tokens` across steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_prompt_tokens: Option<u64>,
    /// Sum of all `completion_tokens` across steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_completion_tokens: Option<u64>,
    /// Sum of all `cached_tokens` across steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_cached_tokens: Option<u64>,
    /// Total monetary cost for the whole trajectory, in USD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_cost_usd: Option<f64>,
    /// Total number of steps (can differ from `steps.len()`; see `notes`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<u64>,
    /// Custom aggregate metrics not covered by the core schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// Root ATIF trajectory object.
///
/// Field declaration order matters beyond readability: `steps` is
/// deliberately declared **last**. `serde_json`'s struct serialization
/// preserves Rust declaration order, so a header-only reader
/// ([`crate::persist::header`]) can find the byte offset of the `"steps"`
/// key and parse everything before it as a small header type without
/// touching the (possibly huge) steps array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    /// ATIF schema/version marker, e.g. `"ATIF-v1.7"`.
    pub schema_version: String,
    /// Run-scoped identifier; MAY be shared across sibling subagents,
    /// continuation segments, or omitted on embedded subagents that inherit
    /// the parent's run identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Per-document identifier, distinct from `session_id`. REQUIRED on any
    /// trajectory embedded in a parent's `subagent_trajectories` array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    /// The agent system that produced this trajectory.
    pub agent: AgentProfile,
    /// Free-form developer notes / discrepancy explanations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Aggregate statistics for the whole trajectory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_metrics: Option<FinalMetrics>,
    /// Reference to a continuation trajectory file, when this trajectory is
    /// split across files (e.g. by a summarization boundary).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continued_trajectory_ref: Option<String>,
    /// Custom root-level metadata not covered by the core schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
    /// Embedded subagent trajectories (single-file multi-agent storage).
    /// Each entry is a complete, independently-valid `Trajectory`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_trajectories: Option<Vec<Trajectory>>,
    /// Ordered interaction steps. Declared last - see the struct doc comment.
    pub steps: Vec<TraceStep>,
}

impl Trajectory {
    /// Construct a trajectory with only the required fields populated.
    pub fn new(schema_version: impl Into<String>, agent: AgentProfile) -> Self {
        Self {
            schema_version: schema_version.into(),
            session_id: None,
            trajectory_id: None,
            agent,
            notes: None,
            final_metrics: None,
            continued_trajectory_ref: None,
            extra: None,
            subagent_trajectories: None,
            steps: Vec::new(),
        }
    }

    /// Steps eligible for supervised fine-tuning: excludes any step with
    /// `is_copied_context == Some(true)`, per the RFC's normative rule.
    pub fn sft_steps(&self) -> impl Iterator<Item = &TraceStep> {
        self.steps
            .iter()
            .filter(|step| !step.is_excluded_from_sft())
    }
}
