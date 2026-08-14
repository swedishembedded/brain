// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `ObservationSchema` / `ObservationResultSchema`: environment feedback (or
//! system-event results) attached to a step.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::content::MessageBody;
use super::subagent::SubagentRef;

/// Container for the results of the actions taken in a step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepObservation {
    /// One entry per tool call, action, or system event this step produced.
    pub results: Vec<ObservationEntry>,
}

impl StepObservation {
    /// Build an observation with a single result.
    pub fn single(result: ObservationEntry) -> Self {
        Self {
            results: vec![result],
        }
    }
}

/// A single element of `ObservationSchema.results`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ObservationEntry {
    /// The `tool_call_id` this result corresponds to; absent for actions that
    /// don't use standard tool calling (bare agent actions, system events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_call_id: Option<String>,
    /// Output/result content. May be omitted when `subagent_trajectory_ref`
    /// is present and the full subagent trajectory supplies the detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageBody>,
    /// References to delegated subagent trajectories (singleton array for a
    /// single subagent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_trajectory_ref: Option<Vec<SubagentRef>>,
    /// Custom observation-result-level metadata (confidence score, retrieval
    /// score, source document ID, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl ObservationEntry {
    /// Build a plain-text result tied to a tool call.
    pub fn for_call(source_call_id: impl Into<String>, content: impl Into<MessageBody>) -> Self {
        Self {
            source_call_id: Some(source_call_id.into()),
            content: Some(content.into()),
            subagent_trajectory_ref: None,
            extra: None,
        }
    }

    /// Build a result that delegates to a subagent trajectory.
    pub fn for_subagent(refs: Vec<SubagentRef>) -> Self {
        Self {
            source_call_id: None,
            content: None,
            subagent_trajectory_ref: Some(refs),
            extra: None,
        }
    }
}
