// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `SubagentTrajectoryRefSchema`: a reference from an observation result to a
//! delegated subagent trajectory, resolved either by `trajectory_id` against
//! the parent's embedded `subagent_trajectories` array, or by
//! `trajectory_path` against an external file.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A reference to a delegated subagent's trajectory.
///
/// Per ATIF v1.7, at least one of `trajectory_id` (embedded form) or
/// `trajectory_path` (file-ref form) MUST be set - enforced by
/// [`crate::validate::validate_trajectory`], not by the type system, since
/// both fields are individually optional on the wire. `session_id` is
/// informational only (run-scoped, not a resolution key - see the RFC's
/// "Why `session_id` is not a resolution mechanism" note).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SubagentRef {
    /// Canonical identifier of the embedded subagent trajectory; matches
    /// `Trajectory.trajectory_id` of an entry in the parent's
    /// `subagent_trajectories` array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    /// Location of the subagent trajectory as an external file/URL/DB ref.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_path: Option<String>,
    /// Informational-only run identity of the delegated subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Custom metadata about the subagent execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl SubagentRef {
    /// Build an embedded-form reference, resolved via `trajectory_id`.
    pub fn by_trajectory_id(trajectory_id: impl Into<String>) -> Self {
        Self {
            trajectory_id: Some(trajectory_id.into()),
            ..Default::default()
        }
    }

    /// Build a file-ref-form reference, resolved via `trajectory_path`.
    pub fn by_trajectory_path(trajectory_path: impl Into<String>) -> Self {
        Self {
            trajectory_path: Some(trajectory_path.into()),
            ..Default::default()
        }
    }

    /// `true` iff this ref sets neither `trajectory_id` nor `trajectory_path`,
    /// making it unresolvable per the ATIF v1.7 rule.
    pub fn is_unresolvable(&self) -> bool {
        self.trajectory_id.is_none() && self.trajectory_path.is_none()
    }
}
