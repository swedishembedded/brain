// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `AgentSchema`: identifies the agent system that produced a trajectory.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Identifies the agent system used to produce a [`crate::Trajectory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Agent system name (e.g. "openhands", "claude-code", "sven").
    pub name: String,
    /// Agent system version (e.g. "1.0.0").
    pub version: String,
    /// Default LLM model for this trajectory; step-level `model_name` overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// OpenAI-style function-calling tool definitions available to the agent.
    /// Modeled loosely as raw JSON values since ATIF only requires each
    /// element to "follow OpenAI's function calling schema" rather than
    /// mandating a closed shape this crate should own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_definitions: Option<Vec<Value>>,
    /// Custom agent configuration not covered by the core schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl AgentProfile {
    /// Construct a minimal agent profile with only the two required fields.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        }
    }

    /// Attach a default model name.
    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }
}
