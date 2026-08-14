// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `ToolCallSchema`: a single function/tool invocation made by the agent.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A single structured tool/function invocation within a [`crate::TraceStep`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInvocation {
    /// Unique identifier for this call; correlates with
    /// `ObservationEntry.source_call_id`.
    pub tool_call_id: String,
    /// Name of the invoked function/tool.
    pub function_name: String,
    /// Arguments passed to the function. Must be a JSON object, may be `{}`.
    pub arguments: Value,
    /// Custom tool-call-level metadata (timeout, retry count, tool version, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl ToolInvocation {
    /// Construct a tool invocation with an empty-object argument map.
    pub fn new(tool_call_id: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            function_name: function_name.into(),
            arguments: Value::Object(Map::new()),
            extra: None,
        }
    }

    /// Set the argument object.
    pub fn with_arguments(mut self, arguments: Value) -> Self {
        self.arguments = arguments;
        self
    }
}
