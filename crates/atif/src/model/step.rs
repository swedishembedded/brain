// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! `StepObject`: a single turn in a trajectory - a system prompt, a user
//! message, or a complete agent turn (LLM inference, action, observation).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::content::MessageBody;
use super::metrics::StepMetrics;
use super::observation::StepObservation;
use super::tool::ToolInvocation;

/// The originator of a [`TraceStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepOrigin {
    /// System prompt or system-initiated operation.
    System,
    /// A message from the human user.
    User,
    /// An agent turn (LLM inference, action, observation).
    Agent,
}

/// `reasoning_effort`: either a qualitative label ("low"/"medium"/"high") or
/// a quantitative score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReasoningEffort {
    /// A qualitative label, e.g. "low", "medium", "high".
    Text(String),
    /// A quantitative effort score.
    Score(f64),
}

/// A single turn in the `steps` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceStep {
    /// Ordinal index of the turn, starting from 1.
    pub step_id: u64,
    /// ISO 8601 timestamp this step occurred at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Originator of this step.
    pub source: StepOrigin,
    /// LLM model used for this turn. Only applicable when `source` is `Agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Effort assigned to this step. Only applicable when `source` is `Agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The dialogue message. Required, but may be an empty string.
    pub message: MessageBody,
    /// The agent's explicit internal reasoning. Only applicable when `source`
    /// is `Agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Structured tool/function invocations. Only applicable when `source` is
    /// `Agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolInvocation>>,
    /// Environment feedback / system-event results for this step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<StepObservation>,
    /// LLM operational metrics. Only applicable when `source` is `Agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<StepMetrics>,
    /// Custom step-level metadata. Applicable to all step types; this is
    /// also where the Section VII `context_management` convention nests
    /// (see [`crate::model::ContextManagement`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
    /// Number of LLM inferences this step represents. `Some(0)` on an
    /// `Agent` step signals deterministic (non-LLM) dispatch - see
    /// [`crate::validate::validate_trajectory`] for the associated rule that
    /// `metrics`/`reasoning_content` MUST then be absent. Applicable to all
    /// step types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_call_count: Option<u64>,
    /// Marks a step copied from a prior trajectory for context purposes.
    /// Steps with `Some(true)` here MUST be excluded from SFT training data
    /// - see [`crate::model::Trajectory::sft_steps`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_copied_context: Option<bool>,
}

impl TraceStep {
    /// Construct a step with only the required fields populated.
    pub fn new(step_id: u64, source: StepOrigin, message: impl Into<MessageBody>) -> Self {
        Self {
            step_id,
            timestamp: None,
            source,
            model_name: None,
            reasoning_effort: None,
            message: message.into(),
            reasoning_content: None,
            tool_calls: None,
            observation: None,
            metrics: None,
            extra: None,
            llm_call_count: None,
            is_copied_context: None,
        }
    }

    /// `true` iff this step must be excluded from SFT training data per the
    /// `is_copied_context` normative rule (absent/`None` counts as `false`).
    pub fn is_excluded_from_sft(&self) -> bool {
        self.is_copied_context.unwrap_or(false)
    }
}
