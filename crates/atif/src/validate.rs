// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Validation of [`crate::model::Trajectory`] values against ATIF v1.7 rules
//! not already enforced structurally by the type system.
//!
//! [`validate_trajectory`] mirrors the RFC's own reference validator
//! (Section VI): it walks the whole document and **collects every
//! violation** rather than short-circuiting on the first one, so a producer
//! can see the full list of problems in one pass.
//!
//! A few rules are intentionally *not* runtime-checked here because the type
//! system already makes the invalid state unrepresentable:
//! - `ContentPartSchema`'s text/image conditional fields - see
//!   [`crate::model::ContentSegment`], a tagged enum.
//! - `ImageSourceSchema.media_type`'s four-value closed set - see
//!   [`crate::model::ImageMediaType`], a closed enum; a JSON value outside
//!   the four allowed MIME types simply fails to deserialize.

use thiserror::Error;

use crate::model::{StepOrigin, Trajectory};

/// A single ATIF v1.7 rule violation, as collected by [`validate_trajectory`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError {
    /// `step_id`s within a single trajectory's `steps` array must be
    /// sequential starting at 1.
    #[error("step at index {index} has step_id {found}, expected {expected} (step_ids must be sequential starting at 1)")]
    StepIdSequence {
        /// Zero-based index into the `steps` array.
        index: usize,
        /// The step_id that was expected at this position.
        expected: u64,
        /// The step_id actually found.
        found: u64,
    },

    /// A field documented as "Only applicable when source is agent" was set
    /// on a non-agent step.
    #[error("step_id={step_id}: field `{field}` is only applicable to agent steps, but source is `{step_source:?}`")]
    AgentOnlyField {
        /// The offending step's `step_id`.
        step_id: u64,
        /// The offending step's actual `source`. Named `step_source` (not
        /// `source`) because `thiserror` treats a field literally named
        /// `source` as the error's `Error::source()` chain.
        step_source: StepOrigin,
        /// Name of the agent-only field that was set.
        field: &'static str,
    },

    /// `llm_call_count == 0` on an agent step (deterministic dispatch) but
    /// `metrics` and/or `reasoning_content` were present.
    #[error("step_id={step_id}: llm_call_count=0 (deterministic dispatch) requires `{field}` to be absent")]
    DeterministicDispatchFieldsPresent {
        /// The offending step's `step_id`.
        step_id: u64,
        /// Name of the field that must be absent (`metrics` or `reasoning_content`).
        field: &'static str,
    },

    /// A `SubagentTrajectoryRef` set neither `trajectory_id` nor `trajectory_path`.
    #[error("step_id={step_id}, observation result #{result_index}: subagent_trajectory_ref[{ref_index}] sets neither trajectory_id nor trajectory_path")]
    UnresolvableSubagentRef {
        /// The step containing the offending observation.
        step_id: u64,
        /// Index into `observation.results`.
        result_index: usize,
        /// Index into `subagent_trajectory_ref`.
        ref_index: usize,
    },

    /// An embedded trajectory in `subagent_trajectories` is missing the
    /// REQUIRED `trajectory_id`.
    #[error("subagent_trajectories[{index}] is missing the required trajectory_id")]
    MissingSubagentTrajectoryId {
        /// Index into the parent's `subagent_trajectories` array.
        index: usize,
    },

    /// Two or more entries in `subagent_trajectories` share a `trajectory_id`.
    #[error("duplicate trajectory_id `{0}` within subagent_trajectories")]
    DuplicateSubagentTrajectoryId(String),

    /// An `ObservationEntry.source_call_id` did not match any
    /// `tool_call_id` among the parent step's `tool_calls`.
    #[error("step_id={step_id}, observation result #{result_index}: source_call_id `{source_call_id}` does not match any tool_call_id in this step's tool_calls")]
    DanglingSourceCallId {
        /// The step containing the offending observation.
        step_id: u64,
        /// Index into `observation.results`.
        result_index: usize,
        /// The `source_call_id` that could not be resolved.
        source_call_id: String,
    },

    /// `schema_version` is empty or does not look like `"ATIF-v..."`.
    #[error("schema_version `{0}` does not look like a valid ATIF version string (expected e.g. \"ATIF-v1.7\")")]
    SchemaVersion(String),

    /// A violation found while recursively validating an embedded subagent
    /// trajectory.
    #[error("subagent_trajectories[{index}]: {inner}")]
    Nested {
        /// Index into the parent's `subagent_trajectories` array.
        index: usize,
        /// The violation found inside that embedded trajectory.
        inner: Box<ValidationError>,
    },
}

/// Validate a trajectory against every ATIF v1.7 rule this crate enforces,
/// collecting **all** violations rather than stopping at the first one.
/// Embedded `subagent_trajectories` are validated recursively (each is a
/// full, independent trajectory with its own `step_id` sequence).
pub fn validate_trajectory(trajectory: &Trajectory) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    validate_schema_version(trajectory, &mut errors);
    validate_step_id_sequence(trajectory, &mut errors);
    for step in &trajectory.steps {
        validate_agent_only_fields(step, &mut errors);
        validate_deterministic_dispatch(step, &mut errors);
        validate_source_call_ids(step, &mut errors);
        validate_subagent_refs(step, &mut errors);
    }
    validate_embedded_subagents(trajectory, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_schema_version(trajectory: &Trajectory, errors: &mut Vec<ValidationError>) {
    let v = &trajectory.schema_version;
    if v.is_empty() || !v.starts_with("ATIF-v") {
        errors.push(ValidationError::SchemaVersion(v.clone()));
    }
}

fn validate_step_id_sequence(trajectory: &Trajectory, errors: &mut Vec<ValidationError>) {
    for (index, step) in trajectory.steps.iter().enumerate() {
        let expected = (index + 1) as u64;
        if step.step_id != expected {
            errors.push(ValidationError::StepIdSequence {
                index,
                expected,
                found: step.step_id,
            });
        }
    }
}

fn validate_agent_only_fields(step: &crate::model::TraceStep, errors: &mut Vec<ValidationError>) {
    if step.source == StepOrigin::Agent {
        return;
    }
    let mut check = |present: bool, field: &'static str| {
        if present {
            errors.push(ValidationError::AgentOnlyField {
                step_id: step.step_id,
                step_source: step.source,
                field,
            });
        }
    };
    check(step.model_name.is_some(), "model_name");
    check(step.reasoning_effort.is_some(), "reasoning_effort");
    check(step.reasoning_content.is_some(), "reasoning_content");
    check(step.tool_calls.is_some(), "tool_calls");
    check(step.metrics.is_some(), "metrics");
}

fn validate_deterministic_dispatch(
    step: &crate::model::TraceStep,
    errors: &mut Vec<ValidationError>,
) {
    if step.source != StepOrigin::Agent || step.llm_call_count != Some(0) {
        return;
    }
    if step.metrics.is_some() {
        errors.push(ValidationError::DeterministicDispatchFieldsPresent {
            step_id: step.step_id,
            field: "metrics",
        });
    }
    if step.reasoning_content.is_some() {
        errors.push(ValidationError::DeterministicDispatchFieldsPresent {
            step_id: step.step_id,
            field: "reasoning_content",
        });
    }
}

fn validate_source_call_ids(step: &crate::model::TraceStep, errors: &mut Vec<ValidationError>) {
    let Some(observation) = &step.observation else {
        return;
    };
    let known_ids: Vec<&str> = step
        .tool_calls
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tc| tc.tool_call_id.as_str())
        .collect();
    for (result_index, result) in observation.results.iter().enumerate() {
        if let Some(source_call_id) = &result.source_call_id {
            if !known_ids.contains(&source_call_id.as_str()) {
                errors.push(ValidationError::DanglingSourceCallId {
                    step_id: step.step_id,
                    result_index,
                    source_call_id: source_call_id.clone(),
                });
            }
        }
    }
}

fn validate_subagent_refs(step: &crate::model::TraceStep, errors: &mut Vec<ValidationError>) {
    let Some(observation) = &step.observation else {
        return;
    };
    for (result_index, result) in observation.results.iter().enumerate() {
        let Some(refs) = &result.subagent_trajectory_ref else {
            continue;
        };
        for (ref_index, r) in refs.iter().enumerate() {
            if r.is_unresolvable() {
                errors.push(ValidationError::UnresolvableSubagentRef {
                    step_id: step.step_id,
                    result_index,
                    ref_index,
                });
            }
        }
    }
}

fn validate_embedded_subagents(trajectory: &Trajectory, errors: &mut Vec<ValidationError>) {
    let Some(children) = &trajectory.subagent_trajectories else {
        return;
    };

    let mut seen_ids: Vec<&str> = Vec::new();
    for (index, child) in children.iter().enumerate() {
        match &child.trajectory_id {
            None => errors.push(ValidationError::MissingSubagentTrajectoryId { index }),
            Some(id) => {
                if seen_ids.contains(&id.as_str()) {
                    errors.push(ValidationError::DuplicateSubagentTrajectoryId(id.clone()));
                } else {
                    seen_ids.push(id.as_str());
                }
            }
        }

        if let Err(child_errors) = validate_trajectory(child) {
            for inner in child_errors {
                errors.push(ValidationError::Nested {
                    index,
                    inner: Box::new(inner),
                });
            }
        }
    }
}
