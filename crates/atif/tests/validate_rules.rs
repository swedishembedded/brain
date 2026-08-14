// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! One test proving each validation rule catches its violation, and one
//! proving valid data passes, per the task brief.

use atif::{
    validate_trajectory, AgentProfile, ContentSegment, ImageMediaType, ObservationEntry,
    StepMetrics, StepObservation, StepOrigin, SubagentRef, ToolInvocation, TraceStep, Trajectory,
    ValidationError,
};

fn base_trajectory() -> Trajectory {
    Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"))
}

// ── step_id sequential-starting-at-1 ────────────────────────────────────────

#[test]
fn step_id_sequence_valid_passes() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    t.steps.push(TraceStep::new(2, StepOrigin::Agent, "hello"));
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn step_id_sequence_gap_fails() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    t.steps.push(TraceStep::new(3, StepOrigin::Agent, "hello")); // skips 2
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::StepIdSequence { .. })));
}

#[test]
fn step_id_not_starting_at_one_fails() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(0, StepOrigin::User, "hi"));
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::StepIdSequence { .. })));
}

// ── agent-only fields on non-agent steps ────────────────────────────────────

#[test]
fn agent_only_fields_on_user_step_fail_and_are_all_collected() {
    let mut step = TraceStep::new(1, StepOrigin::User, "hi");
    step.model_name = Some("gpt-4".into());
    step.reasoning_content = Some("thinking".into());
    step.tool_calls = Some(vec![ToolInvocation::new("call_1", "search")]);
    step.metrics = Some(StepMetrics::default());

    let mut t = base_trajectory();
    t.steps.push(step);
    let errs = validate_trajectory(&t).unwrap_err();

    // All four violations must be reported in a single pass, not just the first.
    let agent_only_count = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::AgentOnlyField { .. }))
        .count();
    assert_eq!(
        agent_only_count, 4,
        "expected 4 agent-only-field violations, got: {errs:?}"
    );
}

#[test]
fn agent_only_fields_absent_on_user_step_passes() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn agent_only_fields_on_agent_step_pass() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "hi");
    step.model_name = Some("gpt-4".into());
    step.reasoning_content = Some("thinking".into());
    step.metrics = Some(StepMetrics::default());
    let mut t = base_trajectory();
    t.steps.push(step);
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn llm_call_count_applicable_to_non_agent_steps_without_error() {
    // llm_call_count is explicitly "Applicable to all step types" per the RFC,
    // so setting it on a system/user step must NOT trigger AgentOnlyField.
    let mut step = TraceStep::new(1, StepOrigin::System, "boot");
    step.llm_call_count = Some(1);
    let mut t = base_trajectory();
    t.steps.push(step);
    assert!(validate_trajectory(&t).is_ok());
}

// ── llm_call_count == 0 deterministic dispatch on agent steps ──────────────

#[test]
fn llm_call_count_zero_agent_step_without_metrics_or_reasoning_passes() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.llm_call_count = Some(0);
    step.tool_calls = Some(vec![ToolInvocation::new("call_1", "graph_dispatch")]);
    let mut t = base_trajectory();
    t.steps.push(step);
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn llm_call_count_zero_agent_step_with_metrics_fails() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.llm_call_count = Some(0);
    step.metrics = Some(StepMetrics::default());
    let mut t = base_trajectory();
    t.steps.push(step);
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DeterministicDispatchFieldsPresent { .. }
    )));
}

#[test]
fn llm_call_count_zero_agent_step_with_reasoning_content_fails() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.llm_call_count = Some(0);
    step.reasoning_content = Some("shouldn't be here".into());
    let mut t = base_trajectory();
    t.steps.push(step);
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DeterministicDispatchFieldsPresent { .. }
    )));
}

// ── ContentPart conditional fields (structurally enforced by the enum) ─────

#[test]
fn content_segment_text_and_image_are_mutually_exclusive_by_construction() {
    let text = ContentSegment::text("hello");
    let image = ContentSegment::image(ImageMediaType::Png, "images/a.png");
    match text {
        ContentSegment::Text { text } => assert_eq!(text, "hello"),
        ContentSegment::Image { .. } => panic!("expected Text"),
    }
    match image {
        ContentSegment::Image { source } => assert_eq!(source.path, "images/a.png"),
        ContentSegment::Text { .. } => panic!("expected Image"),
    }
    // There is no way to express both `text` and `source` (or neither) in a
    // single ContentSegment value - the enum shape is the enforcement.
}

#[test]
fn content_segment_rejects_a_json_object_with_both_text_and_image_type() {
    // A JSON blob that tries to smuggle in an image's `source` on a
    // `type: "text"` segment must fail to parse: our enum has no matching
    // variant shape (Text only ever gets `text`).
    let bad = r#"{"type":"text","source":{"media_type":"image/png","path":"x"}}"#;
    let result: Result<ContentSegment, _> = serde_json::from_str(bad);
    assert!(result.is_err());
}

// ── ImageSource media_type restricted to the 4 allowed MIME types ──────────

#[test]
fn image_media_type_accepts_all_four_allowed_values() {
    for (json, expected) in [
        ("\"image/jpeg\"", ImageMediaType::Jpeg),
        ("\"image/png\"", ImageMediaType::Png),
        ("\"image/gif\"", ImageMediaType::Gif),
        ("\"image/webp\"", ImageMediaType::Webp),
    ] {
        let parsed: ImageMediaType = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, expected);
    }
}

#[test]
fn image_media_type_rejects_disallowed_value() {
    let result: Result<ImageMediaType, _> = serde_json::from_str("\"image/bmp\"");
    assert!(
        result.is_err(),
        "image/bmp is not one of the 4 allowed ATIF media types"
    );
}

// ── SubagentTrajectoryRef resolvability ─────────────────────────────────────

#[test]
fn subagent_ref_with_neither_field_set_fails_validation() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.observation = Some(StepObservation::single(ObservationEntry::for_subagent(
        vec![SubagentRef::default()],
    )));
    let mut t = base_trajectory();
    t.steps.push(step);
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::UnresolvableSubagentRef { .. })));
}

#[test]
fn subagent_ref_with_trajectory_path_only_is_valid_pre_v1_7_back_compat() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.observation = Some(StepObservation::single(ObservationEntry::for_subagent(
        vec![SubagentRef::by_trajectory_path("subagents/child.json")],
    )));
    let mut t = base_trajectory();
    t.steps.push(step);
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn subagent_ref_with_trajectory_id_only_is_valid() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.observation = Some(StepObservation::single(ObservationEntry::for_subagent(
        vec![SubagentRef::by_trajectory_id("child-1")],
    )));
    let mut t = base_trajectory();
    t.trajectory_id = Some("parent".into());
    t.steps.push(step);

    let mut child = Trajectory::new("ATIF-v1.7", AgentProfile::new("sub", "1.0.0"));
    child.trajectory_id = Some("child-1".into());
    child
        .steps
        .push(TraceStep::new(1, StepOrigin::User, "task"));
    t.subagent_trajectories = Some(vec![child]);

    assert!(validate_trajectory(&t).is_ok());
}

// ── embedded subagent_trajectories: required trajectory_id + uniqueness ────

#[test]
fn embedded_subagent_missing_trajectory_id_fails() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    let mut child = Trajectory::new("ATIF-v1.7", AgentProfile::new("sub", "1.0.0"));
    child
        .steps
        .push(TraceStep::new(1, StepOrigin::User, "task")); // no trajectory_id set
    t.subagent_trajectories = Some(vec![child]);

    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::MissingSubagentTrajectoryId { .. })));
}

#[test]
fn embedded_subagent_duplicate_trajectory_id_fails() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    let mk_child = |id: &str| {
        let mut c = Trajectory::new("ATIF-v1.7", AgentProfile::new("sub", "1.0.0"));
        c.trajectory_id = Some(id.into());
        c.steps.push(TraceStep::new(1, StepOrigin::User, "task"));
        c
    };
    t.subagent_trajectories = Some(vec![mk_child("dup"), mk_child("dup")]);

    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::DuplicateSubagentTrajectoryId(_))));
}

#[test]
fn embedded_subagent_step_id_sequence_is_recursively_validated() {
    let mut t = base_trajectory();
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    let mut child = Trajectory::new("ATIF-v1.7", AgentProfile::new("sub", "1.0.0"));
    child.trajectory_id = Some("child-1".into());
    child
        .steps
        .push(TraceStep::new(5, StepOrigin::User, "broken sequence")); // should start at 1
    t.subagent_trajectories = Some(vec![child]);

    let errs = validate_trajectory(&t).unwrap_err();
    assert!(
        errs.iter()
            .any(|e| matches!(e, ValidationError::Nested { .. })),
        "expected the child's own StepIdSequence violation to surface as a Nested error: {errs:?}"
    );
}

// ── source_call_id referential integrity ────────────────────────────────────

#[test]
fn source_call_id_matching_a_tool_call_passes() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.tool_calls = Some(vec![ToolInvocation::new("call_1", "search")]);
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "call_1", "result",
    )));
    let mut t = base_trajectory();
    t.steps.push(step);
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn dangling_source_call_id_fails() {
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.tool_calls = Some(vec![ToolInvocation::new("call_1", "search")]);
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "call_does_not_exist",
        "result",
    )));
    let mut t = base_trajectory();
    t.steps.push(step);
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::DanglingSourceCallId { .. })));
}

// ── schema_version sanity ───────────────────────────────────────────────────

#[test]
fn schema_version_looking_like_atif_v_passes() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    assert!(validate_trajectory(&t).is_ok());
}

#[test]
fn empty_schema_version_fails() {
    let mut t = Trajectory::new("", AgentProfile::new("sven", "1.0.0"));
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::SchemaVersion(_))));
}

#[test]
fn non_atif_schema_version_fails() {
    let mut t = Trajectory::new("not-atif", AgentProfile::new("sven", "1.0.0"));
    t.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
    let errs = validate_trajectory(&t).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::SchemaVersion(_))));
}
