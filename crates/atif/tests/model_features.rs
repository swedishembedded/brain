// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Feature-level coverage for multimodal content, subagent embedding,
//! `llm_call_count = 0` deterministic dispatch, `is_copied_context`
//! filtering, and the Section VII context-management convention.

use atif::{
    validate_trajectory, AgentProfile, ContentSegment, ContextManagement, ImageMediaType,
    MessageBody, ObservationEntry, StepObservation, StepOrigin, SubagentRef, ToolInvocation,
    TraceStep, Trajectory,
};

fn roundtrip(t: &Trajectory) -> Trajectory {
    let json = serde_json::to_string_pretty(t).expect("serialize");
    serde_json::from_str(&json).expect("parse")
}

// ── Multimodal content ──────────────────────────────────────────────────────

#[test]
fn mixed_text_and_image_message_round_trips() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    let mut step = TraceStep::new(
        1,
        StepOrigin::User,
        MessageBody::segments(vec![
            ContentSegment::text("What is in this image?"),
            ContentSegment::image(ImageMediaType::Png, "images/step_1_input.png"),
        ]),
    );
    step.timestamp = Some("2025-10-11T10:30:00Z".to_string());
    t.steps.push(step);

    let restored = roundtrip(&t);
    let parts = restored.steps[0].message.as_segments().expect("segments");
    assert_eq!(parts.len(), 2);
    match &parts[0] {
        ContentSegment::Text { text } => assert_eq!(text, "What is in this image?"),
        _ => panic!("expected text segment first"),
    }
    match &parts[1] {
        ContentSegment::Image { source } => {
            assert_eq!(source.media_type, ImageMediaType::Png);
            assert_eq!(source.path, "images/step_1_input.png");
        }
        _ => panic!("expected image segment second"),
    }
}

#[test]
fn image_source_round_trips_for_each_allowed_media_type() {
    for mt in [
        ImageMediaType::Jpeg,
        ImageMediaType::Png,
        ImageMediaType::Gif,
        ImageMediaType::Webp,
    ] {
        let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
        t.steps.push(TraceStep::new(
            1,
            StepOrigin::User,
            MessageBody::segments(vec![ContentSegment::image(mt, "img.bin")]),
        ));
        let restored = roundtrip(&t);
        let parts = restored.steps[0].message.as_segments().unwrap();
        match &parts[0] {
            ContentSegment::Image { source } => assert_eq!(source.media_type, mt),
            _ => panic!("expected image"),
        }
    }
}

#[test]
fn multimodal_observation_content_round_trips() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.tool_calls = Some(vec![ToolInvocation::new("call_1", "screenshot")]);
    step.observation = Some(StepObservation::single(ObservationEntry {
        source_call_id: Some("call_1".into()),
        content: Some(MessageBody::segments(vec![
            ContentSegment::text("Captured:"),
            ContentSegment::image(ImageMediaType::Jpeg, "images/shot.jpg"),
        ])),
        subagent_trajectory_ref: None,
        extra: None,
    }));
    t.steps.push(step);

    let restored = roundtrip(&t);
    let content = restored.steps[0].observation.as_ref().unwrap().results[0]
        .content
        .as_ref()
        .unwrap();
    assert_eq!(content.as_segments().unwrap().len(), 2);
    assert!(validate_trajectory(&restored).is_ok());
}

// ── Subagent embedding ──────────────────────────────────────────────────────

#[test]
fn embedded_subagent_round_trips_and_validates() {
    let mut child = Trajectory::new("ATIF-v1.7", AgentProfile::new("sub-agent", "0.1.0"));
    child.trajectory_id = Some("child-abc".into());
    child
        .steps
        .push(TraceStep::new(1, StepOrigin::User, "Summarize this."));
    child
        .steps
        .push(TraceStep::new(2, StepOrigin::Agent, "Summary: ..."));

    let mut parent = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    parent.trajectory_id = Some("parent-1".into());
    let mut step = TraceStep::new(1, StepOrigin::Agent, "Delegating to a subagent.");
    step.observation = Some(StepObservation::single(ObservationEntry::for_subagent(
        vec![SubagentRef::by_trajectory_id("child-abc")],
    )));
    parent.steps.push(step);
    parent.subagent_trajectories = Some(vec![child]);

    assert!(validate_trajectory(&parent).is_ok());

    let restored = roundtrip(&parent);
    let children = restored.subagent_trajectories.as_ref().unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].trajectory_id.as_deref(), Some("child-abc"));
    assert_eq!(children[0].steps.len(), 2);
    assert!(validate_trajectory(&restored).is_ok());
}

// ── llm_call_count = 0 deterministic dispatch ───────────────────────────────

#[test]
fn deterministic_dispatch_step_round_trips_and_validates() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    let mut step = TraceStep::new(1, StepOrigin::Agent, "");
    step.llm_call_count = Some(0);
    step.tool_calls = Some(vec![ToolInvocation::new("call_1", "graph_edge_dispatch")]);
    t.steps.push(step);

    assert!(validate_trajectory(&t).is_ok());
    let restored = roundtrip(&t);
    assert_eq!(restored.steps[0].llm_call_count, Some(0));
    assert!(restored.steps[0].metrics.is_none());
    assert!(restored.steps[0].reasoning_content.is_none());
    assert!(validate_trajectory(&restored).is_ok());
}

// ── is_copied_context SFT filtering ─────────────────────────────────────────

#[test]
fn is_copied_context_steps_are_excluded_from_sft_iteration() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    t.steps
        .push(TraceStep::new(1, StepOrigin::User, "original turn"));

    let mut copied = TraceStep::new(2, StepOrigin::Agent, "copied from prior trajectory");
    copied.is_copied_context = Some(true);
    t.steps.push(copied);

    t.steps
        .push(TraceStep::new(3, StepOrigin::Agent, "new turn"));

    let sft_ids: Vec<u64> = t.sft_steps().map(|s| s.step_id).collect();
    assert_eq!(
        sft_ids,
        vec![1, 3],
        "step 2 (is_copied_context=true) must be excluded"
    );
}

#[test]
fn steps_without_is_copied_context_are_all_included() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    t.steps.push(TraceStep::new(1, StepOrigin::User, "a"));
    t.steps.push(TraceStep::new(2, StepOrigin::Agent, "b"));
    let sft_ids: Vec<u64> = t.sft_steps().map(|s| s.step_id).collect();
    assert_eq!(sft_ids, vec![1, 2]);
}

// ── Context management convention (Section VII) ────────────────────────────

#[test]
fn context_management_convention_round_trips_nested_in_extra() {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    let mut step = TraceStep::new(5, StepOrigin::System, "Context compaction performed");
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "n/a",
        "Summary: prior conversation covered topic X...",
    )));
    ContextManagement::new("compaction", "replace").insert_into_extra(&mut step.extra);
    t.steps.push(step);

    let restored = roundtrip(&t);
    let extra = restored.steps[0].extra.as_ref().expect("extra present");
    let cm = ContextManagement::from_extra(extra).expect("context_management present");
    assert_eq!(cm.kind, "compaction");
    assert_eq!(cm.boundary, "replace");
}
