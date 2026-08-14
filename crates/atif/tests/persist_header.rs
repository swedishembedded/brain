// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Header-only quick-read tests: fast path on small and large trajectories,
//! and graceful fallback to a full parse on malformed/truncated input.

use atif::persist::{read_trajectory_header, read_trajectory_header_fast};
use atif::{AgentProfile, FinalMetrics, StepOrigin, TraceStep, Trajectory};

fn big_trajectory(n: u64) -> Trajectory {
    let mut t = Trajectory::new(
        "ATIF-v1.7",
        AgentProfile::new("sven", "1.0.0").with_model("claude"),
    );
    t.session_id = Some("session-xyz".into());
    t.trajectory_id = Some("traj-xyz".into());
    t.notes = Some("a big trajectory".into());
    t.final_metrics = Some(FinalMetrics {
        total_steps: Some(n),
        ..Default::default()
    });
    for i in 1..=n {
        t.steps
            .push(TraceStep::new(i, StepOrigin::User, format!("turn {i}")));
    }
    t
}

#[test]
fn fast_path_header_matches_full_parse_on_small_trajectory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.json");
    let full = big_trajectory(2);
    std::fs::write(&path, serde_json::to_string_pretty(&full).unwrap()).unwrap();

    let header = read_trajectory_header_fast(&path)
        .expect("fast path must succeed on our own serializer output");
    assert_eq!(header.schema_version, full.schema_version);
    assert_eq!(header.session_id, full.session_id);
    assert_eq!(header.trajectory_id, full.trajectory_id);
    assert_eq!(header.agent.name, full.agent.name);
    assert_eq!(header.notes, full.notes);
    assert_eq!(
        header.final_metrics.unwrap().total_steps,
        full.final_metrics.unwrap().total_steps
    );
}

#[test]
fn fast_path_header_matches_full_parse_on_large_trajectory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.json");
    let full = big_trajectory(5_000);
    std::fs::write(&path, serde_json::to_string_pretty(&full).unwrap()).unwrap();

    let header =
        read_trajectory_header_fast(&path).expect("fast path must succeed on a large document too");
    assert_eq!(header.schema_version, "ATIF-v1.7");
    assert_eq!(header.agent.name, "sven");
    assert_eq!(header.final_metrics.unwrap().total_steps, Some(5_000));
}

#[test]
fn public_reader_falls_back_gracefully_when_fast_path_heuristic_misses() {
    // Hand-construct JSON where `steps` is NOT the last field (violates the
    // convention our own serializer follows), so the depth-tracking scan for
    // a *trailing* top-level "steps" key won't find a match the way it would
    // on our own output - this must not be treated as a hard failure: the
    // public reader falls back to a full `Trajectory` parse and still
    // extracts the same header fields.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.json");
    let json = r#"{
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"step_id": 1, "source": "user", "message": "hi"}
        ],
        "agent": {"name": "sven", "version": "1.0.0"},
        "session_id": "s-1"
    }"#;
    std::fs::write(&path, json).unwrap();

    // The fast heuristic (which expects `steps` last) should not find a
    // usable split point here and must report that cleanly...
    assert!(read_trajectory_header_fast(&path).is_err());

    // ...while the defensive public entry point falls back to a full parse
    // and still returns the right header fields.
    let header = read_trajectory_header(&path).expect("fallback must succeed via full parse");
    assert_eq!(header.schema_version, "ATIF-v1.7");
    assert_eq!(header.agent.name, "sven");
    assert_eq!(header.session_id.as_deref(), Some("s-1"));
}

#[test]
fn header_includes_continued_trajectory_ref() {
    // The header type documents itself as "everything except steps and
    // subagent_trajectories" - continued_trajectory_ref is header data (it is
    // how a continued trajectory is discovered without parsing steps).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("continued.json");
    let mut full = big_trajectory(2);
    full.continued_trajectory_ref = Some("next-segment.json".into());
    std::fs::write(&path, serde_json::to_string_pretty(&full).unwrap()).unwrap();

    let header = read_trajectory_header_fast(&path).unwrap();
    assert_eq!(
        header.continued_trajectory_ref.as_deref(),
        Some("next-segment.json")
    );
}

#[test]
fn fast_path_splits_before_subagent_trajectories() {
    // subagent_trajectories is declared before steps, and can dwarf the
    // header. The fast path must stop BEFORE it - proven here by making its
    // contents unparseable: a reader that includes (or parses-and-discards)
    // it would fail.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subagents.json");
    let json = r#"{
        "schema_version": "ATIF-v1.7",
        "session_id": "s-1",
        "agent": {"name": "sven", "version": "1.0.0"},
        "subagent_trajectories": [ {"this is": not even json ],
        "steps": []
    }"#;
    std::fs::write(&path, json).unwrap();

    let header = read_trajectory_header_fast(&path)
        .expect("header read must not touch subagent_trajectories");
    assert_eq!(header.session_id.as_deref(), Some("s-1"));
}

#[test]
fn fast_path_reads_only_the_header_prefix_not_the_whole_file() {
    // A "header-only fast read" must not slurp the entire file. Proven by
    // appending invalid UTF-8 inside the steps array, past the split point:
    // a whole-file read_to_string chokes on it; a streaming prefix read
    // never sees it.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binary-tail.json");
    let mut bytes = br#"{
        "schema_version": "ATIF-v1.7",
        "session_id": "s-2",
        "agent": {"name": "sven", "version": "1.0.0"},
        "steps": ["#
        .to_vec();
    bytes.extend(vec![0xFF; 256 * 1024]); // invalid UTF-8, never valid JSON
    std::fs::write(&path, bytes).unwrap();

    let header = read_trajectory_header_fast(&path)
        .expect("the fast path must stop reading at the steps split point");
    assert_eq!(header.session_id.as_deref(), Some("s-2"));
    assert_eq!(header.agent.name, "sven");
}

#[test]
fn public_reader_errors_cleanly_on_genuinely_truncated_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.json");
    let full = big_trajectory(3);
    let mut json = serde_json::to_string_pretty(&full).unwrap();
    json.truncate(json.len() / 2); // chop the file in half, mid-document
    std::fs::write(&path, json).unwrap();

    // Neither the fast path nor a full parse can make sense of this; the
    // public reader must return an error rather than panic or fabricate data.
    assert!(read_trajectory_header(&path).is_err());
}
