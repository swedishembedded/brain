// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! NDJSON step streaming tests: write N steps as independently-valid lines,
//! read them back in order, tolerate a trailing blank line, and report a
//! clear per-line error on malformed input.

use atif::persist::{read_steps_ndjson, write_steps_ndjson, NdjsonError};
use atif::{StepOrigin, TraceStep};

fn steps(n: u64) -> Vec<TraceStep> {
    (1..=n)
        .map(|i| TraceStep::new(i, StepOrigin::User, format!("turn {i}")))
        .collect()
}

#[test]
fn write_then_read_round_trips_in_order() {
    let mut buf = Vec::new();
    write_steps_ndjson(&mut buf, &steps(5)).unwrap();

    let text = String::from_utf8(buf).unwrap();
    // Every non-empty line must be an independently-valid standalone step object.
    for line in text.lines() {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("each NDJSON line parses alone");
        assert!(v.get("step_id").is_some());
    }

    let read_back = read_steps_ndjson(text.as_bytes()).unwrap();
    assert_eq!(read_back.len(), 5);
    for (i, step) in read_back.iter().enumerate() {
        assert_eq!(step.step_id, (i + 1) as u64);
        assert_eq!(
            step.message.as_text(),
            Some(format!("turn {}", i + 1)).as_deref()
        );
    }
}

#[test]
fn tolerates_trailing_blank_lines() {
    let mut buf = Vec::new();
    write_steps_ndjson(&mut buf, &steps(2)).unwrap();
    let mut text = String::from_utf8(buf).unwrap();
    text.push_str("\n\n\n"); // extra trailing blank lines

    let read_back = read_steps_ndjson(text.as_bytes()).unwrap();
    assert_eq!(read_back.len(), 2);
}

#[test]
fn malformed_line_reports_which_line_failed() {
    let mut buf = Vec::new();
    write_steps_ndjson(&mut buf, &steps(2)).unwrap();
    let mut text = String::from_utf8(buf).unwrap();
    text.push_str("{not valid json\n");

    let err = read_steps_ndjson(text.as_bytes()).unwrap_err();
    match err {
        NdjsonError::Parse { line, .. } => {
            assert_eq!(line, 3, "the malformed line is the 3rd line")
        }
        other => panic!("expected NdjsonError::Parse, got {other:?}"),
    }
}

#[test]
fn empty_input_yields_no_steps() {
    let read_back = read_steps_ndjson(&b""[..]).unwrap();
    assert!(read_back.is_empty());
}
