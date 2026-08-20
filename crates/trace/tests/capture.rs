// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end behaviour of a built subscriber, asserted against REAL captured
//! output rather than against the directives that were supposed to produce it.
//!
//! A directive list can be right while the subscriber built from it is wrong
//! (a filter attached to the wrong layer, a format that drops the target, a
//! level that is off by one). These tests emit real events through the real
//! `tracing` macros and read back what a user would have seen, which is the
//! only thing that proves the chain registry -> directives -> filter ->
//! renderer actually holds.
//!
//! `tracing::subscriber::with_default` is used instead of the global install
//! because the global slot can be claimed exactly once per process, and these
//! cases need several different configurations.

use std::sync::{Arc, Mutex};

use brain_trace::{Config, Format, Sink};

/// Emit one event at every level, from two different components, under the
/// subscriber `cfg` describes; return everything that was written.
fn capture(cfg: &Config) -> String {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let sub = brain_trace::subscriber(cfg, Sink::Buffer(buf.clone())).expect("subscriber builds");
    tracing::subscriber::with_default(sub, || {
        tracing::error!(target: "ltxv::pipeline", step = 1, "boom");
        tracing::warn!(target: "ltxv::pipeline", step = 1, "degraded");
        tracing::info!(target: "ltxv::pipeline", step = 1, "started");
        tracing::debug!(target: "ltxv::dit", layer = 7, "cache miss");
        tracing::trace!(target: "ltxv::dit", layer = 7, "per-iteration detail");
        tracing::error!(target: "gpu_core::devices", "adapter enumeration failed");
    });
    let out = buf.lock().expect("not poisoned").clone();
    String::from_utf8(out).expect("the fmt layer writes utf-8")
}

fn cfg(families: &[(&str, u8)], format: Format) -> Config {
    Config {
        families: families.iter().map(|(n, l)| (n.to_string(), *l)).collect(),
        format,
        ..Default::default()
    }
}

/// The 0-5 scale is a real gradient, not a boolean: each level admits exactly
/// the levels at or above it in severity and nothing below.
#[test]
fn each_level_admits_exactly_the_levels_at_or_above_it() {
    let messages = ["boom", "degraded", "started", "cache miss", "per-iteration detail"];
    for level in 1..=5u8 {
        let out = capture(&cfg(&[("ltxv", level)], Format::Text));
        for (i, m) in messages.iter().enumerate() {
            let want = i < level as usize;
            assert_eq!(out.contains(m), want, "at --trace-ltxv {level}, {m:?} present={} but expected {want}\n{out}", out.contains(m));
        }
    }
}

/// Level 0 must produce NOTHING - no subscriber, no output, no file
/// truncation - so an untraced run behaves exactly as it did before.
#[test]
fn level_zero_emits_nothing_at_all() {
    let off = cfg(&[("ltxv", 0), ("gpu", 0)], Format::Text);
    assert!(off.is_off());
    assert_eq!(brain_trace::install_to(&off, Sink::Buffer(Arc::new(Mutex::new(Vec::new())))), Ok(false));
    assert_eq!(capture(&off), "");
}

/// Asking for one family must not turn on another. `--trace-ltxv 5` while
/// debugging video must not bury the reader in GPU events, and vice versa.
#[test]
fn a_family_is_isolated_from_the_others() {
    let out = capture(&cfg(&[("ltxv", 5)], Format::Text));
    assert!(out.contains("per-iteration detail"), "{out}");
    assert!(!out.contains("adapter enumeration failed"), "ltxv tracing leaked gpu events:\n{out}");

    let out = capture(&cfg(&[("gpu", 5)], Format::Text));
    assert!(out.contains("adapter enumeration failed"), "{out}");
    assert!(!out.contains("per-iteration detail"), "gpu tracing leaked ltxv events:\n{out}");

    let out = capture(&cfg(&[("ltxv", 5), ("gpu", 1)], Format::Text));
    assert!(out.contains("per-iteration detail") && out.contains("adapter enumeration failed"), "{out}");
}

/// Text output must say which component each line came from - the whole
/// reason the target is printed - and must carry the event's fields, not just
/// its message.
#[test]
fn text_output_labels_the_component_and_keeps_fields() {
    let out = capture(&cfg(&[("ltxv", 5)], Format::Text));
    let line = out.lines().find(|l| l.contains("cache miss")).expect("the debug event was emitted");
    assert!(line.contains("ltxv::dit"), "no component label on: {line}");
    assert!(line.contains("DEBUG"), "no level on: {line}");
    assert!(line.contains("layer=7"), "field dropped from: {line}");
    // No ANSI escapes when the sink is not an interactive terminal, or a
    // saved trace is unreadable in anything but a terminal.
    assert!(!out.contains('\u{1b}'), "ANSI escapes leaked into non-terminal output");
}

/// JSON output must be genuinely parseable, one object per line, with the
/// target as a real member - not a message string that happens to mention it.
#[test]
fn json_output_is_parseable_with_the_target_as_a_field() {
    let out = capture(&cfg(&[("ltxv", 5)], Format::Json));
    let mut seen = 0;
    for line in out.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("not JSON: {e}\n{line}"));
        assert!(v.get("target").and_then(|t| t.as_str()).is_some_and(|t| t.starts_with("ltxv")), "target is not a field of {line}");
        assert!(v.get("level").is_some(), "level is not a field of {line}");
        seen += 1;
    }
    assert_eq!(seen, 5, "expected one JSON object per emitted ltxv event:\n{out}");

    let one = out.lines().find(|l| l.contains("cache miss")).expect("the debug event was emitted");
    let v: serde_json::Value = serde_json::from_str(one).expect("parses");
    assert_eq!(v["target"], "ltxv::dit");
    assert_eq!(v["level"], "DEBUG");
    assert_eq!(v["fields"]["layer"], 7);
}

/// A span's name and fields must reach the output, since `#[instrument]` is
/// how a unit of work (a whole denoise run, one streamed forward) is scoped -
/// an event inside it is only interpretable with that context attached.
#[test]
fn span_context_reaches_json_output() {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let sub = brain_trace::subscriber(&cfg(&[("ltxv", 5)], Format::Json), Sink::Buffer(buf.clone())).expect("builds");
    tracing::subscriber::with_default(sub, || {
        let span = tracing::info_span!(target: "ltxv::pipeline", "denoise", steps = 8);
        let _e = span.enter();
        tracing::debug!(target: "ltxv::pipeline", step = 3, "stepping");
    });
    let out = String::from_utf8(buf.lock().expect("not poisoned").clone()).expect("utf-8");
    let line = out.lines().find(|l| l.contains("stepping")).expect("the event was emitted");
    let v: serde_json::Value = serde_json::from_str(line).expect("parses");
    assert_eq!(v["span"]["name"], "denoise");
    assert_eq!(v["span"]["steps"], 8);
}
