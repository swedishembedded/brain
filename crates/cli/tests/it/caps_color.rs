// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain caps` is read by people and by pipes: colour codes belong on a
//! terminal only. A captured stdout (what `Command::output` gives) is never a
//! terminal, so the listing must come out as plain text there.

use std::process::Command;

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

#[test]
fn caps_to_a_pipe_has_no_escape_sequences() {
    let output = Command::new(bin()).args(["caps", "brain/demo"]).env_remove("NO_COLOR").output().expect("run brain caps");
    assert!(output.status.success(), "brain caps failed: {}", String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("brain/demo"), "{text}");
    assert!(!text.contains('\x1b'), "colour codes on a non-terminal stdout: {text:?}");
}
