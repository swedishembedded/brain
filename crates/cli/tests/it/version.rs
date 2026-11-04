// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

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

fn assert_version_flag(flag: &str) {
    let output = Command::new(bin())
        .arg(flag)
        .output()
        .expect("run brain --version");
    assert!(
        output.status.success(),
        "brain {flag} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("brain {}\n", env!("CARGO_PKG_VERSION"))
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn version_flags_report_workspace_version_without_backend_setup() {
    assert_version_flag("--version");
    assert_version_flag("-V");
}
