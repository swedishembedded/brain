// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Running a check alone. Every integration test shares this one binary and
//! runs beside the others, so a check that changes process-wide state (a
//! placement policy, the kernel-upgrade or native-kernel switch, the panic
//! hook) or reads process-wide counters runs in a child process of its own.

/// Run `test`, an ignored test of the module at `module_path` (pass
/// `module_path!()`), alone in a child process of this binary, and require it
/// to have run and passed.
pub fn alone(module_path: &str, test: &str) {
    let name = format!("{}::{test}", module_path.trim_start_matches("it::"));
    let out = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args(["--exact", &name, "--ignored", "--nocapture", "--test-threads=1"])
        .output()
        .expect("re-run this test binary for the child");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{name} failed:\n{text}");
    assert!(text.contains("1 passed"), "{name} did not run:\n{text}");
}
