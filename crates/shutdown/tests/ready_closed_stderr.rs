// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for servers whose readiness signal
// survives a dead log pipe for its clients. If your team needs expertise in
// service lifecycle and graceful shutdown then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `Gate::bound` documents that it never panics. Its log line goes to stderr,
//! and a plain `eprintln!` panics when stderr's reader is gone, which would
//! leave the readiness marker uncreated or the surface thread dead. The gate
//! runs in a child process (this test binary re-run on [`CHILD`]) because a
//! broken pipe on stderr cannot be arranged in-process.

#![cfg(unix)]

use std::io::Write;
use std::process::{Command, Stdio};

const CHILD: &str = "child_binds_the_gate";

/// The child half: reads the marker path from stdin, which the parent writes
/// only after closing its stderr reader, then binds the gate. A panic in
/// `bound` fails this test and the child exits 101.
#[test]
#[ignore = "runs only as the child of the ready test"]
fn child_binds_the_gate() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("the marker path from the parent");
    let path = std::path::PathBuf::from(line.trim_end_matches('\n'));
    let gate = brain_shutdown::ready::Gate::touching(&path, 1).expect("gate");
    gate.bound("child");
    assert!(gate.is_ready());
}

#[test]
fn bound_creates_the_marker_when_the_stderr_reader_is_gone() {
    let dir = std::env::temp_dir().join(format!("brain-ready-closed-stderr-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join("ready");
    let mut child = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--ignored", "--exact", CHILD, "--nocapture", "--test-threads=1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child");
    drop(child.stderr.take()); // every write to the child's stderr now fails with EPIPE
    let mut path = marker.to_str().expect("a UTF-8 temporary path").as_bytes().to_vec();
    path.push(b'\n');
    child.stdin.take().expect("piped stdin").write_all(&path).unwrap();
    let status = child.wait().expect("wait for the child");
    let created = marker.exists();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(status.success(), "Gate::bound must not panic when stderr is closed (child: {status})");
    assert!(created, "the readiness marker must exist");
}
