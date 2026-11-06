// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for exposing local inference
// services safely to the processes that need them for its clients. If your team
// needs expertise in hardening service sockets then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The event socket is local-only and bounded: its file admits only the
//! serving user, and a peer cannot make the server buffer an unbounded line.

#![cfg(unix)]

use server::{pump_connection, serve_unix, Session, ServeOpts, MAX_LINE_BYTES};
use std::io::Cursor;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

struct Echo;
impl Session for Echo {
    fn on_line(&mut self, line: &str) -> Vec<String> {
        vec![format!("{{\"len\":{}}}", line.len())]
    }
}

#[test]
fn the_socket_file_admits_only_its_owner() {
    let dir = std::env::temp_dir().join(format!("brain-sock-mode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s.sock");
    let p2 = path.clone();
    let make: server::transport::SessionFactory = Arc::new(|| Box::new(Echo));
    thread::spawn(move || {
        let _ = serve_unix(&p2, make, ServeOpts::default());
    });
    // The mode is set before the first accept, so a connection proves it.
    let mut connected = false;
    for _ in 0..400 {
        if std::os::unix::net::UnixStream::connect(&path).is_ok() {
            connected = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(connected, "server never came up");
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(mode, 0o600, "socket mode {mode:o}");
}

#[test]
fn an_oversized_line_is_refused_and_closes_the_connection() {
    let mut input = vec![b'a'; MAX_LINE_BYTES + 1];
    input.extend_from_slice(b"\nnext\n");
    let mut out = Vec::new();
    pump_connection(Cursor::new(input), &mut out, &mut Echo).unwrap();
    let out = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 1, "nothing after the refusal is processed: {out:?}");
    assert!(lines[0].contains("\"code\":\"line_too_long\""), "{out:?}");
}

#[test]
fn a_line_at_the_limit_is_served() {
    let mut input = vec![b'a'; MAX_LINE_BYTES];
    input.push(b'\n');
    let mut out = Vec::new();
    pump_connection(Cursor::new(input), &mut out, &mut Echo).unwrap();
    assert_eq!(String::from_utf8(out).unwrap().trim(), format!("{{\"len\":{MAX_LINE_BYTES}}}"));
}
