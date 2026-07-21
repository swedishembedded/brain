// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One protocol, three transports.
//!
//! brain has, historically, spoken two divergent JSONL dialects: the runtime's
//! [`events::Event`] protocol on stdio, and an ad-hoc dialect on the TTS Unix
//! socket. This crate unifies them: **every** transport — stdio, TCP, Unix
//! socket — carries the identical `events::Event` frames, so a client's demux
//! logic is transport-agnostic and `brain-py` can swap the underlying stream
//! without touching its request methods.
//!
//! Design (in-policy: `std::thread` + `std::net`, no async runtime):
//! - A [`Session`] processes one inbound line and returns response lines. The
//!   canonical implementation, [`ControllerSession`], wraps a
//!   [`runtime::Controller`].
//! - [`pump_connection`] is the transport-independent core: greet, then read
//!   lines and write responses until EOF, with **`catch_unwind` panic
//!   isolation** so a panicking model closes only its own connection.
//! - [`serve_unix`] / [`serve_tcp`] accept connections, cap concurrency, and run
//!   one [`Session`] per connection on its own thread. Each connection builds
//!   its own session via a factory, so per-instance model state never crosses
//!   threads (throughput later comes from N replicas, one per worker).

use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub mod controller_session;
pub mod transport;

pub use controller_session::ControllerSession;
pub use transport::{serve_tcp, serve_unix, ServeOpts};

/// A per-connection protocol handler. Stateful across lines within one
/// connection (a `req_id` demux, a streaming turn), independent across
/// connections.
pub trait Session {
    /// Process one inbound JSONL line; return zero or more response lines
    /// (already JSON-encoded, no trailing newline).
    fn on_line(&mut self, line: &str) -> Vec<String>;

    /// Lines to emit once when the connection opens (e.g. a `ready` event).
    fn greeting(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// The transport-independent connection loop: emit the greeting, then read
/// newline-delimited lines and write the response lines for each, flushing per
/// line. Blank lines are ignored. Returns when the reader hits EOF.
///
/// A panic inside [`Session::on_line`] is caught: an `error` line is written and
/// the connection is closed (rather than continuing with a possibly-poisoned
/// session). This isolates a misbehaving model to its own connection.
pub fn pump_connection<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    session: &mut dyn Session,
) -> io::Result<()> {
    for line in session.greeting() {
        writeln!(writer, "{line}")?;
    }
    writer.flush()?;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let responses = match catch_unwind(AssertUnwindSafe(|| session.on_line(&line))) {
            Ok(r) => r,
            Err(_) => {
                let _ = writeln!(writer, "{}", panic_error_line());
                let _ = writer.flush();
                break; // close the connection; session state is untrustworthy
            }
        };
        for r in responses {
            writeln!(writer, "{r}")?;
        }
        writer.flush()?;
    }
    Ok(())
}

/// A structured `error` line emitted when a handler panics.
fn panic_error_line() -> String {
    serde_json::json!({
        "event": "error",
        "code": "internal",
        "message": "handler panicked",
        "retryable": false,
    })
    .to_string()
}

/// Serve a single [`Session`] over blocking stdin/stdout — the classic `brain
/// run` transport, now sharing the unified `Session` core.
pub fn serve_stdio(session: &mut dyn Session) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    pump_connection(stdin.lock(), stdout.lock(), session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A trivial session: echoes each line's length and greets with `ready`.
    struct LenSession;
    impl Session for LenSession {
        fn on_line(&mut self, line: &str) -> Vec<String> {
            vec![format!("{{\"len\":{}}}", line.len())]
        }
        fn greeting(&mut self) -> Vec<String> {
            vec!["{\"event\":\"ready\"}".into()]
        }
    }

    /// A session that panics on a magic line.
    struct PanicSession;
    impl Session for PanicSession {
        fn on_line(&mut self, line: &str) -> Vec<String> {
            if line == "boom" {
                panic!("boom");
            }
            vec![format!("ok:{line}")]
        }
    }

    fn run(session: &mut dyn Session, input: &str) -> String {
        let mut out = Vec::new();
        pump_connection(Cursor::new(input.as_bytes()), &mut out, session).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn greeting_then_line_responses() {
        let out = run(&mut LenSession, "abc\nde\n");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "{\"event\":\"ready\"}");
        assert_eq!(lines[1], "{\"len\":3}");
        assert_eq!(lines[2], "{\"len\":2}");
    }

    #[test]
    fn blank_lines_are_skipped() {
        let out = run(&mut LenSession, "\n\nx\n");
        // greeting + exactly one response for "x"
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2, "{out:?}");
        assert_eq!(lines[1], "{\"len\":1}");
    }

    #[test]
    fn panic_is_isolated_to_the_connection() {
        let out = run(&mut PanicSession, "ok1\nboom\nok2\n");
        // first line handled, panic yields a structured error, then the
        // connection closes (ok2 never processed).
        assert!(out.contains("ok:ok1"));
        assert!(out.contains("\"code\":\"internal\""));
        assert!(!out.contains("ok:ok2"), "must not process after a panic");
    }
}
