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

/// A sink for one already-encoded protocol line, written to the transport and
/// **flushed immediately** — so a session that streams (one chunk per model step)
/// reaches the peer incrementally instead of buffering the whole turn.
pub trait LineSink {
    fn send(&mut self, line: &str) -> io::Result<()>;
}

/// Adapts any [`Write`] into a [`LineSink`] that appends a newline and flushes
/// per line (the live-streaming behavior [`pump_connection`] wants).
struct WriterSink<'a, W: Write> {
    w: &'a mut W,
}

impl<W: Write> LineSink for WriterSink<'_, W> {
    fn send(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.w, "{line}")?;
        self.w.flush()
    }
}

/// A per-connection protocol handler. Stateful across lines within one
/// connection (a `req_id` demux, a streaming turn), independent across
/// connections.
pub trait Session {
    /// Process one inbound JSONL line; return zero or more response lines
    /// (already JSON-encoded, no trailing newline). The simple, buffered form.
    fn on_line(&mut self, line: &str) -> Vec<String>;

    /// Stream response lines to `out` **as they are produced**, flushing per line.
    /// The default forwards the whole [`on_line`](Session::on_line) batch, so a
    /// buffered session streams trivially; a session that can produce output
    /// incrementally (e.g. [`ControllerSession`] over a token stream) overrides
    /// this to flush each chunk to the wire the moment it is generated.
    fn on_line_streaming(&mut self, line: &str, out: &mut dyn LineSink) -> io::Result<()> {
        for l in self.on_line(line) {
            out.send(&l)?;
        }
        Ok(())
    }

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
        // Stream each response line straight to the writer (flushed per line) as
        // the session produces it. `sink` borrows `writer` for the duration of the
        // turn; scope it so `writer` is free again for the panic path below.
        let outcome = {
            let mut sink = WriterSink { w: &mut writer };
            catch_unwind(AssertUnwindSafe(|| session.on_line_streaming(&line, &mut sink)))
        };
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e), // a genuine write/IO error: propagate
            Err(_) => {
                let _ = writeln!(writer, "{}", panic_error_line());
                let _ = writer.flush();
                break; // close the connection; session state is untrustworthy
            }
        }
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

    /// A [`Write`] that records the order of `write`/`flush` calls, so a test can
    /// prove output is flushed per line (streamed) rather than batched.
    #[derive(Default)]
    struct RecordingWriter {
        log: Vec<String>,
    }
    impl Write for RecordingWriter {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.log.push(format!("write:{}", String::from_utf8_lossy(b).trim_end()));
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.log.push("flush".into());
            Ok(())
        }
    }

    /// A session that emits two lines incrementally via the streaming sink.
    struct StreamingSession;
    impl Session for StreamingSession {
        fn on_line(&mut self, _line: &str) -> Vec<String> {
            unreachable!("streaming path must be used")
        }
        fn on_line_streaming(&mut self, _line: &str, out: &mut dyn LineSink) -> io::Result<()> {
            out.send("a")?;
            out.send("b")?;
            Ok(())
        }
    }

    #[test]
    fn streaming_flushes_after_every_line() {
        let mut w = RecordingWriter::default();
        pump_connection(Cursor::new(b"x\n".to_vec()), &mut w, &mut StreamingSession).unwrap();
        // The two streamed lines are each flushed as produced — a flush lands
        // between the two content writes, not one batched flush at the end.
        // (`writeln!` splits into a content write + a newline write, so match the
        // content markers and require a flush between them.)
        let ia = w.log.iter().position(|e| e == "write:a").expect("wrote a");
        let ib = w.log.iter().position(|e| e == "write:b").expect("wrote b");
        assert!(ia < ib, "lines out of order: {:?}", w.log);
        assert!(
            w.log[ia..ib].iter().any(|e| e == "flush"),
            "first line must flush before the second is written (live streaming): {:?}",
            w.log
        );
    }

    #[test]
    fn controller_session_streams_audio_chunks_live() {
        // The whole chain: a fake TTS model → Controller → ControllerSession →
        // wire. A synth request must produce several audio_chunk lines, each
        // flushed to the writer as generated (not one batch at turn end).
        let reg = runtime::Registry {
            synth: Some(Box::new(runtime::FakeSynthModel::default())),
            ..Default::default()
        };
        let mut session = ControllerSession::new(runtime::Controller::new(reg));
        let req = events::encode_line(&events::Event::UserSynthRequest {
            text: "the quick brown fox jumps over the lazy dog twice and then some".into(),
            ref_audio: None,
            ref_text: None,
            language: None,
        });
        let input = format!("{req}\n");
        let mut w = RecordingWriter::default();
        pump_connection(Cursor::new(input.into_bytes()), &mut w, &mut session).unwrap();

        let chunk_idxs: Vec<usize> =
            w.log.iter().enumerate().filter(|(_, e)| e.contains("audio_chunk")).map(|(i, _)| i).collect();
        assert!(chunk_idxs.len() >= 2, "expected several streamed audio chunks: {:?}", w.log);
        // Consecutive chunks are separated by a flush — each reaches the wire
        // before the next is generated (live streaming, not one batch at the end).
        for pair in chunk_idxs.windows(2) {
            assert!(
                w.log[pair[0]..pair[1]].iter().any(|e| e == "flush"),
                "no flush between streamed chunks: {:?}",
                w.log
            );
        }
    }

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
