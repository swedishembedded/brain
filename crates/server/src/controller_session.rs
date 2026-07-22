// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The canonical [`Session`]: a per-connection [`runtime::Controller`] speaking
//! the `events::Event` protocol.

use crate::{LineSink, Session};
use events::{Envelope, Event};
use runtime::{Controller, Emit};
use std::io;

/// Wraps a [`Controller`] as a [`Session`]. Each connection owns one, so the
/// controller's per-turn `req_id` demux and any streaming state are per-client.
pub struct ControllerSession {
    ctrl: Controller,
    greeted: bool,
}

impl ControllerSession {
    pub fn new(ctrl: Controller) -> ControllerSession {
        ControllerSession { ctrl, greeted: false }
    }
}

impl Session for ControllerSession {
    fn greeting(&mut self) -> Vec<String> {
        // announce readiness exactly once, as `brain run` does before its loop
        self.greeted = true;
        vec![events::encode_envelope(&Envelope::bare(Event::Ready))]
    }

    fn on_line(&mut self, line: &str) -> Vec<String> {
        self.ctrl.feed_line(line).iter().map(events::encode_envelope).collect()
    }

    /// Stream the controller's emissions to the wire the moment each is produced,
    /// rather than buffering the whole turn — so a long token/audio stream reaches
    /// the client live. The controller's own `feed_line_streaming` flushes each
    /// envelope to the adapter sink below.
    fn on_line_streaming(&mut self, line: &str, out: &mut dyn LineSink) -> io::Result<()> {
        // Encode each emitted envelope to a JSONL line and push it to `out`. The
        // runtime `Emit::emit` can't return an error, so we stash the first IO
        // error and report it after the turn (a broken pipe stops the connection).
        struct EnvSink<'a> {
            out: &'a mut dyn LineSink,
            err: io::Result<()>,
        }
        impl Emit for EnvSink<'_> {
            fn emit(&mut self, env: Envelope) {
                if self.err.is_ok() {
                    self.err = self.out.send(&events::encode_envelope(&env));
                }
            }
        }
        let mut sink = EnvSink { out, err: Ok(()) };
        // No transport-level control source yet: a `cancel` is delivered as the
        // next line between turns (still recoverable). Mid-stream wire cancel needs
        // a non-blocking side channel — the controller seam (`Control`) is ready.
        self.ctrl.feed_line_streaming(line, &mut sink, &mut ());
        sink.err
    }
}
