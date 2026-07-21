// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The canonical [`Session`]: a per-connection [`runtime::Controller`] speaking
//! the `events::Event` protocol.

use crate::Session;
use events::{Envelope, Event};
use runtime::Controller;

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
}
