// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Building and installing the one global `tracing_subscriber`.
//!
//! Deliberately thin: an [`EnvFilter`] built from [`Config::directives`] plus
//! `tracing_subscriber`'s own `fmt` layer. The per-line component label, the
//! level filtering and the JSON encoding are all the subscriber's - nothing
//! here re-implements them.

use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex, MutexGuard};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::{Config, Format, Output};

/// Where a built subscriber writes. `Buffer` exists so the format and level
/// behaviour can be asserted against real captured output rather than
/// inspected indirectly, and so an embedder can collect a trace in-process.
#[derive(Clone)]
pub enum Sink {
    Stdout,
    File(Arc<Mutex<File>>),
    Buffer(Arc<Mutex<Vec<u8>>>),
}

/// A [`Sink`]'s per-event writer. The lock is taken for the WHOLE event and
/// released when the formatter drops this, so two threads' events cannot
/// interleave mid-line.
pub enum SinkWriter<'a> {
    Stdout(io::StdoutLock<'static>),
    File(MutexGuard<'a, File>),
    Buffer(MutexGuard<'a, Vec<u8>>),
}

impl Write for SinkWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            SinkWriter::Stdout(w) => w.write(buf),
            SinkWriter::File(w) => w.write(buf),
            SinkWriter::Buffer(w) => w.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            SinkWriter::Stdout(w) => w.flush(),
            SinkWriter::File(w) => w.flush(),
            SinkWriter::Buffer(w) => w.flush(),
        }
    }
}

impl<'a> MakeWriter<'a> for Sink {
    type Writer = SinkWriter<'a>;
    fn make_writer(&'a self) -> Self::Writer {
        match self {
            // A poisoned lock here means some other thread panicked WHILE
            // formatting an event. The trace is a debugging aid, so recover
            // the guard and keep writing rather than turning a panic anywhere
            // in the process into a second panic inside the logger.
            Sink::Stdout => SinkWriter::Stdout(io::stdout().lock()),
            Sink::File(f) => SinkWriter::File(f.lock().unwrap_or_else(|e| e.into_inner())),
            Sink::Buffer(b) => SinkWriter::Buffer(b.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }
}

/// Build the subscriber `cfg` describes, writing to `sink`. Separate from
/// [`install_to`] so a test can run it under
/// `tracing::subscriber::with_default` instead of claiming the process-global
/// slot.
pub fn subscriber(cfg: &Config, sink: Sink) -> Result<Box<dyn tracing::Subscriber + Send + Sync>, String> {
    // Everything is off unless a directive turns it on, so a family that was
    // not asked for stays silent even when some other family is at level 5 -
    // and no third-party crate's own events leak into brain's trace.
    let mut filter = EnvFilter::new("off");
    for d in cfg.directives() {
        let directive = d.parse().map_err(|e| format!("brain-trace: bad filter directive {d:?}: {e}"))?;
        filter = filter.add_directive(directive);
    }

    // Colour only when a human is actually looking at a terminal: a file (or
    // a piped stdout) would otherwise get ANSI escapes baked into it, which
    // is exactly what breaks grepping a trace after the fact.
    let ansi = matches!(sink, Sink::Stdout) && io::stdout().is_terminal();
    let layer = match cfg.format {
        Format::Text => tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_ansi(ansi)
            .with_writer(sink)
            .boxed(),
        Format::Json => tracing_subscriber::fmt::layer()
            .json()
            .with_target(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_ansi(false)
            .with_writer(sink)
            .boxed(),
    };
    Ok(Box::new(tracing_subscriber::registry().with(filter).with(layer)))
}

/// Install the global subscriber for `cfg`, opening [`Output::File`] if that
/// is where output goes.
///
/// Returns whether a subscriber was installed: `Ok(false)` means every family
/// is at level 0, so nothing was installed and the process runs exactly as it
/// did before this feature existed - `tracing`'s no-subscriber path, not a
/// subscriber that filters everything out.
pub fn install(cfg: &Config) -> Result<bool, String> {
    let sink = match &cfg.output {
        Output::Stdout => Sink::Stdout,
        Output::File(path) => {
            if cfg.is_off() {
                // Nothing would be written; do not truncate a file the user
                // named on a run that traces nothing.
                return Ok(false);
            }
            let f = File::create(path).map_err(|e| format!("--trace-output {}: {e}", path.display()))?;
            Sink::File(Arc::new(Mutex::new(f)))
        }
    };
    install_to(cfg, sink)
}

/// [`install`] against an already-built [`Sink`].
pub fn install_to(cfg: &Config, sink: Sink) -> Result<bool, String> {
    if cfg.is_off() {
        return Ok(false);
    }
    let sub = subscriber(cfg, sink)?;
    tracing::subscriber::set_global_default(sub).map_err(|e| format!("brain-trace: a tracing subscriber is already installed ({e})"))?;
    Ok(true)
}
