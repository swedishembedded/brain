// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One concise per-architecture weight-load line, driven by
//! `brain_checkpoint::load_progress`.
//!
//! An arch command's pipeline prints its own coarse phase
//! (`ltxv [0/14] build transformer...`) and then goes quiet for the seconds
//! to minutes a multi-gigabyte checkpoint takes to stream from disk. This
//! module keeps that stretch alive: it installs the load-progress observer
//! once, at dispatch, under the architecture's own label, and renders what
//! the checkpoint readers report as one line on stderr:
//!
//! ```text
//! ltxv load ltx-2.5-dit.gguf 34% (2.1 GiB/6.2 GiB)
//! ```
//!
//! On a terminal the line redraws in place, at most every
//! [`REDRAW_EVERY`] (a tensor-by-tensor stream produces far more events than
//! a human can read), and commits itself into the scrollback with a
//! newline once a file's stream completes - so a finished load stays a
//! finished fact, and the pipeline's next in-place phase line starts on a
//! fresh line. Not a terminal (a pipe, a log): one plain line per completed
//! file, nothing else - the same sparse-into-logs rule `brain pull` follows.
//!
//! A file that has already reported completion is never rendered again:
//! `RealDit` and friends keep their checkpoint source alive across denoise
//! and re-stream cold blocks on a cache miss, and those re-reads must not
//! fight the per-step `denoise sigma=...` line for the terminal. A stream
//! that ends below 100% because the pipeline moved on (block weights read
//! lazily on first forward, not at build) simply leaves its last drawn
//! percentage on the line until something else replaces it - that is the
//! honest state.
//!
//! Infrastructure verbs (`serve`, `pull`) never install this: they own
//! their output surfaces.

use std::io::{IsTerminal, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use checkpoint::load_progress::{self, LoadEvent};

/// Minimum spacing between two in-place redraws. A completion (or the first
/// sight of a stream) always renders immediately.
const REDRAW_EVERY: Duration = Duration::from_millis(150);

/// The one rendered line, clamped at 100% - a re-read pushes the cumulative
/// counter past the file's total by design.
fn render_line(label: &str, file: &str, done: u64, total: u64) -> String {
    let pct = done.min(total) * 100 / total.max(1);
    format!(
        "{label} load {file} {pct}% ({}/{})",
        crate::pull_cli::human_bytes(done),
        crate::pull_cli::human_bytes(total)
    )
}

/// The file name only - the stream's `file` is a path, and the label already
/// says which model is loading.
fn basename(file: &str) -> &str {
    file.rsplit('/').next().unwrap_or(file)
}

struct Line {
    out: Box<dyn Write + Send>,
    label: String,
    terminal: bool,
    last_draw: Option<Instant>,
    /// The stream currently on the line: (file, done, total).
    current: Option<(String, u64, u64)>,
    /// Files that have reported completion - their later events are re-reads.
    completed: std::collections::HashSet<String>,
}

impl Line {
    fn with_sink(label: String, terminal: bool, out: Box<dyn Write + Send>) -> Line {
        Line { out, label, terminal, last_draw: None, current: None, completed: Default::default() }
    }

    fn on_event(&mut self, e: &LoadEvent, now: Instant) {
        if e.total == 0 || self.completed.contains(&e.file) {
            return;
        }
        if !self.terminal {
            // Piped: only the completion is worth a line, one per file.
            if e.done >= e.total {
                self.completed.insert(e.file.clone());
                let _ = writeln!(self.out, "{}", render_line(&self.label, basename(&e.file), e.done, e.total));
            }
            return;
        }
        if e.done >= e.total {
            self.current = Some((e.file.clone(), e.done, e.total));
            self.finish();
            return;
        }
        match &self.current {
            Some((f, _, _)) if *f == e.file => {
                self.current = Some((e.file.clone(), e.done, e.total));
                if self.last_draw.is_none_or(|t| now.duration_since(t) >= REDRAW_EVERY) {
                    self.draw(now);
                }
            }
            _ => {
                let had_open = self.current.is_some();
                self.current = Some((e.file.clone(), e.done, e.total));
                if had_open && self.last_draw.is_some_and(|t| now.duration_since(t) < REDRAW_EVERY) {
                    // Inside the redraw window the open line is left as is;
                    // the next eligible draw replaces it with this stream.
                    return;
                }
                if had_open {
                    let _ = writeln!(self.out);
                }
                self.draw(now);
            }
        }
    }

    /// Redraw the current stream in place.
    fn draw(&mut self, now: Instant) {
        if let Some((f, done, total)) = &self.current {
            let _ = write!(self.out, "\r{}", render_line(&self.label, basename(f), *done, *total));
            let _ = self.out.flush();
            self.last_draw = Some(now);
        }
    }

    /// Commit the current stream's line into the scrollback and close it.
    fn finish(&mut self) {
        let Some((f, done, total)) = &self.current else { return };
        let _ = write!(self.out, "\r{}", render_line(&self.label, basename(f), *done, *total));
        let _ = writeln!(self.out);
        let _ = self.out.flush();
        self.completed.insert(f.clone());
        self.current = None;
        self.last_draw = None;
    }
}

/// Install the load-progress observer under `label` (the architecture id).
/// The observer is process-global and last-wins; an arch command installs
/// exactly once, here, before its handler runs.
pub fn install(label: &str) {
    let terminal = std::io::stderr().is_terminal();
    let line = Mutex::new(Line::with_sink(label.to_string(), terminal, Box::new(std::io::stderr())));
    load_progress::observe(Box::new(move |e| {
        if let Ok(mut l) = line.lock() {
            l.on_event(e, Instant::now());
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    /// A shared in-memory sink the tests read back after driving events.
    #[derive(Clone)]
    struct Recorder(Arc<StdMutex<Vec<u8>>>);

    impl Write for Recorder {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn ev(file: &str, done: u64, total: u64) -> LoadEvent {
        LoadEvent { file: file.to_string(), done, total }
    }

    fn out(rec: &Recorder) -> String {
        String::from_utf8(rec.0.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn the_line_is_concise_and_clamps_at_100() {
        assert_eq!(render_line("ltxv", "dit.gguf", 16, 100), "ltxv load dit.gguf 16% (16 B/100 B)");
        assert_eq!(
            render_line("ltxv", "dit.gguf", 140, 100),
            "ltxv load dit.gguf 100% (140 B/100 B)"
        );
        assert_eq!(basename("/models/x/dit.gguf"), "dit.gguf");
    }

    #[test]
    fn terminal_mode_redraws_throttled_and_commits_on_completion() {
        let rec = Recorder(Default::default());
        let mut l = Line::with_sink("ltxv".into(), true, Box::new(rec.clone()));
        let t0 = Instant::now();
        l.on_event(&ev("/x/dit.gguf", 16, 100), t0);
        // A redraw inside the window is skipped - tensor reads arrive far
        // faster than a human reads.
        l.on_event(&ev("/x/dit.gguf", 32, 100), t0 + Duration::from_millis(50));
        l.on_event(&ev("/x/dit.gguf", 60, 100), t0 + Duration::from_millis(200));
        // Completion commits the line even inside the redraw window...
        l.on_event(&ev("/x/dit.gguf", 100, 100), t0 + Duration::from_millis(250));
        // ...and everything after it is a re-read, never rendered again.
        l.on_event(&ev("/x/dit.gguf", 140, 100), t0 + Duration::from_millis(400));
        assert_eq!(
            out(&rec),
            "\rltxv load dit.gguf 16% (16 B/100 B)"
                .to_string()
                + "\rltxv load dit.gguf 60% (60 B/100 B)"
                + "\rltxv load dit.gguf 100% (100 B/100 B)\n"
        );
    }

    #[test]
    fn a_stream_switch_commits_the_open_line_and_shows_the_new_one() {
        let rec = Recorder(Default::default());
        let mut l = Line::with_sink("ltxv".into(), true, Box::new(rec.clone()));
        let t0 = Instant::now();
        l.on_event(&ev("/x/dit.gguf", 16, 100), t0);
        // The pipeline moved on before dit's stream finished (block weights
        // are read lazily on first forward): the partial line is committed
        // as-is and the new stream takes the terminal.
        l.on_event(&ev("/x/te.gguf", 4, 100), t0 + Duration::from_millis(200));
        assert_eq!(
            out(&rec),
            "\rltxv load dit.gguf 16% (16 B/100 B)"
                .to_string()
                + "\n\rltxv load te.gguf 4% (4 B/100 B)"
        );
    }

    #[test]
    fn pipe_mode_prints_only_one_line_per_completed_file() {
        let rec = Recorder(Default::default());
        let mut l = Line::with_sink("ltxv".into(), false, Box::new(rec.clone()));
        let t0 = Instant::now();
        l.on_event(&ev("/x/dit.gguf", 16, 100), t0);
        l.on_event(&ev("/x/dit.gguf", 60, 100), t0);
        l.on_event(&ev("/x/dit.gguf", 100, 100), t0);
        l.on_event(&ev("/x/dit.gguf", 140, 100), t0);
        l.on_event(&ev("/x/te.gguf", 60, 100), t0);
        assert_eq!(
            out(&rec),
            "ltxv load dit.gguf 100% (100 B/100 B)\n"
        );
    }
}
