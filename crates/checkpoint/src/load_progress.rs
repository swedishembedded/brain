// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-load progress, reported by the streaming readers themselves.
//!
//! Loading a multi-gigabyte checkpoint from disk is the longest quiet
//! stretch of an interactive run: a pipeline reports its coarse phase
//! (`build transformer`) and then says nothing until the weights are on the
//! device. The mmap readers in this crate are the one place every
//! architecture's weight bytes flow through, so they are also the one place
//! a load can be observed without threading a callback through every
//! model's builder: [`MmapSafetensors`](crate::mmap::MmapSafetensors) and
//! [`MmapGguf`](crate::gguf::MmapGguf) report each tensor they hand out as
//! on-disk bytes against the file's tensor-byte total ([`note`], crate
//! internal), and a frontend that wants to show progress installs an
//! observer with [`observe`] and renders one concise line.
//!
//! The observer is optional, and the steady-state cost without one is a
//! single relaxed atomic add per tensor read. With one installed, the
//! rendering (throttling, in-place redraw, when to newline) belongs to the
//! frontend, not here: a server that loads on request must not redraw a
//! terminal line into a log, and a one-shot CLI wants exactly that. The
//! cumulative counters live per opened file, so a sharded checkpoint
//! reports one stream per shard file.

use std::sync::Mutex;

/// One read report: `file`'s weights have advanced to `done` of `total`
/// on-disk bytes. `done` is cumulative per opened file and monotonic; a
/// tensor re-read (a block re-streamed on a cache miss) genuinely decodes
/// its bytes from the mapping again and adds them again, so a renderer
/// clamps its fraction at 100% rather than trusting the counter to cap
/// itself.
pub struct LoadEvent {
    pub file: String,
    pub done: u64,
    pub total: u64,
}

type Observer = Box<dyn Fn(&LoadEvent) + Send + Sync>;

static OBSERVER: Mutex<Option<Observer>> = Mutex::new(None);

/// Install (or replace) the load-progress observer. One process-wide
/// observer, because the readers have no channel to choose between several:
/// the last installer wins, which is the right rule for a CLI that installs
/// once at dispatch and for a test that replaces what it installed.
pub fn observe(f: Observer) {
    *OBSERVER.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(f);
}

/// Remove the observer. A short-lived CLI just exits with it in place and
/// nothing else; tests need this so the process-global state they set does
/// not leak into the next test.
pub fn clear() {
    *OBSERVER.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Report one read: `file`'s cumulative decoded bytes are now `done` of
/// `total`. No-op without an observer.
///
/// The lock is taken poison-tolerantly: this runs on every tensor read of
/// every model build, including inside tests that panic on purpose, and a
/// poisoned global here would turn one failed test into dozens of
/// unrelated PoisonError failures.
pub(crate) fn note(file: &str, done: u64, total: u64) {
    let guard = OBSERVER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(f) = guard.as_ref() {
        f(&LoadEvent { file: file.to_string(), done, total });
    }
}

/// One opened file's cumulative byte counter. Each mmap reader owns one,
/// built from the file's path and its tensor-byte total at open; every leaf
/// read adds the bytes it walked.
pub(crate) struct LoadMeter {
    file: String,
    total: u64,
    seen: std::sync::atomic::AtomicU64,
}

impl LoadMeter {
    pub(crate) fn new(file: String, total: u64) -> Self {
        Self { file, total, seen: std::sync::atomic::AtomicU64::new(0) }
    }

    /// Record a read of `bytes` on-disk bytes and report the new cumulative
    /// figure. A file with no tensor bytes never reports (an empty stream is
    /// nothing to show progress for).
    pub(crate) fn note(&self, bytes: u64) {
        if self.total == 0 {
            return;
        }
        let done = self.seen.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed) + bytes;
        note(&self.file, done, self.total);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// The observer is process-global and cargo runs a binary's tests on
    /// parallel threads: every test that installs one holds this for its
    /// whole body and clears it on the way out.
    static OBSERVE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    pub(crate) fn observe_lock() -> MutexGuard<'static, ()> {
        OBSERVE_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn the_observer_sees_every_note_and_clear_stops_them() {
        let _serial = observe_lock();
        let got = std::sync::Arc::new(Mutex::new(Vec::new()));
        let sink = got.clone();
        // The closure itself never panics (a panicking observer would unwind
        // through whatever read invoked it), so a failed assertion below
        // cannot poison anything other tests touch.
        super::observe(Box::new(move |e| {
            if let Ok(mut v) = sink.lock() {
                v.push((e.file.clone(), e.done, e.total));
            }
        }));
        super::note("dit.gguf", 10, 100);
        super::note("dit.gguf", 30, 100);
        let seen = got.lock().unwrap().clone();
        let dit_count = seen.iter().filter(|(f, _, _)| f == "dit.gguf").count();
        assert_eq!(dit_count, 2);
        super::clear();
        super::note("dit.gguf", 40, 100);
        // The observer is process-global and the suite runs its tests
        // concurrently: only this test's own synthetic stream ("dit.gguf",
        // a name no real fixture uses) is asserted on.
        let seen_after = got.lock().unwrap().clone();
        let after = seen_after.iter().filter(|(f, _, _)| f == "dit.gguf");
        assert_eq!(
            after.count(),
            2,
            "clear() must stop the reports: no dit.gguf event past the 2 above"
        );
    }

    /// Without an observer a note is a no-op: the readers must be usable
    /// (and cheap) in a process that never asked for progress.
    #[test]
    fn a_note_without_an_observer_is_a_no_op() {
        let _serial = observe_lock();
        super::clear();
        super::note("dit.gguf", 1, 1);
    }
}
