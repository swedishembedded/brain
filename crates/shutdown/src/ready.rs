// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Readiness — the start-of-life mirror of [`crate::Shutdown`].
//!
//! `brain serve` can be asked for several independent listeners at once (an HTTP
//! surface per dialect, plus the D-Bus name), each bound from a different runtime
//! on a different thread. "The process is ready" is the AND of all of them, which
//! no single listener can answer for itself, and which no HTTP route could ever
//! answer for the D-Bus surface — a `/healthz` on port 8788 says nothing about
//! port 8787 or about `com.swedishembedded.Brain1`. [`Gate`] is that AND: a
//! counting latch that creates one marker file on the LAST bind and never before.
//!
//! The failure mode is deliberately one-sided. A surface that fails to bind
//! simply never calls [`Gate::bound`], so the count never reaches `expected` and
//! the marker never appears — "not ready" needs no error path and cannot be
//! reported by accident. A waiter must still bound its wait and check the process
//! is alive: "never appears" converts a wrong answer into a hang, not a signal.
//!
//! The marker is intentionally empty and holds no secret: not a key (that is
//! `--api-keys-out`'s job), not a pid, not an address. If it had content, a
//! caller would end up checking `[ -s FILE ]` instead of `[ -e FILE ]`, which
//! reintroduces exactly the partial-write race a plain existence check avoids.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A counting readiness latch shared by every serving surface in one process.
/// Cheap to clone; every clone shares one count and one marker path.
#[derive(Clone, Default)]
pub struct Gate(Option<Arc<Inner>>);

struct Inner {
    path: PathBuf,
    expected: usize,
    seen: AtomicUsize,
    fired: AtomicBool,
}

impl Gate {
    /// A gate that creates `path` once `expected` distinct surfaces have reported
    /// bound. `expected` of 0 is treated as 1 (a gate with nothing to wait for
    /// would otherwise never fire).
    ///
    /// Prepares `path` eagerly rather than lazily at the first [`Gate::bound`]
    /// call, because both failures here are silent-hang bugs otherwise: a
    /// **stale** marker left over from a previous run is removed now (a waiter
    /// must never mistake it for this run's), and a missing/unwritable parent
    /// directory is an error NOW — at flag-parsing time — rather than a surprise
    /// minutes later, after the model directory scan, when the marker is finally
    /// due.
    pub fn touching(path: impl Into<PathBuf>, expected: usize) -> std::io::Result<Gate> {
        let path = path.into();
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        // Fail fast if the parent directory does not exist / is not writable,
        // rather than discovering it only once every surface has already bound.
        // `OpenOptions::create_new` + immediate delete is a real probe (not just
        // an `exists()` check on the directory) without leaving anything behind
        // for a racing waiter to see.
        let probe = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        drop(probe);
        std::fs::remove_file(&path)?;
        Ok(Gate(Some(Arc::new(Inner { path, expected: expected.max(1), seen: AtomicUsize::new(0), fired: AtomicBool::new(false) }))))
    }

    /// A gate that does nothing — `--ready-file` was not given. Every method is a
    /// no-op, so call sites need no `Option` and no branch.
    pub fn disabled() -> Gate {
        Gate(None)
    }

    /// Report that one surface's listener now exists. `what` names it only for
    /// the diagnostic printed if the marker cannot be created; it plays no role
    /// in the count.
    ///
    /// Exactly one caller — the one whose increment reaches `expected` — creates
    /// the file, via `create_new`, so "exactly once" is enforced by the OS as
    /// well as by the counter. Safe to call from any thread or runtime; never
    /// blocks on anything but a local file create, never panics, and harmless if
    /// called more than `expected` times (a surface that somehow binds twice).
    pub fn bound(&self, what: &str) {
        let Some(inner) = &self.0 else { return };
        if inner.seen.fetch_add(1, Ordering::SeqCst) + 1 != inner.expected {
            return; // not the last one — exactly one caller ever sees equality
        }
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&inner.path) {
            Ok(_) => {
                inner.fired.store(true, Ordering::SeqCst);
                eprintln!("brain serve: ready — all {} surface(s) bound ({what} last); touched {}", inner.expected, inner.path.display());
            }
            Err(e) => eprintln!(
                "brain serve: every requested surface is bound but --ready-file {} could not be created ({e}); readiness will never be signalled",
                inner.path.display()
            ),
        }
    }

    /// Whether the marker has actually been created. For diagnostics and tests —
    /// callers should watch the file on disk, not poll this in-process.
    pub fn is_ready(&self) -> bool {
        self.0.as_ref().is_some_and(|i| i.fired.load(Ordering::SeqCst))
    }

    /// The marker path, if this gate has one.
    pub fn path(&self) -> Option<&Path> {
        self.0.as_ref().map(|i| i.path.as_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-ready-test-{}-{n}", std::process::id()))
    }

    #[test]
    fn marker_appears_only_on_the_last_bound_call() {
        let path = tmp_path();
        let gate = Gate::touching(&path, 3).unwrap();
        gate.bound("a");
        assert!(!path.exists(), "must not appear after 1/3");
        gate.bound("b");
        assert!(!path.exists(), "must not appear after 2/3");
        gate.bound("c");
        assert!(path.exists(), "must appear after 3/3");
        assert!(gate.is_ready());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn marker_is_empty() {
        let path = tmp_path();
        let gate = Gate::touching(&path, 1).unwrap();
        gate.bound("only");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn extra_bound_calls_are_harmless() {
        let path = tmp_path();
        let gate = Gate::touching(&path, 2).unwrap();
        gate.bound("a");
        gate.bound("b");
        gate.bound("c");
        gate.bound("d");
        assert!(path.exists());
        assert!(gate.is_ready());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn concurrent_bound_calls_fire_exactly_once() {
        let path = tmp_path();
        let gate = Gate::touching(&path, 8).unwrap();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let g = gate.clone();
                std::thread::spawn(move || g.bound(&format!("t{i}")))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(path.exists());
        assert!(gate.is_ready());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_disabled_gate_creates_nothing() {
        let gate = Gate::disabled();
        gate.bound("x");
        assert!(gate.path().is_none());
        assert!(!gate.is_ready());
    }

    #[test]
    fn a_stale_marker_is_removed_at_construction() {
        let path = tmp_path();
        std::fs::write(&path, b"stale from a previous run").unwrap();
        let gate = Gate::touching(&path, 2).unwrap();
        assert!(!path.exists(), "stale marker must be gone immediately after touching()");
        gate.bound("a");
        assert!(!path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn touching_errors_when_the_parent_directory_is_missing() {
        let res = Gate::touching("/nonexistent-dir-for-brain-ready-tests-xyz/ready", 1);
        assert!(res.is_err());
    }

    #[test]
    fn a_gate_that_never_reaches_expected_never_creates_the_marker() {
        let path = tmp_path();
        let gate = Gate::touching(&path, 2).unwrap();
        gate.bound("only-one-of-two");
        assert!(!path.exists());
        assert!(!gate.is_ready());
        drop(gate);
        assert!(!path.exists(), "dropping an unfired gate must not create the marker");
    }
}
