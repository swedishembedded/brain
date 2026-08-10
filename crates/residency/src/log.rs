// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A tiny, dependency-free leveled-logging primitive for the serving path.
//!
//! `residency` stays a leaf crate (see this crate's own module doc) — no `log`/
//! `tracing` dependency, just a process-global level checked before an
//! `eprintln!`. Level 0 (the default) means "errors only" — real failures
//! already print unconditionally via their own `eprintln!` call sites
//! throughout this codebase and are NOT routed through here; this module only
//! gates the optional, higher-volume tiers (`warn`/`info`/`debug`) a caller
//! opts into with `--verbose`/`-v` (`crates/cli/src/run_cli.rs`).
//!
//! `set_verbosity` is called at most once, at process startup, before the
//! `Executor` (or anything else) is built — nothing here is meant to change
//! at runtime, so a plain `AtomicU8` (not a `OnceLock`) is enough: reads from
//! any thread always see SOME valid, already-published level.

use std::sync::atomic::{AtomicU8, Ordering};

static VERBOSITY: AtomicU8 = AtomicU8::new(0);

/// Set the process-wide verbosity level (0-3 — see this module's doc for what
/// each tier gates). Values above 3 are clamped, not rejected, so a caller
/// forwarding a user-supplied `-vvvv` doesn't need to validate first.
pub fn set_verbosity(level: u8) {
    VERBOSITY.store(level.min(3), Ordering::Relaxed);
}

/// The current verbosity level (0-3).
pub fn verbosity() -> u8 {
    VERBOSITY.load(Ordering::Relaxed)
}

/// Level 1+: a condition worth the operator's attention but not a failure —
/// e.g. a model family whose weights aren't configured, so it's silently not
/// served. Hidden at the default level 0 ("errors only").
pub fn warn(msg: &str) {
    if verbosity() >= 1 {
        eprintln!("[warn] {msg}");
    }
}

/// Level 2+: routine lifecycle events an operator would want when actually
/// watching the server — model activate/resident/evict, the events this
/// module exists for (see `executor.rs`'s `on_msg`/`assign` call sites).
pub fn info(msg: &str) {
    if verbosity() >= 2 {
        eprintln!("[info] {msg}");
    }
}

/// Level 3+: fine-grained detail (per-claim budget numbers, scheduling
/// decisions) — noisy enough that even an operator debugging a residency
/// issue only wants it on demand.
pub fn debug(msg: &str) {
    if verbosity() >= 3 {
        eprintln!("[debug] {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_verbosity_clamps_above_3_rather_than_panicking() {
        set_verbosity(200);
        assert_eq!(verbosity(), 3);
        set_verbosity(0); // restore, since VERBOSITY is process-global
    }
}
