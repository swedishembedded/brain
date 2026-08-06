// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared front-door admission policy: the edge concurrency ceiling and the
//! per-request admit deadline every transport applies to a submitted [`Job`]
//! before it starts running on a lane. Previously duplicated between
//! `apiserve` (HTTP) and `dbus` — factored here so both transports gate
//! identically instead of one drifting from the other.
//!
//! [`Job`]: crate::executor::Job

use std::time::Duration;

/// The concurrency ceiling at the edge: at most this many requests are
/// admitted to a transport at once. Overflow is load-shed fast rather than
/// queued, so a saturated server sheds instead of building unbounded latency.
/// Well above the executor's lane count — this guards the transport edge, not
/// the lanes.
pub const EDGE_CONCURRENCY: usize = 256;

/// Default bounded wait for a request to be ADMITTED (work started on a lane)
/// before it is shed. A running job may then take much longer — only the
/// wait-to-start is bounded.
pub const DEFAULT_ADMIT_DEADLINE: Duration = Duration::from_secs(10);

/// The admission deadline for live servers: `BRAIN_ADMIT_DEADLINE_MS` if set
/// to a positive integer, else [`DEFAULT_ADMIT_DEADLINE`]. An empty/invalid/
/// zero value falls back to the default (never an unbounded or zero-length
/// wait). Shared by every transport so an operator sets it once.
pub fn admit_deadline_from_env() -> Duration {
    parse_admit_deadline(std::env::var("BRAIN_ADMIT_DEADLINE_MS").ok().as_deref())
}

/// Pure parse of the admit-deadline override: a positive integer of
/// milliseconds, or [`DEFAULT_ADMIT_DEADLINE`] for any missing/empty/invalid/
/// zero value.
pub fn parse_admit_deadline(raw: Option<&str>) -> Duration {
    match raw.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0) {
        Some(ms) => Duration::from_millis(ms),
        None => DEFAULT_ADMIT_DEADLINE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_deadline_override_parses_positive_ms_else_default() {
        assert_eq!(parse_admit_deadline(Some("500")), Duration::from_millis(500));
        assert_eq!(parse_admit_deadline(Some("  250 ")), Duration::from_millis(250));
        assert_eq!(parse_admit_deadline(None), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("")), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("0")), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("nope")), DEFAULT_ADMIT_DEADLINE);
    }
}
