// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping accelerator fleets
// serving through driver faults for its clients. If your team needs expertise
// in fault isolation and recovery policy then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The per-device circuit breaker. Time is always passed in, so the same code
//! runs under the model checker's discrete clock and the real one.

use std::time::{Duration, Instant};

/// When a quarantined device is probed and how many clean probes readmit it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BreakerPolicy {
    /// Faults older than this no longer count toward tripping.
    pub window: Duration,
    /// The fault count inside [`Self::window`] that trips the breaker. Below
    /// it a fault is a restart: the probe (a fresh worker opening the device)
    /// is due at once and one clean probe readmits the device.
    pub trip_after: usize,
    /// How long a tripped breaker waits before each probe.
    pub probe_interval: Duration,
    /// Clean probes, each a [`Self::probe_interval`] apart, that close a
    /// tripped breaker.
    pub probes_to_close: usize,
}

impl Default for BreakerPolicy {
    /// Restart on the first two faults in ten minutes; the third trips the
    /// breaker, which then probes every five minutes and closes after two
    /// clean probes. Without the trip a card whose driver is genuinely wedged
    /// would cost every request a full deadline before failing.
    fn default() -> BreakerPolicy {
        BreakerPolicy { window: Duration::from_secs(600), trip_after: 3, probe_interval: Duration::from_secs(300), probes_to_close: 2 }
    }
}

/// Whether a device takes work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BreakerState {
    /// Healthy: work is admitted.
    Closed,
    /// Quarantined after a fault: nothing is admitted; a probe is due at
    /// [`Breaker::retry_at`].
    Open,
    /// A probe is running: nothing else is admitted until it answers.
    HalfOpen,
}

/// One device's breaker. See [`BreakerPolicy`] for the transitions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Breaker {
    policy: BreakerPolicy,
    state: BreakerState,
    /// Fault times inside the window, oldest first.
    faults: Vec<Instant>,
    /// When the next probe may start (meaningful while open).
    retry_at: Option<Instant>,
    tripped: bool,
    clean_probes: usize,
    cause: Option<String>,
}

impl Breaker {
    pub fn new(policy: BreakerPolicy) -> Breaker {
        Breaker { policy, state: BreakerState::Closed, faults: Vec::new(), retry_at: None, tripped: false, clean_probes: 0, cause: None }
    }

    pub fn state(&self) -> BreakerState {
        self.state
    }

    /// Whether work may start on the device. Only a closed breaker admits.
    pub fn admits(&self) -> bool {
        self.state == BreakerState::Closed
    }

    /// The fault that opened the breaker, while it is not closed.
    pub fn cause(&self) -> Option<&str> {
        self.cause.as_deref()
    }

    /// Whether repeated faults tripped the breaker (spaced probes, more than one to close).
    pub fn tripped(&self) -> bool {
        self.tripped
    }

    /// When the next probe may start, while open.
    pub fn retry_at(&self) -> Option<Instant> {
        self.retry_at.filter(|_| self.state == BreakerState::Open)
    }

    /// Record a fault at `now` and open the breaker, whatever state it was in.
    /// Returns true when this fault is the one that trips it, so the caller
    /// logs the trip once rather than once per fault.
    pub fn record_fault(&mut self, now: Instant, cause: &str) -> bool {
        let window = self.policy.window;
        self.faults.retain(|&t| now.saturating_duration_since(t) < window);
        self.faults.push(now);
        let was_tripped = self.tripped;
        self.tripped = self.tripped || self.faults.len() >= self.policy.trip_after.max(1);
        self.state = BreakerState::Open;
        self.clean_probes = 0;
        self.retry_at = Some(if self.tripped { now + self.policy.probe_interval } else { now });
        self.cause = Some(cause.to_string());
        self.tripped && !was_tripped
    }

    /// Whether a probe may start at `now`.
    pub fn probe_due(&self, now: Instant) -> bool {
        self.state == BreakerState::Open && self.retry_at.is_some_and(|t| now >= t)
    }

    /// Move to half-open for one probe, if one is due. False leaves the state
    /// unchanged: closed (nothing to probe), half-open (one probe at a time),
    /// or open but not yet due.
    pub fn begin_probe(&mut self, now: Instant) -> bool {
        if !self.probe_due(now) {
            return false;
        }
        self.state = BreakerState::HalfOpen;
        true
    }

    /// The running probe passed. Closes the breaker once enough clean probes
    /// have passed, else waits a probe interval for the next one. Ignored
    /// unless a probe is running.
    pub fn probe_passed(&mut self, now: Instant) {
        if self.state != BreakerState::HalfOpen {
            return;
        }
        self.clean_probes += 1;
        let needed = if self.tripped { self.policy.probes_to_close.max(1) } else { 1 };
        if self.clean_probes >= needed {
            // The fault history stays: a device that faults again soon after
            // closing is still within the window and trips sooner.
            self.state = BreakerState::Closed;
            self.tripped = false;
            self.clean_probes = 0;
            self.retry_at = None;
            self.cause = None;
        } else {
            self.state = BreakerState::Open;
            self.retry_at = Some(now + self.policy.probe_interval);
        }
    }

    /// The running probe failed: that is a fault like any other. Ignored
    /// unless a probe is running.
    pub fn probe_failed(&mut self, now: Instant, cause: &str) -> bool {
        if self.state != BreakerState::HalfOpen {
            return false;
        }
        self.record_fault(now, cause)
    }
}
