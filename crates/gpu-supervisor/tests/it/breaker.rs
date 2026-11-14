// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping accelerator fleets
// serving through driver faults for its clients. If your team needs expertise
// in fault isolation and recovery policy then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The per-device circuit breaker's policy: a fault quarantines the device at
//! once, a passing probe readmits it, and repeated faults inside the window
//! slow the probes down instead of costing every request a full deadline.

use std::time::{Duration, Instant};

use gpu_supervisor::{Breaker, BreakerPolicy, BreakerState};

fn policy() -> BreakerPolicy {
    BreakerPolicy { window: Duration::from_secs(600), trip_after: 3, probe_interval: Duration::from_secs(300), probes_to_close: 2 }
}

#[test]
fn a_fault_quarantines_the_device_until_a_probe_passes() {
    let t0 = Instant::now();
    let mut b = Breaker::new(policy());
    assert!(b.admits(), "a fresh breaker admits work");
    b.record_fault(t0, "gpu0: run exceeded its deadline");
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.admits(), "an open breaker must not admit work");
    assert_eq!(b.cause(), Some("gpu0: run exceeded its deadline"));
    // Below the trip count a fault is a restart: the probe is due at once.
    assert!(b.probe_due(t0));
    assert!(b.begin_probe(t0));
    assert_eq!(b.state(), BreakerState::HalfOpen);
    assert!(!b.admits(), "work must wait while the probe runs");
    b.probe_passed(t0);
    assert_eq!(b.state(), BreakerState::Closed);
    assert!(b.admits());
    assert_eq!(b.cause(), None);
}

#[test]
fn a_failed_probe_keeps_the_device_quarantined() {
    let t0 = Instant::now();
    let mut b = Breaker::new(policy());
    b.record_fault(t0, "crash");
    assert!(b.begin_probe(t0));
    b.probe_failed(t0, "probe crashed too");
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.admits());
    assert_eq!(b.cause(), Some("probe crashed too"));
}

#[test]
fn the_third_fault_in_the_window_trips_and_needs_two_spaced_probes() {
    let t0 = Instant::now();
    let s = Duration::from_secs;
    let mut b = Breaker::new(policy());
    for k in 0..2 {
        assert!(!b.record_fault(t0 + s(k), "wedge"), "fault {k} is below the trip count");
        assert!(b.begin_probe(t0 + s(k)));
        b.probe_passed(t0 + s(k));
        assert!(b.admits());
    }
    assert!(b.record_fault(t0 + s(2), "wedge"), "the third fault inside 600 s trips the breaker");
    assert!(b.tripped());
    assert!(!b.probe_due(t0 + s(2)), "a tripped breaker waits a probe interval");
    assert!(!b.begin_probe(t0 + s(301)));
    assert!(b.begin_probe(t0 + s(302)));
    b.probe_passed(t0 + s(302));
    assert_eq!(b.state(), BreakerState::Open, "one clean probe is not enough once tripped");
    assert!(!b.probe_due(t0 + s(303)));
    assert!(b.begin_probe(t0 + s(602)));
    b.probe_passed(t0 + s(602));
    assert_eq!(b.state(), BreakerState::Closed, "two clean probes close a tripped breaker");
    assert!(!b.tripped());
}

#[test]
fn faults_outside_the_window_do_not_trip() {
    let t0 = Instant::now();
    let mut b = Breaker::new(policy());
    for k in 0..5u64 {
        let t = t0 + Duration::from_secs(k * 601);
        assert!(!b.record_fault(t, "wedge"), "fault {k} is alone in its window");
        assert!(b.begin_probe(t));
        b.probe_passed(t);
    }
}

#[test]
fn a_probe_cannot_start_while_one_runs_or_while_closed() {
    let t0 = Instant::now();
    let mut b = Breaker::new(policy());
    assert!(!b.begin_probe(t0), "a closed breaker has nothing to probe");
    b.record_fault(t0, "wedge");
    assert!(b.begin_probe(t0));
    assert!(!b.begin_probe(t0), "one probe at a time");
}
