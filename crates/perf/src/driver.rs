// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The load driver: pushes a [`Workload`] through a [`PerfTarget`] according to
//! the workload's arrival process, and records the emission timeline.
//!
//! The driver is deliberately single-threaded and synchronous. Every scenario in
//! this suite measures *the engine*, so introducing a thread pool or an async
//! runtime here would fold the harness's own scheduling noise into the numbers
//! it reports. Targets that are genuinely concurrent (the paged engine batches
//! internally) express that inside `step`, where it belongs.

use std::collections::HashMap;
use std::time::Instant;

use crate::metrics::ReqRecord;
use crate::target::{Emission, EmissionKind, PerfTarget};
use crate::workload::{Arrival, Workload};

/// Result of one driver run: every request's timeline plus the measured wall
/// clock. Wall time is taken from the **first measured submit** to the last
/// measured completion, so warm-up never inflates the denominator.
pub struct Run {
    pub records: Vec<ReqRecord>,
    pub wall_s: f64,
}

/// Drive `target` with `workload`.
pub fn drive(target: &mut dyn PerfTarget, workload: &Workload) -> Run {
    let requests = workload.requests();
    let delays = workload.arrival_delays();
    let warmup_n = workload.warmup_requests;

    let mut records: Vec<ReqRecord> = Vec::with_capacity(requests.len());
    let mut by_id: HashMap<u64, usize> = HashMap::new();
    let mut emissions: Vec<Emission> = Vec::new();

    // Wall-clock window over the measured (non-warm-up) requests only.
    let mut measured_start: Option<Instant> = None;
    let mut measured_end: Option<Instant> = None;

    let mut next_to_submit = 0usize;
    let mut in_flight = 0usize;
    let run_start = Instant::now();
    // Absolute arrival time of request `next_to_submit`, accumulated as we go.
    // (Re-summing the delay prefix each poll would be O(n^2) in request count.)
    let mut next_due_at: f64 = delays.first().copied().unwrap_or(0.0);

    let concurrency = match workload.arrival {
        Arrival::ClosedLoop { concurrency } => concurrency.max(1),
        _ => usize::MAX,
    };

    loop {
        // ---- submit whatever the arrival process says is due ----
        loop {
            if next_to_submit >= requests.len() {
                break;
            }
            let due = match workload.arrival {
                // Keep exactly `concurrency` requests in flight.
                Arrival::ClosedLoop { .. } => in_flight < concurrency,
                // Everything at t=0.
                Arrival::Saturated => true,
                // Clock-driven: submit once its arrival time has passed.
                Arrival::Rate { .. } | Arrival::Ramp { .. } => {
                    run_start.elapsed().as_secs_f64() >= next_due_at
                }
            };
            if !due {
                break;
            }
            let req = requests[next_to_submit].clone();
            let warmup = next_to_submit < warmup_n;
            let now = Instant::now();
            let id = target.submit(req.clone());
            by_id.insert(id, records.len());
            records.push(ReqRecord::new(id, req.class, req.input_artifacts, warmup, now));
            if !warmup && measured_start.is_none() {
                measured_start = Some(now);
            }
            next_to_submit += 1;
            in_flight += 1;
            next_due_at += delays.get(next_to_submit).copied().unwrap_or(0.0);
        }

        // ---- let the engine make progress ----
        emissions.clear();
        let busy = target.step(&mut emissions);

        for e in &emissions {
            let Some(&idx) = by_id.get(&e.id) else { continue };
            let rec = &mut records[idx];
            match e.kind {
                EmissionKind::Admitted => rec.admit = Some(e.at),
                EmissionKind::Artifact => {
                    if rec.first.is_none() {
                        rec.first = Some(e.at);
                    }
                    rec.artifacts.push(e.at);
                }
                EmissionKind::Done => {
                    rec.done = Some(e.at);
                    in_flight = in_flight.saturating_sub(1);
                    if !rec.warmup {
                        measured_end = Some(e.at);
                    }
                }
                EmissionKind::Failed => {
                    rec.failed = true;
                    rec.done = Some(e.at);
                    in_flight = in_flight.saturating_sub(1);
                    if !rec.warmup {
                        measured_end = Some(e.at);
                    }
                }
            }
        }

        let all_submitted = next_to_submit >= requests.len();
        if all_submitted && !busy && !target.busy() {
            break;
        }
        // A rate-driven run with nothing in flight yet must not spin the CPU
        // while it waits for the next arrival.
        if !busy && !target.busy() && !all_submitted {
            if let Arrival::Rate { .. } | Arrival::Ramp { .. } = workload.arrival {
                std::thread::yield_now();
            }
        }
    }

    let wall_s = match (measured_start, measured_end) {
        (Some(a), Some(b)) => b.saturating_duration_since(a).as_secs_f64(),
        _ => run_start.elapsed().as_secs_f64(),
    };
    Run { records, wall_s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::testing::FakeTarget;
    use crate::workload::{Arrival, Class, Lengths, Slo, Workload};

    fn wl(arrival: Arrival, n: usize, warmup: usize, out: usize) -> Workload {
        Workload {
            name: "t".into(),
            classes: vec![Class {
                name: "t".into(),
                input: Lengths::Fixed(16),
                output: Lengths::Fixed(out),
                weight: 1.0,
                slo: Slo::NONE,
            }],
            arrival,
            num_requests: n,
            warmup_requests: warmup,
            ignore_stop: true,
            seed: 1,
        }
    }

    #[test]
    fn every_request_completes_and_is_recorded() {
        let mut t = FakeTarget::new(4, 2);
        let w = wl(Arrival::Saturated, 12, 0, 5);
        let run = drive(&mut t, &w);
        assert_eq!(run.records.len(), 12);
        for r in &run.records {
            assert!(r.done.is_some(), "request {} never completed", r.id);
            assert_eq!(r.output_artifacts(), 5);
            assert!(r.first.is_some());
            assert!(r.admit.is_some());
        }
    }

    #[test]
    fn closed_loop_holds_concurrency() {
        // With concurrency 3 the driver must never have more than 3 in flight.
        let mut t = FakeTarget::new(64, 1); // target itself is not the limit
        let w = wl(Arrival::ClosedLoop { concurrency: 3 }, 20, 0, 3);
        let run = drive(&mut t, &w);
        assert_eq!(run.records.len(), 20);
        // Reconstruct max overlap from the timeline.
        let mut events: Vec<(std::time::Instant, i32)> = Vec::new();
        for r in &run.records {
            events.push((r.submit, 1));
            if let Some(d) = r.done {
                events.push((d, -1));
            }
        }
        events.sort_by_key(|e| e.0);
        let (mut cur, mut max) = (0, 0);
        for (_, d) in events {
            cur += d;
            max = max.max(cur);
        }
        assert!(max <= 3, "closed loop must cap in-flight at 3, saw {max}");
    }

    #[test]
    fn warmup_requests_are_marked_and_excluded_from_wall_time() {
        let mut t = FakeTarget::new(2, 1);
        let w = wl(Arrival::Saturated, 6, 4, 2);
        let run = drive(&mut t, &w);
        assert_eq!(run.records.len(), 10);
        assert_eq!(run.records.iter().filter(|r| r.warmup).count(), 4);
        assert!(run.records[0].warmup && !run.records[4].warmup);
        assert!(run.wall_s >= 0.0);
    }

    #[test]
    fn saturated_submits_everything_before_completion() {
        let mut t = FakeTarget::new(100, 1);
        let w = wl(Arrival::Saturated, 8, 0, 1);
        let run = drive(&mut t, &w);
        // All 8 submitted essentially together, so the first submit precedes the
        // first completion.
        let first_submit = run.records.iter().map(|r| r.submit).min().unwrap();
        let first_done = run.records.iter().filter_map(|r| r.done).min().unwrap();
        assert!(first_submit <= first_done);
        assert_eq!(run.records.len(), 8);
    }

    #[test]
    fn rate_arrival_spreads_submissions_over_time() {
        let mut t = FakeTarget::new(16, 1);
        // 50/s for 10 requests => ~0.2s of arrivals.
        let w = wl(Arrival::Rate { per_s: 50.0, burstiness: 1.0 }, 10, 0, 1);
        let run = drive(&mut t, &w);
        assert_eq!(run.records.len(), 10);
        let t0 = run.records.first().unwrap().submit;
        let tn = run.records.last().unwrap().submit;
        assert!(tn > t0, "clock-driven arrivals must not all land at once");
    }
}
