// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `overload` — does the server stay useful past its capacity?
//!
//! Every engine has a load beyond which it cannot meet its SLO. What separates a
//! good one from a bad one is what happens *next*: a server that queues without
//! bound turns a brief spike into minutes of uniformly-missed deadlines, having
//! spent real compute on answers nobody can still use. A server that sheds early
//! keeps serving the traffic it can actually satisfy.
//!
//! So this scenario measures capacity first, then offers **multiples of it** and
//! reports SLO goodput — not completions. An engine is rewarded for rejecting
//! work it provably cannot finish in time, and penalised for producing tokens
//! past their deadline.
//!
//! It also measures **recovery**: after the spike ends, how long until goodput
//! returns to the pre-spike level. A server that needs minutes to drain is still
//! failing long after the cause is gone.

use serde_json::{json, Value};

use crate::stats::r3;
use crate::workload::{Arrival, Class, Slo, Workload};

/// The offered-load ladder, as multiples of measured capacity.
pub const MULTIPLES: &[f64] = &[0.5, 0.8, 1.0, 1.2, 2.0, 4.0];

/// What to do with work that arrives beyond capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Accept everything and queue without bound — the default, and the one
    /// that collapses.
    UnboundedQueue,
    /// Refuse once the queue exceeds a depth.
    MaxQueueDepth(usize),
    /// Refuse work that provably cannot meet its deadline given the queue.
    DeadlineAware,
}

impl Admission {
    pub fn name(&self) -> &'static str {
        match self {
            Admission::UnboundedQueue => "unbounded_queue",
            Admission::MaxQueueDepth(_) => "max_queue_depth",
            Admission::DeadlineAware => "deadline_aware",
        }
    }

    /// Should a request arriving with `queued` already waiting be admitted?
    /// `service_ms` is the observed mean time to serve one request.
    pub fn admit(&self, queued: usize, service_ms: f64, deadline_ms: Option<f64>) -> bool {
        match *self {
            Admission::UnboundedQueue => true,
            Admission::MaxQueueDepth(n) => queued < n,
            Admission::DeadlineAware => match deadline_ms {
                // Everything ahead of it must clear first; if that alone blows
                // the deadline, the work is already known to be wasted.
                Some(d) => (queued as f64) * service_ms <= d,
                None => true,
            },
        }
    }
}

/// One rung of the ladder.
#[derive(Clone, Debug)]
pub struct Point {
    pub multiple: f64,
    pub offered_per_s: f64,
    pub admitted: usize,
    pub rejected: usize,
    pub completed: usize,
    pub met_slo: usize,
    pub goodput_per_s: f64,
    pub queue_p99_ms: Option<f64>,
    /// Requests admitted that then missed their deadline — pure waste, the
    /// number an admission policy exists to drive down.
    pub wasted: usize,
}

impl Point {
    /// The share of admitted work that turned out to be useless.
    pub fn waste_fraction(&self) -> f64 {
        if self.admitted == 0 {
            return 0.0;
        }
        self.wasted as f64 / self.admitted as f64
    }
    /// Of the work that was refused, how much would genuinely have missed its
    /// deadline. 1.0 = every rejection was justified.
    pub fn rejection_accuracy(&self, would_have_missed: usize) -> Option<f64> {
        (self.rejected > 0).then(|| would_have_missed as f64 / self.rejected as f64)
    }
    pub fn to_json(&self) -> Value {
        json!({
            "multiple_of_capacity": self.multiple,
            "offered_per_s": r3(self.offered_per_s),
            "admitted": self.admitted,
            "rejected": self.rejected,
            "completed": self.completed,
            "met_slo": self.met_slo,
            "goodput_per_s": r3(self.goodput_per_s),
            "queue_ms_p99": self.queue_p99_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "wasted_admissions": self.wasted,
            "waste_fraction": r3(self.waste_fraction()),
        })
    }
}

/// Build the workload for one rung: an open-loop Poisson arrival at
/// `capacity * multiple` requests/s.
pub fn workload(class: Class, capacity_per_s: f64, multiple: f64, num_requests: usize, seed: u64) -> Workload {
    let rate = (capacity_per_s * multiple).max(0.01);
    Workload {
        name: format!("overload/x{multiple}"),
        classes: vec![class],
        // Open loop is essential here: a closed loop *cannot* overload a server,
        // because it only submits when something finishes. Overload is by
        // definition arrival-driven.
        arrival: Arrival::Rate { per_s: rate, burstiness: 1.0 },
        num_requests,
        warmup_requests: 0,
        ignore_stop: true,
        seed,
    }
}

/// Summarise the ladder: peak goodput, where it collapses, and recovery.
pub fn to_json(points: &[Point], policy: Admission, recovery_ms: Option<f64>) -> Value {
    let peak = points.iter().fold(None::<&Point>, |acc, p| match acc {
        Some(a) if a.goodput_per_s >= p.goodput_per_s => Some(a),
        _ => Some(p),
    });
    // Collapse = the first rung past saturation whose goodput falls below half the peak.
    let collapse = peak.and_then(|pk| {
        points
            .iter()
            .find(|p| p.multiple > 1.0 && p.goodput_per_s < pk.goodput_per_s * 0.5)
            .map(|p| p.multiple)
    });
    json!({
        "admission_policy": policy.name(),
        "peak_goodput_per_s": peak.map(|p| Value::from(r3(p.goodput_per_s))).unwrap_or(Value::Null),
        "peak_at_multiple": peak.map(|p| Value::from(p.multiple)).unwrap_or(Value::Null),
        "collapse_at_multiple": collapse.map(Value::from).unwrap_or(Value::Null),
        "recovery_ms": recovery_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
        "ladder": points.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
    })
}

pub fn render(points: &[Point], policy: Admission) -> String {
    let mut s = format!("\n  admission policy: {}\n", policy.name());
    s.push_str(&format!(
        "\n{:>7} {:>11} {:>10} {:>10} {:>12} {:>9}\n",
        "offered", "admitted", "rejected", "met SLO", "goodput/s", "waste"
    ));
    s.push_str(&format!("{:-<66}\n", ""));
    for p in points {
        s.push_str(&format!(
            "{:>6.1}x {:>11} {:>10} {:>10} {:>12.1} {:>8.0}%\n",
            p.multiple,
            p.admitted,
            p.rejected,
            p.met_slo,
            p.goodput_per_s,
            p.waste_fraction() * 100.0
        ));
    }
    s
}

/// The class overload is offered, derived from a base workload's shape.
pub fn class_from(base: &Workload) -> Class {
    base.classes.first().cloned().unwrap_or(Class {
        name: "overload".into(),
        input: crate::workload::Lengths::Fixed(128),
        output: crate::workload::Lengths::Fixed(64),
        weight: 1.0,
        slo: Slo::ttfa(1000.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(multiple: f64, admitted: usize, rejected: usize, met: usize, goodput: f64, wasted: usize) -> Point {
        Point {
            multiple,
            offered_per_s: 10.0 * multiple,
            admitted,
            rejected,
            completed: admitted,
            met_slo: met,
            goodput_per_s: goodput,
            queue_p99_ms: None,
            wasted,
        }
    }

    #[test]
    fn unbounded_queue_admits_everything() {
        let a = Admission::UnboundedQueue;
        assert!(a.admit(0, 10.0, Some(100.0)));
        assert!(a.admit(10_000, 10.0, Some(1.0)), "that is precisely the failure mode");
    }

    #[test]
    fn max_queue_depth_sheds_past_its_bound() {
        let a = Admission::MaxQueueDepth(4);
        assert!(a.admit(3, 10.0, None));
        assert!(!a.admit(4, 10.0, None));
    }

    #[test]
    fn deadline_aware_refuses_provably_late_work() {
        let a = Admission::DeadlineAware;
        // 10 queued x 50ms service = 500ms before this one starts.
        assert!(a.admit(10, 50.0, Some(600.0)), "fits inside the deadline");
        assert!(!a.admit(10, 50.0, Some(400.0)), "cannot possibly meet it");
        assert!(a.admit(10_000, 50.0, None), "no deadline means nothing to violate");
    }

    #[test]
    fn open_loop_arrival_is_used_because_closed_loop_cannot_overload() {
        let c = class_from(&crate::workload::standard("chat", Arrival::Saturated, 1, 0).unwrap());
        let w = workload(c, 10.0, 2.0, 100, 1);
        match w.arrival {
            Arrival::Rate { per_s, .. } => assert_eq!(per_s, 20.0),
            other => panic!("overload must be arrival-driven, got {other:?}"),
        }
    }

    #[test]
    fn waste_fraction_reports_admitted_work_that_missed() {
        let p = pt(2.0, 100, 0, 40, 40.0, 60);
        assert_eq!(p.waste_fraction(), 0.6);
        assert_eq!(pt(1.0, 0, 0, 0, 0.0, 0).waste_fraction(), 0.0);
    }

    #[test]
    fn rejection_accuracy_needs_rejections() {
        assert_eq!(pt(1.0, 10, 0, 10, 10.0, 0).rejection_accuracy(0), None);
        assert_eq!(pt(2.0, 10, 10, 10, 10.0, 0).rejection_accuracy(10), Some(1.0));
    }

    #[test]
    fn collapse_is_detected_when_goodput_halves_past_capacity() {
        let pts = vec![
            pt(0.5, 50, 0, 50, 50.0, 0),
            pt(1.0, 100, 0, 100, 100.0, 0),
            pt(2.0, 200, 0, 40, 40.0, 160),
        ];
        let j = to_json(&pts, Admission::UnboundedQueue, None);
        assert_eq!(j["peak_goodput_per_s"], 100.0);
        assert_eq!(j["collapse_at_multiple"], 2.0);
    }

    #[test]
    fn a_server_that_holds_up_reports_no_collapse() {
        let pts = vec![pt(1.0, 100, 0, 100, 100.0, 0), pt(4.0, 100, 300, 100, 100.0, 0)];
        let j = to_json(&pts, Admission::MaxQueueDepth(8), Some(120.0));
        assert!(j["collapse_at_multiple"].is_null());
        assert_eq!(j["recovery_ms"], 120.0);
    }

    #[test]
    fn ladder_covers_below_and_well_past_capacity() {
        assert!(MULTIPLES.first().unwrap() < &1.0);
        assert!(MULTIPLES.last().unwrap() >= &4.0);
    }
}
