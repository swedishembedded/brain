// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `faults` — inject failures and measure recovery.
//!
//! An inference engine should be tested by actively breaking it. The properties
//! that matter under failure are not throughput but: how fast is the fault
//! *noticed*, how much work is lost, does a partial or corrupted answer ever
//! reach a client, and does one failed component freeze everything else.
//!
//! The last is the one worth designing for. A single hung rank that stalls every
//! other rank turns a one-device failure into a total outage, and no
//! steady-state benchmark will ever show it.

use serde_json::{json, Value};

use crate::stats::r3;

/// What to break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// A device worker dies mid-request.
    WorkerDeath,
    /// A kernel dispatch fails.
    KernelFailure,
    /// Device allocation fails (VRAM exhausted).
    DeviceOom,
    /// Host allocation fails.
    HostOom,
    /// A weight read from storage fails.
    WeightReadFailure,
    /// A rank stops responding without dying — the worst case, because there is
    /// nothing to observe except absence.
    HungRank,
    /// A collective times out.
    CollectiveTimeout,
    /// A KV transfer arrives corrupted.
    CorruptKvTransfer,
}

impl Fault {
    pub fn name(&self) -> &'static str {
        match self {
            Fault::WorkerDeath => "worker_death",
            Fault::KernelFailure => "kernel_failure",
            Fault::DeviceOom => "device_oom",
            Fault::HostOom => "host_oom",
            Fault::WeightReadFailure => "weight_read_failure",
            Fault::HungRank => "hung_rank",
            Fault::CollectiveTimeout => "collective_timeout",
            Fault::CorruptKvTransfer => "corrupt_kv_transfer",
        }
    }
    /// Faults expressible against a single-process engine. The distributed ones
    /// need a multi-rank harness and are listed but not injectable here — saying
    /// so is better than silently reporting a pass.
    pub fn single_process() -> &'static [Fault] {
        &[Fault::KernelFailure, Fault::DeviceOom, Fault::HostOom, Fault::WeightReadFailure]
    }
    pub fn requires_multi_rank(&self) -> bool {
        matches!(self, Fault::HungRank | Fault::CollectiveTimeout | Fault::WorkerDeath | Fault::CorruptKvTransfer)
    }
}

/// What one injection did.
#[derive(Clone, Debug)]
pub struct Injection {
    pub fault: &'static str,
    /// Whether the fault could actually be injected here.
    pub injected: bool,
    /// Time from injection to the engine noticing.
    pub detect_ms: Option<f64>,
    /// Time from detection to serving correctly again.
    pub recovery_ms: Option<f64>,
    /// Requests that died with the fault.
    pub lost: usize,
    /// Requests that returned a partial or wrong answer — the worst outcome,
    /// strictly worse than an error, because a client cannot tell.
    pub corrupted: usize,
    /// Requests unrelated to the fault that still completed.
    pub survived: usize,
    pub expected_survivors: usize,
    /// Did an error surface to the caller at all?
    pub reported: bool,
}

impl Injection {
    pub fn skipped(fault: Fault, why: &str) -> Injection {
        let _ = why;
        Injection {
            fault: fault.name(),
            injected: false,
            detect_ms: None,
            recovery_ms: None,
            lost: 0,
            corrupted: 0,
            survived: 0,
            expected_survivors: 0,
            reported: false,
        }
    }
    /// The engine behaved acceptably: the failure was reported, nothing was
    /// silently corrupted, and unrelated work kept flowing.
    pub fn acceptable(&self) -> bool {
        !self.injected || (self.reported && self.corrupted == 0 && self.survived >= self.expected_survivors)
    }
    pub fn to_json(&self) -> Value {
        json!({
            "fault": self.fault,
            "injected": self.injected,
            "detect_ms": self.detect_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "recovery_ms": self.recovery_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "lost_requests": self.lost,
            "corrupted_responses": self.corrupted,
            "unrelated_survived": self.survived,
            "unrelated_expected": self.expected_survivors,
            "error_reported": self.reported,
            "acceptable": self.acceptable(),
        })
    }
}

#[derive(Debug, Default)]
pub struct Report {
    pub injections: Vec<Injection>,
}

impl Report {
    pub fn all_acceptable(&self) -> bool {
        self.injections.iter().all(|i| i.acceptable())
    }
    pub fn injected_count(&self) -> usize {
        self.injections.iter().filter(|i| i.injected).count()
    }
    pub fn to_json(&self) -> Value {
        json!({
            "injections": self.injections.iter().map(|i| i.to_json()).collect::<Vec<_>>(),
            "injected": self.injected_count(),
            "skipped": self.injections.len() - self.injected_count(),
            "all_acceptable": self.all_acceptable(),
        })
    }
}

pub fn render(r: &Report) -> String {
    let mut s = format!("\n{:<22} {:>9} {:>11} {:>11} {:>11}\n", "fault", "injected", "detect ms", "corrupted", "acceptable");
    s.push_str(&format!("{:-<68}\n", ""));
    for i in &r.injections {
        s.push_str(&format!(
            "{:<22} {:>9} {:>11} {:>11} {:>11}\n",
            i.fault,
            if i.injected { "yes" } else { "skipped" },
            i.detect_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()),
            i.corrupted,
            if i.acceptable() { "yes" } else { "NO" },
        ));
    }
    let skipped = r.injections.len() - r.injected_count();
    if skipped > 0 {
        s.push_str(&format!(
            "\n{skipped} fault(s) skipped: they need a multi-rank harness, which this scenario\ndoes not build. They are listed rather than silently reported as passing.\n"
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inj(fault: Fault, corrupted: usize, survived: usize, expected: usize, reported: bool) -> Injection {
        Injection {
            fault: fault.name(),
            injected: true,
            detect_ms: Some(5.0),
            recovery_ms: Some(50.0),
            lost: 1,
            corrupted,
            survived,
            expected_survivors: expected,
            reported,
        }
    }

    #[test]
    fn a_reported_failure_that_spares_neighbours_is_acceptable() {
        assert!(inj(Fault::DeviceOom, 0, 8, 8, true).acceptable());
    }

    #[test]
    fn silent_corruption_is_never_acceptable() {
        let i = inj(Fault::CorruptKvTransfer, 1, 8, 8, true);
        assert!(!i.acceptable(), "a wrong answer is worse than an error");
    }

    #[test]
    fn an_unreported_failure_is_not_acceptable() {
        assert!(!inj(Fault::KernelFailure, 0, 8, 8, false).acceptable());
    }

    #[test]
    fn taking_down_unrelated_requests_is_not_acceptable() {
        assert!(!inj(Fault::DeviceOom, 0, 3, 8, true).acceptable());
    }

    #[test]
    fn skipped_faults_do_not_count_as_passes_or_failures() {
        let s = Injection::skipped(Fault::HungRank, "needs multiple ranks");
        assert!(!s.injected);
        assert!(s.acceptable(), "not injected means nothing was proven either way");
        let r = Report { injections: vec![s] };
        assert_eq!(r.injected_count(), 0);
        assert!(render(&r).contains("skipped"));
    }

    #[test]
    fn distributed_faults_are_declared_as_needing_multiple_ranks() {
        assert!(Fault::HungRank.requires_multi_rank());
        assert!(!Fault::DeviceOom.requires_multi_rank());
        for f in Fault::single_process() {
            assert!(!f.requires_multi_rank(), "{} should be injectable here", f.name());
        }
    }

    #[test]
    fn report_summarises_acceptability() {
        let r = Report {
            injections: vec![inj(Fault::DeviceOom, 0, 8, 8, true), inj(Fault::HostOom, 2, 8, 8, true)],
        };
        assert!(!r.all_acceptable());
        assert_eq!(r.to_json()["injected"], 2);
    }
}
