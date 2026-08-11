// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Turning the emission timeline into the reported numbers.
//!
//! Every metric here is derived from one per-request record ([`ReqRecord`]) that
//! the driver fills in as emissions arrive. The definitions are deliberately
//! model-agnostic: "artifact" is a token for a decoder, an audio chunk for
//! TTS, a frame for detection.
//!
//! Two rules are enforced here rather than left to the caller, because getting
//! them wrong silently produces a flattering number:
//!
//! * **Warm-up requests never enter any statistic.** They are recorded and
//!   discarded, not merely "run first".
//! * **A failed request is never goodput** — and is never quietly dropped from
//!   the denominator either.

use std::time::Instant;

use serde_json::{json, Value};

use crate::stats::{jain_fairness, r3, Dist};
use crate::workload::Slo;

/// Everything observed about one request.
#[derive(Clone, Debug)]
pub struct ReqRecord {
    pub id: u64,
    pub class: usize,
    pub input_artifacts: usize,
    pub warmup: bool,
    pub submit: Instant,
    pub admit: Option<Instant>,
    pub first: Option<Instant>,
    pub done: Option<Instant>,
    pub failed: bool,
    /// The engine's error string when `failed` (see `Emission::error`).
    pub error: Option<String>,
    /// Refused by the engine's admission policy — terminal, never serviced,
    /// and deliberately NOT a failure (see `EmissionKind::Rejected`).
    pub rejected: bool,
    /// Timestamp of every artifact, in order.
    pub artifacts: Vec<Instant>,
}

impl ReqRecord {
    pub fn new(id: u64, class: usize, input_artifacts: usize, warmup: bool, submit: Instant) -> ReqRecord {
        ReqRecord {
            id,
            class,
            input_artifacts,
            warmup,
            submit,
            admit: None,
            first: None,
            done: None,
            failed: false,
            error: None,
            rejected: false,
            artifacts: Vec::new(),
        }
    }

    pub fn queue_ms(&self) -> Option<f64> {
        self.admit.map(|a| ms(self.submit, a))
    }
    /// Time to first artifact — TTFT for a decoder. Includes queue + prefill.
    pub fn ttfa_ms(&self) -> Option<f64> {
        self.first.map(|f| ms(self.submit, f))
    }
    pub fn e2e_ms(&self) -> Option<f64> {
        self.done.map(|d| ms(self.submit, d))
    }
    /// Gaps between successive artifacts — ITL for a decoder.
    pub fn ial_ms(&self) -> Vec<f64> {
        self.artifacts.windows(2).map(|w| ms(w[0], w[1])).collect()
    }
    /// Mean time per output artifact after the first — TPOT for a decoder.
    /// `None` for a one-shot model, where it is not defined.
    pub fn tpoa_ms(&self) -> Option<f64> {
        let n = self.artifacts.len();
        if n < 2 {
            return None;
        }
        let last = *self.artifacts.last().unwrap();
        Some(ms(self.artifacts[0], last) / (n - 1) as f64)
    }
    pub fn output_artifacts(&self) -> usize {
        self.artifacts.len()
    }

    /// Whether this request met its class SLO. A failed or unfinished request
    /// never counts, whatever its latencies were.
    pub fn meets(&self, slo: &Slo) -> bool {
        if self.failed || self.done.is_none() {
            return false;
        }
        if let Some(budget) = slo.ttfa_ms {
            match self.ttfa_ms() {
                Some(t) if t <= budget => {}
                _ => return false,
            }
        }
        if let Some(budget) = slo.ial_ms {
            // Every gap must be inside the budget: one long stall is exactly the
            // user-visible failure an averaged ITL would hide.
            if self.ial_ms().iter().any(|&g| g > budget) {
                return false;
            }
        }
        if let Some(budget) = slo.e2e_ms {
            match self.e2e_ms() {
                Some(t) if t <= budget => {}
                _ => return false,
            }
        }
        true
    }
}

fn ms(a: Instant, b: Instant) -> f64 {
    b.saturating_duration_since(a).as_secs_f64() * 1000.0
}

/// The aggregate `performance` + `scheduling` blocks for a set of records.
#[derive(Debug)]
pub struct Summary {
    pub wall_s: f64,
    pub requests: usize,
    pub completed: usize,
    pub failed: usize,
    /// Refused at admission — never serviced, so excluded from the SLO
    /// denominator (refusing hopeless work must not read as missing SLOs).
    pub rejected: usize,
    pub input_artifacts: usize,
    pub output_artifacts: usize,
    pub requests_per_s: f64,
    pub input_per_s: f64,
    pub output_per_s: f64,
    pub goodput_per_s: f64,
    pub slo_attainment: f64,
    pub ttfa: Dist,
    pub ial: Dist,
    pub tpoa: Dist,
    pub e2e: Dist,
    pub queue: Dist,
    pub slo: Slo,
    /// A sample of DISTINCT failure reasons (first few, in arrival order) —
    /// the artifact is the deliverable, and "N failed" with the reasons
    /// discarded is unrecoverable after the fact.
    pub errors: Vec<String>,
}

impl Summary {
    /// Build from records. **Warm-up records are dropped here**, so no caller
    /// can accidentally include them.
    pub fn build(records: &[ReqRecord], wall_s: f64, slo: Slo) -> Summary {
        let measured: Vec<&ReqRecord> = records.iter().filter(|r| !r.warmup).collect();

        let mut ttfa = Dist::new();
        let mut ial = Dist::new();
        let mut tpoa = Dist::new();
        let mut e2e = Dist::new();
        let mut queue = Dist::new();

        let mut completed = 0usize;
        let mut failed = 0usize;
        let mut rejected = 0usize;
        let mut in_arts = 0usize;
        let mut out_arts = 0usize;
        let mut met = 0usize;

        for r in &measured {
            if r.rejected {
                rejected += 1;
                continue; // never serviced: no latency samples, no artifacts
            }
            if r.failed {
                failed += 1;
            } else if r.done.is_some() {
                completed += 1;
            }
            in_arts += r.input_artifacts;
            out_arts += r.output_artifacts();
            if let Some(v) = r.ttfa_ms() {
                ttfa.push(v);
            }
            if let Some(v) = r.e2e_ms() {
                e2e.push(v);
            }
            if let Some(v) = r.queue_ms() {
                queue.push(v);
            }
            if let Some(v) = r.tpoa_ms() {
                tpoa.push(v);
            }
            for g in r.ial_ms() {
                ial.push(g);
            }
            if r.meets(&slo) {
                met += 1;
            }
        }

        let n = measured.len();
        let secs = wall_s.max(1e-9);
        // Goodput counts only the output of SLO-satisfying requests: work that
        // missed its deadline is not useful work, however fast it was produced.
        let good_arts: usize =
            measured.iter().filter(|r| r.meets(&slo)).map(|r| r.output_artifacts()).sum();

        // SLO attainment is over ADMITTED work: a policy that sheds hopeless
        // requests must not have its refusals scored as SLO misses.
        let n_admitted = n - rejected;
        // First few DISTINCT failure reasons, so an all-failed run's artifact
        // says WHY instead of only how many.
        const MAX_ERROR_SAMPLES: usize = 5;
        let mut errors: Vec<String> = Vec::new();
        for r in &measured {
            if let Some(e) = r.error.as_ref().filter(|_| r.failed) {
                if !errors.contains(e) {
                    errors.push(e.clone());
                    if errors.len() >= MAX_ERROR_SAMPLES {
                        break;
                    }
                }
            }
        }
        Summary {
            wall_s,
            requests: n,
            completed,
            failed,
            rejected,
            input_artifacts: in_arts,
            output_artifacts: out_arts,
            requests_per_s: completed as f64 / secs,
            input_per_s: in_arts as f64 / secs,
            output_per_s: out_arts as f64 / secs,
            goodput_per_s: good_arts as f64 / secs,
            slo_attainment: if n_admitted == 0 { 1.0 } else { met as f64 / n_admitted as f64 },
            ttfa,
            ial,
            tpoa,
            e2e,
            queue,
            slo,
            errors,
        }
    }

    pub fn to_json(&mut self) -> Value {
        json!({
            "wall_s": r3(self.wall_s),
            "requests": self.requests,
            "completed": self.completed,
            "failed": self.failed,
            "errors": self.errors,
            "rejected": self.rejected,
            "requests_per_s": r3(self.requests_per_s),
            "input_artifacts_per_s": r3(self.input_per_s),
            "output_artifacts_per_s": r3(self.output_per_s),
            "goodput_per_s": r3(self.goodput_per_s),
            "slo_attainment": r3(self.slo_attainment),
            "slo": self.slo.to_json(),
            "ttfa_ms": self.ttfa.to_json(),
            "ial_ms": self.ial.to_json(),
            "tpoa_ms": self.tpoa.to_json(),
            "e2e_ms": self.e2e.to_json(),
        })
    }

    pub fn scheduling_json(&mut self, alone_e2e_ms: Option<f64>, per_class_rate: &[f64]) -> Value {
        // Normalised slowdown: how much slower a request is under load than the
        // same request run alone. The single most honest contention number.
        let slowdown = match (alone_e2e_ms, self.e2e.percentile(0.50)) {
            (Some(alone), Some(loaded)) if alone > 0.0 => Value::from(r3(loaded / alone)),
            _ => Value::Null,
        };
        json!({
            "queue_ms": self.queue.to_json(),
            "normalised_slowdown": slowdown,
            "jain_fairness": jain_fairness(per_class_rate).map(Value::from).unwrap_or(Value::Null),
            "starvation_ms_max": self.queue.max().map(r3).map(Value::from).unwrap_or(Value::Null),
            "preemptions": Value::Null,
            "decode_stall_ms_from_prefill": Value::Null,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    /// A request submitted at 0, admitted at 10, first artifact at 100, then one
    /// artifact every 10ms up to 5 artifacts, done at 140.
    fn sample(base: Instant, id: u64, warmup: bool) -> ReqRecord {
        let mut r = ReqRecord::new(id, 0, 32, warmup, base);
        r.admit = Some(at(base, 10));
        r.first = Some(at(base, 100));
        r.artifacts = vec![at(base, 100), at(base, 110), at(base, 120), at(base, 130), at(base, 140)];
        r.done = Some(at(base, 140));
        r
    }

    #[test]
    fn timeline_derivations_match_the_definitions() {
        let base = Instant::now();
        let r = sample(base, 0, false);
        assert_eq!(r.queue_ms(), Some(10.0));
        assert_eq!(r.ttfa_ms(), Some(100.0));
        assert_eq!(r.e2e_ms(), Some(140.0));
        assert_eq!(r.ial_ms(), vec![10.0, 10.0, 10.0, 10.0]);
        // TPOA = (last - first) / (n-1) = 40/4
        assert_eq!(r.tpoa_ms(), Some(10.0));
        assert_eq!(r.output_artifacts(), 5);
    }

    #[test]
    fn one_shot_request_has_no_tpoa_and_ttfa_equals_e2e() {
        let base = Instant::now();
        let mut r = ReqRecord::new(0, 0, 1, false, base);
        r.first = Some(at(base, 50));
        r.artifacts = vec![at(base, 50)];
        r.done = Some(at(base, 50));
        assert_eq!(r.tpoa_ms(), None, "TPOA is undefined for a single artifact");
        assert!(r.ial_ms().is_empty());
        assert_eq!(r.ttfa_ms(), r.e2e_ms());
    }

    #[test]
    fn warmup_records_are_excluded_from_every_statistic() {
        let base = Instant::now();
        let recs = vec![sample(base, 0, true), sample(base, 1, true), sample(base, 2, false)];
        let s = Summary::build(&recs, 1.0, Slo::NONE);
        assert_eq!(s.requests, 1, "only the measured request counts");
        assert_eq!(s.output_artifacts, 5, "warm-up artifacts must not inflate the rate");
    }

    /// A rejected request is terminal but neither completed nor failed, and it
    /// must not drag down SLO attainment: refusing hopeless work is the
    /// behaviour the overload scenario exists to reward, not to penalise.
    #[test]
    fn rejected_requests_leave_the_slo_denominator() {
        let base = Instant::now();
        let mut rej = ReqRecord::new(1, 0, 32, false, base);
        rej.rejected = true;
        let recs = vec![sample(base, 0, false), rej];
        let s = Summary::build(&recs, 1.0, Slo::ttfa(1000.0));
        assert_eq!(s.requests, 2);
        assert_eq!(s.rejected, 1);
        assert_eq!(s.completed, 1);
        assert_eq!(s.failed, 0, "a refusal is not an error");
        assert_eq!(s.slo_attainment, 1.0, "attainment is over admitted work only");
    }

    #[test]
    fn a_single_stall_fails_an_ial_slo_even_when_the_mean_is_fine() {
        let base = Instant::now();
        let mut r = ReqRecord::new(0, 0, 1, false, base);
        r.first = Some(at(base, 10));
        // Nine 1ms gaps and one 500ms stall: mean gap ~50ms, one gap catastrophic.
        let mut t = 10u64;
        r.artifacts.push(at(base, t));
        for i in 0..10 {
            t += if i == 5 { 500 } else { 1 };
            r.artifacts.push(at(base, t));
        }
        r.done = Some(at(base, t));
        let slo = Slo { ttfa_ms: None, ial_ms: Some(50.0), e2e_ms: None };
        assert!(!r.meets(&slo), "a single stall must fail the SLO — an averaged ITL would hide it");
    }

    #[test]
    fn failed_and_unfinished_requests_are_never_goodput() {
        let base = Instant::now();
        let mut failed = sample(base, 0, false);
        failed.failed = true;
        let mut unfinished = sample(base, 1, false);
        unfinished.done = None;
        assert!(!failed.meets(&Slo::NONE));
        assert!(!unfinished.meets(&Slo::NONE));

        let s = Summary::build(&[failed, unfinished], 1.0, Slo::NONE);
        assert_eq!(s.failed, 1);
        assert_eq!(s.completed, 0);
        assert_eq!(s.goodput_per_s, 0.0);
        assert_eq!(s.requests, 2, "they stay in the denominator");
    }

    #[test]
    fn goodput_counts_only_slo_satisfying_output() {
        let base = Instant::now();
        let fast = sample(base, 0, false); // ttfa 100ms
        let mut slow = sample(base, 1, false);
        slow.first = Some(at(base, 900)); // ttfa 900ms — misses a 500ms budget
        slow.artifacts[0] = at(base, 900);
        let slo = Slo::ttfa(500.0);
        let s = Summary::build(&[fast, slow], 1.0, slo);
        assert_eq!(s.output_artifacts, 10, "throughput counts everything");
        assert_eq!(s.goodput_per_s, 5.0, "goodput counts only the request that met its SLO");
        assert_eq!(s.slo_attainment, 0.5);
    }

    #[test]
    fn empty_run_is_null_not_a_fake_zero() {
        let mut s = Summary::build(&[], 1.0, Slo::NONE);
        let j = s.to_json();
        assert!(j["ttfa_ms"]["p50"].is_null());
        assert_eq!(j["requests"], 0);
    }
}
