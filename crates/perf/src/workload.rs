// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Workload definitions: request shapes and **arrival processes**.
//!
//! Requests that all exist at t=0 exercise a different engine than requests
//! arriving independently over time. Both are legitimate measurements and they
//! are never mixed — the arrival process is part of the workload, recorded in
//! the artifact, and `compare` warns when two results used different ones.
//!
//! Every distribution is driven by `data::rng::Rng` from an explicit seed, so a
//! workload replays exactly.

use data::rng::Rng;
use serde_json::{json, Value};

use crate::target::PerfRequest;

/// How requests arrive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arrival {
    /// A fixed number of requests in flight; a new one is submitted as soon as
    /// one finishes. The most deterministic engine pressure — the default.
    ClosedLoop { concurrency: usize },
    /// Everything available at t=0. Saturation; an optimistic upper bound.
    Saturated,
    /// Independent arrivals at a mean rate, with Gamma-distributed inter-arrival
    /// times. `burstiness == 1.0` is Poisson; `< 1` is burstier, `> 1` smoother.
    Rate { per_s: f64, burstiness: f64 },
    /// Linear ramp from `from_per_s` to `to_per_s` over the run.
    Ramp { from_per_s: f64, to_per_s: f64 },
}

impl Arrival {
    pub fn name(&self) -> &'static str {
        match self {
            Arrival::ClosedLoop { .. } => "closed_loop",
            Arrival::Saturated => "saturated",
            Arrival::Rate { .. } => "rate",
            Arrival::Ramp { .. } => "ramp",
        }
    }
    /// The concurrency cap, when the process defines one.
    pub fn concurrency(&self) -> Option<usize> {
        match self {
            Arrival::ClosedLoop { concurrency } => Some(*concurrency),
            _ => None,
        }
    }
}

/// A length distribution for input or output artifact counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lengths {
    Fixed(usize),
    /// Uniform over `[lo, hi]`.
    Uniform { lo: usize, hi: usize },
    /// Log-normal-ish: a Gaussian in log space, clamped to `[lo, hi]`. Closer to
    /// real prompt-length distributions than uniform.
    LogNormal { median: usize, sigma: f64, lo: usize, hi: usize },
}

impl Lengths {
    pub fn sample(&self, rng: &mut Rng) -> usize {
        match *self {
            Lengths::Fixed(n) => n,
            Lengths::Uniform { lo, hi } => {
                if hi <= lo {
                    lo
                } else {
                    lo + (rng.next_u64() as usize % (hi - lo + 1))
                }
            }
            Lengths::LogNormal { median, sigma, lo, hi } => {
                let g = rng.next_gaussian();
                let v = (median as f64) * (g * sigma).exp();
                (v.round() as usize).clamp(lo, hi)
            }
        }
    }
    pub fn to_json(&self) -> Value {
        match *self {
            Lengths::Fixed(n) => json!({ "dist": "fixed", "value": n }),
            Lengths::Uniform { lo, hi } => json!({ "dist": "uniform", "lo": lo, "hi": hi }),
            Lengths::LogNormal { median, sigma, lo, hi } => {
                json!({ "dist": "lognormal", "median": median, "sigma": sigma, "lo": lo, "hi": hi })
            }
        }
    }
}

/// One traffic class: a request shape plus the latency contract it must meet.
#[derive(Clone, Debug)]
pub struct Class {
    pub name: String,
    pub input: Lengths,
    pub output: Lengths,
    /// Share of total requests belonging to this class (weights are normalised).
    pub weight: f64,
    pub slo: Slo,
}

/// The latency contract a class must satisfy to count as goodput.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slo {
    /// P99 time-to-first-artifact budget, ms. `None` = no TTFA requirement.
    pub ttfa_ms: Option<f64>,
    /// P99 inter-artifact latency budget, ms. `None` = no IAL requirement.
    pub ial_ms: Option<f64>,
    /// End-to-end budget, ms.
    pub e2e_ms: Option<f64>,
}

impl Slo {
    pub const NONE: Slo = Slo { ttfa_ms: None, ial_ms: None, e2e_ms: None };
    pub fn ttfa(ms: f64) -> Slo {
        Slo { ttfa_ms: Some(ms), ial_ms: None, e2e_ms: None }
    }
    pub fn interactive(ttfa_ms: f64, ial_ms: f64) -> Slo {
        Slo { ttfa_ms: Some(ttfa_ms), ial_ms: Some(ial_ms), e2e_ms: None }
    }
    pub fn to_json(&self) -> Value {
        json!({
            "ttfa_ms_p99": self.ttfa_ms.map(Value::from).unwrap_or(Value::Null),
            "ial_ms_p99": self.ial_ms.map(Value::from).unwrap_or(Value::Null),
            "e2e_ms": self.e2e_ms.map(Value::from).unwrap_or(Value::Null),
        })
    }
}

/// A complete workload: what to send, how it arrives, and how much.
#[derive(Clone, Debug)]
pub struct Workload {
    pub name: String,
    pub classes: Vec<Class>,
    pub arrival: Arrival,
    pub num_requests: usize,
    pub warmup_requests: usize,
    /// Force the full requested output length instead of letting a request stop
    /// early. Essential in synthetic runs: without it, requests that stop early
    /// silently shorten the workload and inflate the rate.
    pub ignore_stop: bool,
    pub seed: u64,
}

impl Workload {
    /// Generate the request stream. Deterministic in `seed`.
    pub fn requests(&self) -> Vec<PerfRequest> {
        let mut rng = Rng::new(self.seed);
        let total: f64 = self.classes.iter().map(|c| c.weight).sum();
        let n = self.num_requests + self.warmup_requests;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            // Pick a class by weight.
            let mut pick = rng.next_f64() * total;
            let mut ci = 0;
            for (idx, c) in self.classes.iter().enumerate() {
                pick -= c.weight;
                if pick <= 0.0 {
                    ci = idx;
                    break;
                }
                ci = idx;
            }
            let c = &self.classes[ci];
            out.push(PerfRequest {
                input_artifacts: c.input.sample(&mut rng).max(1),
                output_artifacts: c.output.sample(&mut rng).max(1),
                class: ci,
                seed: self.seed.wrapping_add(i as u64),
            });
        }
        out
    }

    /// Inter-arrival delays in seconds, parallel to [`Workload::requests`].
    /// Empty for closed-loop and saturated processes, which are driven by
    /// completion rather than by a clock.
    pub fn arrival_delays(&self) -> Vec<f64> {
        let n = self.num_requests + self.warmup_requests;
        let mut rng = Rng::new(self.seed ^ 0x9E37_79B9_7F4A_7C15);
        match self.arrival {
            Arrival::ClosedLoop { .. } | Arrival::Saturated => Vec::new(),
            Arrival::Rate { per_s, burstiness } => {
                (0..n).map(|_| gamma_interarrival(&mut rng, per_s, burstiness)).collect()
            }
            Arrival::Ramp { from_per_s, to_per_s } => (0..n)
                .map(|i| {
                    let f = if n <= 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
                    let rate = from_per_s + (to_per_s - from_per_s) * f;
                    gamma_interarrival(&mut rng, rate.max(1e-6), 1.0)
                })
                .collect(),
        }
    }

    pub fn slo_for(&self, class: usize) -> Slo {
        self.classes.get(class).map(|c| c.slo).unwrap_or(Slo::NONE)
    }

    pub fn to_json(&self) -> Value {
        let (rate, burst) = match self.arrival {
            Arrival::Rate { per_s, burstiness } => (Value::from(per_s), Value::from(burstiness)),
            Arrival::Ramp { from_per_s, to_per_s } => {
                (json!({ "from": from_per_s, "to": to_per_s }), Value::Null)
            }
            _ => (Value::Null, Value::Null),
        };
        let single = self.classes.len() == 1;
        json!({
            "name": self.name,
            "arrival": self.arrival.name(),
            "concurrency": self.arrival.concurrency().map(Value::from).unwrap_or(Value::Null),
            "request_rate": rate,
            "burstiness": burst,
            "num_requests": self.num_requests,
            "warmup_requests": self.warmup_requests,
            "ignore_stop": self.ignore_stop,
            "seed": self.seed,
            "input_artifacts": if single { self.classes[0].input.to_json() } else { Value::Null },
            "output_artifacts": if single { self.classes[0].output.to_json() } else { Value::Null },
            "classes": self.classes.iter().map(|c| json!({
                "name": c.name,
                "weight": c.weight,
                "input": c.input.to_json(),
                "output": c.output.to_json(),
                "slo": c.slo.to_json(),
            })).collect::<Vec<_>>(),
        })
    }
}

/// Gamma-distributed inter-arrival time with mean `1/rate`. `shape == 1` is
/// exponential (a Poisson process); smaller shape is burstier.
fn gamma_interarrival(rng: &mut Rng, rate: f64, shape: f64) -> f64 {
    let mean = 1.0 / rate.max(1e-9);
    if (shape - 1.0).abs() < 1e-9 {
        // Exponential via inverse transform.
        let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
        return -mean * (1.0 - u).ln();
    }
    // Sum-of-exponentials approximation for integer-ish shape, then scale so the
    // mean stays 1/rate. Good enough to shape burstiness; exactness is not the
    // point, reproducibility is.
    let k = shape.max(0.05);
    let n = k.round().max(1.0) as usize;
    let mut acc = 0.0;
    for _ in 0..n {
        let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
        acc += -(1.0 - u).ln();
    }
    mean * acc / n as f64
}

// ===================== the standard workload matrix =====================

/// The eight standard shapes. Every one runs on every device, so a single grid
/// characterises a model on a machine. See `docs/performance/benchmarking.md`.
pub fn standard(name: &str, arrival: Arrival, num_requests: usize, seed: u64) -> Option<Workload> {
    let (input, output, slo) = match name {
        "interactive" => (128, 256, Slo::interactive(500.0, 50.0)),
        "chat" => (1024, 256, Slo::interactive(2000.0, 50.0)),
        "rag" => (4096, 128, Slo::interactive(4000.0, 50.0)),
        "rag_long" => (16384, 256, Slo::interactive(10000.0, 80.0)),
        "agent" => (8192, 1024, Slo::interactive(6000.0, 50.0)),
        "decode_heavy" => (128, 2048, Slo::interactive(500.0, 40.0)),
        "prefill_heavy" => (32768, 64, Slo::ttfa(20000.0)),
        "shared_prefix" => (8192, 256, Slo::interactive(6000.0, 50.0)),
        _ => return None,
    };
    Some(Workload {
        name: name.to_string(),
        classes: vec![Class {
            name: name.to_string(),
            input: Lengths::Fixed(input),
            output: Lengths::Fixed(output),
            weight: 1.0,
            slo,
        }],
        arrival,
        num_requests,
        warmup_requests: 0,
        ignore_stop: true,
        seed,
    })
}

/// The names in [`standard`], in matrix order.
pub const STANDARD: &[&str] = &[
    "interactive",
    "chat",
    "rag",
    "rag_long",
    "agent",
    "decode_heavy",
    "prefill_heavy",
    "shared_prefix",
];

/// Scale a workload down by `div` so the same eight shapes run on a small
/// device (integrated GPU, NPU, laptop CPU) without OOM. The artifact records
/// the factor, so a scaled run is never silently compared to a full one.
pub fn scaled(w: &Workload, div: usize) -> Workload {
    let d = div.max(1);
    let shrink = |l: Lengths| match l {
        Lengths::Fixed(n) => Lengths::Fixed((n / d).max(1)),
        Lengths::Uniform { lo, hi } => Lengths::Uniform { lo: (lo / d).max(1), hi: (hi / d).max(1) },
        Lengths::LogNormal { median, sigma, lo, hi } => LogNormalScaled(median, sigma, lo, hi, d).into(),
    };
    let mut out = w.clone();
    out.name = format!("{}/div{d}", w.name);
    for c in &mut out.classes {
        c.input = shrink(c.input);
        c.output = shrink(c.output);
    }
    out
}

struct LogNormalScaled(usize, f64, usize, usize, usize);
impl From<LogNormalScaled> for Lengths {
    fn from(s: LogNormalScaled) -> Lengths {
        Lengths::LogNormal {
            median: (s.0 / s.4).max(1),
            sigma: s.1,
            lo: (s.2 / s.4).max(1),
            hi: (s.3 / s.4).max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_stream_is_deterministic_in_the_seed() {
        let w = standard("chat", Arrival::ClosedLoop { concurrency: 4 }, 50, 7).unwrap();
        let a = w.requests();
        let b = w.requests();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.input_artifacts, y.input_artifacts);
            assert_eq!(x.output_artifacts, y.output_artifacts);
        }
    }

    #[test]
    fn different_seeds_give_different_streams_when_random() {
        let mk = |seed| Workload {
            name: "t".into(),
            classes: vec![Class {
                name: "t".into(),
                input: Lengths::Uniform { lo: 1, hi: 10_000 },
                output: Lengths::Fixed(4),
                weight: 1.0,
                slo: Slo::NONE,
            }],
            arrival: Arrival::Saturated,
            num_requests: 64,
            warmup_requests: 0,
            ignore_stop: true,
            seed,
        };
        let a: Vec<_> = mk(1).requests().iter().map(|r| r.input_artifacts).collect();
        let b: Vec<_> = mk(2).requests().iter().map(|r| r.input_artifacts).collect();
        assert_ne!(a, b, "seed must actually drive the stream");
    }

    #[test]
    fn warmup_requests_are_generated_on_top_of_the_measured_ones() {
        let mut w = standard("chat", Arrival::Saturated, 10, 1).unwrap();
        w.warmup_requests = 5;
        assert_eq!(w.requests().len(), 15);
    }

    #[test]
    fn standard_matrix_shapes_match_the_documented_table() {
        let w = standard("prefill_heavy", Arrival::Saturated, 1, 0).unwrap();
        let r = &w.requests()[0];
        assert_eq!(r.input_artifacts, 32768);
        assert_eq!(r.output_artifacts, 64);
        assert!(standard("nope", Arrival::Saturated, 1, 0).is_none());
        assert_eq!(STANDARD.len(), 8);
        for n in STANDARD {
            assert!(standard(n, Arrival::Saturated, 1, 0).is_some(), "{n} must be defined");
        }
    }

    #[test]
    fn closed_loop_and_saturated_have_no_arrival_clock() {
        let w = standard("chat", Arrival::ClosedLoop { concurrency: 8 }, 20, 1).unwrap();
        assert!(w.arrival_delays().is_empty());
    }

    #[test]
    fn rate_arrival_has_the_requested_mean() {
        let mut w = standard("chat", Arrival::Rate { per_s: 10.0, burstiness: 1.0 }, 20_000, 3).unwrap();
        w.num_requests = 20_000;
        let d = w.arrival_delays();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        // Mean inter-arrival should be ~1/10 s. Loose bound: this is a sampled
        // process, the assertion is that the rate is honoured, not exact.
        assert!((mean - 0.1).abs() < 0.02, "mean inter-arrival {mean} should be near 0.1s");
    }

    #[test]
    fn scaling_shrinks_every_class_and_renames() {
        let w = standard("rag_long", Arrival::Saturated, 4, 1).unwrap();
        let s = scaled(&w, 8);
        let r = &s.requests()[0];
        assert_eq!(r.input_artifacts, 16384 / 8);
        assert!(s.name.contains("div8"), "a scaled run must be labelled so it is never compared to a full one");
    }

    #[test]
    fn slo_none_never_constrains() {
        let s = Slo::NONE;
        assert!(s.ttfa_ms.is_none() && s.ial_ms.is_none() && s.e2e_ms.is_none());
    }
}
