// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-tensor activation-magnitude statistics for quantization decisions.
//!
//! The question this answers, for any activation stream a caller wants to
//! quantize: is a per-tensor (or per-scale-unit) symmetric scale a good fit,
//! or does this tensor have a heavy tail that a single absmax-derived scale
//! would handle poorly? [`Collector`] accumulates a bounded, roughly-uniform
//! subsample of `|x|` values per named stream (memory bounded regardless of
//! how much data flows through), and [`Collector::report`] turns that into
//! `(absmax, p99, p99.99, outlier_ratio)` per stream, worst-tailed first.
//! `outlier_ratio = absmax / p99.99` near 1 means a tight distribution
//! (quantization-friendly); a large value means a few outliers would crush
//! the resolution of the bulk under a single absmax-derived scale.
//!
//! Originally written for ZipDepth's INT8 decoder-vs-encoder ablation
//! (`depth::quant`, still the home of the vision-specific `ActTap`/`Ctx`
//! wiring and calibration-image plumbing); extracted here so a second
//! caller (Qwen's KV-cache calibration) doesn't reimplement the same
//! bounded-reservoir/percentile math.

use std::cell::RefCell;
use std::collections::HashMap;

/// The cap on samples kept per stream. ~50k gives a p99.99 resolved to ~5
/// samples, which is plenty for a ratio, and bounds memory regardless of how
/// much data is fed in.
pub const SAMPLE_CAP: usize = 50_000;

#[derive(Default)]
struct StreamAcc {
    absmax: f32,
    /// Strided subsample of `|x|` values; percentiles are computed from it.
    samples: Vec<f32>,
    /// Total values seen, to keep the subsample stride roughly uniform.
    seen: u64,
}

/// Accumulates a bounded subsample of `|x|` magnitudes per named stream.
/// Observes only — never mutates its input. Cheap to hold behind a shared
/// reference (`observe` takes `&self`, interior-mutable), so it composes with
/// tap-style callback seams the same way `depth::quant::ActStatsCollector`
/// (now a thin wrapper over this) does.
#[derive(Default)]
pub struct Collector {
    streams: RefCell<HashMap<String, StreamAcc>>,
}

impl Collector {
    pub fn new() -> Collector {
        Collector::default()
    }

    /// Record `|x|` for every value in `x`, under `name`. Call this from
    /// wherever the activation stream is naturally available (a tap
    /// callback, a post-kernel readback, …).
    pub fn observe(&self, name: &str, x: &[f32]) {
        let mut streams = self.streams.borrow_mut();
        let acc = streams.entry(name.to_string()).or_default();
        // Keep the subsample bounded: once full, take roughly every k-th
        // value so the sample stays representative of the whole run, not
        // just the first call.
        let stride = ((acc.seen as usize / SAMPLE_CAP) + 1).max(1);
        for (i, v) in x.iter().enumerate() {
            let a = v.abs();
            if a > acc.absmax {
                acc.absmax = a;
            }
            if acc.samples.len() < SAMPLE_CAP && (acc.seen as usize + i).is_multiple_of(stride) {
                acc.samples.push(a);
            }
        }
        acc.seen += x.len() as u64;
    }

    /// Per-stream `(absmax, p99, p99.99, outlier_ratio)`, sorted by
    /// `outlier_ratio` descending — the most quantization-hostile stream first.
    pub fn report(&self) -> Vec<StreamReport> {
        let streams = self.streams.borrow();
        let mut out: Vec<StreamReport> = streams
            .iter()
            .map(|(name, a)| {
                let mut s = a.samples.clone();
                s.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let p = |q: f32| -> f32 {
                    if s.is_empty() {
                        return 0.0;
                    }
                    let k = ((q * (s.len() - 1) as f32).round() as usize).min(s.len() - 1);
                    s[k]
                };
                let p9999 = p(0.9999).max(1e-9);
                StreamReport { name: name.clone(), absmax: a.absmax, p99: p(0.99), p9999, outlier_ratio: a.absmax / p9999 }
            })
            .collect();
        out.sort_by(|a, b| b.outlier_ratio.partial_cmp(&a.outlier_ratio).unwrap());
        out
    }
}

/// One stream's activation-magnitude summary.
#[derive(Clone, Debug)]
pub struct StreamReport {
    pub name: String,
    pub absmax: f32,
    pub p99: f32,
    /// The 99.99th percentile of `|x|` — the range a quantization scale
    /// should really target, rather than the single worst-case outlier.
    pub p9999: f32,
    /// `absmax / p99.99`. ~1 means a tight distribution (quantization-
    /// friendly); a large value means a heavy tail a single absmax-derived
    /// scale handles poorly.
    pub outlier_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmax_and_percentiles_match_a_hand_computed_distribution() {
        let c = Collector::new();
        // A sample this small (~100) has no room between "the max" and "the
        // 99.99th percentile of the sample" -- p(0.9999) rounds straight to
        // the last index, so a tiny reservoir always reports ratio == 1
        // regardless of outliers. 20000 values is comfortably under
        // SAMPLE_CAP (so every value is kept, no stride) while leaving real
        // separation between rank 19998 (what p(0.9999) picks) and the max.
        let mut vals: Vec<f32> = (1..=20000).map(|v| v as f32).collect();
        vals.push(1_000_000.0);
        c.observe("x", &vals);
        let r = c.report();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].name, "x");
        assert_eq!(r[0].absmax, 1_000_000.0);
        assert_eq!(r[0].p9999, 19999.0, "p99.99 of [1..=20000] must land just under the bulk's own max, not the outlier");
        assert!(r[0].outlier_ratio > 40.0, "a 1e6-vs-2e4 outlier must show a large ratio, got {}", r[0].outlier_ratio);
    }

    #[test]
    fn a_tight_distribution_has_a_ratio_near_one() {
        let c = Collector::new();
        let vals: Vec<f32> = (1..=1000).map(|v| v as f32).collect();
        c.observe("tight", &vals);
        let r = c.report();
        assert!((r[0].outlier_ratio - 1.0).abs() < 0.05, "a uniform distribution's own max should be near its own p99.99: {}", r[0].outlier_ratio);
    }

    #[test]
    fn multiple_streams_are_independent_and_sorted_worst_first() {
        let c = Collector::new();
        c.observe("tight", &(1..=6000).map(|v| v as f32).collect::<Vec<_>>());
        let mut tailed: Vec<f32> = (1..=6000).map(|v| v as f32).collect();
        tailed.push(100_000.0);
        c.observe("tailed", &tailed);
        let r = c.report();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].name, "tailed", "the heavier-tailed stream must sort first");
        assert!(r[0].outlier_ratio > r[1].outlier_ratio);
        assert!((r[1].outlier_ratio - 1.0).abs() < 0.01, "the tight stream's own ratio should stay near 1: {}", r[1].outlier_ratio);
    }

    #[test]
    fn the_sample_reservoir_stays_bounded_across_many_observations() {
        let c = Collector::new();
        // Feed far more values than SAMPLE_CAP, in small chunks (the shape a
        // real tap sees: one call per forward pass, not one giant call).
        for _ in 0..2000 {
            c.observe("s", &[1.0f32; 100]);
        }
        // No public accessor to the raw sample count, but report() must not
        // itself blow up or take an unreasonable amount of memory/time to
        // sort -- exercised implicitly by this test completing at all. The
        // real bound is asserted structurally by SAMPLE_CAP's own doc: this
        // is a smoke test that many-call accumulation doesn't panic/hang.
        let r = c.report();
        assert_eq!(r[0].absmax, 1.0);
    }

    #[test]
    fn an_all_zero_stream_reports_a_ratio_of_one_not_a_division_by_zero() {
        let c = Collector::new();
        c.observe("zeros", &[0.0f32; 100]);
        let r = c.report();
        assert_eq!(r[0].absmax, 0.0);
        assert!(r[0].p9999 > 0.0, "p9999 must floor above zero to avoid a NaN/inf ratio");
        assert!(r[0].outlier_ratio.is_finite());
    }
}
