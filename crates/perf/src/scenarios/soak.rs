// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `soak` — long-duration drift.
//!
//! Short benchmarks hide everything that accumulates: allocator fragmentation,
//! host memory creep, cache degradation, descriptor and thread leaks, scheduler
//! overhead that grows with the number of sequences ever seen. A server that is
//! 5% faster for ten minutes and degrades after twelve hours is not the better
//! server, and no amount of re-running a 60-second benchmark will show it.
//!
//! So this samples the same handful of numbers once per interval over hours and
//! reports **drift**: the trend, not the mean. The deliverable is
//! "throughput fell 18% and P99 doubled over 6 hours, host RSS grew 2.3 GB",
//! which is a different kind of statement from any single-point measurement.

use serde_json::{json, Value};

use crate::stats::r3;

/// One periodic sample.
#[derive(Clone, Debug)]
pub struct Sample {
    pub elapsed_s: f64,
    pub output_per_s: f64,
    pub ttfa_p99_ms: Option<f64>,
    pub ial_p99_ms: Option<f64>,
    pub host_mem_mb: Option<f64>,
    pub kv_free_blocks: Option<u32>,
    pub open_fds: Option<usize>,
    pub threads: Option<usize>,
    pub errors: usize,
}

/// Least-squares slope of `y` against elapsed hours — the drift per hour.
fn slope_per_hour(samples: &[Sample], y: impl Fn(&Sample) -> Option<f64>) -> Option<f64> {
    let pts: Vec<(f64, f64)> =
        samples.iter().filter_map(|s| y(s).map(|v| (s.elapsed_s / 3600.0, v))).collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    (denom.abs() > 1e-12).then(|| (n * sxy - sx * sy) / denom)
}

/// Percentage change from the first sample to the last.
fn total_change_pct(samples: &[Sample], y: impl Fn(&Sample) -> Option<f64>) -> Option<f64> {
    let vals: Vec<f64> = samples.iter().filter_map(&y).collect();
    let (first, last) = (vals.first()?, vals.last()?);
    (first.abs() > 1e-12).then(|| (last - first) / first * 100.0)
}

#[derive(Debug, Default)]
pub struct Report {
    pub samples: Vec<Sample>,
    pub restarts: usize,
}

impl Report {
    pub fn duration_s(&self) -> f64 {
        self.samples.last().map(|s| s.elapsed_s).unwrap_or(0.0)
    }

    /// A run shorter than this cannot support an hourly extrapolation: the
    /// per-hour slope of a few seconds of samples is dominated by warm-up and
    /// scheduling noise, and printing it as "%/h" invites a completely wrong
    /// conclusion. Below it, drift is reported as `null`.
    pub const MIN_TREND_S: f64 = 600.0;

    /// True when the run was long enough for a trend to mean anything.
    pub fn trend_valid(&self) -> bool {
        self.duration_s() >= Self::MIN_TREND_S && self.samples.len() >= 3
    }

    /// Throughput drift, % per hour. Negative means the engine is slowing down.
    /// `None` when the run was too short to extrapolate.
    pub fn throughput_drift_pct_per_h(&self) -> Option<f64> {
        if !self.trend_valid() {
            return None;
        }
        let base = self.samples.first()?.output_per_s;
        let slope = slope_per_hour(&self.samples, |s| Some(s.output_per_s))?;
        (base.abs() > 1e-12).then(|| slope / base * 100.0)
    }

    /// Host memory growth, MB per hour. Sustained positive growth is a leak.
    pub fn memory_growth_mb_per_h(&self) -> Option<f64> {
        if !self.trend_valid() {
            return None;
        }
        slope_per_hour(&self.samples, |s| s.host_mem_mb)
    }

    /// P99 latency drift, % per hour. `None` when the run was too short.
    pub fn latency_drift_pct_per_h(&self) -> Option<f64> {
        if !self.trend_valid() {
            return None;
        }
        let base = self.samples.first()?.ttfa_p99_ms?;
        let slope = slope_per_hour(&self.samples, |s| s.ttfa_p99_ms)?;
        (base.abs() > 1e-12).then(|| slope / base * 100.0)
    }

    /// Did KV capacity shrink over the run? A pool that loses free blocks it
    /// never gets back is leaking sequences.
    pub fn kv_leak_blocks(&self) -> Option<i64> {
        let first = self.samples.first()?.kv_free_blocks? as i64;
        let last = self.samples.last()?.kv_free_blocks? as i64;
        Some(first - last)
    }

    pub fn total_errors(&self) -> usize {
        self.samples.iter().map(|s| s.errors).sum()
    }

    /// A soak passes when nothing is trending the wrong way beyond tolerance.
    pub fn healthy(&self, max_slowdown_pct_per_h: f64, max_mem_growth_mb_per_h: f64) -> bool {
        let slow_ok = self.throughput_drift_pct_per_h().map(|d| d >= -max_slowdown_pct_per_h).unwrap_or(true);
        let mem_ok = self.memory_growth_mb_per_h().map(|g| g <= max_mem_growth_mb_per_h).unwrap_or(true);
        let kv_ok = self.kv_leak_blocks().map(|l| l <= 0).unwrap_or(true);
        slow_ok && mem_ok && kv_ok && self.restarts == 0
    }

    pub fn to_json(&self) -> Value {
        json!({
            "duration_s": r3(self.duration_s()),
            "samples": self.samples.len(),
            "trend_valid": self.trend_valid(),
            "min_trend_s": Self::MIN_TREND_S,
            "restarts": self.restarts,
            "errors": self.total_errors(),
            "throughput_drift_pct_per_h": self.throughput_drift_pct_per_h().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "throughput_total_change_pct": total_change_pct(&self.samples, |s| Some(s.output_per_s)).map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "latency_drift_pct_per_h": self.latency_drift_pct_per_h().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "host_mem_growth_mb_per_h": self.memory_growth_mb_per_h().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "kv_leaked_blocks": self.kv_leak_blocks().map(Value::from).unwrap_or(Value::Null),
            "series": self.samples.iter().map(|s| json!({
                "elapsed_s": r3(s.elapsed_s),
                "output_artifacts_per_s": r3(s.output_per_s),
                "ttfa_ms_p99": s.ttfa_p99_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
                "ial_ms_p99": s.ial_p99_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
                "host_mem_mb": s.host_mem_mb.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
                "kv_free_blocks": s.kv_free_blocks.map(Value::from).unwrap_or(Value::Null),
                "errors": s.errors,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Host RSS in MB, for the memory-growth series.
pub fn host_mem_mb() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: f64 = text.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096.0 / 1024.0 / 1024.0)
}

/// Open file descriptors, for the leak series.
pub fn open_fds() -> Option<usize> {
    Some(std::fs::read_dir("/proc/self/fd").ok()?.count())
}

pub fn render(r: &Report) -> String {
    let f = |v: Option<f64>, unit: &str| {
        v.map(|x| format!("{x:+.2}{unit}")).unwrap_or_else(|| "—".into())
    };
    let mut s = format!(
        "\n  soak {:.1} min, {} samples, {} errors, {} restarts\n",
        r.duration_s() / 60.0,
        r.samples.len(),
        r.total_errors(),
        r.restarts
    );
    s.push_str(&format!("  throughput drift {}   latency drift {}\n", f(r.throughput_drift_pct_per_h(), "%/h"), f(r.latency_drift_pct_per_h(), "%/h")));
    s.push_str(&format!("  host memory {}   kv leaked blocks {}\n",
        f(r.memory_growth_mb_per_h(), " MB/h"),
        r.kv_leak_blocks().map(|v| v.to_string()).unwrap_or_else(|| "—".into())));
    if !r.healthy(5.0, 64.0) {
        s.push_str("  ! trending the wrong way — see the series\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_s: f64, tput: f64, mem: f64, kv: u32) -> Sample {
        Sample {
            elapsed_s: t_s,
            output_per_s: tput,
            ttfa_p99_ms: Some(100.0),
            ial_p99_ms: Some(10.0),
            host_mem_mb: Some(mem),
            kv_free_blocks: Some(kv),
            open_fds: Some(20),
            threads: Some(8),
            errors: 0,
        }
    }

    #[test]
    fn a_flat_run_shows_no_drift() {
        let r = Report {
            samples: (0..5).map(|i| sample(i as f64 * 3600.0, 100.0, 500.0, 1000)).collect(),
            restarts: 0,
        };
        assert!(r.throughput_drift_pct_per_h().unwrap().abs() < 1e-9);
        assert!(r.healthy(5.0, 64.0));
    }

    #[test]
    fn slowdown_shows_as_negative_drift() {
        // 100 -> 70 over 3 hours = -10 per hour on a base of 100 => -10%/h.
        let r = Report {
            samples: vec![
                sample(0.0, 100.0, 500.0, 1000),
                sample(3600.0, 90.0, 500.0, 1000),
                sample(7200.0, 80.0, 500.0, 1000),
                sample(10800.0, 70.0, 500.0, 1000),
            ],
            restarts: 0,
        };
        let d = r.throughput_drift_pct_per_h().unwrap();
        assert!((d + 10.0).abs() < 1e-6, "got {d}");
        assert!(!r.healthy(5.0, 64.0), "a 10%/h slowdown must fail a 5%/h tolerance");
    }

    #[test]
    fn memory_growth_is_reported_per_hour() {
        let r = Report {
            samples: vec![
                sample(0.0, 100.0, 500.0, 1000),
                sample(1800.0, 100.0, 600.0, 1000),
                sample(3600.0, 100.0, 700.0, 1000),
            ],
            restarts: 0,
        };
        assert!((r.memory_growth_mb_per_h().unwrap() - 200.0).abs() < 1e-6);
        assert!(!r.healthy(5.0, 64.0), "200 MB/h growth must fail a 64 MB/h tolerance");
    }

    #[test]
    fn kv_blocks_that_never_come_back_are_a_leak() {
        let r = Report {
            samples: vec![
                sample(0.0, 100.0, 500.0, 1000),
                sample(1800.0, 100.0, 500.0, 970),
                sample(3600.0, 100.0, 500.0, 940),
            ],
            restarts: 0,
        };
        assert_eq!(r.kv_leak_blocks(), Some(60));
        assert!(!r.healthy(50.0, 10_000.0), "a KV leak alone must fail the run");
    }

    #[test]
    fn a_restart_fails_the_soak_outright() {
        let r = Report { samples: vec![sample(0.0, 100.0, 500.0, 1000)], restarts: 1 };
        assert!(!r.healthy(100.0, 100_000.0));
    }

    #[test]
    fn a_short_run_refuses_to_extrapolate_an_hourly_trend() {
        // 10 seconds of samples say nothing about an hour.
        let r = Report {
            samples: (0..5).map(|i| sample(i as f64 * 2.0, 100.0 - i as f64 * 10.0, 500.0, 1000)).collect(),
            restarts: 0,
        };
        assert!(!r.trend_valid());
        assert_eq!(r.throughput_drift_pct_per_h(), None, "a 10s run must not report %/h");
        assert_eq!(r.latency_drift_pct_per_h(), None);
        assert_eq!(r.memory_growth_mb_per_h(), None);
    }

    #[test]
    fn one_sample_cannot_establish_a_trend() {
        let r = Report { samples: vec![sample(0.0, 100.0, 500.0, 1000)], restarts: 0 };
        assert_eq!(r.throughput_drift_pct_per_h(), None);
        assert_eq!(r.memory_growth_mb_per_h(), None);
    }

    #[test]
    fn host_memory_is_readable_on_linux() {
        #[cfg(target_os = "linux")]
        {
            assert!(host_mem_mb().unwrap_or(0.0) > 0.0);
            assert!(open_fds().unwrap_or(0) > 0);
        }
    }
}
