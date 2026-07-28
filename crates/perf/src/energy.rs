// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Energy accounting — joules per artifact, and cost at SLO.
//!
//! Raw artifacts/s rewards configurations that burn far more power or need far
//! more expensive hardware. For edge deployment the deciding number is usually
//! *energy per useful request*, and that is the number tokens/s misleads on
//! hardest.
//!
//! Sampling is deliberately external (`nvidia-smi --query-gpu=power.draw`) and
//! optional: the engine has no power counters, brain must not gain a hard
//! dependency on a vendor tool, and a machine with no readable power meter must
//! report `null` rather than a fabricated zero. A sampler that cannot read power
//! is not an error — it simply yields `None`, and the `resources` block stays
//! `null`, which is exactly the "not measured" the schema requires.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A background power sampler over one or more GPUs.
pub struct PowerSampler {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<(Instant, f64)>>>,
    handle: Option<std::thread::JoinHandle<()>>,
    started: Instant,
}

/// What a completed sampling window measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyReport {
    /// Mean total board power over the window, watts.
    pub mean_watts: f64,
    /// Integrated energy over the window, joules.
    pub joules: f64,
    pub seconds: f64,
    pub samples: usize,
}

impl EnergyReport {
    pub fn j_per_artifact(&self, artifacts: usize) -> Option<f64> {
        (artifacts > 0).then(|| self.joules / artifacts as f64)
    }
}

impl PowerSampler {
    /// Start sampling the given GPU indices, or return `None` when power cannot
    /// be read (no NVML/`nvidia-smi`, no GPU, a CPU-only run).
    pub fn start(gpus: &[u32], interval: Duration) -> Option<PowerSampler> {
        if gpus.is_empty() || read_power(gpus).is_none() {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let (s2, samp2, ids) = (stop.clone(), samples.clone(), gpus.to_vec());
        let handle = std::thread::spawn(move || {
            while !s2.load(Ordering::Relaxed) {
                if let Some(w) = read_power(&ids) {
                    if let Ok(mut v) = samp2.lock() {
                        v.push((Instant::now(), w));
                    }
                }
                std::thread::sleep(interval);
            }
        });
        Some(PowerSampler { stop, samples, handle: Some(handle), started: Instant::now() })
    }

    /// Stop sampling and integrate. Trapezoidal over the samples, which is the
    /// honest reading of a coarse external meter — `nvidia-smi` reports an
    /// instantaneous board draw, not an accumulator.
    pub fn finish(mut self) -> Option<EnergyReport> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let v = self.samples.lock().ok()?.clone();
        if v.len() < 2 {
            return None;
        }
        let mut joules = 0.0;
        for w in v.windows(2) {
            let dt = w[1].0.saturating_duration_since(w[0].0).as_secs_f64();
            joules += 0.5 * (w[0].1 + w[1].1) * dt;
        }
        let seconds = self.started.elapsed().as_secs_f64();
        let mean_watts = v.iter().map(|(_, w)| *w).sum::<f64>() / v.len() as f64;
        Some(EnergyReport { mean_watts, joules, seconds, samples: v.len() })
    }
}

/// Total instantaneous board power across `gpus`, watts.
fn read_power(gpus: &[u32]) -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,power.draw", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut total = 0.0;
    let mut seen = 0;
    for line in text.lines() {
        let mut it = line.split(',');
        let idx: u32 = it.next()?.trim().parse().ok()?;
        let w: f64 = match it.next()?.trim().parse() {
            Ok(v) => v,
            // "[N/A]" on cards without a power sensor.
            Err(_) => continue,
        };
        if gpus.contains(&idx) {
            total += w;
            seen += 1;
        }
    }
    (seen > 0).then_some(total)
}

/// The `resources` energy fields for an artifact, or nulls when unmeasured.
pub fn to_json(report: Option<&EnergyReport>, output_artifacts: usize) -> (serde_json::Value, serde_json::Value, serde_json::Value) {
    match report {
        Some(r) => (
            serde_json::json!(crate::stats::r3(r.joules)),
            serde_json::json!(crate::stats::r3(r.mean_watts)),
            r.j_per_artifact(output_artifacts)
                .map(|v| serde_json::json!(crate::stats::r3(v)))
                .unwrap_or(serde_json::Value::Null),
        ),
        None => (serde_json::Value::Null, serde_json::Value::Null, serde_json::Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_power_meter_yields_none_not_zero() {
        // No GPUs selected => nothing to sample, and the caller must be able to
        // tell that apart from "measured 0 J".
        assert!(PowerSampler::start(&[], Duration::from_millis(10)).is_none());
        let (j, w, per) = to_json(None, 100);
        assert!(j.is_null() && w.is_null() && per.is_null());
    }

    #[test]
    fn energy_per_artifact_needs_artifacts() {
        let r = EnergyReport { mean_watts: 100.0, joules: 50.0, seconds: 0.5, samples: 10 };
        assert_eq!(r.j_per_artifact(0), None);
        assert_eq!(r.j_per_artifact(100), Some(0.5));
    }

    #[test]
    fn json_reports_measured_energy() {
        let r = EnergyReport { mean_watts: 120.0, joules: 60.0, seconds: 0.5, samples: 8 };
        let (j, w, per) = to_json(Some(&r), 120);
        assert_eq!(j, 60.0);
        assert_eq!(w, 120.0);
        assert_eq!(per, 0.5);
    }
}
