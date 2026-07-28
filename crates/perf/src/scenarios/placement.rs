// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `placement` — heterogeneous CPU / GPU / Vulkan / NPU scheduling.
//!
//! brain runs the *same WGSL* on three backends plus a separate whole-graph NPU
//! path, so it can ask a question most engines cannot: **given a machine with
//! mixed devices, does the engine place work well?**
//!
//! The measure is against an oracle rather than in absolute terms, because
//! absolute throughput conflates "the placer is smart" with "the hardware is
//! fast":
//!
//! ```text
//! placement_efficiency = observed_goodput / oracle_goodput
//! ```
//!
//! where the oracle is the best single placement found by measuring each device
//! on its own. An efficiency near 1 on a heterogeneous box means the scheduler
//! is genuinely exploiting the mix; well under 1 means it is merely *capable* of
//! running operators on several backends, which is a much weaker property and
//! the one that is easy to mistake for the first.
//!
//! Efficiency above 1 is meaningful too: it means combining devices beat every
//! single device, i.e. the work really was distributed.

use serde_json::{json, Value};

use crate::stats::r3;

/// One device's measured standalone performance.
#[derive(Clone, Debug)]
pub struct DeviceResult {
    /// The `--device` spec that produced it (`gpu0`, `cpu`, `gpu0,gpu1`).
    pub spec: String,
    /// Resolved hardware label, e.g. `gpu[0] Tesla`.
    pub label: String,
    pub output_per_s: f64,
    pub goodput_per_s: f64,
    pub ttfa_p99_ms: Option<f64>,
    /// True when this "GPU" was actually a software rasteriser.
    pub software: bool,
}

/// The scenario's outcome.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Every single-device measurement.
    pub singles: Vec<DeviceResult>,
    /// Measurements where more than one device was schedulable.
    pub combined: Vec<DeviceResult>,
}

impl Report {
    /// The best single placement — the oracle.
    pub fn oracle(&self) -> Option<&DeviceResult> {
        self.singles.iter().fold(None, |acc, d| match acc {
            Some(a) if a.goodput_per_s >= d.goodput_per_s => Some(a),
            _ => Some(d),
        })
    }

    /// The best combined placement actually achieved.
    pub fn best_combined(&self) -> Option<&DeviceResult> {
        self.combined.iter().fold(None, |acc, d| match acc {
            Some(a) if a.goodput_per_s >= d.goodput_per_s => Some(a),
            _ => Some(d),
        })
    }

    /// `observed / oracle`. `None` unless both a combined run and a single-device
    /// oracle exist — with only one device there is nothing to place.
    pub fn efficiency(&self) -> Option<f64> {
        let o = self.oracle()?;
        let c = self.best_combined()?;
        (o.goodput_per_s > 0.0).then(|| c.goodput_per_s / o.goodput_per_s)
    }

    /// True when at least one measurement ran on a software rasteriser, which
    /// makes the whole comparison suspect.
    pub fn any_software(&self) -> bool {
        self.singles.iter().chain(self.combined.iter()).any(|d| d.software)
    }

    pub fn to_json(&self) -> Value {
        let dev = |d: &DeviceResult| {
            json!({
                "spec": d.spec,
                "label": d.label,
                "output_artifacts_per_s": r3(d.output_per_s),
                "goodput_per_s": r3(d.goodput_per_s),
                "ttfa_ms_p99": d.ttfa_p99_ms.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
                "adapter_is_software": d.software,
            })
        };
        json!({
            "singles": self.singles.iter().map(dev).collect::<Vec<_>>(),
            "combined": self.combined.iter().map(dev).collect::<Vec<_>>(),
            "oracle": self.oracle().map(dev).unwrap_or(Value::Null),
            "best_combined": self.best_combined().map(dev).unwrap_or(Value::Null),
            "placement_efficiency": self.efficiency().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "any_software_adapter": self.any_software(),
        })
    }
}

/// The device specs to measure on a machine with `gpus` GPUs. Singles first,
/// then the combinations worth trying.
pub fn specs_for(gpus: u32) -> (Vec<String>, Vec<String>) {
    let mut singles = vec!["cpu".to_string()];
    for i in 0..gpus {
        singles.push(format!("gpu{i}"));
    }
    let mut combined = Vec::new();
    if gpus >= 2 {
        combined.push((0..gpus).map(|i| format!("gpu{i}")).collect::<Vec<_>>().join(","));
    }
    if gpus >= 1 {
        combined.push("gpu,cpu".to_string());
    }
    (singles, combined)
}

pub fn render(r: &Report) -> String {
    let mut s = format!("\n{:<14} {:<22} {:>12} {:>12} {:>11}\n", "spec", "hardware", "out/s", "goodput/s", "ttfa p99");
    s.push_str(&format!("{:-<76}\n", ""));
    for d in r.singles.iter().chain(r.combined.iter()) {
        s.push_str(&format!(
            "{:<14} {:<22} {:>12.1} {:>12.1} {:>11}\n",
            d.spec.chars().take(14).collect::<String>(),
            d.label.chars().take(22).collect::<String>(),
            d.output_per_s,
            d.goodput_per_s,
            d.ttfa_p99_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()),
        ));
    }
    match (r.efficiency(), r.oracle()) {
        (Some(e), Some(o)) => {
            s.push_str(&format!(
                "\nplacement efficiency {:.2} (best combined / best single, oracle = {})\n",
                e, o.spec
            ));
            if e < 0.95 {
                s.push_str(
                    "  ! combining devices did not beat the best single one — the engine can run\n    \
                     on several backends but is not exploiting them together\n",
                );
            }
        }
        _ => s.push_str("\nplacement efficiency — (needs both a single-device oracle and a combined run)\n"),
    }
    if r.any_software() {
        s.push_str("  ! a measurement used a software rasteriser; not a hardware comparison\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(spec: &str, goodput: f64) -> DeviceResult {
        DeviceResult {
            spec: spec.into(),
            label: spec.into(),
            output_per_s: goodput,
            goodput_per_s: goodput,
            ttfa_p99_ms: Some(10.0),
            software: false,
        }
    }

    #[test]
    fn oracle_is_the_best_single_device() {
        let r = Report { singles: vec![d("cpu", 30.0), d("gpu0", 100.0)], combined: vec![] };
        assert_eq!(r.oracle().unwrap().spec, "gpu0");
    }

    #[test]
    fn efficiency_needs_both_a_single_and_a_combined_run() {
        let only_singles = Report { singles: vec![d("cpu", 30.0)], combined: vec![] };
        assert_eq!(only_singles.efficiency(), None, "one device means nothing to place");
        let full = Report { singles: vec![d("gpu0", 100.0)], combined: vec![d("gpu0,gpu1", 180.0)] };
        assert_eq!(full.efficiency(), Some(1.8));
    }

    #[test]
    fn efficiency_below_one_means_combining_did_not_pay() {
        let r = Report { singles: vec![d("gpu0", 100.0)], combined: vec![d("gpu,cpu", 90.0)] };
        assert_eq!(r.efficiency(), Some(0.9));
        assert!(render(&r).contains("not exploiting them together"));
    }

    #[test]
    fn software_adapters_are_surfaced_not_hidden() {
        let mut sw = d("gpu0", 5.0);
        sw.software = true;
        let r = Report { singles: vec![sw], combined: vec![] };
        assert!(r.any_software());
        assert!(render(&r).contains("software rasteriser"));
    }

    #[test]
    fn specs_cover_each_card_alone_and_together() {
        let (singles, combined) = specs_for(2);
        assert!(singles.contains(&"cpu".to_string()));
        assert!(singles.contains(&"gpu0".to_string()) && singles.contains(&"gpu1".to_string()));
        assert!(combined.contains(&"gpu0,gpu1".to_string()));
        assert!(combined.contains(&"gpu,cpu".to_string()));
    }

    #[test]
    fn a_gpuless_machine_still_measures_the_cpu() {
        let (singles, combined) = specs_for(0);
        assert_eq!(singles, vec!["cpu".to_string()]);
        assert!(combined.is_empty(), "nothing to combine without a GPU");
    }

    #[test]
    fn json_carries_the_efficiency_and_the_oracle() {
        let r = Report { singles: vec![d("gpu0", 100.0)], combined: vec![d("gpu,cpu", 150.0)] };
        let j = r.to_json();
        assert_eq!(j["placement_efficiency"], 1.5);
        assert_eq!(j["oracle"]["spec"], "gpu0");
    }
}
