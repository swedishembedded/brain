// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `frontend` — host-side saturation.
//!
//! The accelerator is not always the bottleneck. Tokenisation, chat-template
//! rendering, JSONL framing, image decode, audio resampling, detokenisation and
//! stream flushing all run on the host, and any one of them can cap the engine
//! well below what the device could do. End-to-end latency tells you something
//! is slow; it does not tell you it was the tokenizer.
//!
//! The deliverable is **host cores required per saturated device**. If serving
//! one GPU at full rate takes six cores of JSON and tokenisation, that is a
//! capacity-planning fact that a tokens/s number hides completely.

use serde_json::{json, Value};

use crate::stats::{r3, Dist};

/// One host-side pipeline stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Tokenise,
    ChatTemplate,
    JsonlEncode,
    JsonlDecode,
    Detokenise,
    ImageDecode,
    AudioResample,
    StreamFlush,
}

impl Stage {
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Tokenise => "tokenise",
            Stage::ChatTemplate => "chat_template",
            Stage::JsonlEncode => "jsonl_encode",
            Stage::JsonlDecode => "jsonl_decode",
            Stage::Detokenise => "detokenise",
            Stage::ImageDecode => "image_decode",
            Stage::AudioResample => "audio_resample",
            Stage::StreamFlush => "stream_flush",
        }
    }
}

/// Per-stage timing over many iterations.
#[derive(Clone, Debug)]
pub struct StageResult {
    pub stage: &'static str,
    pub per_op_us: Dist,
    pub ops: usize,
    /// Bytes or tokens processed, for a rate.
    pub units: usize,
}

impl StageResult {
    pub fn new(stage: Stage) -> StageResult {
        StageResult { stage: stage.name(), per_op_us: Dist::new(), ops: 0, units: 0 }
    }
    pub fn record(&mut self, micros: f64, units: usize) {
        self.per_op_us.push(micros);
        self.ops += 1;
        self.units += units;
    }
    /// Mean microseconds of host CPU per unit processed.
    pub fn us_per_unit(&self) -> Option<f64> {
        let total: f64 = self.per_op_us.mean()? * self.ops as f64;
        (self.units > 0).then(|| total / self.units as f64)
    }
}

/// The scenario outcome.
#[derive(Debug, Default)]
pub struct Report {
    pub stages: Vec<StageResult>,
    /// Device artifact rate the host must keep up with.
    pub device_artifacts_per_s: Option<f64>,
}

impl Report {
    /// Host CPU-seconds needed per second of device work, i.e. **cores per
    /// saturated device**. Above 1.0 means a single core cannot feed the device.
    pub fn cores_per_device(&self) -> Option<f64> {
        let rate = self.device_artifacts_per_s?;
        if rate <= 0.0 {
            return None;
        }
        // Sum of per-artifact host cost across every stage, in seconds.
        let per_artifact_s: f64 = self.stages.iter().filter_map(|s| s.us_per_unit()).sum::<f64>() / 1e6;
        Some(per_artifact_s * rate)
    }

    /// The stage costing the most host time per unit — where to look first.
    pub fn bottleneck(&self) -> Option<&StageResult> {
        self.stages.iter().filter(|s| s.us_per_unit().is_some()).fold(None, |acc, s| match acc {
            Some(a) if StageResult::us_per_unit(a) >= s.us_per_unit() => Some(a),
            _ => Some(s),
        })
    }

    pub fn to_json(&mut self) -> Value {
        let cores = self.cores_per_device();
        let bottleneck = self.bottleneck().map(|s| s.stage.to_string());
        let stages: Vec<Value> = self
            .stages
            .iter_mut()
            .map(|s| {
                let per_unit = s.us_per_unit();
                json!({
                    "stage": s.stage,
                    "ops": s.ops,
                    "units": s.units,
                    "us_per_op": s.per_op_us.to_json(),
                    "us_per_unit": per_unit.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
                })
            })
            .collect();
        json!({
            "stages": stages,
            "device_artifacts_per_s": self.device_artifacts_per_s.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "host_cores_per_saturated_device": cores.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "bottleneck_stage": bottleneck.map(Value::from).unwrap_or(Value::Null),
        })
    }
}

pub fn render(r: &mut Report) -> String {
    let cores = r.cores_per_device();
    let bottleneck = r.bottleneck().map(|s| s.stage.to_string());
    let mut s = format!("\n{:<16} {:>10} {:>14} {:>14}\n", "stage", "ops", "us/op p50", "us/unit");
    s.push_str(&format!("{:-<58}\n", ""));
    for st in r.stages.iter_mut() {
        let p50 = st.per_op_us.percentile(0.50).map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
        let pu = st.us_per_unit().map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
        s.push_str(&format!("{:<16} {:>10} {:>14} {:>14}\n", st.stage, st.ops, p50, pu));
    }
    match cores {
        Some(c) => {
            s.push_str(&format!("\nhost cores per saturated device: {c:.2}\n"));
            if c > 1.0 {
                s.push_str("  ! one core cannot feed the device — the frontend is the ceiling\n");
            }
        }
        None => s.push_str("\nhost cores per saturated device — (needs a device rate to compare against)\n"),
    }
    if let Some(b) = bottleneck {
        s.push_str(&format!("bottleneck stage: {b}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(s: Stage, per_op_us: f64, ops: usize, units_per_op: usize) -> StageResult {
        let mut r = StageResult::new(s);
        for _ in 0..ops {
            r.record(per_op_us, units_per_op);
        }
        r
    }

    #[test]
    fn per_unit_cost_divides_by_units_not_ops() {
        // 10 ops x 100us, each covering 10 units => 10us per unit.
        let s = stage(Stage::Tokenise, 100.0, 10, 10);
        assert_eq!(s.us_per_unit(), Some(10.0));
    }

    #[test]
    fn a_stage_with_no_units_reports_none() {
        let s = StageResult::new(Stage::JsonlEncode);
        assert_eq!(s.us_per_unit(), None, "nothing measured must not read as free");
    }

    #[test]
    fn cores_per_device_scales_with_the_device_rate() {
        let mut r = Report {
            // 1000us = 1ms of host work per artifact.
            stages: vec![stage(Stage::Tokenise, 1000.0, 10, 1)],
            device_artifacts_per_s: Some(1000.0),
        };
        // 1ms host per artifact x 1000 artifacts/s = 1.0 core.
        assert_eq!(r.cores_per_device(), Some(1.0));
        r.device_artifacts_per_s = Some(4000.0);
        assert_eq!(r.cores_per_device(), Some(4.0));
        assert!(render(&mut r).contains("cannot feed the device"));
    }

    #[test]
    fn without_a_device_rate_there_is_nothing_to_compare_against() {
        let mut r = Report { stages: vec![stage(Stage::Tokenise, 10.0, 5, 1)], device_artifacts_per_s: None };
        assert_eq!(r.cores_per_device(), None);
        assert!(r.to_json()["host_cores_per_saturated_device"].is_null());
    }

    #[test]
    fn bottleneck_is_the_costliest_stage_per_unit() {
        let r = Report {
            stages: vec![
                stage(Stage::JsonlEncode, 10.0, 4, 1),
                stage(Stage::Tokenise, 900.0, 4, 1),
                stage(Stage::Detokenise, 50.0, 4, 1),
            ],
            device_artifacts_per_s: Some(10.0),
        };
        assert_eq!(r.bottleneck().unwrap().stage, "tokenise");
    }

    #[test]
    fn stage_names_are_stable_for_the_artifact() {
        assert_eq!(Stage::ChatTemplate.name(), "chat_template");
        assert_eq!(Stage::StreamFlush.name(), "stream_flush");
    }
}
