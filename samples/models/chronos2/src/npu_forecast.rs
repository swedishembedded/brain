// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `npu-forecast`: forecast a context series with the Chronos-2 transformer core
//! running on the Intel NPU through OpenVINO. The host (the model's compute
//! backend) does the scaler, patch, embed and REG assembly and the head
//! rearrange and denormalisation; the exported ONNX core (`emb` + `kmask` to
//! `qhead`) runs on the accelerator through the model's pluggable-core seam.
//! `verify` also runs the pure device path and reports the largest difference.
//!
//! The graph is compiled for the exact context length the series needs. Needs
//! OpenVINO and an Intel NPU at run time, and says so when it has neither.
//!
//! Swedish Embedded AB implements probabilistic forecasting on NPUs and other
//! edge accelerators for its clients. If your team needs expertise in
//! exporting transformer cores for OpenVINO then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::cell::RefCell;
use std::time::Instant;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use npu::openvino::Chronos2Session;
use serde_json::json;

/// Longest context or horizon one run may ask for.
const MAX_BARS: f64 = 100_000.0;
/// Shortest context the model forecasts from, points.
const MIN_CONTEXT: usize = 8;

pub fn spec() -> ActionSpec {
    let mut spec = ActionSpec::new("npu-forecast", "forecast a context series with the Chronos-2 transformer core on the Intel NPU through OpenVINO")
        .param(ParamSpec::new("horizon", ParamType::Int, "steps to forecast").min(1.0).max(MAX_BARS).default(json!(8)))
        .param(ParamSpec::new("context_len", ParamType::Int, "length of the synthetic sinusoid used when no series is given").min(1.0).max(MAX_BARS).default(json!(128)))
        .param(ParamSpec::new("verify", ParamType::Bool, "also run the device path and report the largest difference from the NPU core").default(json!(false)))
        .input(BlobSpec::new("series", Media::Text, "the context: whitespace or comma separated numbers (default: a synthetic sinusoid)"))
        .output(BlobSpec::new("report", Media::Text, "what ran: the device, the core time, the median and p10/p90 paths"))
        .output(BlobSpec::new("forecast", Media::Bytes, "the quantile forecast as raw f32-LE, quantile-major; meta {shape: [levels, horizon]}"));
    spec.params.extend(npu::run_options::params());
    spec
}

pub struct NpuForecast {
    weights: String,
}

impl NpuForecast {
    pub fn new(weights: String) -> NpuForecast {
        NpuForecast { weights }
    }
}

/// The numbers of a whitespace or comma separated text, unparseable tokens skipped.
fn parse_series(text: &str) -> Vec<f32> {
    text.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect()
}

fn rounded(row: &[f32]) -> Vec<f32> {
    row.iter().map(|v| (v * 100.0).round() / 100.0).collect()
}

impl Action for NpuForecast {
    fn spec(&self) -> ActionSpec {
        spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let horizon = inv.get_i64("horizon").unwrap_or(8).max(1) as usize;
        let context: Vec<f32> = match inv.get_blob("series") {
            Some(b) => parse_series(std::str::from_utf8(&b.bytes).map_err(|_| "chronos2 npu-forecast: series is not UTF-8 text".to_string())?),
            None => (0..inv.get_i64("context_len").unwrap_or(128).max(1) as usize).map(|i| 100.0 + (i as f32 * 0.1).sin() * 5.0).collect(),
        };
        if context.len() < MIN_CONTEXT {
            return Err(format!("chronos2 npu-forecast: context too short ({} points, need {MIN_CONTEXT})", context.len()));
        }
        let cfg = npu::run_options::config(inv)?;
        let model = chronos2::model::Chronos2::load(&self.weights).map_err(|e| format!("chronos2 npu-forecast: load {}: {e}", self.weights))?;
        let (d, q) = (model.config().d_model, model.config().num_quantiles);

        // The pluggable core compiles the ONNX for the exact (S, n_out) this
        // context needs, then infers. A compile or infer failure surfaces as a
        // panic out of the closure; it is turned into this action's error
        // rather than unwinding the program.
        let device = RefCell::new(String::new());
        let core_ms = RefCell::new(0.0f64);
        let t0 = Instant::now();
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.forecast_quantiles_with_core(&context, horizon, |emb, mask, n_out| {
                let s = emb.len() / d;
                let bytes = npu::chronos2_export::export_onnx(&self.weights, s, n_out, npu::qwen_topology::Quant::F32).expect("export chronos2 core onnx");
                let mut sess = Chronos2Session::load_bytes(&bytes, &cfg).expect("compile chronos2 core");
                *device.borrow_mut() = sess.device().to_string();
                let tc = Instant::now();
                let r = sess.run(emb, mask).expect("npu infer");
                *core_ms.borrow_mut() = tc.elapsed().as_secs_f64() * 1e3;
                r
            })
        }))
        .map_err(|_| "chronos2 npu-forecast: the NPU core failed (export, compile or infer error: see stderr)".to_string())?;
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut report = format!("chronos2 core on {} · horizon {horizon} · {} context pts · core {:.1} ms (total {total_ms:.1} ms)\n", device.borrow(), context.len(), core_ms.borrow());
        let row = |qi: usize| &out[qi * horizon..qi * horizon + horizon];
        report.push_str(&format!("  median : {:?}\n", rounded(row(q / 2))));
        if q >= 3 {
            report.push_str(&format!("  p10    : {:?}\n  p90    : {:?}\n", rounded(row(0)), rounded(row(q - 1))));
        }
        let mut result = Outcome::new().set("device", json!(*device.borrow())).set("core_ms", json!(*core_ms.borrow())).set("total_ms", json!(total_ms));
        if inv.get_bool("verify").unwrap_or(false) {
            let cpu = model.forecast_quantiles(&context, horizon);
            let maxdiff = out.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            report.push_str(&format!("  vs device core: max abs diff {maxdiff:.5} (rel {:.2e})\n", maxdiff / scale));
            result = result.set("max_abs_diff", json!(maxdiff));
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(result.blob("report", Blob::new(Media::Text, report.into_bytes())).blob("forecast", Blob::new(Media::Bytes, bytes).with_meta(json!({"shape": [q, horizon], "dtype": "f32le"}))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_series_is_whitespace_or_comma_separated_numbers_and_junk_is_skipped() {
        assert_eq!(parse_series("1, 2.5\n3  x 4"), [1.0, 2.5, 3.0, 4.0]);
        assert!(parse_series("").is_empty());
    }

    #[test]
    fn a_context_shorter_than_the_model_takes_is_refused_before_any_load() {
        let action = NpuForecast::new("/nonexistent".into());
        let inv = Invocation::new().blob("series", Blob::new(Media::Text, b"1 2 3".to_vec()));
        let e = action.run(&inv, &mut |_| {}).unwrap_err();
        assert!(e.contains("context too short"), "{e}");
    }

    #[test]
    fn the_action_offers_the_npu_options() {
        let names: Vec<String> = spec().params.into_iter().map(|p| p.name).collect();
        for want in ["horizon", "context_len", "verify", "ov_device", "allow_fallback"] {
            assert!(names.iter().any(|n| n == want), "{want} in {names:?}");
        }
    }
}
