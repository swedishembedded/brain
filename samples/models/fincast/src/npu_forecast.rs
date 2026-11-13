// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `npu-forecast`: forecast a context series with the FinCast decoder and
//! sparse-MoE transformer core running on the Intel NPU through OpenVINO. The
//! host (the model's compute backend) does the patch-embed and frequency
//! assembly and the head rearrange and denormalisation; the exported ONNX core
//! (`emb` + `amask` to `qhead`) runs on the accelerator through the model's
//! pluggable-core seam. `verify` also runs the pure device path and reports the
//! largest difference.
//!
//! The ~1 B-parameter core's single-protobuf ONNX exceeds protobuf's 2 GB
//! read-from-buffer limit, so it is exported with an external-data sidecar into
//! a private scratch directory and compiled from the file. Needs OpenVINO and
//! an Intel NPU at run time, and says so when it has neither.
//!
//! Swedish Embedded AB implements financial forecasting on NPUs and other edge
//! accelerators for its clients. If your team needs expertise in exporting
//! mixture-of-experts transformers for OpenVINO then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::cell::RefCell;
use std::time::Instant;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use npu::openvino::{FincastSession, NpuConfig};
use serde_json::json;

/// Longest context or horizon one run may ask for.
const MAX_BARS: f64 = 100_000.0;
/// Shortest context the model forecasts from, points.
const MIN_CONTEXT: usize = 8;
/// The synthetic context's length when no series is given.
const SYNTHETIC_CONTEXT: usize = 512;

pub fn spec() -> ActionSpec {
    let mut spec = ActionSpec::new("npu-forecast", "forecast a context series with the FinCast transformer core on the Intel NPU through OpenVINO")
        .param(ParamSpec::new("horizon", ParamType::Int, "steps to forecast").min(1.0).max(MAX_BARS).default(json!(8)))
        .param(ParamSpec::new("freq", ParamType::Int, "frequency bucket: 0 daily, 1 weekly, 2 monthly").min(0.0).max(2.0).default(json!(0)))
        .param(ParamSpec::new("verify", ParamType::Bool, "also run the device path and report the largest difference from the NPU core").default(json!(false)))
        .input(BlobSpec::new("series", Media::Text, "the context: whitespace or comma separated numbers (default: a synthetic trend with a sinusoid)"))
        .output(BlobSpec::new("report", Media::Text, "what ran: the device, the core time, the mean and p10/p90 paths"))
        .output(BlobSpec::new("forecast", Media::Bytes, "the forecast as raw f32-LE, [horizon, 1 + levels] (column 0 the mean); meta {shape}"));
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

fn rounded(row: impl Iterator<Item = f32>) -> Vec<f32> {
    row.map(|v| (v * 100.0).round() / 100.0).collect()
}

/// Compile the core for a context of `s` patch tokens: exported with an
/// external-data sidecar into a private scratch directory that is removed
/// whether or not the compile worked.
fn compile_core(weights: &str, s: usize, cfg: &NpuConfig) -> Result<FincastSession, String> {
    let dir = capability::scratch::scratch_dir("fincast-npu").map_err(|e| format!("a scratch directory: {e}"))?;
    let onnx = dir.join("core.onnx");
    let compiled = npu::fincast_export::export_external(weights, s, npu::qwen_topology::Quant::F32, &onnx.to_string_lossy())
        .and_then(|()| FincastSession::load_path(&onnx.to_string_lossy(), cfg).map_err(|e| format!("compile fincast core: {e}")));
    let _ = std::fs::remove_dir_all(&dir);
    compiled
}

impl Action for NpuForecast {
    fn spec(&self) -> ActionSpec {
        spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let horizon = inv.get_i64("horizon").unwrap_or(8).max(1) as usize;
        let freq = inv.get_i64("freq").unwrap_or(0).max(0) as usize;
        let context: Vec<f32> = match inv.get_blob("series") {
            Some(b) => parse_series(std::str::from_utf8(&b.bytes).map_err(|_| "fincast npu-forecast: series is not UTF-8 text".to_string())?),
            None => (0..SYNTHETIC_CONTEXT).map(|i| 100.0 + 0.05 * i as f32 + (i as f32 * 0.1).sin() * 5.0).collect(),
        };
        if context.len() < MIN_CONTEXT {
            return Err(format!("fincast npu-forecast: context too short ({} points, need {MIN_CONTEXT})", context.len()));
        }
        let cfg = npu::run_options::config(inv)?;
        let model = fincast::model::Fincast::load(&self.weights).map_err(|e| format!("fincast npu-forecast: load {}: {e}", self.weights))?;
        let no = model.config().num_outputs();

        // A compile or infer failure surfaces as a panic out of the pluggable
        // core's closure; it is turned into this action's error rather than
        // unwinding the program.
        let device = RefCell::new(String::new());
        let core_ms = RefCell::new(0.0f64);
        let t0 = Instant::now();
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.forecast_full_with_core(&context, freq, horizon, |emb, amask| {
                let s = (amask.len() as f64).sqrt() as usize;
                let mut sess = compile_core(&self.weights, s, &cfg).expect("fincast core");
                *device.borrow_mut() = sess.device().to_string();
                let tc = Instant::now();
                let r = sess.run(emb, amask).expect("npu infer");
                *core_ms.borrow_mut() = tc.elapsed().as_secs_f64() * 1e3;
                r
            })
        }))
        .map_err(|_| "fincast npu-forecast: the NPU core failed (export, compile or infer error: see stderr)".to_string())?;
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut report = format!("fincast core on {} · horizon {horizon} · {} context pts · core {:.1} ms (total {total_ms:.1} ms)\n", device.borrow(), context.len(), core_ms.borrow());
        // `out` is [horizon, num_outputs] step-major: column 0 the mean, then the quantiles.
        report.push_str(&format!("  mean : {:?}\n", rounded((0..horizon).map(|t| out[t * no]))));
        if no >= 10 {
            report.push_str(&format!("  p10  : {:?}\n  p90  : {:?}\n", rounded((0..horizon).map(|t| out[t * no + 1])), rounded((0..horizon).map(|t| out[t * no + 9]))));
        }
        let mut result = Outcome::new().set("device", json!(*device.borrow())).set("core_ms", json!(*core_ms.borrow())).set("total_ms", json!(total_ms));
        if inv.get_bool("verify").unwrap_or(false) {
            let cpu = model.forecast_full(&context, freq, horizon);
            let maxdiff = out.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            report.push_str(&format!("  vs device core: max abs diff {maxdiff:.5} (rel {:.2e})\n", maxdiff / scale));
            result = result.set("max_abs_diff", json!(maxdiff));
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(result.blob("report", Blob::new(Media::Text, report.into_bytes())).blob("forecast", Blob::new(Media::Bytes, bytes).with_meta(json!({"shape": [horizon, no], "dtype": "f32le"}))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_series_is_whitespace_or_comma_separated_numbers_and_junk_is_skipped() {
        assert_eq!(parse_series("1, 2.5\n3  x 4"), [1.0, 2.5, 3.0, 4.0]);
    }

    #[test]
    fn a_context_shorter_than_the_model_takes_is_refused_before_any_load() {
        let action = NpuForecast::new("/nonexistent".into());
        let inv = Invocation::new().blob("series", Blob::new(Media::Text, b"1 2 3".to_vec()));
        let e = action.run(&inv, &mut |_| {}).unwrap_err();
        assert!(e.contains("context too short"), "{e}");
    }

    #[test]
    fn the_action_offers_the_frequency_bucket_and_the_npu_options() {
        let names: Vec<String> = spec().params.into_iter().map(|p| p.name).collect();
        for want in ["horizon", "freq", "verify", "ov_device", "allow_fallback"] {
            assert!(names.iter().any(|n| n == want), "{want} in {names:?}");
        }
    }
}
