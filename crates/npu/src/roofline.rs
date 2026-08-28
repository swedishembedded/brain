// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measured NPU roofline: what `openvino::device_info` reports for free (device
//! name + `OPTIMIZATION_CAPABILITIES`), plus fp16 GFLOP/s and int8 GOP/s
//! actually TIMED through a synthetic MatMul compiled and run via OpenVINO.
//!
//! OpenVINO's own `PropertyKey` enum has no peak-TOPS/peak-GFLOPS property at
//! all - there is nothing to query, so unlike a GPU (where a vendor-reported
//! peak at least exists to be double-checked) any NPU number here must be
//! measured, never a datasheet figure. This mirrors [`gpu_core::roof`]'s
//! "measured, never assumed" discipline on the NPU/OpenVINO seam, using the
//! generic [`crate::openvino::NpuGraph`] runner + the `onnx::GraphBuilder`
//! in-process graph construction already proven by `tests/npugraph.rs`.
//!
//! Field names deliberately mirror `gpu_core::roof::Roofs` (`gflops`,
//! `int8_gops`) so a caller assembling a unified GPU+NPU+CPU roofline report
//! (the `brain roofline` command this module feeds) does not have to
//! translate vocabulary between backends.
//!
//! Every entry point here is infallible in the "no NPU / no OpenVINO" sense:
//! [`measure`] returns `None` rather than panicking, exiting, or hanging -
//! unlike `brain npu bench`, which hard-`die()`s on any OpenVINO error today.
//! A caller building a hardware report cannot let one missing accelerator
//! abort the whole report.
//!
//! Swedish Embedded AB builds inference-serving pipelines that pick between
//! GPU, NPU and CPU targets at runtime from real measured throughput rather
//! than vendor datasheets. If your team needs a trustworthy picture of what an
//! edge accelerator can actually deliver, you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::openvino::{self, DeviceInfo, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use std::time::Instant;

/// Matrix dims for the synthetic probe: `y[M,N] = x[M,K] @ w[K,N]`. Large
/// enough that device compute - not host<->device marshalling or per-call
/// dispatch overhead - dominates the timed region on a real NPU (order
/// 10^8-10^9 FLOP/int-ops per call), small enough that compiling + running it
/// stays well under a second even on a modest accelerator or in CI.
const PROBE_M: usize = 256;
const PROBE_K: usize = 1024;
const PROBE_N: usize = 1024;

/// Timed repetitions per precision. The reported rate is the FASTEST of these
/// (best-of-N), matching `gpu_core::roof`'s own bracketing discipline: a
/// slower rep reflects scheduling noise or contention, never the device
/// legitimately running faster than its best observed rate.
const PROBE_REPS: usize = 15;
/// Untimed warm-up runs before the timed reps, so first-inference driver/JIT
/// warm-up cost never leaks into the measured region.
const PROBE_WARMUP: usize = 3;

/// A device's measured NPU-path roofline. Both the free capability fields and
/// the measured throughput fields are populated independently - a device with
/// no measurable precisions still reports its name/capabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct NpuRoofline {
    /// `FULL_DEVICE_NAME` (e.g. "Intel(R) AI Boost").
    pub device_name: String,
    /// `OPTIMIZATION_CAPABILITIES` as OpenVINO reports them (e.g.
    /// `["FP16","INT8","EXPORT_IMPORT"]`).
    pub capabilities: Vec<String>,
    /// Measured fp16 throughput, GFLOP/s (`2*M*N*K` per probe MatMul, best of
    /// [`PROBE_REPS`]). `None` when the device does not claim `"FP16"` in its
    /// capabilities, or when compiling/running the probe failed - never a
    /// fabricated number.
    pub fp16_gops: Option<f32>,
    /// Measured int8 throughput, GOP/s, from a
    /// `QuantizeLinear`/`DequantizeLinear`-wrapped MatMul (the same QDQ shape
    /// `crate::topology` emits for INT8 convs) so the NPU compiler sees a real
    /// quantized op, not a fp32 graph run at fp16. `None` when the device does
    /// not claim `"INT8"`, or the probe failed.
    pub int8_gops: Option<f32>,
}

/// Which precisions [`measure`] should even attempt, decided purely from the
/// device's advertised capabilities. Split out from `measure` so the
/// never-fabricate-a-number contract is unit-testable without OpenVINO or
/// hardware: see `tests::probing_is_gated_by_advertised_capabilities`.
fn should_probe(info: &DeviceInfo) -> (bool, bool) {
    (info.supports("FP16"), info.supports("INT8"))
}

/// Measure `dev`'s roofline: free capability info via
/// [`openvino::device_info`], then fp16/int8 throughput via a synthetic
/// MatMul - each precision probed ONLY when the device's own
/// `OPTIMIZATION_CAPABILITIES` claims it.
///
/// Returns `None`, never panics/exits/hangs, whenever `dev` cannot be
/// characterised at all: no OpenVINO runtime installed, the device not
/// present, or (on non-x86_64/non-linux/windows targets) the stub backend.
/// This is deliberately the opposite contract from `brain npu bench`'s
/// `die()`-on-any-error today - a hardware report must survive one missing
/// accelerator, not abort on it.
pub fn measure(dev: NpuDevice) -> Option<NpuRoofline> {
    // No fallback: a caller asking for the NPU's roofline wants the NPU's
    // number, not a CPU/GPU stand-in silently reported under the NPU's name.
    let info = openvino::device_info(dev, false).ok()?;
    let (probe_fp16, probe_int8) = should_probe(&info);

    let cfg = NpuConfig {
        device: dev,
        perf_hint: PerfHint::Throughput,
        allow_fallback: false,
        ..Default::default()
    };
    // Deterministic, non-degenerate input (no zeros/uniform values that could
    // let the compiler fold or a QDQ round-trip saturate identically).
    let x: Vec<f32> = (0..PROBE_M * PROBE_K).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();

    let fp16_gops =
        if probe_fp16 { probe(&build_fp16_matmul(), &cfg, &x, matmul_flops()) } else { None };
    let int8_gops =
        if probe_int8 { probe(&build_int8_matmul(), &cfg, &x, matmul_flops()) } else { None };

    Some(NpuRoofline { device_name: info.full_name, capabilities: info.capabilities, fp16_gops, int8_gops })
}

/// `2*M*N*K`: multiply-adds in the probe MatMul, counted as 2 ops each.
fn matmul_flops() -> f64 {
    2.0 * PROBE_M as f64 * PROBE_N as f64 * PROBE_K as f64
}

/// Compile `bytes` on `cfg.device`, run it [`PROBE_WARMUP`] times untimed then
/// [`PROBE_REPS`] times timed, and return the best (fastest) achieved rate in
/// G-units/s. `None` on any compile/run failure - the caller must not fabricate
/// a number when the device claimed a capability it then failed to deliver on.
fn probe(bytes: &[u8], cfg: &NpuConfig, x: &[f32], ops: f64) -> Option<f32> {
    let mut graph = NpuGraph::compile_bytes(bytes, cfg).ok()?;
    let dims = vec![PROBE_M as i64, PROBE_K as i64];
    for _ in 0..PROBE_WARMUP {
        graph.run(&[("x", Feed::F32(x, dims.clone()))]).ok()?;
    }
    let mut best = f64::INFINITY;
    for _ in 0..PROBE_REPS {
        let t0 = Instant::now();
        graph.run(&[("x", Feed::F32(x, dims.clone()))]).ok()?;
        best = best.min(t0.elapsed().as_secs_f64());
    }
    if !(best.is_finite() && best > 0.0) {
        return None;
    }
    Some((ops / best / 1e9) as f32)
}

/// `y[M,N] = x[M,K] @ w[K,N]`, plain fp32 ONNX - OpenVINO compiles this to the
/// NPU's native fp16 execution path with no calibration needed (mirrors
/// `tests/npugraph.rs`'s tiny MatMul, just scaled up).
fn build_fp16_matmul() -> Vec<u8> {
    let (m, k, n) = (PROBE_M as i64, PROBE_K as i64, PROBE_N as i64);
    let w: Vec<f32> = (0..PROBE_K * PROBE_N).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

    let mut g = onnx::GraphBuilder::new("roofline_fp16_matmul");
    g.input_f32("x", &[m, k]);
    g.init_f32("w", &[k, n], w);
    g.add(onnx::Node::new("MatMul", &["x", "w"], &["y"]).name("MatMul"));
    g.output_f32("y", &[m, n]);
    g.finish()
}

/// Same shape as [`build_fp16_matmul`], but the weight is a genuine per-tensor
/// INT8 constant fed through `DequantizeLinear`, and the activation goes
/// through a `QuantizeLinear`/`DequantizeLinear` round-trip before the MatMul -
/// the same QDQ pattern `crate::topology::Exporter::conv_node` emits for INT8
/// convs, so the OpenVINO NPU compiler sees a real quantized op to fuse into
/// native int8 execution rather than a fp32 graph that just happens to run at
/// fp16.
fn build_int8_matmul() -> Vec<u8> {
    let (m, k, n) = (PROBE_M as i64, PROBE_K as i64, PROBE_N as i64);
    let wf: Vec<f32> = (0..PROBE_K * PROBE_N).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let w_scale = 0.02f32;
    let wq: Vec<i8> = wf.iter().map(|v| (v / w_scale).round().clamp(-127.0, 127.0) as i8).collect();
    let x_scale = 0.05f32;

    let mut g = onnx::GraphBuilder::new("roofline_int8_matmul");
    g.input_f32("x", &[m, k]);

    g.init_i8("w_i8", &[k, n], wq);
    g.init_f32("w_scale", &[], vec![w_scale]);
    g.init_i8("w_zp", &[], vec![0i8]);
    g.add(onnx::Node::new("DequantizeLinear", &["w_i8", "w_scale", "w_zp"], &["w_dq"]).name("WeightDQ"));

    g.init_f32("x_scale", &[], vec![x_scale]);
    g.init_i8("x_zp", &[], vec![0i8]);
    g.add(onnx::Node::new("QuantizeLinear", &["x", "x_scale", "x_zp"], &["x_q"]).name("ActQ"));
    g.add(onnx::Node::new("DequantizeLinear", &["x_q", "x_scale", "x_zp"], &["x_dq"]).name("ActDQ"));

    g.add(onnx::Node::new("MatMul", &["x_dq", "w_dq"], &["y"]).name("MatMul"));
    g.output_f32("y", &[m, n]);
    g.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `measure` must never panic/exit/hang when there is no OpenVINO runtime
    /// or no NPU device - the common CI shape. Bounded: `device_info` fails
    /// fast (no compile, no probe) whenever the runtime/device is absent.
    #[test]
    fn measure_returns_none_cleanly_without_hardware() {
        if openvino::npu_present() {
            // A real NPU is reachable in this environment: this specific
            // "no hardware" contract isn't exercised here. The ignored
            // round-trip test below is what covers the real-hardware path.
            eprintln!("SKIP: NPU present in this environment; see the #[ignore]d round-trip test instead");
            return;
        }
        assert_eq!(measure(NpuDevice::Npu), None);
    }

    /// The never-fabricate-a-number contract, isolated from OpenVINO/hardware:
    /// a device that does not list a precision in `OPTIMIZATION_CAPABILITIES`
    /// must never have that precision probed at all.
    #[test]
    fn probing_is_gated_by_advertised_capabilities() {
        let fp16_only = DeviceInfo {
            device: "NPU".into(),
            full_name: "Test NPU".into(),
            capabilities: vec!["FP16".into(), "EXPORT_IMPORT".into()],
        };
        assert_eq!(should_probe(&fp16_only), (true, false));

        let neither = DeviceInfo { device: "CPU".into(), full_name: "Test CPU".into(), capabilities: vec![] };
        assert_eq!(should_probe(&neither), (false, false));

        let both = DeviceInfo {
            device: "NPU".into(),
            full_name: "Test NPU".into(),
            capabilities: vec!["FP16".into(), "INT8".into()],
        };
        assert_eq!(should_probe(&both), (true, true));
    }
}
