// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! OpenVINO NPU runtime seam.
//!
//! OpenVINO is a DEFAULT dependency (not behind a cargo feature): on x86_64
//! linux/windows the real implementation is always compiled in, so `--device
//! npu` works out of the box with no special rebuild. The `openvino` crate's
//! `runtime-linking` defers loading the OpenVINO shared library to run time, so
//! the build stays green on machines without OpenVINO installed — a missing
//! runtime surfaces as [`NpuError::RuntimeNotFound`] only when a session is
//! actually opened. On other targets (wasm/aarch64/macos) a stub with the same
//! API reports [`NpuError::Unsupported`].

use std::fmt;

/// Errors from the NPU runtime path.
#[derive(Debug)]
pub enum NpuError {
    /// The NPU path is not supported on this target (non x86_64-linux/windows).
    Unsupported(String),
    /// OpenVINO runtime could not be loaded (not installed / not on the loader
    /// path). Install OpenVINO and source `setupvars.sh`.
    RuntimeNotFound(String),
    /// The requested device (e.g. "NPU") is not present on this machine.
    DeviceUnavailable(String),
    /// Any other OpenVINO / IO error.
    Other(String),
}

impl fmt::Display for NpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpuError::Unsupported(s) => write!(
                f,
                "the NPU/OpenVINO path is not supported on this platform ({s}); \
                 it requires x86_64 linux or windows with an Intel NPU"
            ),
            NpuError::RuntimeNotFound(s) => write!(
                f,
                "OpenVINO runtime not found ({s}). Install it with `make requirements` \
                 (the `openvino` pip wheel) — brain auto-discovers it inside an active \
                 virtualenv. Otherwise set LD_LIBRARY_PATH to the dir containing \
                 libopenvino_c.so (or source an OpenVINO setupvars.sh). The Intel NPU \
                 also needs its user-mode driver."
            ),
            NpuError::DeviceUnavailable(s) => write!(f, "device unavailable: {s}"),
            NpuError::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for NpuError {}

/// Which OpenVINO device to target. Maps to the OV device string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuDevice {
    Npu,
    Cpu,
    Gpu,
    Auto,
}

impl NpuDevice {
    pub fn ov_str(self) -> &'static str {
        match self {
            NpuDevice::Npu => "NPU",
            NpuDevice::Cpu => "CPU",
            NpuDevice::Gpu => "GPU",
            NpuDevice::Auto => "AUTO",
        }
    }
    pub fn parse(s: &str) -> Option<NpuDevice> {
        match s.to_ascii_lowercase().as_str() {
            "npu" => Some(NpuDevice::Npu),
            "cpu" => Some(NpuDevice::Cpu),
            "gpu" => Some(NpuDevice::Gpu),
            "auto" => Some(NpuDevice::Auto),
            _ => None,
        }
    }
}

/// Latency- vs throughput-oriented compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerfHint {
    Latency,
    Throughput,
}

/// NPU compile/run configuration → OpenVINO property map.
#[derive(Clone, Debug)]
pub struct NpuConfig {
    pub device: NpuDevice,
    pub perf_hint: PerfHint,
    pub cache_dir: Option<std::path::PathBuf>,
    pub turbo: bool,
    pub tiles: Option<i32>,
    pub compilation_params: Option<String>,
    pub qdq_opt: bool,
    pub profiling: bool,
    /// If the requested device is absent, fall back to CPU/GPU so the same
    /// compile→run path can be exercised without an NPU.
    pub allow_fallback: bool,
}

impl Default for NpuConfig {
    fn default() -> Self {
        NpuConfig {
            device: NpuDevice::Npu,
            perf_hint: PerfHint::Latency,
            cache_dir: None,
            turbo: false,
            tiles: None,
            compilation_params: Some("optimization-level=2 performance-hint-override=latency".into()),
            qdq_opt: true,
            profiling: false,
            allow_fallback: false,
        }
    }
}

/// What a resolved OpenVINO device reports about itself — used to print, at
/// startup, exactly which hardware path a run actually takes and whether the
/// requested weight precision is a *native* device capability.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// The device actually resolved to (e.g. "NPU", or a CPU/GPU fallback).
    pub device: String,
    /// `FULL_DEVICE_NAME` (e.g. "Intel(R) AI Boost").
    pub full_name: String,
    /// `OPTIMIZATION_CAPABILITIES` (e.g. `["FP16","INT8","EXPORT_IMPORT"]`). The
    /// Intel NPU lists `INT8` but **not** `INT4`: an INT4-weight graph still
    /// compiles and runs there, but as weight-*compression* (the 4-bit weights are
    /// decompressed to a native type for the MAC), not native 4-bit compute.
    pub capabilities: Vec<String>,
}

impl DeviceInfo {
    /// Does the device advertise `cap` (case-insensitive) as a native optimization
    /// capability? e.g. `supports("INT4")`.
    pub fn supports(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c.eq_ignore_ascii_case(cap))
    }
}

/// Raw model outputs handed back to brain's host decode: one entry per graph
/// output, `(name, shape, data)` with NCHW layout preserved from the ONNX graph.
pub struct HeadOutputs {
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

/// Latency/throughput measurement from [`bench`].
#[derive(Clone, Debug)]
pub struct BenchResult {
    pub device: String,
    pub iters: usize,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub throughput_fps: f64,
}

// On x86_64 linux/windows the OpenVINO runtime is always available (loaded at
// run time); elsewhere a stub with the identical API reports `Unsupported`.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
mod real;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
pub use real::*;

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))))]
mod stub;
#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))))]
pub use stub::*;

/// The YOLO NPU session is a whole-graph [`GraphBackend`]: compile an ONNX graph
/// once for a device, then run frames through it. This is the second backend
/// contract (distinct from the eager per-step `backend_api::Backend`); the same
/// impl covers the real OpenVINO session and the stub, since both expose
/// `load_bytes`/`run`/`device`.
impl backend_api::GraphBackend for NpuSession {
    type Config = NpuConfig;
    type Output = HeadOutputs;
    type Error = NpuError;

    fn compile(onnx: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        NpuSession::load_bytes(onnx, cfg)
    }
    fn run(&mut self, input: &[f32], shape: [usize; 4]) -> Result<HeadOutputs, NpuError> {
        NpuSession::run(self, input, shape)
    }
    fn device(&self) -> &str {
        NpuSession::device(self)
    }
}
