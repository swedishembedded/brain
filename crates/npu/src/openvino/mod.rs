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
    /// path). Install OpenVINO and source `setupvars.sh`. See docs/yolo/NPU.md.
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
                 also needs its user-mode driver. See docs/yolo/NPU.md"
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
