// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stub OpenVINO runtime for targets where OpenVINO is unavailable (wasm,
//! aarch64, macOS). The API matches [`super::real`] so the CLI compiles
//! everywhere; every entry point reports the platform is unsupported.

use super::{BenchResult, HeadOutputs, NpuConfig, NpuError};
use std::path::Path;

fn unsupported<T>() -> Result<T, NpuError> {
    Err(NpuError::Unsupported(std::env::consts::ARCH.to_string()))
}

/// Always empty on unsupported targets.
pub fn available_devices() -> Result<Vec<String>, NpuError> {
    Ok(Vec::new())
}

/// Always false on unsupported targets.
pub fn npu_present() -> bool {
    false
}

/// A compiled NPU model. Never constructible on unsupported targets.
pub struct NpuSession {
    _priv: (),
}

impl NpuSession {
    pub fn load(_onnx_path: &Path, _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn input_shape(&self) -> [usize; 4] {
        [0; 4]
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn run(&mut self, _input_chw: &[f32], _shape: [usize; 4]) -> Result<HeadOutputs, NpuError> {
        unsupported()
    }
}

pub fn bench(
    _session: &mut NpuSession,
    _input_chw: &[f32],
    _shape: [usize; 4],
    _warmup: usize,
    _iters: usize,
) -> Result<BenchResult, NpuError> {
    unsupported()
}

/// A compiled decoder model. Never constructible on unsupported targets.
pub struct DecoderSession {
    _priv: (),
}

impl DecoderSession {
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn load_path(_p: &Path, _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn seq_len(&self) -> usize {
        0
    }
    pub fn vocab(&self) -> usize {
        0
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn run_ids(&mut self, _ids: &[i64]) -> Result<Vec<f32>, NpuError> {
        unsupported()
    }
}
