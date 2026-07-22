// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stub OpenVINO runtime for targets where OpenVINO is unavailable (wasm,
//! aarch64, macOS). The API matches [`super::real`] so the CLI compiles
//! everywhere; every entry point reports the platform is unsupported.

use super::{BenchResult, DeviceInfo, HeadOutputs, NpuConfig, NpuDevice, NpuError};
use std::path::Path;

fn unsupported<T>() -> Result<T, NpuError> {
    Err(NpuError::Unsupported(std::env::consts::ARCH.to_string()))
}

/// Always empty on unsupported targets.
pub fn available_devices() -> Result<Vec<String>, NpuError> {
    Ok(Vec::new())
}

/// Unsupported on stub targets.
pub fn device_info(_device: NpuDevice, _allow_fallback: bool) -> Result<DeviceInfo, NpuError> {
    unsupported()
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

/// A compiled input-embedding graph. Never constructible on unsupported targets.
pub struct EmbedSession {
    _priv: (),
}

impl EmbedSession {
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn load_path(_p: &Path, _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn seq_len(&self) -> usize {
        0
    }
    pub fn d_in(&self) -> usize {
        0
    }
    pub fn d_out(&self) -> usize {
        0
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn run_embeds(&mut self, _embeds: &[f32]) -> Result<Vec<f32>, NpuError> {
        unsupported()
    }
}

/// The Chronos-2 transformer core graph. Never constructible on unsupported targets.
pub struct Chronos2Session {
    _priv: (),
}

impl Chronos2Session {
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn seq_len(&self) -> usize {
        0
    }
    pub fn n_out(&self) -> usize {
        0
    }
    pub fn run(&mut self, _emb: &[f32], _kmask: &[f32]) -> Result<Vec<f32>, NpuError> {
        unsupported()
    }
}

/// The Kronos decode_s1 core graph. Never constructible on unsupported targets.
pub struct KronosS1Session {
    _priv: (),
}

impl KronosS1Session {
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn seq_len(&self) -> usize {
        0
    }
    pub fn s1_vocab(&self) -> usize {
        0
    }
    pub fn run(&mut self, _x: &[f32]) -> Result<(Vec<f32>, Vec<f32>), NpuError> {
        unsupported()
    }
}

/// The Kronos decode_s2 dependency graph. Never constructible on unsupported targets.
pub struct KronosS2Session {
    _priv: (),
}

impl KronosS2Session {
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn seq_len(&self) -> usize {
        0
    }
    pub fn s2_vocab(&self) -> usize {
        0
    }
    pub fn run(&mut self, _ctx: &[f32], _sib: &[f32]) -> Result<Vec<f32>, NpuError> {
        unsupported()
    }
}

/// A compiled KV-cache decode-step graph. Never constructible on unsupported targets.
pub struct KvSession {
    _priv: (),
}

impl KvSession {
    #[allow(clippy::too_many_arguments)]
    pub fn load_path(
        _p: &Path,
        _cfg: &NpuConfig,
        _n_layers: usize,
        _d: usize,
        _nkv: usize,
        _hd: usize,
        _cap: usize,
    ) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    #[allow(clippy::type_complexity)]
    pub fn run_step(
        &mut self,
        _x: &[f32],
        _cos: &[f32],
        _sin: &[f32],
        _mask: &[f32],
        _past_k: &[Vec<f32>],
        _past_v: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), NpuError> {
        unsupported()
    }
}

/// A compiled prefill graph. Never constructible on unsupported targets.
pub struct PrefillSession {
    _priv: (),
}

impl PrefillSession {
    #[allow(clippy::too_many_arguments)]
    pub fn load_path(
        _p: &Path,
        _cfg: &NpuConfig,
        _n_layers: usize,
        _d: usize,
        _nkv: usize,
        _hd: usize,
        _cap: usize,
    ) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    #[allow(clippy::type_complexity)]
    pub fn run(&mut self, _embeds: &[f32]) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), NpuError> {
        unsupported()
    }
}

/// A compiled codec-decoder graph. Never constructible on unsupported targets.
pub struct CodecSession {
    _priv: (),
}

impl CodecSession {
    pub fn load_bytes(_bytes: &[u8], _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn load_path(_p: &Path, _cfg: &NpuConfig) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn nq(&self) -> usize {
        0
    }
    pub fn code_len(&self) -> usize {
        0
    }
    pub fn out_len(&self) -> usize {
        0
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn run_codes(&mut self, _codes: &[i64]) -> Result<Vec<f32>, NpuError> {
        unsupported()
    }
}

pub struct BackStreamSession {
    _priv: (),
}

impl BackStreamSession {
    pub fn load_path(
        _p: &Path,
        _cfg: &NpuConfig,
        _bufs: Vec<(String, i64, i64)>,
        _latent_dim: usize,
        _chunk: usize,
    ) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn zero_buffers(&self) -> Vec<Vec<f32>> {
        Vec::new()
    }
    pub fn run(&mut self, _latent: &[f32], _bufins: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<Vec<f32>>), NpuError> {
        unsupported()
    }
}

pub struct FusedMtpSession {
    _priv: (),
}

impl FusedMtpSession {
    pub fn load_path(_p: &Path, _cfg: &NpuConfig, _emb: usize, _nres: usize) -> Result<Self, NpuError> {
        unsupported()
    }
    pub fn device(&self) -> &str {
        ""
    }
    pub fn run(&mut self, _talker_hidden: &[f32], _cb0_embed: &[f32]) -> Result<(Vec<u32>, Vec<f32>), NpuError> {
        unsupported()
    }
}
