// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The networks on OpenVINO: [`crate::export`]'s graph compiled for an Intel
//! NPU, GPU or CPU by `openvino_rt`.
//!
//! What this buys over brain's own kernels is a whole-graph compiler: the
//! NPU has no other way in, and on an Intel iGPU the compiled graph runs the
//! small MobileNetV3 layers far faster than one dispatch per op does today.
//! The output is checked against the same goldens as [`crate::Net`].

use std::path::Path;

use openvino_rt::{NpuConfig, NpuDevice, NpuSession};

use crate::config::NetConfig;
use crate::infer::{Infer, Output};

/// One network compiled for one OpenVINO device.
pub struct OvNet {
    cfg: NetConfig,
    session: NpuSession,
}

/// Which OpenVINO device a name means: `npu`, `gpu`, `cpu`.
pub fn device(name: &str) -> Option<NpuDevice> {
    match name {
        "npu" => Some(NpuDevice::Npu),
        "gpu" => Some(NpuDevice::Gpu),
        "cpu" => Some(NpuDevice::Cpu),
        _ => None,
    }
}

impl OvNet {
    /// Import `dir/<cfg.file>`, export it, and compile it for `device`. A
    /// missing runtime or device is an error naming it; there is no
    /// fallback to another device.
    pub fn load(dir: &Path, cfg: NetConfig, device: NpuDevice) -> Result<OvNet, String> {
        let t = crate::import::import(&dir.join(cfg.file), &cfg)?;
        OvNet::compile(&t, cfg, device)
    }

    /// Compile already-imported tensors.
    pub fn compile(t: &onnx::walk::Tensors, cfg: NetConfig, device: NpuDevice) -> Result<OvNet, String> {
        let bytes = crate::export::export(t, &cfg)?;
        let ov = NpuConfig { device, allow_fallback: false, compilation_params: None, ..NpuConfig::default() };
        let session = NpuSession::load_bytes(&bytes, &ov).map_err(|e| format!("compiling {} for OpenVINO {}: {e}", cfg.file, device.ov_str()))?;
        Ok(OvNet { cfg, session })
    }
}

impl Infer for OvNet {
    fn config(&self) -> &NetConfig {
        &self.cfg
    }

    fn infer(&mut self, x: &[f32]) -> Result<Output, String> {
        let n = self.cfg.input as usize;
        let out = self.session.run(x, [1, 3, n, n]).map_err(|e| format!("{}: {e}", self.cfg.file))?;
        let names: Vec<String> = out.tensors.iter().map(|t| t.0.clone()).collect();
        let mut map = None;
        let mut pooled = None;
        for (name, _, data) in out.tensors {
            match name.as_str() {
                crate::export::OUTPUT => map = Some(data),
                crate::export::POOLED => pooled = Some(data),
                _ => {}
            }
        }
        let map = map.ok_or_else(|| format!("{}: the compiled graph returned no `{}` (it returned {names:?})", self.cfg.file, crate::export::OUTPUT))?;
        if self.cfg.peak_pool && pooled.is_none() {
            return Err(format!("{}: the compiled graph returned no `{}`", self.cfg.file, crate::export::POOLED));
        }
        Ok(Output { map, pooled })
    }

    fn device(&self) -> String {
        format!("OpenVINO {}", self.session.device())
    }
}
