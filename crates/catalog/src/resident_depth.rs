// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Monocular depth (ZipDepth) behind the residency scheduler.
//!
//! The actions are ZipDepth's own (`zipdepth::caps`): on a GPU or the CPU an
//! instance is the served provider over the checkpoint, run on the device the
//! scheduler assigned. `infer` may also be placed on an NPU, where the
//! instance is the compiled ZipDepth graph instead and draws the same outputs
//! (`zipdepth::caps::infer_outcome`).
//!
//! The weights are a host-RAM model: the imported map lives in system
//! memory and the per-call device buffers are transient, so it is budgeted
//! against the RAM pool.

use std::collections::HashMap;

use capability::{ActionResult, Invocation, Manifest, Progress, Provider};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use zipdepth::ZipConfig;

/// ZipDepth behind the scheduler, over one checkpoint.
pub struct DepthResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models
    /// (mirrors `resident_llm.rs::GptResident`).
    id: String,
    path: String,
}

/// Instance key of the actions that run only on the engine (training,
/// calibration); `infer`'s may be placed on an NPU.
const ENGINE_KEY: &str = "engine";

impl DepthResident {
    pub fn from_env() -> Option<DepthResident> {
        let path = std::env::var("BRAIN_ZIPDEPTH_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        // See resident_llm.rs::GptResident::from_env's comment: env-loaded,
        // no upstream vendor/repo provenance.
        Some(Self::from_card(&path, &checkpoint::st::ModelCard::new(zipdepth::caps::MODEL, "depth"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused: depth has no
    /// text vocab. The checkpoint's variant (base or npu-blend) is detected
    /// from its own tensor shapes when it loads.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, _tokenizer: Option<&str>) -> DepthResident {
        DepthResident { id: card.id.clone(), path: path.to_string() }
    }
}

impl ResidentModel for DepthResident {
    fn manifest(&self) -> Manifest {
        let mut m = zipdepth::caps::manifest();
        m.model = self.id.clone();
        m
    }
    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), if action == "infer" { "default" } else { ENGINE_KEY })
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // ZipDepth (~6.1M params) is imported into a host-RAM weight map and runs
        // via brain's engine; the Hot footprint is the weights in RAM (~1.3x the
        // checkpoint file, allowing for the f32 unpack + index overhead).
        let ram = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
        if key.config == ENGINE_KEY {
            return MemCost::new(0, ram);
        }
        // `infer` has an NPU path (`crates/npu` depth topology, OpenVINO):
        // advertising an NPU footprint lets the scheduler place it on an NPU
        // when one is budgeted (see `place::pick_device`). The compiled fp16
        // graph and its activation scratch fit well within 256 MB.
        MemCost::new(0, ram).with_npu(256 << 20)
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let (Device::Npu(_), false) = (device, key.config == ENGINE_KEY) {
            // Auto-detect the variant from the checkpoint's own tensor names so
            // the strict importer's shapes match.
            let cfg = zipdepth::cfg_for_checkpoint(&self.path)?;
            if cfg.upsample_unfold {
                return Err("depth on NPU needs the blend ('npu') ZipDepth checkpoint".into());
            }
            let init = zipdepth::import::load(&self.path, &cfg)?;
            let side = if cfg.input > 0 { cfg.input } else { 384 };
            let model = DepthNpuModel { cfg, init, side };
            let ov = npu::openvino::NpuConfig { device: npu::openvino::NpuDevice::Npu, allow_fallback: true, ..Default::default() };
            let graph = <DepthNpuModel as npu::NpuModel>::compile(&model, &ov)?;
            eprintln!("depth: compiled ZipDepth {side}x{side} on {}", graph.device());
            return Ok(Box::new(DepthNpuInstance { graph, side }));
        }
        Ok(Box::new(DepthInstance { provider: zipdepth::caps::DepthProvider::new(Some(self.path.clone())), device }))
    }
}

/// The ZipDepth graph as a generic [`npu::NpuModel`]: build the depth ONNX for a
/// fixed `side × side` input. This is the *only* depth-specific NPU code - compile /
/// cache / infer / evict all reuse `npu::openvino::NpuGraph`.
struct DepthNpuModel {
    cfg: ZipConfig,
    init: HashMap<String, Vec<f32>>,
    side: u32,
}

impl npu::NpuModel for DepthNpuModel {
    fn build(&self, g: &mut onnx::GraphBuilder) -> Result<(), String> {
        npu::build_depth_graph_hw(&self.cfg, &self.init, self.side, self.side, g);
        Ok(())
    }
    fn cache_key(&self) -> String {
        format!("zipdepth-{}x{}", self.side, self.side)
    }
}

/// A depth instance running on the NPU: the compiled [`NpuGraph`] + the fixed input
/// side. Per call: resize the image to `side×side` CHW, infer on the NPU, resize the
/// `[1,1,2·side,2·side]` inverse-depth map back to the frame grid.
struct DepthNpuInstance {
    graph: npu::openvino::NpuGraph,
    side: u32,
}

impl Instance for DepthNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        use npu::openvino::Feed;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let s = self.side;
        // resize to the compiled square, pack CHW
        let resized = imaging::resize_bilinear_hwc(&hwc, 3, w, h, s, s);
        let hw = (s * s) as usize;
        let mut chw = vec![0f32; 3 * hw];
        for y in 0..s as usize {
            for x in 0..s as usize {
                for c in 0..3 {
                    chw[c * hw + y * s as usize + x] = resized[(y * s as usize + x) * 3 + c];
                }
            }
        }
        let out = self.graph.run(&[("input", Feed::F32(&chw, vec![1, 3, s as i64, s as i64]))]).map_err(|e| e.to_string())?;
        let (_name, oshape, data) = out.into_iter().next().ok_or("depth NPU: no output")?;
        // output is [1,1,H,W] inverse-depth; resize back to the frame grid.
        let (oh, ow) = (oshape[oshape.len() - 2] as u32, oshape[oshape.len() - 1] as u32);
        let depth = imaging::resize_bilinear_hwc(&data, 1, ow, oh, w, h);
        Ok(zipdepth::caps::infer_outcome(inv, &hwc, w, h, depth)?.set("device", serde_json::json!("npu")))
    }
}

/// A resident ZipDepth instance on the engine: the served provider over the
/// checkpoint, whose first call imports the weights onto the assigned device
/// and keeps them; dropping the instance frees them.
struct DepthInstance {
    provider: zipdepth::caps::DepthProvider,
    device: Device,
}

impl Instance for DepthInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.provider.action(action).ok_or_else(|| format!("zipdepth: unknown action '{action}'"))?;
        crate::resident_llm::on_device(self.device, || act.run(inv, progress))?
    }
}
