// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Monocular depth (ZipDepth) behind the residency scheduler.
//!
//! Mirrors the yolo adapter: a [`ResidentModel`] whose `activate` loads a released
//! ZipDepth `.pth` (`BRAIN_ZIPDEPTH_WEIGHTS`) once, and whose [`Instance`] owns the
//! resident weights - dropping it frees them. One action, `depth`.
//!
//! The instance keeps the imported weight map in host RAM (the model's "Hot"
//! footprint) plus a live [`Gpu`] backend, and per call rebuilds a [`ParamStore`]
//! and a [`Predictor`] to run the exact same aspect-preserving preprocess ->
//! forward -> unwarp pipeline `brain depth --image` uses (`Predictor::predict`).
//! This is a RAM-resident model: the weights live in system memory, the per-call
//! device buffers are transient, so it is budgeted against the RAM pool like yolo.

use std::collections::HashMap;

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use zipdepth::{Predictor, ZipConfig};
use gpu_core::Gpu;
use paramstore::ParamStore;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

/// ZipDepth monocular depth behind the scheduler. Loads a brain-format ZipDepth
/// checkpoint (`BRAIN_ZIPDEPTH_WEIGHTS`); the resident instance holds the weights in
/// RAM - dropping it frees them. One action, `depth`.
pub struct DepthResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models
    /// (mirrors `resident_llm.rs::GptResident`).
    id: String,
    path: String,
}

impl DepthResident {
    pub fn from_env() -> Option<DepthResident> {
        let path = std::env::var("BRAIN_ZIPDEPTH_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        // See resident_llm.rs::GptResident::from_env's comment: env-loaded,
        // no upstream vendor/repo provenance.
        Some(Self::from_card(&path, &checkpoint::st::ModelCard::new("brain/zipdepth", "depth"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused -- depth has no
    /// text vocab. The checkpoint's real variant (base vs npu-blend) is
    /// auto-detected from its own tensor shapes at `activate()` time
    /// (`zipdepth::cfg_for_checkpoint`), not from anything carried here.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, _tokenizer: Option<&str>) -> DepthResident {
        DepthResident { id: card.id.clone(), path: path.to_string() }
    }

    fn depth_spec() -> ActionSpec {
        ActionSpec::new("depth", "monocular depth from a single image (ZipDepth)")
            .param(
                ParamSpec::new("input", ParamType::Int, "model input (shorter side, x32); 0 = the checkpoint's native 384")
                    .default(json!(0)),
            )
            .input(BlobSpec::new("image", Media::Image, "the image to estimate depth for").required())
    }
}

impl ResidentModel for DepthResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "monocular depth (ZipDepth)", vec![Self::depth_spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // ZipDepth (~6.1M params) is imported into a host-RAM weight map and runs
        // via brain's engine; the Hot footprint is the weights in RAM (~1.3x the
        // checkpoint file, allowing for the f32 unpack + index overhead).
        let ram = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
        // ZipDepth has an NPU path (`crates/npu` depth topology → OpenVINO). Advertising
        // an NPU footprint (`npu > 0`) makes the scheduler auto-place depth on the NPU
        // when one is budgeted (see `place::pick_device`). The compiled fp16 graph +
        // activation scratch is small (~256 MB is a generous bound for the 6.1M-param net).
        MemCost::new(0, ram).with_npu(256 << 20)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Auto-detect the checkpoint variant from its own tensor names, exactly like
        // `brain depth` (see `zipdepth::cfg_for_checkpoint`), so the strict importer's
        // shapes match without the caller passing a variant.
        let cfg = zipdepth::cfg_for_checkpoint(&self.path).unwrap_or_else(|_| ZipConfig::base());
        let init = zipdepth::import::load(&self.path, &cfg)?;

        // Placed on the NPU → compile the ZipDepth ONNX graph ONCE (via the generic
        // `npu::NpuModel` seam) for a fixed square input and run it through the
        // reusable `NpuGraph`. Every other device → the existing engine path.
        if let Device::Npu(_) = device {
            if cfg.upsample_unfold {
                return Err("depth on NPU needs the blend ('npu') ZipDepth checkpoint (BRAIN_ZIPDEPTH_WEIGHTS)".into());
            }
            let side = if cfg.input > 0 { cfg.input } else { 384 };
            let model = DepthNpuModel { cfg: cfg.clone(), init, side };
            let ov = npu::openvino::NpuConfig { device: npu::openvino::NpuDevice::Npu, allow_fallback: true, ..Default::default() };
            let graph = <DepthNpuModel as npu::NpuModel>::compile(&model, &ov)?;
            eprintln!("depth: compiled ZipDepth {side}x{side} on {}", graph.device());
            return Ok(Box::new(DepthNpuInstance { graph, side }));
        }

        // Build the engine once (honours the process backend / `--device`), and
        // keep the imported weights resident in host RAM. `on_device` scopes
        // the build onto the dispatcher's assigned device, like every sibling
        // adapter (`resident_scrfd.rs`, `resident_upscale.rs`, ...) - a bare
        // `Gpu::new` here bound whatever the thread-local default card was,
        // so the manager could budget depth against a device it wasn't on.
        let gpu = crate::resident_llm::on_device(device, || Gpu::new(zipdepth::net::PIPELINES))?;
        Ok(Box::new(DepthInstance { gpu, init, cfg }))
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

        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &depth {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let range = (mx - mn).max(1e-6);
        let norm: Vec<f32> = depth.iter().map(|&v| ((v - mn) / range).clamp(0.0, 1.0)).collect();
        Ok(Outcome::new()
            .set("width", json!(w))
            .set("height", json!(h))
            .set("device", json!("npu"))
            .blob("depth", capability::blob::image_blob(&norm, w, h, 1)))
    }
}

/// A resident ZipDepth instance: the engine plus the imported weights (RAM). Each
/// call materialises a [`ParamStore`] + [`Predictor`] from these and runs one
/// inference; dropping the instance frees the resident weights.
struct DepthInstance {
    gpu: Gpu,
    init: HashMap<String, Vec<f32>>,
    cfg: ZipConfig,
}

impl Instance for DepthInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        // Optional smaller input (shorter side): the net is fully convolutional, so
        // any x32 input is valid and faster - the predictor rounds it. 0 keeps native.
        let mut cfg = self.cfg.clone();
        if let Some(n) = inv.get_i64("input") {
            if n > 0 {
                cfg.input = n as u32;
            }
        }

        // Materialise the param store on the engine from the resident weight map,
        // then run the reference pipeline (aspect-preserving resize -> forward ->
        // unwarp to the frame grid). `predict` returns a `[h*w]` inverse-depth map.
        let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(name, shape)| (name, shape.iter().product())).collect();
        let ps = ParamStore::new(&self.gpu, params, &self.init);
        let predictor = Predictor::new(&self.gpu, cfg.clone(), ps);
        let depth = predictor.predict(&hwc, w, h);

        // Normalise the depth to [0,1] (min-max over the frame) and emit it as a
        // single-channel HWC f32 image blob, mirroring `emit_image` for c=1.
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &depth {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        let range = (mx - mn).max(1e-6);
        let norm: Vec<f32> = depth.iter().map(|&v| ((v - mn) / range).clamp(0.0, 1.0)).collect();

        Ok(Outcome::new()
            .set("width", json!(w))
            .set("height", json!(h))
            .blob("depth", capability::blob::image_blob(&norm, w, h, 1)))
    }
}
