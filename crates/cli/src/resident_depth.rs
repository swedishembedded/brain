// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Monocular depth (ZipDepth) behind the residency scheduler.
//!
//! Mirrors the yolo adapter: a [`ResidentModel`] whose `activate` loads a released
//! ZipDepth `.pth` (`BRAIN_DEPTH_WEIGHTS`) once, and whose [`Instance`] owns the
//! resident weights — dropping it frees them. One action, `depth`.
//!
//! The instance keeps the imported weight map in host RAM (the model's "Hot"
//! footprint) plus a live [`Gpu`] backend, and per call rebuilds a [`ParamStore`]
//! and a [`Predictor`] to run the exact same aspect-preserving preprocess ->
//! forward -> unwarp pipeline `brain depth --image` uses (`Predictor::predict`).
//! This is a RAM-resident model: the weights live in system memory, the per-call
//! device buffers are transient, so it is budgeted against the RAM pool like yolo.

use std::collections::HashMap;

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use depth::{Predictor, ZipConfig};
use gpu_core::Gpu;
use paramstore::ParamStore;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

/// ZipDepth monocular depth behind the scheduler. Loads a brain-format ZipDepth
/// checkpoint (`BRAIN_DEPTH_WEIGHTS`); the resident instance holds the weights in
/// RAM — dropping it frees them. One action, `depth`.
pub struct DepthResident {
    path: String,
}

impl DepthResident {
    pub fn from_env() -> Option<DepthResident> {
        std::env::var("BRAIN_DEPTH_WEIGHTS").ok().filter(|p| !p.is_empty()).map(|path| DepthResident { path })
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
        Manifest::new("depth", "monocular depth (ZipDepth)", vec![Self::depth_spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("depth", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // ZipDepth (~6.1M params) is imported into a host-RAM weight map and runs
        // via brain's engine; the Hot footprint is the weights in RAM (~1.3x the
        // checkpoint file, allowing for the f32 unpack + index overhead).
        let ram = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
        MemCost::new(0, ram)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // Auto-detect the checkpoint variant from its own tensor names, exactly like
        // `brain depth` (`where_conv.*` -> blend/npu upsampler, else unfold/base), so
        // the strict importer's shapes match without the caller passing a variant.
        let names = depth::import::tensor_names(&self.path).unwrap_or_default();
        let blend = names.iter().any(|n| n.contains("where_conv"));
        let cfg = ZipConfig { upsample_unfold: !blend, ..ZipConfig::base() };

        // Build the engine once (honours the process backend / `--device`), and
        // import the weights once into a host-RAM map the instance keeps resident.
        let gpu = Gpu::new(depth::net::PIPELINES);
        let init = depth::import::load(&self.path, &cfg)?;
        Ok(Box::new(DepthInstance { gpu, init, cfg }))
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

/// Decode an image blob (HWC f32 + `{w,h}` meta) into `(pixels, w, h)` — the same
/// contract yolo's input uses.
fn image_of(inv: &Invocation) -> Result<(Vec<f32>, u32, u32), String> {
    let blob = inv.get_blob("image").ok_or("depth: missing input 'image'")?;
    let w = blob.meta.get("w").and_then(|v| v.as_u64()).ok_or("depth: image meta needs w")? as u32;
    let h = blob.meta.get("h").and_then(|v| v.as_u64()).ok_or("depth: image meta needs h")? as u32;
    let hwc: Vec<f32> = blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    Ok((hwc, w, h))
}

impl Instance for DepthInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (hwc, w, h) = image_of(inv)?;

        // Optional smaller input (shorter side): the net is fully convolutional, so
        // any x32 input is valid and faster — the predictor rounds it. 0 keeps native.
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
        let bytes: Vec<u8> = depth.iter().flat_map(|&v| ((v - mn) / range).clamp(0.0, 1.0).to_le_bytes()).collect();

        Ok(Outcome::new()
            .set("width", json!(w))
            .set("height", json!(h))
            .blob("depth", Blob::new(Media::Image, bytes).with_meta(json!({"w": w, "h": h, "c": 1}))))
    }
}
