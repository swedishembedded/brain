// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 behind the residency scheduler.
//!
//! The actions are YOLOv8's own (`yolov8::caps`): on a GPU or the CPU an
//! instance is the served provider over the checkpoint, run on the device the
//! scheduler assigned. `detect` may also be placed on an NPU, where the
//! instance is the compiled graph instead ([`npu::YoloSession`]) and draws the
//! same detections; the identity stage has no NPU path, so a `detect` that asks
//! for one is refused there rather than quietly run without it.
//!
//! Swedish Embedded AB implements object detection on edge accelerators for
//! its clients. If your team needs expertise in scheduling detectors across
//! GPUs, CPUs and NPUs then you can procure our services by sending an email
//! to info@swedishembedded.com.

use capability::{ActionResult, Invocation, Manifest, Progress, Provider};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// YOLOv8 behind the scheduler, over one checkpoint.
pub struct YoloResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models
    /// (mirrors `resident_llm.rs::GptResident`).
    id: String,
    path: String,
}

/// Instance key of the actions that run only on the engine (training,
/// evaluation); `detect`'s may be placed on an NPU.
const ENGINE_KEY: &str = "engine";

impl YoloResident {
    /// The checkpoint the model store resolves, under the model's own id.
    pub fn from_store(models_dir: Option<&std::path::Path>) -> Option<YoloResident> {
        let path = loader::served::served_assembly(models_dir, "yolov8", &yolov8::spec::YoloSpec, &[])?.role_path("weights").ok()?;
        Some(Self::from_card(&path, &checkpoint::st::ModelCard::new(yolov8::caps::MODEL, "yolo"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused: the class names
    /// of the published weights are a fixed COCO-80 table, not a tokenizer.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, _tokenizer: Option<&str>) -> YoloResident {
        YoloResident { id: card.id.clone(), path: path.to_string() }
    }
}

impl ResidentModel for YoloResident {
    fn manifest(&self) -> Manifest {
        let mut m = yolov8::caps::manifest();
        m.model = self.id.clone();
        m
    }
    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), if action == "detect" { "default" } else { ENGINE_KEY })
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // YOLOv8n is small and runs on the CPU in brain: a modest RAM footprint.
        let cost = MemCost::new(0, 128 << 20);
        if key.config == ENGINE_KEY {
            return cost;
        }
        // `detect` has an NPU path (`npu::YoloSession`, OpenVINO): advertising an
        // NPU footprint lets the scheduler place it on an NPU when one is
        // budgeted (see `place::pick_device`).
        cost.with_npu(256 << 20)
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let (Device::Npu(_), false) = (device, key.config == ENGINE_KEY) {
            let cfg = npu::openvino::NpuConfig { device: npu::openvino::NpuDevice::Npu, allow_fallback: true, ..Default::default() };
            let session = npu::YoloSession::from_weights(&self.path, None, &cfg)?;
            eprintln!("yolov8: compiled the detector on {}", session.device());
            return Ok(Box::new(YoloNpuInstance { session }));
        }
        Ok(Box::new(YoloInstance { provider: yolov8::caps::YoloProvider::new(Some(self.path.clone()), Some(std::sync::Arc::new(yolov8_faces::StoreFaces::new(crate::components())))), device }))
    }
}

/// A resident YOLOv8 on the engine: the served provider over the checkpoint,
/// whose first call loads the weights onto the assigned device and keeps them;
/// dropping the instance frees them.
struct YoloInstance {
    provider: yolov8::caps::YoloProvider,
    device: Device,
}

impl Instance for YoloInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.provider.action(action).ok_or_else(|| format!("yolov8: unknown action '{action}'"))?;
        crate::resident_llm::on_device(self.device, || act.run(inv, progress))?
    }
}

/// `detect` on the NPU: the compiled graph, with the box decode and NMS on the
/// host.
struct YoloNpuInstance {
    session: npu::YoloSession,
}

impl Instance for YoloNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.get_blob("identity_ref").is_some() || inv.get_str("identity_ref_path").is_some_and(|p| !p.is_empty()) {
            return Err("yolov8 detect: identity verification has no NPU path (the ArcFace embedder runs on the engine)".into());
        }
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let conf = inv.get_f64("conf").unwrap_or(0.25) as f32;
        let iou = inv.get_f64("iou").unwrap_or(0.45) as f32;
        let dets = self.session.detect(&hwc, w, h, conf, iou)?;
        Ok(yolov8::caps::detect_outcome(&yolov8::caps::unlabeled(&dets))?.set("device", serde_json::json!("npu")))
    }
}
