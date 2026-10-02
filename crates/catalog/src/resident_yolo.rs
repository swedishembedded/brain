// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 object detection behind the residency scheduler.

use capability::{ActionResult, Invocation, Manifest, Media, Outcome, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::{json, Value};

/// COCO-80 class names (index → label), so detections carry human labels (dog = 16).
const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// YOLO detection behind the scheduler. Loads a brain-format YOLOv8 checkpoint
/// (`BRAIN_YOLOV8`); the resident instance holds the model on the CPU (brain's yolo
/// default) - dropping it frees the RAM. One action, `detect`.
pub struct YoloResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models
    /// (mirrors `resident_llm.rs::GptResident`).
    id: String,
    path: String,
}

impl YoloResident {
    pub fn from_env() -> Option<YoloResident> {
        let path = std::env::var("BRAIN_YOLOV8").ok().filter(|p| !p.is_empty())?;
        // See resident_llm.rs::GptResident::from_env's comment: env-loaded,
        // no upstream vendor/repo provenance.
        Some(Self::from_card(&path, &checkpoint::st::ModelCard::new("brain/yolov8", "yolo"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused -- yolo's class
    /// names are a fixed COCO-80 table, not learned from a tokenizer.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, _tokenizer: Option<&str>) -> YoloResident {
        YoloResident { id: card.id.clone(), path: path.to_string() }
    }

    fn detect_spec() -> capability::ActionSpec {
        use capability::{BlobSpec, ParamSpec, ParamType};
        capability::ActionSpec::new("detect", "detect objects in an image (YOLOv8, COCO-80 classes)")
            .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold").default(json!(0.25)))
            .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").default(json!(0.45)))
            .input(BlobSpec::new("image", Media::Image, "the image to run detection on").required())
    }
}

impl ResidentModel for YoloResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "object detection (YOLOv8, COCO-80)", vec![Self::detect_spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // YOLOv8n is small and runs on the CPU in brain → a modest RAM footprint.
        MemCost::new(0, 128 << 20)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // `BRAIN_YOLOV8_BATCH` (default 1) sets the forward batch: >1 enables a TRUE
        // batched forward (one detect over N images) when the scheduler groups jobs.
        let batch = std::env::var("BRAIN_YOLOV8_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1u32).max(1);
        Ok(Box::new(YoloInstance { yolo: yolov8::Yolo::load(&self.path, batch), batch: batch as usize }))
    }
}

struct YoloInstance {
    yolo: yolov8::Yolo,
    batch: usize,
}

fn detections_outcome(dets: &[[f32; 6]]) -> Outcome {
    let objects: Vec<Value> = dets
        .iter()
        .map(|d| {
            let cls = d[5] as usize;
            json!({"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "class": cls, "label": COCO.get(cls).copied().unwrap_or("?")})
        })
        .collect();
    Outcome::new().set("count", json!(objects.len())).set("detections", json!(objects))
}

impl Instance for YoloInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    /// TRUE batched forward: chunk the invocations to the model's batch and run one
    /// `detect_batch` per chunk (the last chunk padded to the batch, its padding
    /// results discarded). With batch 1 this is one forward per image.
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let b = self.batch;
        let mut out: Vec<ActionResult> = Vec::with_capacity(invs.len());
        for chunk in invs.chunks(b) {
            // Decode this chunk's images (errors become per-job error results).
            let mut imgs: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(chunk.len());
            let mut errs: Vec<Option<String>> = Vec::with_capacity(chunk.len());
            for inv in chunk {
                match capability::blob::decode_image(inv, "image") {
                    Ok(im) => {
                        imgs.push(im);
                        errs.push(None);
                    }
                    Err(e) => errs.push(Some(e)),
                }
            }
            if imgs.is_empty() {
                out.extend(errs.into_iter().map(|e| Err(e.unwrap_or_default())));
                continue;
            }
            // NMS thresholds from the first valid invocation (post-forward, per-image).
            let (conf, iou) = chunk
                .iter()
                .find_map(|i| i.get_blob("image").map(|_| (i.get_f64("conf").unwrap_or(0.25) as f32, i.get_f64("iou").unwrap_or(0.45) as f32)))
                .unwrap_or((0.25, 0.45));
            // Pad to the model's batch by repeating the last image; drop the padding.
            let last = imgs.last().unwrap().clone();
            while imgs.len() < b {
                imgs.push(last.clone());
            }
            let refs: Vec<(&[f32], u32, u32)> = imgs.iter().map(|(p, w, h)| (p.as_slice(), *w, *h)).collect();
            let batched = self.yolo.detect_batch(&refs, conf, iou);
            // Zip results back to the (possibly-erroring) chunk jobs, in order.
            let mut valid = batched.into_iter();
            for e in errs {
                match e {
                    Some(msg) => out.push(Err(msg)),
                    None => out.push(Ok(detections_outcome(&valid.next().unwrap_or_default()))),
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------- z-image
