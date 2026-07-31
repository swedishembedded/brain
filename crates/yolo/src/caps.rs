// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLO's capabilities behind the generalized [`capability`] interface — what
//! makes `brain caps yolo` / `brain do yolo detect …` (and the perf suite's
//! `CapabilityTarget`) work with no yolo-specific plumbing in the CLI.
//!
//! One action, `detect`: the same infer path `brain yolo detect` runs
//! ([`Yolo::load`] + [`Yolo::detect`], letterbox → forward → DFL decode → NMS).
//! One-shot: no `Progress` stream — the detections are the single artifact. The
//! manifest is static (no weights); the model loads lazily on the first run and
//! stays resident across calls (keyed by weights path), mirroring `zimage::caps`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use serde_json::{json, Value};

use crate::model::Yolo;

/// The model id used on the CLI (`brain do yolo …`) and the event API.
pub const MODEL: &str = "yolo";

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let detect = ActionSpec::new("detect", "detect objects in an image (letterbox → forward → DFL decode → NMS)")
        .param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format YOLO checkpoint (.safetensors)").required())
        .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold").default(json!(0.25)))
        .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").default(json!(0.45)))
        .input(BlobSpec::new("image", Media::Image, "the image to run detection on").required())
        .output(BlobSpec::new("detections", Media::Text, "JSON array of {bbox:[x1,y1,x2,y2], conf, class} in image coords"));
    Manifest::new(MODEL, "YOLOv8-style anchor-free object detector — single-image detection.", vec![detect])
}

/// The executable YOLO model behind the manifest. Construction is free — the
/// checkpoint loads lazily on the first `detect` and stays resident (RAM/CPU by
/// default, like `brain yolo detect`).
#[derive(Default)]
pub struct YoloProvider {
    hot: Arc<Mutex<Option<(String, Yolo)>>>,
}

impl YoloProvider {
    pub fn new() -> YoloProvider {
        YoloProvider::default()
    }
}

impl Provider for YoloProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "detect").then(|| Arc::new(DetectAction { hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct DetectAction {
    hot: Arc<Mutex<Option<(String, Yolo)>>>,
}

impl Action for DetectAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "detect").expect("known action")
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = inv.get_str("weights").ok_or("yolo detect: missing required param 'weights'")?;
        if !Path::new(&weights).exists() {
            return Err(format!("yolo detect: weights not found at '{weights}'"));
        }
        let conf = inv.get_f64("conf").unwrap_or(0.25) as f32;
        let iou = inv.get_f64("iou").unwrap_or(0.45) as f32;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        // Hot path: keep the loaded model resident; rebuild only on a new path.
        let mut guard = self.hot.lock().map_err(|_| "yolo: hot model lock poisoned")?;
        if !matches!(&*guard, Some((p, _)) if *p == weights) {
            *guard = None; // free the old resident model before loading new
            *guard = Some((weights.clone(), Yolo::load(&weights, 1)));
        }
        let model = &guard.as_ref().unwrap().1;

        let dets = model.detect(&hwc, w, h, conf, iou);
        let objects: Vec<Value> = dets
            .iter()
            .map(|d| json!({"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "class": d[5] as u32}))
            .collect();
        let text = serde_json::to_string(&objects).map_err(|e| e.to_string())?;
        Ok(Outcome::new()
            .set("count", json!(objects.len()))
            .set("detections", json!(objects))
            .blob("detections", Blob::new(Media::Text, text.into_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YoloConfig;
    use capability::Registry;
    use model::ModelConfig;

    #[test]
    fn manifest_declares_detect() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        let d = &m.actions[0];
        assert_eq!(d.name, "detect");
        assert!(!d.streaming, "detect is one-shot (the result is the single artifact)");
        assert!(d.params.iter().any(|p| p.name == "weights" && p.required));
        assert_eq!(d.params.iter().find(|p| p.name == "conf").unwrap().default, Some(json!(0.25)));
        assert!(d.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert_eq!(d.outputs[0].media, Media::Text);
        // validation fills defaults / rejects a missing image, without weights.
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = d.validate(Invocation::new().set("weights", json!("w")).blob("image", img)).unwrap();
        assert_eq!(inv.get_f64("iou"), Some(0.45));
        assert!(d.validate(Invocation::new().set("weights", json!("w"))).is_err());
        assert_eq!(manifest().to_json()["actions"][0]["name"], "detect");
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(YoloProvider::new()));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let err = reg
            .run(MODEL, "detect", Invocation::new().set("weights", json!("/nonexistent/yolo.safetensors")).blob("image", img), &mut |_| {})
            .unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
    }

    /// End-to-end on a tiny synthetic checkpoint (64px input, 1 class): save
    /// `init_weights` to disk, then drive `detect` through the Registry. A
    /// random-init model detects nothing meaningful — the assertion is that the
    /// full load → letterbox → forward → NMS path runs and emits the outputs.
    #[test]
    fn tiny_checkpoint_detects_end_to_end() {
        let mut cfg = YoloConfig::tiny(1);
        cfg.input = 64; // fully convolutional: a small input keeps the test fast
        let init = crate::init::init_model(&cfg, 3);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
                (name, vec![n as u64], v)
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("yolo-caps-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);

        let mut reg = Registry::new();
        reg.register(Arc::new(YoloProvider::new()));
        let px = vec![0.5f32; 48 * 32 * 3];
        let bytes: Vec<u8> = px.iter().flat_map(|f| f.to_le_bytes()).collect();
        let img = Blob::new(Media::Image, bytes).with_meta(json!({"w": 48, "h": 32, "c": 3}));
        let inv = Invocation::new().set("weights", json!(path.to_str().unwrap())).blob("image", img);
        let out = reg.run(MODEL, "detect", inv, &mut |_| {}).unwrap();
        assert!(out.outputs["count"].is_u64());
        assert!(out.outputs["detections"].is_array());
        let text = String::from_utf8(out.blobs["detections"].bytes.clone()).unwrap();
        assert!(serde_json::from_str::<Value>(&text).unwrap().is_array());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
