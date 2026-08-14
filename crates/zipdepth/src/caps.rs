// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth's capabilities behind the generalized [`capability`] interface —
//! what makes `brain caps depth` / `brain do depth infer …` (and the perf
//! suite's `CapabilityTarget`) work with no depth-specific plumbing in the CLI.
//!
//! One action, `infer`: the same single-image path `brain depth --image` runs —
//! variant auto-detect ([`crate::import::cfg_for_checkpoint`]), strict import,
//! then [`crate::Predictor::predict`] (aspect-preserving resize → forward →
//! unwarp to the frame grid). One-shot: the depth map is the single artifact.
//!
//! Residency follows the `DepthResident` pattern: the engine + the imported
//! host-RAM weight map stay resident across calls (keyed by weights path); the
//! per-call [`paramstore::ParamStore`] / [`crate::Predictor`] are transient
//! device state.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;

use crate::{Predictor, ZipConfig};

/// The model id used on the CLI (`brain do depth …`) and the event API.
pub const MODEL: &str = "brain/zipdepth";

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let infer = ActionSpec::new("infer", "dense relative inverse depth from a single image (ZipDepth)")
        .param(ParamSpec::new("weights", ParamType::Str, "path to a ZipDepth .pth checkpoint (variant auto-detected)").required())
        .param(ParamSpec::new("input", ParamType::Int, "model input (shorter side, x32); 0 = the checkpoint's native 384").default(json!(0)))
        .input(BlobSpec::new("image", Media::Image, "the image to estimate depth for").required())
        .output(BlobSpec::new("depth", Media::Image, "min-max-normalized inverse-depth map on the frame grid (single channel)"));
    Manifest::new(MODEL, "ZipDepth monocular depth — dense relative inverse-depth from one image.", vec![infer])
}

/// The resident state: the engine plus the imported host-RAM weight map for one
/// checkpoint (the model's Hot footprint). Per call a transient `ParamStore` +
/// `Predictor` are materialised from it — the `DepthResident` pattern.
struct Hot {
    weights: String,
    gpu: Gpu,
    init: HashMap<String, Vec<f32>>,
    cfg: ZipConfig,
}

/// The executable ZipDepth model behind the manifest. Construction is free —
/// the checkpoint imports lazily on the first `infer` and stays resident.
#[derive(Default)]
pub struct DepthProvider {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl DepthProvider {
    pub fn new() -> DepthProvider {
        DepthProvider::default()
    }
}

impl Provider for DepthProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "infer").then(|| Arc::new(InferAction { hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct InferAction {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl Action for InferAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "infer").expect("known action")
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = inv.get_str("weights").ok_or("depth infer: missing required param 'weights'")?;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        // Hot path: engine + imported weight map resident per checkpoint path.
        let mut guard = self.hot.lock().map_err(|_| "depth: hot model lock poisoned")?;
        if !matches!(&*guard, Some(hot) if hot.weights == weights) {
            *guard = None; // free the old resident weights before importing new
            let cfg = crate::import::cfg_for_checkpoint(&weights)?;
            let gpu = Gpu::new(crate::net::PIPELINES);
            let init = crate::import::load(&weights, &cfg)?;
            *guard = Some(Hot { weights: weights.clone(), gpu, init, cfg });
        }
        let hot = guard.as_ref().unwrap();

        // Optional smaller input (fully convolutional: any x32 side is valid).
        let mut cfg = hot.cfg.clone();
        if let Some(n) = inv.get_i64("input") {
            if n > 0 {
                cfg.input = n as u32;
            }
        }

        // Transient device state from the resident host weights, then the same
        // reference pipeline `brain depth --image` runs.
        let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(name, s)| (name, s.iter().product())).collect();
        let ps = paramstore::ParamStore::new(&hot.gpu, params, &hot.init);
        let predictor = Predictor::new(&hot.gpu, cfg, ps);
        let depth = predictor.predict(&hwc, w, h);

        // Min-max normalize to [0,1] for the blob; report the raw bounds so the
        // relative map stays recoverable.
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &depth {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let range = (mx - mn).max(1e-6);
        let bytes: Vec<u8> = depth.iter().flat_map(|&v| ((v - mn) / range).clamp(0.0, 1.0).to_le_bytes()).collect();
        Ok(Outcome::new()
            .set("width", json!(w))
            .set("height", json!(h))
            .set("min", json!(mn))
            .set("max", json!(mx))
            .blob("depth", Blob::new(Media::Image, bytes).with_meta(json!({"w": w, "h": h, "c": 1, "min": mn, "max": mx}))))
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_infer() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "infer");
        assert!(!a.streaming, "infer is one-shot (the depth map is the single artifact)");
        assert!(a.params.iter().any(|p| p.name == "weights" && p.required));
        assert_eq!(a.params.iter().find(|p| p.name == "input").unwrap().default, Some(json!(0)));
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert_eq!(a.outputs[0].name, "depth");
        // validation without weights: defaults fill, missing image rejected.
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = a.validate(Invocation::new().set("weights", json!("w")).blob("image", img)).unwrap();
        assert_eq!(inv.get_i64("input"), Some(0));
        assert!(a.validate(Invocation::new().set("weights", json!("w"))).is_err());
        assert_eq!(manifest().to_json()["actions"][0]["name"], "infer");
    }

    /// The released ZipDepth `.pth` is not on every box: a missing checkpoint
    /// must surface as a clean `ActionResult` error, not a panic.
    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(DepthProvider::new()));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let err = reg
            .run(MODEL, "infer", Invocation::new().set("weights", json!("/nonexistent/zipdepth.pth")).blob("image", img), &mut |_| {})
            .unwrap_err();
        assert!(!err.is_empty(), "expected a descriptive error, got: {err}");
    }
}
