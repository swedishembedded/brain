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

/// One `infer` call's result: the min-max-normalized `[0,1]` inverse-depth
/// map on the frame's own grid, plus the raw bounds it was normalized from
/// (so the relative map stays recoverable) - [`Session::predict`]'s and
/// [`InferAction::run`]'s shared typed output, the same "CLI/D-Bus and SDK
/// call the same code" split `codeformer::caps::RestoreOutput`/`sam2::caps::
/// SegmentOutput` already established.
pub struct DepthOutput {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub min: f32,
    pub max: f32,
}

/// The shared core: build a transient `ParamStore`/[`Predictor`] from
/// `gpu`+`cfg`+`init`, run one forward pass, then min-max normalize - exactly
/// what [`InferAction::run`] did inline before this was factored out, called
/// from there (via `Hot`'s residency) and from [`Session::predict`] (via its
/// own owned state) alike.
fn predict_normalized(gpu: &Gpu, cfg: &ZipConfig, init: &HashMap<String, Vec<f32>>, hwc: &[f32], w: u32, h: u32) -> DepthOutput {
    let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(name, s)| (name, s.iter().product())).collect();
    let ps = paramstore::ParamStore::new(gpu, params, init);
    let predictor = Predictor::new(gpu, cfg.clone(), ps);
    let depth = predictor.predict(hwc, w, h);

    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &depth {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let range = (mx - mn).max(1e-6);
    let values: Vec<f32> = depth.iter().map(|&v| ((v - mn) / range).clamp(0.0, 1.0)).collect();
    DepthOutput { values, width: w, height: h, min: mn, max: mx }
}

/// A bound ZipDepth session: one resolved checkpoint, imported once, ready to
/// [`Session::predict`] on any number of frames - the shape `crates/sdk`'s
/// `DepthPipeline` needs (a builder resolves the checkpoint once via
/// `zipdepth::spec::ZipdepthSpec`, `load()` imports it, then every call is
/// just a forward pass), distinct from [`DepthProvider`]'s own `Hot`
/// residency (keyed by weights path, shared across many different
/// checkpoints over one long-lived D-Bus/CLI process).
pub struct Session {
    gpu: Gpu,
    init: HashMap<String, Vec<f32>>,
    cfg: ZipConfig,
}

impl Session {
    /// The checkpoint's own native model input (shorter side, already a
    /// multiple of 32) - the same value `input: 0` on the `infer` action
    /// resolves to.
    pub fn native_input(&self) -> u32 {
        self.cfg.input
    }

    /// Predict depth for an interleaved-RGB HWC frame in `[0,1]`, returning a
    /// `[h*w]` min-max-normalized inverse-depth map on the frame's own grid.
    /// `input`, when `Some` and non-zero, overrides the checkpoint's native
    /// model input side - the same knob the `infer` action's `input` param
    /// exposes (fully convolutional: any multiple-of-32 side is valid).
    pub fn predict(&self, hwc: &[f32], w: u32, h: u32, input: Option<u32>) -> DepthOutput {
        let mut cfg = self.cfg.clone();
        if let Some(n) = input {
            if n > 0 {
                cfg.input = n;
            }
        }
        predict_normalized(&self.gpu, &cfg, &self.init, hwc, w, h)
    }
}

/// Resolve `weights`' variant and import it into a fresh [`Session`] - the
/// same [`crate::import::cfg_for_checkpoint`] + [`crate::import::load`] pair
/// [`InferAction::run`]'s own `Hot` construction runs, on its own [`Gpu`].
pub fn load(weights: &str) -> Result<Session, String> {
    let cfg = crate::import::cfg_for_checkpoint(weights)?;
    let gpu = Gpu::new(crate::net::PIPELINES);
    let init = crate::import::load(weights, &cfg)?;
    Ok(Session { gpu, init, cfg })
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let infer = ActionSpec::new("infer", "dense relative inverse depth from a single image (ZipDepth)")
        .param(ParamSpec::new("weights", ParamType::Str, "path to a ZipDepth .pth checkpoint (variant auto-detected)").required().host_env("BRAIN_ZIPDEPTH_WEIGHTS"))
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
        let out = predict_normalized(&hot.gpu, &cfg, &hot.init, &hwc, w, h);

        let bytes: Vec<u8> = out.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(Outcome::new()
            .set("width", json!(out.width))
            .set("height", json!(out.height))
            .set("min", json!(out.min))
            .set("max", json!(out.max))
            .blob("depth", Blob::new(Media::Image, bytes).with_meta(json!({"w": out.width, "h": out.height, "c": 1, "min": out.min, "max": out.max}))))
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
