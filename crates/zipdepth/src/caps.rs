// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth's served actions: `infer` (depth from one image, and optionally
//! a view of it), `train` (from a checkpoint or from scratch) and `calib`
//! (the per-layer INT8 outlier report).
//!
//! `infer` is variant auto-detect ([`crate::import::cfg_for_checkpoint`]),
//! strict import, then [`crate::Predictor::predict`] (aspect-preserving
//! resize, forward, unwarp to the frame grid); a view is drawn by
//! [`crate::view::render`]. The engine and the imported host-RAM weight map
//! stay resident across calls (keyed by checkpoint path); the per-call
//! [`paramstore::ParamStore`] and [`crate::Predictor`] are transient device
//! state.
//!
//! Swedish Embedded AB implements monocular depth estimation on embedded
//! cameras for its clients. If your team needs expertise in depth models on
//! its own hardware then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;
use std::sync::Arc;

use capability::{
    hot::Hot, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;

use imaging::viz::Bounds;

use crate::view::{render, Style, View};
use crate::{Predictor, ZipConfig};

/// The model id the catalog serves ZipDepth under.
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

/// A transient [`Predictor`] over the resident host weights `init`.
fn predictor<'g>(gpu: &'g Gpu, cfg: &ZipConfig, init: &HashMap<String, Vec<f32>>) -> Predictor<'g> {
    let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(name, s)| (name, s.iter().product())).collect();
    Predictor::new(gpu, cfg.clone(), paramstore::ParamStore::new(gpu, params, init))
}

/// `depth` min-max normalized to `[0,1]`, with the bounds it came from.
fn normalized(depth: Vec<f32>, w: u32, h: u32) -> DepthOutput {
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
        normalized(predictor(&self.gpu, &cfg, &self.init).predict(hwc, w, h), w, h)
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

fn bench_param() -> ParamSpec {
    ParamSpec::new("bench", ParamType::Int, "after the first (cold) inference, run this many more and report the steady-state per-frame ms (min, median, mean)")
        .default(json!(0))
        .min(0.0)
        .max(MAX_BENCH)
        .host_resolved()
}

/// Most warm re-inferences one `infer` call may time.
const MAX_BENCH: f64 = 1_000.0;
/// Most stereogram pattern repeats across a frame.
const MAX_STRIPES: f64 = 64.0;
/// Most calibration images one `calib` call reads.
const MAX_CALIB_IMAGES: f64 = 100_000.0;

/// The names the `view` parameter takes: `none`, then every [`View`].
fn view_names() -> Vec<String> {
    std::iter::once("none").chain(View::ALL.iter().map(|v| v.name())).map(str::to_string).collect()
}

pub fn infer_spec() -> ActionSpec {
    ActionSpec::new("infer", "dense relative inverse depth from a single image, and optionally a view of it: the map colorized beside the photo, a depth effect or a stereogram").cancellable()
        .param(ParamSpec::new("input", ParamType::Int, "model input (shorter side, x32); 0 = the checkpoint's native 384. The net is fully convolutional: work scales with the square, so 256 is about 2.3x quicker and mildly softer").min(0.0).max(capability::bounds::SIZE_PX).default(json!(0)))
        .param(ParamSpec::new("view", ParamType::Enum(view_names()), "also render the depth as: side (photo | colorized depth), depth (colorized), fog or blur (depth effects on the photo), stereo (random-dot autostereogram), stereo-image (one patterned with the photo), stereo-dual (cross-eye pair); none = only the map").default(json!("none")))
        .param(ParamSpec::new("colormap", ParamType::Enum(vec!["turbo".into(), "gray".into(), "grayinv".into()]), "how the view colors depth").default(json!("turbo")))
        .param(ParamSpec::new("stripes", ParamType::Int, "stereogram pattern repeats across the frame; fewer = wider slices").default(json!(5)).min(1.0).max(MAX_STRIPES))
        .param(bench_param())
        .input(BlobSpec::new("image", Media::Image, "the image to estimate depth for").required())
        .output(BlobSpec::new("depth", Media::Image, "min-max-normalized inverse-depth map on the frame grid (single channel)"))
        .output(BlobSpec::new("view", Media::Image, "the depth shown as `view` asks (RGB); absent for view none"))
}

pub fn train_spec() -> ActionSpec {
    ActionSpec::new("train", "train ZipDepth on RGB to depth pairs (a paired directory, or synthetic scenes) with the MiDaS loss (scale-and-shift-invariant MSE plus multi-scale gradient matching on inverse depth), backward and AdamW: from the resolved checkpoint (fine-tune), or from freshly initialized weights when there is none or `fresh` asks").cancellable()
        .training()
        .streaming()
        .from_scratch()
        .param(ParamSpec::new("save", ParamType::Str, "where to write the trained checkpoint (.safetensors) on the serving machine").required().host_resolved())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(50)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("batch", ParamType::Int, "pairs per step").default(json!(2)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("width", ParamType::Int, "training width, px (a multiple of 32)").default(json!(64)).min(32.0).max(capability::bounds::SIZE_PX))
        .param(ParamSpec::new("height", ParamType::Int, "training height, px (a multiple of 32)").default(json!(64)).min(32.0).max(capability::bounds::SIZE_PX))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(1e-3)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("wd", ParamType::Float, "AdamW weight decay").default(json!(0.0)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("data", ParamType::Str, "directory of paired images on the serving machine (manifest.json and <stem>.png beside <stem>.depth.png); synthetic scenes when absent").host_resolved())
        .param(ParamSpec::new("grad_weight", ParamType::Float, "weight of the gradient-matching term of the MiDaS loss").default(json!(0.5)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the fresh weights, the synthetic pairs and the epoch shuffles (omit for random)").min(0.0))
        .param(ParamSpec::new("fresh", ParamType::Bool, "start from freshly initialized weights even when a checkpoint is resolved").default(json!(false)))
        .param(ParamSpec::new("activations", ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()), "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward for the same gradients at lower memory (default keep)"))
}

/// Most pairs one training step may hold.
const MAX_BATCH: f64 = 1_024.0;

pub fn calib_spec() -> ActionSpec {
    ActionSpec::new("calib", "per-layer activation outlier ratios (absmax / p99.99) over a folder of images, encoder against decoder: the data for choosing which layers keep full precision under INT8").cancellable()
        .param(ParamSpec::new("images", ParamType::Str, "folder of calibration images on the serving machine (read in name order; files that are not images are skipped)").required().host_resolved())
        .param(ParamSpec::new("max", ParamType::Int, "most images to read").default(json!(100)).min(1.0).max(MAX_CALIB_IMAGES))
        .param(ParamSpec::new("top", ParamType::Int, "layers listed, most INT8-hostile first").default(json!(20)).min(1.0).max(MAX_CALIB_IMAGES))
        .param(ParamSpec::new("input", ParamType::Int, "model input (shorter side, x32) the images are prepared at, as `infer` would; 0 = the checkpoint's native").min(0.0).max(capability::bounds::SIZE_PX).default(json!(0)))
}

/// The full, static capability manifest: safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "ZipDepth monocular depth: dense relative inverse depth from one image, its training and its INT8 calibration report.", vec![infer_spec(), train_spec(), calib_spec()])
}

/// The resident state: the engine plus the imported host-RAM weight map for one
/// checkpoint (the model's Hot footprint). Per call a transient `ParamStore` +
/// `Predictor` are materialised from it.
struct DepthModel {
    gpu: Gpu,
    init: HashMap<String, Vec<f32>>,
    cfg: ZipConfig,
}

impl DepthModel {
    fn load(weights: &str) -> Result<DepthModel, String> {
        let cfg = crate::import::cfg_for_checkpoint(weights)?;
        let gpu = Gpu::new(crate::net::PIPELINES);
        let init = crate::import::load(weights, &cfg)?;
        Ok(DepthModel { gpu, init, cfg })
    }
}

/// The executable ZipDepth model behind the manifest, over one checkpoint
/// (none for a model trained from scratch). Construction is free: the
/// checkpoint imports on the first call that needs it and stays resident.
#[derive(Default)]
pub struct DepthProvider {
    weights: Option<String>,
    hot: Arc<Hot<String, DepthModel>>,
}

impl DepthProvider {
    pub fn new(weights: Option<String>) -> DepthProvider {
        DepthProvider { weights, hot: Arc::default() }
    }
}

/// ZipDepth over `assembly`'s `weights` role; an assembly without one serves
/// only the actions that run from scratch.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(DepthProvider::new(assembly.role_path("weights").ok())))
}

impl Provider for DepthProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let (weights, hot) = (self.weights.clone(), self.hot.clone());
        match name {
            "infer" => Some(Arc::new(InferAction { weights, hot })),
            "train" => Some(Arc::new(TrainAction { weights })),
            "calib" => Some(Arc::new(CalibAction { weights, hot })),
            _ => None,
        }
    }
}

fn need(weights: &Option<String>, action: &str) -> Result<String, String> {
    weights.clone().ok_or_else(|| format!("zipdepth {action}: no ZipDepth checkpoint (pull one, or name it with --weights)"))
}

/// The checkpoint's config with an optional smaller model input.
fn with_input(cfg: &ZipConfig, inv: &Invocation) -> ZipConfig {
    let mut cfg = cfg.clone();
    if let Some(n) = inv.get_i64("input").filter(|&n| n > 0) {
        cfg.input = n as u32;
    }
    cfg
}

struct InferAction {
    weights: Option<String>,
    hot: Arc<Hot<String, DepthModel>>,
}

impl Action for InferAction {
    fn spec(&self) -> ActionSpec {
        infer_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = need(&self.weights, "infer")?;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let bench = inv.get_i64("bench").unwrap_or(0).max(0) as usize;
        self.hot.try_get(weights.clone(), || DepthModel::load(&weights), |hot| {
            let cfg = with_input(&hot.cfg, inv);
            let predictor = predictor(&hot.gpu, &cfg, &hot.init);
            let depth = predictor.predict(&hwc, w, h);
            let outcome = infer_outcome(inv, &hwc, w, h, depth)?;
            if bench == 0 {
                return Ok(outcome);
            }
            // The first predict paid the one-time costs (pipeline build,
            // packing); these repeats are the steady state a stream sees.
            let mut ms: Vec<f64> = Vec::new();
            for _ in 0..bench {
                inv.cancel.check()?;
                let t = std::time::Instant::now();
                let _ = predictor.predict(&hwc, w, h);
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ms.sort_by(f64::total_cmp);
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            Ok(outcome.set("bench_ms", json!({"frames": ms.len(), "min": ms[0], "median": ms[ms.len() / 2], "mean": mean})))
        })
    }
}

/// What `infer` returns for `depth`, the raw inverse depth of the `w x h`
/// RGB frame `hwc`: the normalized map with its bounds, and the view `inv`
/// asks for. Shared by every device an `infer` runs on.
pub fn infer_outcome(inv: &Invocation, hwc: &[f32], w: u32, h: u32, depth: Vec<f32>) -> Result<Outcome, String> {
    let view = match inv.get_str("view").as_deref() {
        None | Some("none") => None,
        Some(name) => Some(View::parse(name)?),
    };
    let mut outcome = Outcome::new();
    if let Some(view) = view {
        let colormap = imaging::viz::parse_colormap(&inv.get_str("colormap").unwrap_or_else(|| "turbo".into()))?;
        let stripes = inv.get_i64("stripes").unwrap_or(5).max(1) as u32;
        let rgb8: Vec<u8> = hwc.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        let style = Style { bounds: Bounds::from_percentiles(&depth, 0.02, 0.98), colormap, stereo: crate::StereoOpts::with_stripes(w, stripes) };
        let (cw, ch) = view.canvas(w, h);
        let pixels: Vec<f32> = render(&rgb8, &depth, w, h, view, &style).iter().map(|&b| b as f32 / 255.0).collect();
        outcome = outcome.blob("view", capability::blob::image_blob(&pixels, cw, ch, 3));
    }
    let out = normalized(depth, w, h);
    Ok(outcome
        .set("width", json!(out.width))
        .set("height", json!(out.height))
        .set("min", json!(out.min))
        .set("max", json!(out.max))
        .blob("depth", Blob::new(Media::Image, out.values.iter().flat_map(|v| v.to_le_bytes()).collect()).with_meta(json!({"w": out.width, "h": out.height, "c": 1, "min": out.min, "max": out.max}))))
}

struct TrainAction {
    weights: Option<String>,
}

impl Action for TrainAction {
    fn spec(&self) -> ActionSpec {
        train_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let save = inv.get_str("save").ok_or("zipdepth train: missing required param 'save'")?;
        let (w, h) = (inv.get_i64("width").unwrap_or(64) as u32, inv.get_i64("height").unwrap_or(64) as u32);
        if w % 32 != 0 || h % 32 != 0 {
            return Err(format!("zipdepth train: width and height must be multiples of 32 (got {w}x{h})"));
        }
        let saved = inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?;
        let seed = inv.get_i64("seed").map_or_else(data::rng::random_seed, |s| s as u64);
        let start = if inv.get_bool("fresh").unwrap_or(false) { None } else { self.weights.clone() };
        // Fine-tuning builds the checkpoint's own variant; a fresh model is the
        // base (unfold-upsampler) layout.
        let (cfg, init) = match &start {
            Some(path) => {
                let cfg = crate::import::cfg_for_checkpoint(path)?;
                let init = crate::import::load(path, &cfg)?;
                (cfg, init)
            }
            None => (ZipConfig::base(), crate::init_weights(&ZipConfig::base(), seed)),
        };
        let variant = if cfg.upsample_unfold { "base" } else { "npu" };
        let t = crate::train::TrainCfg {
            steps: inv.get_i64("steps").unwrap_or(50).max(1) as u32,
            batch: inv.get_i64("batch").unwrap_or(2).max(1) as u32,
            h,
            w,
            lr: inv.get_f64("lr").unwrap_or(1e-3) as f32,
            wd: inv.get_f64("wd").unwrap_or(0.0) as f32,
            seed,
            fixed_batch: false,
            grad_weight: inv.get_f64("grad_weight").map_or(crate::train::DEFAULT_GRAD_WEIGHT, |a| a as f32),
            saved,
        };
        // Paired images when `data` names a directory, synthetic scenes otherwise.
        let mut source: Box<dyn crate::train::BatchSource> = match inv.get_str("data") {
            Some(dir) => Box::new(crate::dataset::DepthDataset::open(std::path::Path::new(&dir), seed)?),
            None => Box::new(crate::train::Synthetic::new(seed)),
        };
        // The checkpoint keeps every tensor's own shape: the model store reads
        // the variant and its dimensions back from them.
        let shapes: HashMap<String, Vec<u64>> = cfg.param_list().into_iter().map(|(name, shape)| (name, shape.iter().map(|&d| d as u64).collect())).collect();
        let gpu = Gpu::new(crate::net::PIPELINES);
        let steps = t.steps;
        let (ps, res) = crate::train::train_loop(&gpu, cfg, &t, &init, source.as_mut(), |step, loss| {
            progress(Progress::step(step + 1, steps, format!("loss {loss:.4}")));
            !inv.cancel.is_cancelled()
        })?;
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = ps
            .params
            .iter()
            .map(|(name, _)| {
                let n = ps.numel(name);
                let shape = shapes.get(name).cloned().unwrap_or_else(|| vec![n as u64]);
                (name.clone(), shape, gpu.read(ps.w(name), n))
            })
            .collect();
        // The card's id is the one the model store files a ZipDepth checkpoint
        // under, so the saved file is found like a pulled one.
        let card = checkpoint::st::ModelCard::new(MODEL, "depth");
        checkpoint::st::save_safetensors(&save, &tensors, &json!({"model": "zipdepth", "variant": variant}), Some(&card)).map_err(|e| format!("zipdepth train: writing {save}: {e}"))?;
        Ok(Outcome::new()
            .set("first_loss", json!(res.first_loss))
            .set("last_loss", json!(res.last_loss))
            .set("seed", json!(seed))
            .set("from", json!(start.unwrap_or_else(|| "fresh weights".into())))
            .set("saved", json!(save)))
    }
}

struct CalibAction {
    weights: Option<String>,
    hot: Arc<Hot<String, DepthModel>>,
}

impl Action for CalibAction {
    fn spec(&self) -> ActionSpec {
        calib_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = need(&self.weights, "calib")?;
        let dir = inv.get_str("images").ok_or("zipdepth calib: missing required param 'images'")?;
        let max = inv.get_i64("max").unwrap_or(100).max(1) as usize;
        let top = inv.get_i64("top").unwrap_or(20).max(1) as usize;
        self.hot.try_get(weights.clone(), || DepthModel::load(&weights), |hot| {
            let cfg = with_input(&hot.cfg, inv);
            let images = calibration_images(&dir, cfg.input, max, &inv.cancel)?;
            progress(Progress::step(0, 1, format!("calibrating on {} images", images.len())));
            let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(name, s)| (name, s.iter().product())).collect();
            let ps = paramstore::ParamStore::new(&hot.gpu, params, &hot.init);
            let report = crate::collect_activation_stats_sized(&hot.gpu, &cfg, &ps, &images).report();
            let mean = |encoder: bool| {
                let ratios: Vec<f32> = report.iter().filter(|r| r.is_encoder() == encoder).map(|r| r.outlier_ratio).collect();
                if ratios.is_empty() { 0.0 } else { ratios.iter().sum::<f32>() / ratios.len() as f32 }
            };
            let (encoder, decoder) = (mean(true), mean(false));
            let verdict = if decoder > 1.5 * encoder.max(1e-3) {
                "the decoder's tails dominate: keeping the decoder in full precision is likely worth it"
            } else {
                "encoder and decoder tails are comparable: one INT8 policy for both may suffice"
            };
            let layers: Vec<_> = report.iter().take(top).map(|r| json!({"layer": r.name, "absmax": r.absmax, "p99_99": r.p9999, "ratio": r.outlier_ratio})).collect();
            Ok(Outcome::new()
                .set("images", json!(images.len()))
                .set("layers", json!(layers))
                .set("encoder_mean_ratio", json!(encoder))
                .set("decoder_mean_ratio", json!(decoder))
                .set("verdict", json!(verdict)))
        })
    }
}

/// Up to `max` images from `dir`, in name order, each prepared exactly as the
/// predictor prepares a frame (aspect-preserving resize so the shorter side is
/// `input`, both sides rounded to 32, no pad): INT8 scales fitted to any
/// other geometry describe activations the model never produces.
fn calibration_images(dir: &str, input: u32, max: usize, cancel: &capability::CancelToken) -> Result<Vec<crate::CalibImage>, String> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir).map_err(|e| format!("zipdepth calib: {dir}: {e}"))?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
    paths.sort();
    let mut out = Vec::new();
    for path in paths {
        if out.len() >= max {
            break;
        }
        cancel.check()?;
        // Not every file in a folder is an image; those are skipped.
        let Ok(img) = imaging::load(&path) else { continue };
        let hwc: Vec<f32> = img.px.iter().map(|&b| b as f32 / 255.0).collect();
        let (chw, h, w) = crate::predict::preprocess_chw(&hwc, img.w, img.h, input);
        out.push(crate::CalibImage { chw, h, w });
    }
    if out.is_empty() {
        return Err(format!("zipdepth calib: no images in {dir}"));
    }
    Ok(out)
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    fn tiny_image() -> Blob {
        Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}))
    }

    #[test]
    fn manifest_declares_infer_train_and_calib() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["infer", "train", "calib"]);
        let infer = &m.actions[0];
        assert!(!infer.streaming, "infer is one-shot");
        // validation without weights: defaults fill, a missing image is refused
        let inv = infer.validate(Invocation::new().blob("image", tiny_image())).unwrap();
        assert_eq!(inv.get_i64("input"), Some(0));
        assert_eq!(inv.get_str("view").as_deref(), Some("none"));
        assert!(infer.validate(Invocation::new()).is_err());
        assert!(infer.validate(Invocation::new().blob("image", tiny_image()).set("view", json!("sideways"))).is_err());
        let train = &m.actions[1];
        assert!(train.training && train.streaming && train.from_scratch);
        assert!(!m.actions[2].from_scratch, "calibration measures a checkpoint");
    }

    /// With no checkpoint resolved, inference and calibration refuse by
    /// name, and a missing file is an error rather than a panic.
    #[test]
    fn a_missing_checkpoint_is_a_clean_error() {
        let run = |weights: Option<&str>| {
            let mut reg = Registry::new();
            reg.register(Arc::new(DepthProvider::new(weights.map(str::to_string))));
            reg.run(MODEL, "infer", Invocation::new().blob("image", tiny_image()), &mut |_| {}).unwrap_err()
        };
        assert!(run(None).contains("no ZipDepth checkpoint"), "{}", run(None));
        assert!(!run(Some("/nonexistent/zipdepth.pth")).is_empty());
    }

    #[test]
    fn training_refuses_a_size_the_net_cannot_take() {
        let inv = train_spec().validate(Invocation::new().set("save", json!(std::env::temp_dir().join("x.safetensors").to_string_lossy())).set("width", json!(50))).unwrap();
        let err = TrainAction { weights: None }.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("multiples of 32"), "{err}");
    }

    #[test]
    fn calibration_refuses_a_folder_without_images() {
        let dir = std::env::temp_dir().join(format!("zipdepth-calib-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"not an image").unwrap();
        let err = calibration_images(dir.to_str().unwrap(), 384, 10, &capability::CancelToken::default()).err().expect("refused");
        assert!(err.contains("no images"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
