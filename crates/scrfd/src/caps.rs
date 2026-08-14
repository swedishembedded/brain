// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Face detection behind the generalized [`capability`] interface - what makes
//! `brain caps scrfd` / `brain do scrfd detect …`, the D-Bus `Run` method and
//! `brain perf`'s `CapabilityTarget` work with no face-specific plumbing in the
//! CLI or the transports.
//!
//! One action:
//!
//! * **`detect`** - image in, SCRFD boxes + scores + 5 landmarks out, in
//!   **source-image pixels** (the detector's own letterbox is undone by
//!   `detect::decode`'s `det_scale`, before NMS, exactly as insightface does).
//!
//! Self-sufficient: this half of the antelopev2 stack needs only
//! `scrfd_10g_bnkps.onnx` and nothing from the identity embedder. (The reverse
//! is not true - the embedder's default aligned path detects first, and depends
//! on this crate to do it.)
//!
//! # One `Preprocess`, two unit systems
//!
//! [`crate::blob_from_bgr_u8`] is the OpenCV-shaped entry point the parity test
//! replays: interleaved **BGR u8**, `(x - mean)/std` with `mean/std` in 0..255
//! units. Over the wire brain carries **RGB f32 in `[0,1]`**
//! (`capability::blob`). [`unit_norm`] re-expresses the SAME constants in those
//! units - `(255·x − mean)/std = (x − mean/255)/(std/255)` - so there is one
//! pair of constants ([`Preprocess`]), one normalise kernel (`film_chan`, via
//! `imaging::Ctx`), and no second copy of the arithmetic.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Border, Ctx, Filter, Shape};
use serde_json::json;

use crate::config::{Preprocess, ScrfdConfig};
use crate::detect::Face;
use crate::model::Scrfd;

/// The model id used on the CLI (`brain do scrfd …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/scrfd";

/// [`crate::model::PIPELINES`] plus the two kernels only the *serving* path
/// dispatches: the detector letterbox (`resize_bilinear` for the aspect-
/// preserving resize, `pad2d` for the zero canvas insightface pads with).
///
/// Appended, never reordered: `crate::model::kernel` and
/// `vision::ConvKernelIds::resolve` index [`crate::model::PIPELINES`]
/// positionally, so extending it at the tail keeps ONE kernel index space and
/// leaves every existing index valid. The test below pins that invariant.
///
/// A `const` array (not a `Vec`) so it is `'static`: `gpu_core::testgpu::dev`
/// and the weak device pool key on a `&'static` kernel set.
const N_MODEL: usize = crate::model::PIPELINES.len();
pub const SERVING_PIPELINES: [(&str, &str); N_MODEL + 2] = serving_set();

const fn serving_set() -> [(&'static str, &'static str); N_MODEL + 2] {
    let mut k = [("", ""); N_MODEL + 2];
    let mut i = 0;
    while i < N_MODEL {
        k[i] = crate::model::PIPELINES[i];
        i += 1;
    }
    k[N_MODEL] = ("resize_bilinear", kernels::RESIZE_BILINEAR);
    k[N_MODEL + 1] = ("pad2d", kernels::PAD2D);
    k
}

/// A model's `(x − mean)/std` re-expressed for RGB f32 input in `[0,1]`.
fn unit_norm(pre: &Preprocess) -> imaging::Normalization {
    imaging::Normalization { mean: [pre.mean / 255.0; 3], std: [pre.std / 255.0; 3] }
}

pub fn detect_spec() -> ActionSpec {
    ActionSpec::new("detect", "detect faces: boxes, scores and 5-point landmarks (SCRFD-10GF)")
        .param(ParamSpec::new("max_faces", ParamType::Int, "keep at most N faces, highest score first (0 = all)").default(json!(0)))
        .input(BlobSpec::new("image", Media::Image, "the image to detect faces in").required())
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Face detection (insightface antelopev2 SCRFD-10GF): boxes, scores and 5-point landmarks.",
        vec![detect_spec()],
    )
}

// ===================== the shared work =====================

/// The detector on one device - the single implementation of `detect`, shared by
/// the [`ScrfdProvider`], the CLI's residency adapter, and the ArcFace embed
/// action's aligned path (which owns one of these to find the face first).
pub struct ScrfdSession {
    gpu: Gpu,
    scrfd: Scrfd,
    cfg: ScrfdConfig,
}

impl ScrfdSession {
    /// Import `scrfd_10g_bnkps.onnx` from `dir` and build it on `gpu`.
    ///
    /// `gpu` is passed in rather than built here: a process holds ONE device
    /// (`AGENTS.md`), and a caller that also runs another model shares it
    /// (`Gpu::share` for this same kernel set, `Gpu::new_like` for a different
    /// one on the same device). `dir` comes from a CLI flag or `BRAIN_SCRFD_DIR`,
    /// never a baked-in path.
    pub fn load(dir: &str, gpu: Gpu) -> Result<ScrfdSession, String> {
        let w = crate::import::import_dir(std::path::Path::new(dir))?;
        let cfg = ScrfdConfig::scrfd_10g_bnkps();
        let scrfd = Scrfd::new(gpu.share(), cfg.clone(), &w);
        Ok(ScrfdSession { gpu, scrfd, cfg })
    }

    /// Decode the invocation's image to source-resolution CHW RGB in `[0,1]`.
    fn source_chw(inv: &Invocation) -> Result<(Vec<f32>, u32, u32), String> {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        Ok((imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize), w, h))
    }

    /// insightface's detector letterbox: aspect-preserving resize into the
    /// TOP-LEFT of a zero `size × size` canvas, `det_scale = new_h / h`.
    ///
    /// Not `imaging::letterbox_rgb`: that one CENTRES the content (and is a host
    /// nearest-neighbour resampler tuned to YOLO's convention). insightface pads
    /// bottom/right only and resizes bilinearly, and the pad position moves every
    /// box - so this is the reference's geometry, built from the device
    /// `resize_bilinear` + `pad2d` kernels rather than a host loop.
    fn det_canvas(&self, chw: &[f32], w: u32, h: u32) -> (Vec<f32>, f32) {
        let size = self.cfg.image_size;
        let im_ratio = h as f32 / w as f32;
        // model_ratio is 1.0 (the detector canvas is square).
        let (new_w, new_h) = if im_ratio > 1.0 {
            (((size as f32 / im_ratio) as u32).max(1), size)
        } else {
            (size, ((size as f32 * im_ratio) as u32).max(1))
        };
        let det_scale = new_h as f32 / h as f32;

        let ctx = Ctx::new(&self.gpu);
        let src = ctx.upload("scrfd.caps.src", chw);
        let (small, sshape) = ctx.resize(&src, Shape::new(1, 3, h, w), new_h, new_w, Filter::Bilinear, AlignCorners::HalfPixel);
        let border = Border { left: 0, top: 0, right: size - new_w, bottom: size - new_h };
        let (canvas, cshape) = ctx.pad_zero(&small, sshape, border);
        let norm = ctx.normalize(&canvas, cshape, &unit_norm(&self.cfg.pre));
        (ctx.download(&norm, cshape.numel()), det_scale)
    }

    /// Every face in the image, in source-image pixels, highest score first.
    pub fn faces(&self, chw: &[f32], w: u32, h: u32) -> Vec<Face> {
        let (blob, det_scale) = self.det_canvas(chw, w, h);
        let t = self.scrfd.forward(&blob);
        let mut faces = crate::detect::decode(&self.cfg, &t.out_score, &t.out_bbox, &t.out_kps, det_scale);
        faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        faces
    }

    /// Run one `detect` invocation.
    pub fn detect(&self, inv: &Invocation) -> ActionResult {
        let (chw, w, h) = ScrfdSession::source_chw(inv)?;
        let mut faces = self.faces(&chw, w, h);
        let limit = inv.get_i64("max_faces").unwrap_or(0);
        if limit > 0 && faces.len() > limit as usize {
            faces.truncate(limit as usize);
        }
        let list: Vec<serde_json::Value> = faces
            .iter()
            .map(|f| json!({
                "bbox": f.bbox,
                "score": f.score,
                "kps": f.kps.iter().map(|p| json!([p[0], p[1]])).collect::<Vec<_>>(),
            }))
            .collect();
        Ok(Outcome::new().set("width", json!(w)).set("height", json!(h)).set("count", json!(list.len())).set("faces", json!(list)))
    }

    /// Dispatch by action name - the seam the residency `Instance` uses.
    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "detect" => self.detect(inv),
            other => Err(format!("scrfd: unknown action '{other}'")),
        }
    }
}

// ===================== the provider =====================

/// The executable detector behind the manifest. Construction is free - the ONNX
/// graph imports lazily on the first call and stays resident.
pub struct ScrfdProvider {
    dir: String,
    hot: Arc<Mutex<Option<(String, ScrfdSession)>>>,
}

impl ScrfdProvider {
    /// `dir` holds `scrfd_10g_bnkps.onnx` (the antelopev2 release layout).
    pub fn new(dir: impl Into<String>) -> ScrfdProvider {
        ScrfdProvider { dir: dir.into(), hot: Arc::new(Mutex::new(None)) }
    }
    /// The released file this detector needs, by its antelopev2 name.
    pub const RELEASE_FILES: [&str; 1] = [crate::import::RELEASE_FILE];

    /// `BRAIN_SCRFD_DIR` - `None` when unset or when the directory does not hold
    /// the released graph.
    pub fn from_env() -> Option<ScrfdProvider> {
        let dir = std::env::var("BRAIN_SCRFD_DIR").ok().filter(|p| !p.is_empty())?;
        let d = std::path::Path::new(&dir);
        ScrfdProvider::RELEASE_FILES.iter().all(|f| d.join(f).exists()).then(|| ScrfdProvider::new(dir))
    }
}

impl Provider for ScrfdProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "detect")
            .then(|| Arc::new(DetectAction { dir: self.dir.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct DetectAction {
    dir: String,
    hot: Arc<Mutex<Option<(String, ScrfdSession)>>>,
}

impl Action for DetectAction {
    fn spec(&self) -> ActionSpec {
        detect_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut guard = self.hot.lock().map_err(|_| "scrfd: hot model lock poisoned")?;
        if !matches!(&*guard, Some((d, _)) if *d == self.dir) {
            *guard = None; // free the old build before importing another directory
            let gpu = Gpu::new(&SERVING_PIPELINES);
            *guard = Some((self.dir.clone(), ScrfdSession::load(&self.dir, gpu)?));
        }
        guard.as_ref().expect("built above").1.run("detect", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::{Blob, Registry};

    #[test]
    fn manifest_declares_detect_only() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1, "the detector serves exactly one action");
        let d = m.actions.iter().find(|a| a.name == "detect").expect("detect");
        assert!(d.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        // defaults fill
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = d.validate(Invocation::new().blob("image", img)).unwrap();
        assert_eq!(inv.get_i64("max_faces"), Some(0));
        assert!(d.validate(Invocation::new()).is_err(), "the image is required");
    }

    /// The serving pipelines must EXTEND the model's list, never reorder it:
    /// `crate::model::kernel` returns positional indices into `PIPELINES`.
    #[test]
    fn serving_pipelines_only_append() {
        let base = crate::model::PIPELINES;
        let ext = SERVING_PIPELINES;
        assert_eq!(&ext[..base.len()], base, "the shared prefix must stay identical");
        for (name, _) in &ext[base.len()..] {
            assert!(!base.iter().any(|(n, _)| n == name), "{name} is already registered - appending it would duplicate a pipeline");
        }
        let mut names: Vec<&str> = ext.iter().map(|(n, _)| *n).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate kernel name in the serving pipeline list");
    }

    /// `(255·x − mean)/std` and `(x − mean/255)/(std/255)` must agree.
    #[test]
    fn unit_norm_is_the_same_affine_in_zero_to_one_units() {
        let pre = ScrfdConfig::scrfd_10g_bnkps().pre;
        let n = unit_norm(&pre);
        for u8v in [0.0f32, 37.0, 255.0] {
            let want = (u8v - pre.mean) / pre.std;
            let got = (u8v / 255.0 - n.mean[0]) / n.std[0];
            assert!((want - got).abs() < 1e-4, "{want} vs {got} for {u8v}");
        }
    }

    /// The antelopev2 directory is not on every box: a missing one must surface
    /// as a clean `ActionResult` error, not a panic.
    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(ScrfdProvider::new("/nonexistent/antelopev2")));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let err = reg.run(MODEL, "detect", Invocation::new().blob("image", img), &mut |_| {}).unwrap_err();
        assert!(!err.is_empty(), "expected a descriptive error, got: {err}");
    }
}
