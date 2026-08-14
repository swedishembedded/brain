// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Face identity embedding behind the generalized [`capability`] interface -
//! what makes `brain caps arcface` / `brain do arcface embed …`, the D-Bus `Run`
//! method and `brain perf`'s `CapabilityTarget` work with no face-specific
//! plumbing in the CLI or the transports.
//!
//! One action:
//!
//! * **`embed`** - image in, one L2-normalised 512-d ArcFace vector out. With
//!   `align = true` (the default) the primary face is detected, similarity-
//!   aligned to the 112² template and embedded; with `align = false` the input
//!   is taken to be an already-aligned face and only resized.
//!
//! # The detector, and how it shares the device
//!
//! `align = true` cannot align a face it has not found, so this session owns a
//! `scrfd::caps::ScrfdSession`. The two models run on ONE device (`AGENTS.md`:
//! one GPU device per process) with **different kernel sets** - the detector
//! registers no PReLU and no margin head, this crate registers no pooling and no
//! nearest-resize - which is exactly what `Gpu::new_like` is for: a second
//! kernel set on the same adapter and queue. Sharing the handle itself
//! (`Gpu::share`) is not an option across two kernel lists: each crate resolves
//! its kernel indices against the list IT registered, so a model built on the
//! other's handle would bind the wrong pipelines and be silently wrong.
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
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Ctx, Filter, Shape};
use scrfd::caps::ScrfdSession;
use scrfd::Face;
use serde_json::json;

use crate::config::{ArcFaceConfig, Preprocess};
use crate::model::ArcFace;

/// The model id used on the CLI (`brain do arcface …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/arcface";

/// [`crate::model::PIPELINES`] plus the one kernel only the *serving* path
/// dispatches: `resize_bilinear`, for the `align = false` path that takes the
/// input to be an aligned face and only resizes it to 112².
///
/// Appended, never reordered: `crate::model::kernel` and
/// `vision::ConvKernelIds::resolve` index [`crate::model::PIPELINES`]
/// positionally, so extending it at the tail keeps ONE kernel index space and
/// leaves every existing index valid. The test below pins that invariant.
///
/// A `const` array (not a `Vec`) so it is `'static`: `gpu_core::testgpu::dev`
/// and the weak device pool key on a `&'static` kernel set.
const N_MODEL: usize = crate::model::PIPELINES.len();
pub const SERVING_PIPELINES: [(&str, &str); N_MODEL + 1] = serving_set();

const fn serving_set() -> [(&'static str, &'static str); N_MODEL + 1] {
    let mut k = [("", ""); N_MODEL + 1];
    let mut i = 0;
    while i < N_MODEL {
        k[i] = crate::model::PIPELINES[i];
        i += 1;
    }
    k[N_MODEL] = ("resize_bilinear", kernels::RESIZE_BILINEAR);
    k
}

/// The detector graph, by its antelopev2 name - read from the embedder's own
/// directory for the `align = true` path. Named by the detector crate, so a
/// release that renames the file cannot leave the two spellings disagreeing.
const DETECTOR_FILE: &str = scrfd::caps::ScrfdProvider::RELEASE_FILES[0];

/// A model's `(x − mean)/std` re-expressed for RGB f32 input in `[0,1]`.
fn unit_norm(pre: &Preprocess) -> imaging::Normalization {
    imaging::Normalization { mean: [pre.mean / 255.0; 3], std: [pre.std / 255.0; 3] }
}

pub fn embed_spec() -> ActionSpec {
    ActionSpec::new("embed", "512-d ArcFace identity embedding of the primary face (L2-normalised)")
        .param(ParamSpec::new(
            "align",
            ParamType::Bool,
            "detect + 5-point similarity-align the primary face first; false = the input is already an aligned 112x112 face",
        ).default(json!(true)))
        .param(ParamSpec::new(
            "select",
            ParamType::Enum(vec!["largest".into(), "score".into()]),
            "which detected face to embed when several are present",
        ).default(json!("largest")))
        .input(BlobSpec::new("image", Media::Image, "the photo (align = true) or the aligned face (align = false)").required())
        .output(BlobSpec::new("embedding", Media::Bytes, "512 f32 little-endian, L2-normalised (cosine-ready)"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Face identity embedding (insightface antelopev2): ArcFace IResNet-100 512-d vectors.",
        vec![embed_spec()],
    )
}

// ===================== the shared work =====================

/// The embedder, plus the detector its default path aligns with - the single
/// implementation of `embed`, shared by the [`ArcFaceProvider`] and the CLI's
/// residency adapter.
pub struct ArcFaceSession {
    gpu: Gpu,
    arcface: ArcFace,
    cfg: ArcFaceConfig,
    /// The detector for `align = true`. `None` when `dir` holds no
    /// `scrfd_10g_bnkps.onnx`: `align = false` still works on a pre-aligned
    /// crop, so a missing detector is a per-request error rather than a refusal
    /// to serve the model at all.
    det: Option<ScrfdSession>,
}

impl ArcFaceSession {
    /// Import `glintr100.onnx` from `dir` and build it on `gpu`, plus the
    /// detector from the same directory when it is there (the antelopev2 release
    /// ships both files together).
    ///
    /// `gpu` is passed in rather than built here: a process holds ONE device
    /// (`AGENTS.md`). The detector gets `Gpu::new_like` - the same device, its
    /// own kernel set. `dir` comes from a CLI flag or `BRAIN_ARCFACE_DIR`, never
    /// a baked-in path.
    pub fn load(dir: &str, gpu: Gpu) -> Result<ArcFaceSession, String> {
        let w = crate::import::import_dir(std::path::Path::new(dir))?;
        let cfg = ArcFaceConfig::iresnet100();
        let arcface = ArcFace::new(gpu.share(), cfg.clone(), &w);
        let det = if std::path::Path::new(dir).join(DETECTOR_FILE).exists() {
            Some(ScrfdSession::load(dir, gpu.new_like(&scrfd::caps::SERVING_PIPELINES))?)
        } else {
            None
        };
        Ok(ArcFaceSession { gpu, arcface, cfg, det })
    }

    /// Decode the invocation's image to source-resolution CHW RGB in `[0,1]`.
    fn source_chw(inv: &Invocation) -> Result<(Vec<f32>, u32, u32), String> {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        Ok((imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize), w, h))
    }

    /// The RAW ArcFace embedding for one image, plus the face it was taken
    /// from - the shared core of the `embed` action and of PuLID's ID
    /// conditioning.
    ///
    /// **Raw, deliberately.** The released graph has no L2 norm (`‖e‖ ≈ 15–20`),
    /// and the two consumers want different things: the `embed` action
    /// normalises so its output is cosine-ready, while PuLID concatenates the
    /// UN-normalised vector (`pipeline_flux.py` uses insightface's
    /// `face_info['embedding']`, not `normed_embedding`, and only the EVA-CLIP
    /// half of `id_cond` is divided by its norm). Normalising here would make
    /// PuLID's first 512 components ~20x too small with nothing to catch it -
    /// the dumped golden's `‖id_cond[:512]‖` is 20.11 against `‖id_cond[512:]‖`
    /// of exactly 1.0.
    pub fn embed_raw_chw(
        &self,
        chw: &[f32],
        w: u32,
        h: u32,
        align: bool,
        select_largest: bool,
    ) -> Result<(Vec<f32>, Option<Face>), String> {
        let side = self.cfg.image_size;
        let ctx = Ctx::new(&self.gpu);

        let (aligned, face) = if align {
            let det = self.det.as_ref().ok_or(
                "arcface embed: align=true needs the SCRFD detector next to the embedder \
                 (scrfd_10g_bnkps.onnx in the same directory); pass align=false for an \
                 already-aligned 112x112 crop",
            )?;
            let faces = det.faces(chw, w, h);
            if faces.is_empty() {
                return Err("arcface embed: no face detected (pass align=false for an already-aligned 112x112 crop)".into());
            }
            // `faces` is score-sorted; "largest" re-picks by box area, which is
            // the reference recipe's primary-face rule (and PuLID's).
            let f = if select_largest {
                *faces.iter().max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap_or(std::cmp::Ordering::Equal)).expect("non-empty")
            } else {
                faces[0]
            };
            let lmk: Vec<f32> = f.kps.iter().flat_map(|p| [p[0], p[1]]).collect();
            // Umeyama solve on the host + the `grid_sample` kernel - the crate's
            // one alignment implementation.
            let (crop, _m) = crate::align::norm_crop_chw(&self.gpu, chw, 3, h, w, &lmk, side)?;
            (crop, Some(f))
        } else {
            let src = ctx.upload("arcface.caps.face", chw);
            let (small, sshape) = ctx.resize(&src, Shape::new(1, 3, h, w), side, side, Filter::Bilinear, AlignCorners::HalfPixel);
            (ctx.download(&small, sshape.numel()), None)
        };

        let shape = Shape::new(1, 3, side, side);
        let up = ctx.upload("arcface.caps.aligned", &aligned);
        let blob = ctx.normalize(&up, shape, &unit_norm(&self.cfg.pre));
        Ok((self.arcface.embed_blob(&ctx.download(&blob, shape.numel())), face))
    }

    /// Run one `embed` invocation.
    pub fn embed(&self, inv: &Invocation) -> ActionResult {
        let (chw, w, h) = ArcFaceSession::source_chw(inv)?;
        let align = inv.get_bool("align").unwrap_or(true);
        let largest = !matches!(inv.get_str("select").as_deref(), Some("score"));
        let (raw, face) = self.embed_raw_chw(&chw, w, h, align, largest)?;
        // Consumers normalise for cosine, and `model::hostmath` is the one place
        // that lives.
        let e = model::hostmath::l2_normalize(&raw);
        let bytes: Vec<u8> = e.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut out = Outcome::new().set("dim", json!(e.len()));
        if let Some(f) = face {
            out = out.set("bbox", json!(f.bbox)).set("score", json!(f.score));
        }
        Ok(out.blob("embedding", Blob::new(Media::Bytes, bytes).with_meta(json!({"dim": e.len(), "dtype": "f32"}))))
    }

    /// Dispatch by action name - the seam the residency `Instance` uses.
    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "embed" => self.embed(inv),
            other => Err(format!("arcface: unknown action '{other}'")),
        }
    }
}

// ===================== the provider =====================

/// The executable embedder behind the manifest. Construction is free - the ONNX
/// graphs import lazily on the first call and stay resident.
pub struct ArcFaceProvider {
    dir: String,
    hot: Arc<Mutex<Option<(String, ArcFaceSession)>>>,
}

impl ArcFaceProvider {
    /// `dir` holds `glintr100.onnx` (the antelopev2 release layout), and - for
    /// the default aligned path - `scrfd_10g_bnkps.onnx` beside it.
    pub fn new(dir: impl Into<String>) -> ArcFaceProvider {
        ArcFaceProvider { dir: dir.into(), hot: Arc::new(Mutex::new(None)) }
    }
    /// The released file this embedder needs. The detector's graph is NOT in
    /// this list: it is needed by `align = true` only, so a directory without it
    /// still serves `align = false` and the action reports the difference.
    pub const RELEASE_FILES: [&str; 1] = [crate::import::RELEASE_FILE];

    /// `BRAIN_ARCFACE_DIR` - `None` when unset or when the directory does not
    /// hold the released graph.
    pub fn from_env() -> Option<ArcFaceProvider> {
        let dir = std::env::var("BRAIN_ARCFACE_DIR").ok().filter(|p| !p.is_empty())?;
        let d = std::path::Path::new(&dir);
        ArcFaceProvider::RELEASE_FILES.iter().all(|f| d.join(f).exists()).then(|| ArcFaceProvider::new(dir))
    }
}

impl Provider for ArcFaceProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "embed")
            .then(|| Arc::new(EmbedAction { dir: self.dir.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct EmbedAction {
    dir: String,
    hot: Arc<Mutex<Option<(String, ArcFaceSession)>>>,
}

impl Action for EmbedAction {
    fn spec(&self) -> ActionSpec {
        embed_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut guard = self.hot.lock().map_err(|_| "arcface: hot model lock poisoned")?;
        if !matches!(&*guard, Some((d, _)) if *d == self.dir) {
            *guard = None; // free the old build before importing another directory
            let gpu = Gpu::new(&SERVING_PIPELINES);
            *guard = Some((self.dir.clone(), ArcFaceSession::load(&self.dir, gpu)?));
        }
        guard.as_ref().expect("built above").1.run("embed", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_embed_only() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1, "the embedder serves exactly one action");
        let e = m.actions.iter().find(|a| a.name == "embed").expect("embed");
        assert_eq!(e.outputs[0].name, "embedding");
        assert_eq!(e.outputs[0].media, Media::Bytes);
        // defaults fill
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = e.validate(Invocation::new().blob("image", img)).unwrap();
        assert_eq!(inv.get_bool("align"), Some(true));
        assert_eq!(inv.get_str("select").as_deref(), Some("largest"));
        assert!(e.validate(Invocation::new()).is_err(), "the image is required");
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

    /// The two models' kernel sets are DIFFERENT lists, which is why the
    /// detector is built with `Gpu::new_like` rather than `Gpu::share`. If they
    /// ever became the same list this test would say so.
    #[test]
    fn the_detectors_kernel_set_is_a_different_list() {
        let mine: Vec<&str> = SERVING_PIPELINES.iter().map(|(n, _)| *n).collect();
        let theirs: Vec<&str> = scrfd::caps::SERVING_PIPELINES.iter().map(|(n, _)| *n).collect();
        assert_ne!(mine, theirs, "same list => share() would do, and new_like is then misleading");
        // ... and every shared name sits at a different index in at least one of
        // them, which is exactly what makes a shared handle unsafe here.
        assert!(
            mine.iter().any(|n| theirs.contains(n)),
            "the two sets overlap (film_chan at least) - that overlap is what a positional index would get wrong"
        );
    }

    /// `(255·x − mean)/std` and `(x − mean/255)/(std/255)` must agree.
    #[test]
    fn unit_norm_is_the_same_affine_in_zero_to_one_units() {
        let pre = ArcFaceConfig::iresnet100().pre;
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
        reg.register(Arc::new(ArcFaceProvider::new("/nonexistent/antelopev2")));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let err = reg.run(MODEL, "embed", Invocation::new().blob("image", img), &mut |_| {}).unwrap_err();
        assert!(!err.is_empty(), "expected a descriptive error, got: {err}");
    }
}
