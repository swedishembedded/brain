// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PuLID-conditioned FLUX.1 behind the generalized [`capability`] interface -
//! what makes `brain caps flux1-pulid` / `brain do flux1-pulid text2image
//! ...`, the D-Bus `Run` method and `brain perf`'s `CapabilityTarget` work
//! with no PuLID-specific plumbing in the CLI or the transports.
//!
//! One action: **`text2image`** - a prompt plus a face photo in, an HWC RGB
//! image out. This composes five already-existing pieces and adds none of
//! its own numerics:
//!
//! 1. `arcface::caps::ArcFaceSession::embed_raw_chw` - the raw ArcFace
//!    embedding (detect + align, `crate::idcond`'s documented convention).
//! 2. `clip::model::EvaVision` - the EVA-CLIP-L/336 CLS embedding and its 5
//!    tapped hidden states (`clip::EvaVisionConfig::PULID_TAPS`), the same
//!    tower `clip::caps::Session::embed_image` uses.
//! 3. `crate::idcond::compose` - joins the two into `id_cond`.
//! 4. `crate::model::IdFormer` - `id_cond` + the 5 taps -> 32 ID tokens.
//! 5. `crate::adapter::PulidAdapter` (a `crate::model::PulidCa` driven
//!    through `flux1::inject::BlockInject`) - handed to
//!    `flux1::pipeline::Flux1::generate_injected`, which is `Flux1::generate`
//!    with every DiT step routed through `forward_injected` instead of
//!    `forward`. **No denoise loop, no VAE glue and no text conditioning is
//!    duplicated here** - `flux1::pipeline` already owns all of that.
//!
//! # A real, documented preprocessing gap
//!
//! The reference PuLID pipeline prepares the EVA-CLIP input with facexlib
//! RetinaFace alignment plus a BiSeNet face parse (background whitened, face
//! greyscaled) before the tower ever sees it - two models this workspace does
//! not have (`crate`'s own crate-level docs and `Cargo.toml` say so). This
//! action instead resizes the SAME face crop `embed_raw_chw` used for ArcFace
//! straight to EVA-CLIP-L/336's input, with no parsing. `IdCond::from_image`
//! and `crate::idcond`'s parity tests take a caller-supplied EVA cls
//! precisely because this preprocessing step is not reproduced - this is
//! that caller, choosing the closest available approximation rather than
//! refusing to run.
//!
//! # No batching, size fixed at build time, and no end-to-end fixture
//!
//! Same reasoning as `flux1::caps` (`crates/cli/src/resident_flux1.rs`'s
//! module docs) for both: every request is its own multi-step sample, and
//! `Flux1::load` records the DiT's token budget for one `(variant, h, w)`.
//! `crate`'s own docs are explicit that end-to-end generation is not gated
//! here (`crates/flux1` had no sampler loop when that was written); it does
//! now, via this file, but there is still no reference dump of a full
//! PuLID-conditioned generation in this workspace to check it against - see
//! `flux1::pipeline`'s own honest note on verification, which applies here
//! doubly (this composes flux1's unverified glue with PuLID's own unverified
//! injection wiring).

use std::sync::{Arc, Mutex};

use arcface::caps::ArcFaceSession;
use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use clip::config::EvaVisionConfig;
use clip::model::EvaVision;
use flux1::config::Flux1Config;
use flux1::pipeline::{Flux1, GenerateOptions};
use gpu_core::Gpu;
use serde_json::json;

use crate::adapter::PulidAdapter;
use crate::config::PulidConfig;
use crate::model::{IdFormer, PulidCa};

/// The model id used on the CLI (`brain do flux1-pulid ...`), over D-Bus and
/// in the residency manifest.
pub const MODEL: &str = "brain/flux1-pulid";

/// FLUX.1 variants PuLID has been validated against a reference for (`dev`
/// only - `crate`'s own docs: "it is built on FLUX.1-dev, not Kontext").
/// `kontext-dev`/`schnell` are architecturally identical enough to run, but
/// nothing here or upstream checks them.
const VARIANTS: [&str; 3] = ["dev", "kontext-dev", "schnell"];

const DEFAULT_MAX_LEN: u32 = 512;

fn text2image_spec() -> ActionSpec {
    ActionSpec::new("text2image", "Generate an image from a text prompt, conditioned on a face's identity (PuLID + FLUX.1).")
        .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 16)").default(json!(1024)).min(256.0).max(2048.0).step(16.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 16)").default(json!(1024)).min(256.0).max(2048.0).step(16.0))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps; 0 = variant default").default(json!(0)).min(0.0).max(150.0).step(1.0))
        .param(ParamSpec::new("guidance", ParamType::Float, "guidance_in scalar -- dev/kontext-dev only, schnell ignores it").default(json!(3.5)).min(0.0).max(10.0).step(0.1))
        .param(ParamSpec::new("id_weight", ParamType::Float, "identity conditioning strength").default(json!(0.8)).min(0.0).max(2.0).step(0.05))
        .param(ParamSpec::new("max_len", ParamType::Int, "T5-XXL context length").default(json!(DEFAULT_MAX_LEN)).min(32.0).max(512.0).step(1.0))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "FLUX.1 variant -- only dev is validated against a PuLID reference").default(json!("dev")))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for 0)"))
        .input(capability::BlobSpec::new("face_image", capability::Media::Image, "a photo of the identity to condition on").required())
        .output(capability::BlobSpec::new("image", capability::Media::Image, "the generated image"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "FLUX.1 text-to-image conditioned on a face's identity via PuLID (ArcFace + EVA-CLIP -> injected cross-attention).",
        vec![text2image_spec()],
    )
}

struct Req {
    prompt: String,
    variant: String,
    opts: GenerateOptions,
    max_len: usize,
    id_weight: f32,
}

fn req_from(inv: &Invocation) -> Req {
    Req {
        prompt: inv.get_str("prompt").unwrap_or_default(),
        variant: inv.get_str("variant").unwrap_or_else(|| "dev".into()),
        opts: GenerateOptions {
            steps: {
                let s = inv.get_i64("steps").unwrap_or(0);
                (s > 0).then_some(s as usize)
            },
            guidance: inv.get_f64("guidance").unwrap_or(3.5) as f32,
            seed: inv.get_i64("seed").unwrap_or(0) as u64,
            height: inv.get_i64("height").unwrap_or(1024).max(16) as u32,
            width: inv.get_i64("width").unwrap_or(1024).max(16) as u32,
        },
        max_len: inv.get_i64("max_len").unwrap_or(DEFAULT_MAX_LEN as i64).max(1) as usize,
        id_weight: inv.get_f64("id_weight").unwrap_or(0.8) as f32,
    }
}

// ===================== the pipeline =====================

/// One fully-built (FLUX.1 + PuLID) pair for a `(variant, h, w)`. Everything
/// except the DiT and `PulidCa` is size-independent, but is rebuilt with them
/// anyway - see the module docs' "no batching, size fixed at build time"
/// note. PuLID serving is inherently a heavy, infrequent call, not a hot
/// path worth a finer-grained cache for.
struct Bundle {
    flux1: Flux1,
    arcface: ArcFaceSession,
    eva: EvaVision,
    idformer: IdFormer,
    // `PulidCa` is moved into `PulidAdapter` at construction
    // (`PulidAdapter::new` takes it by value); every subsequent request
    // reuses this ONE adapter via `set_id`/`set_id_weight` (both `&self`,
    // interior GPU writes) rather than rebuilding it.
    adapter: PulidAdapter,
    pulid_cfg: PulidConfig,
}

impl Bundle {
    fn load(flux1_root: &str, pulid_root: &str, arcface_root: &str, clip_root: &str, variant: &str, h: u32, w: u32) -> Result<Bundle, String> {
        let flux1 = Flux1::load(flux1_root, variant, h, w)?;
        let fcfg = Flux1Config::from_name(variant)?;
        let n_gen = ((h / 16) * (w / 16)) as usize;

        let gpu = Gpu::new(crate::model::KERNELS);
        let arcface = ArcFaceSession::load(arcface_root, gpu.new_like(&arcface::caps::SERVING_PIPELINES))?;

        let eva_cfg = EvaVisionConfig::eva02_l336();
        let eva_path = std::path::Path::new(clip_root).join(clip::caps::EVA_FILE);
        let eva_p = eva_path.to_str().ok_or("pulid: non-UTF8 EVA checkpoint path")?;
        let eva_tensors = checkpoint::torchpt::read(eva_p).map_err(|e| format!("pulid: reading {eva_p}: {e}"))?;
        let (eva_init, _report) = clip::import::import_eva_visual(eva_tensors, &eva_cfg)?;
        let eva_map: std::collections::HashMap<String, Vec<f32>> =
            eva_init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        let eva = EvaVision::new_on(gpu.new_like(clip::model::TEXT_PIPELINES), eva_cfg, 1, &eva_map);

        let pulid_cfg = PulidConfig::v0_9_1();
        let w_pulid = crate::import::read(pulid_root, &pulid_cfg)?;
        let idformer = IdFormer::new(gpu.new_like(crate::model::KERNELS), pulid_cfg.clone(), w_pulid.encoder);
        let ca = PulidCa::new(gpu.new_like(crate::model::KERNELS), pulid_cfg.clone(), w_pulid.num_ca, n_gen, w_pulid.ca);
        // `id_weight` given here is a placeholder; every request overwrites it
        // via `set_id_weight` before use (read at step-build time, so this is
        // a field write, not a graph rebuild - see `PulidAdapter`'s docs).
        let adapter = PulidAdapter::new(ca, &pulid_cfg, fcfg.depth_double, fcfg.depth_single, 0.8);

        Ok(Bundle { flux1, arcface, eva, idformer, adapter, pulid_cfg })
    }

    /// Face photo (HWC RGB `[0,1]`) -> the 32 projected ID tokens.
    fn id_tokens(&self, chw: &[f32], w: u32, h: u32) -> Result<Vec<f32>, String> {
        let (arc_raw, _face) = self.arcface.embed_raw_chw(chw, w, h, true, true)?;

        // The documented preprocessing gap: a plain resize to EVA-CLIP-L/336's
        // input, not the reference's RetinaFace+BiSeNet crop - see the module
        // docs.
        let side = EvaVisionConfig::eva02_l336().image_size;
        let ctx = imaging::Ctx::new(&self.eva.gpu);
        let src = ctx.upload("pulid.face", chw);
        let (dst, _) = ctx.resize(&src, imaging::Shape::new(1, 3, h, w), side, side, imaging::Filter::Bilinear, imaging::AlignCorners::HalfPixel);
        let resized = ctx.download(&dst, 3 * side * side);
        self.eva.set_pixels(&resized);
        self.eva.forward();
        let eva_cls = self.eva.read_cls_embed_l2norm();

        let cond = crate::idcond::compose(&self.pulid_cfg, &arc_raw, &eva_cls)?;

        let taps: Vec<Vec<f32>> = EvaVisionConfig::PULID_TAPS.iter().map(|&l| self.eva.read_x((l + 1) as usize)).collect();
        self.idformer.set_inputs(&cond, &taps);
        self.idformer.forward();
        Ok(self.idformer.read_id_embedding())
    }

    fn generate(&self, req: &Req, id: &[f32]) -> Result<Vec<f32>, String> {
        self.adapter.set_id(id);
        self.adapter.set_id_weight(req.id_weight);
        self.flux1.generate_injected(&req.prompt, &req.opts, req.max_len, Some(&self.adapter))
    }
}

// ===================== the shared work =====================

pub struct Session {
    flux1_root: String,
    pulid_root: String,
    arcface_root: String,
    clip_root: String,
    built: Mutex<std::collections::HashMap<(String, u32, u32), Bundle>>,
}

impl Session {
    pub fn new(flux1_root: impl Into<String>, pulid_root: impl Into<String>, arcface_root: impl Into<String>, clip_root: impl Into<String>) -> Session {
        Session {
            flux1_root: flux1_root.into(),
            pulid_root: pulid_root.into(),
            arcface_root: arcface_root.into(),
            clip_root: clip_root.into(),
            built: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "text2image" => self.text2image(inv),
            other => Err(format!("pulid: unknown action '{other}'")),
        }
    }

    fn text2image(&self, inv: &Invocation) -> ActionResult {
        let req = req_from(inv);
        let (h, w) = (req.opts.height, req.opts.width);
        let (hwc, fw, fh, c) = capability::blob::decode_hwc(inv, "face_image")?;
        if c != 3 {
            return Err(format!("pulid: face_image must be RGB (3 channels), got {c}"));
        }
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, fh as usize, fw as usize);

        let key = (req.variant.clone(), h, w);
        let mut guard = self.built.lock().map_err(|_| "pulid: pipeline lock poisoned")?;
        let b = match guard.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(Bundle::load(
                &self.flux1_root,
                &self.pulid_root,
                &self.arcface_root,
                &self.clip_root,
                &req.variant,
                h,
                w,
            )?),
        };
        let id = b.id_tokens(&chw, fw, fh)?;
        let hwc_out = b.generate(&req, &id)?;
        Ok(Outcome::new().blob("image", capability::blob::image_blob(&hwc_out, w, h, 3)))
    }
}

// ===================== the provider =====================

type Roots = (String, String, String, String);
type HotSession = Arc<Mutex<Option<(Roots, Arc<Session>)>>>;

pub struct PulidProvider {
    roots: Roots,
    hot: HotSession,
}

impl PulidProvider {
    pub fn new(flux1_root: impl Into<String>, pulid_root: impl Into<String>, arcface_root: impl Into<String>, clip_root: impl Into<String>) -> PulidProvider {
        PulidProvider { roots: (flux1_root.into(), pulid_root.into(), arcface_root.into(), clip_root.into()), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_FLUX1_DIR` + `BRAIN_PULID_DIR` (a `pulid_flux_v0.9.1.safetensors`
    /// file or its directory) + `BRAIN_ARCFACE_DIR` + `BRAIN_CLIP_DIR` (for the
    /// EVA-CLIP-L/336 file, the same variable `clip::caps` uses) - `None`
    /// unless every one of the four is set and the FLUX.1 directory holds a
    /// released `transformer/`.
    pub fn from_env() -> Option<PulidProvider> {
        let get = |k: &str| std::env::var(k).ok().filter(|p| !p.is_empty());
        let (flux1_root, pulid_root, arcface_root, clip_root) =
            (get("BRAIN_FLUX1_DIR")?, get("BRAIN_PULID_DIR")?, get("BRAIN_ARCFACE_DIR")?, get("BRAIN_CLIP_DIR")?);
        std::path::Path::new(&flux1_root)
            .join("transformer")
            .exists()
            .then(|| PulidProvider::new(flux1_root, pulid_root, arcface_root, clip_root))
    }
}

impl Provider for PulidProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "text2image")
            .then(|| Arc::new(PulidAction { roots: self.roots.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct PulidAction {
    roots: Roots,
    hot: HotSession,
}

impl Action for PulidAction {
    fn spec(&self) -> ActionSpec {
        text2image_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let session = {
            let mut guard = self.hot.lock().map_err(|_| "pulid: hot session lock poisoned")?;
            if !matches!(&*guard, Some((r, _)) if *r == self.roots) {
                *guard = None;
                let (a, b, c, d) = self.roots.clone();
                *guard = Some((self.roots.clone(), Arc::new(Session::new(a, b, c, d))));
            }
            guard.as_ref().expect("built above").1.clone()
        };
        session.run("text2image", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;

    #[test]
    fn manifest_declares_text2image() {
        let m = manifest();
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["text2image"]);
    }

    #[test]
    fn an_unknown_action_is_named_not_ignored() {
        let p = PulidProvider::new("/nonexistent", "/nonexistent", "/nonexistent", "/nonexistent");
        assert!(p.action("edit").is_none());
    }

    #[test]
    fn from_env_declines_unless_all_four_directories_are_set() {
        assert!(
            PulidProvider::from_env().is_none()
                || [
                    std::env::var("BRAIN_FLUX1_DIR").is_ok(),
                    std::env::var("BRAIN_PULID_DIR").is_ok(),
                    std::env::var("BRAIN_ARCFACE_DIR").is_ok(),
                    std::env::var("BRAIN_CLIP_DIR").is_ok(),
                ]
                .into_iter()
                .all(|x| x)
        );
    }

    #[test]
    fn id_weight_carries_ui_range() {
        let spec = text2image_spec();
        let p = spec.params.iter().find(|p| p.name == "id_weight").expect("id_weight param");
        assert_eq!(p.min, Some(0.0));
        assert_eq!(p.max, Some(2.0));
    }
}
