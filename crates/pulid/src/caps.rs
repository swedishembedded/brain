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
//! # Reference face preprocessing (`crates/bisenet`)
//!
//! The reference PuLID pipeline prepares the EVA-CLIP input with facexlib's
//! RetinaFace-template alignment plus a real BiSeNet face parse (background
//! whitened, face greyscaled) before the tower ever sees it. This action
//! reproduces that exactly: SCRFD's own landmarks (already computed by
//! `embed_raw_chw` for ArcFace's 112px alignment) are aligned a second time
//! to BiSeNet's 512px FFHQ template (`bisenet::align::norm_crop_512`),
//! parsed (`bisenet::BiSeNet::forward`, the real official
//! `parsing_bisenet` weights), and masked (`bisenet::mask::whiten_and_gray`)
//! - all gated at cosine 1.0000000000 against real `facexlib` output on a
//! real photo, see `crates/bisenet/tests/parity.rs`. Only the resulting
//! 512x512 masked image is then bicubic-resized to EVA-CLIP-L/336, replacing
//! what used to be a plain resize of the raw face crop. `IdCond::from_image`
//! and `crate::idcond`'s parity tests still take a caller-supplied EVA cls
//! (they predate this crate's own preprocessing and gate the composition
//! math independently of it), which remains a valid, narrower way to drive
//! the same pipeline.
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
use flux1::pipeline::{Flux1, GenerateOptions, TrueCfg};
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
/// nothing here or upstream checks them. `krea-dev` IS an upstream-supported
/// PuLID combination (unlike kontext-dev/schnell, which are not) - but this
/// crate still has neither the checkpoint nor a reference dump for it, so it
/// carries the identical unvalidated status, not a stronger claim.
const VARIANTS: [&str; 4] = ["dev", "kontext-dev", "krea-dev", "schnell"];

/// Bounded by `flux1::pipeline::MAX_TXT_LEN`, the length every loaded
/// `Flux1` (and hence every `Bundle`) is actually sized for.
const DEFAULT_MAX_LEN: u32 = flux1::pipeline::MAX_TXT_LEN;

/// The BiSeNet face-parsing checkpoint filename under `BRAIN_BISENET_DIR` -
/// `tools/goldens/pulid_face_parsing_dump_reference.py`'s re-serialization
/// of facexlib's own release (see `crates/bisenet/src/import.rs`'s module
/// docs for why the released `.pth` needs converting first).
pub const BISENET_FILE: &str = "parsing_bisenet.safetensors";

/// The DiT precision enum, in manifest order - mirrors `flux1::caps`'s own
/// (fp32 doesn't fit a 24 GiB card with PuLID's extra residency on top, so
/// this action defaults the OTHER way).
const PRECISIONS: [&str; 2] = ["fp32", "int8"];

/// Optional auxiliary identity photos beyond the required `face_image`
/// primary - upstream PuLID v1.1's own "one primary + up to three auxiliary"
/// shape (see `Bundle::id_tokens`'s doc for how they are fused, and its
/// honest caveat about NOT being upstream's own fusion algorithm).
const EXTRA_FACE_REFS: [&str; 3] = ["face_image1", "face_image2", "face_image3"];

fn text2image_spec() -> ActionSpec {
    let mut spec = ActionSpec::new("text2image", "Generate an image from a text prompt, conditioned on a face's identity (PuLID + FLUX.1).")
        .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 16)").default(json!(1024)).min(256.0).max(2048.0).step(16.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 16)").default(json!(1024)).min(256.0).max(2048.0).step(16.0))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps; 0 = variant default").default(json!(0)).min(0.0).max(150.0).step(1.0))
        .param(ParamSpec::new("start_step", ParamType::Int, "denoising step at which identity injection begins; 0 = every step. Upstream PuLID-FLUX guidance: smaller injects identity sooner (more fidelity, less freedom for the base structure) -- ~4 for photorealism, ~0-1 for stylization").default(json!(0)).min(0.0).max(150.0).step(1.0))
        .param(ParamSpec::new("guidance", ParamType::Float, "guidance_in scalar -- dev/kontext-dev/krea-dev only, schnell ignores it").default(json!(3.5)).min(0.0).max(10.0).step(0.1))
        .param(ParamSpec::new("id_weight", ParamType::Float, "identity conditioning strength").default(json!(1.0)).min(0.0).max(3.0).step(0.05))
        .param(ParamSpec::new("max_len", ParamType::Int, "T5-XXL context length").default(json!(DEFAULT_MAX_LEN)).min(32.0).max(flux1::pipeline::MAX_TXT_LEN as f64).step(1.0))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "FLUX.1 variant -- only dev is validated against a PuLID reference").default(json!("dev")))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"))
        .param(ParamSpec::new("precision", ParamType::Enum(PRECISIONS.iter().map(|s| s.to_string()).collect()), "DiT numeric tier -- int8 is what fits a 24 GiB card, fp32 needs ~48 GiB").default(json!("int8")))
        .param(ParamSpec::new("negative_prompt", ParamType::Str, "negative conditioning for true CFG; ignored unless true_cfg > 0"))
        .param(ParamSpec::new("true_cfg", ParamType::Float, "true classifier-free guidance scale on top of the distilled guidance scalar; 0 = disabled (default, single forward/step). Runs a SECOND, un-injected DiT forward per step from cfg_start_step onward -- doubles cost while active. NOT parity-gated against upstream's own true-CFG branch (no reference dump exists in this workspace)").default(json!(0.0)).min(0.0).max(10.0).step(0.1))
        .param(ParamSpec::new("cfg_start_step", ParamType::Int, "denoising step at which true CFG begins; steps before it use the identity-conditioned prediction alone").default(json!(0)).min(0.0).max(150.0).step(1.0))
        .input(capability::BlobSpec::new("face_image", capability::Media::Image, "a photo of the identity to condition on").required());
    for r in EXTRA_FACE_REFS {
        spec = spec.input(capability::BlobSpec::new(r, capability::Media::Image, "additional photo of the SAME identity (optional; mean-pooled with the others, see Bundle::id_tokens's doc on how)"));
    }
    spec.output(capability::BlobSpec::new("image", capability::Media::Image, "the generated image"))
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
    negative_prompt: Option<String>,
    variant: String,
    opts: GenerateOptions,
    max_len: usize,
    id_weight: f32,
    precision: flux1::Precision,
}

fn req_from(inv: &Invocation) -> Req {
    let true_cfg_scale = inv.get_f64("true_cfg").unwrap_or(0.0) as f32;
    Req {
        prompt: inv.get_str("prompt").unwrap_or_default(),
        negative_prompt: inv.get_str("negative_prompt").filter(|s| !s.is_empty()),
        variant: inv.get_str("variant").unwrap_or_else(|| "dev".into()),
        opts: GenerateOptions {
            steps: {
                let s = inv.get_i64("steps").unwrap_or(0);
                (s > 0).then_some(s as usize)
            },
            guidance: inv.get_f64("guidance").unwrap_or(3.5) as f32,
            seed: inv.get_i64("seed").map(|s| s as u64).unwrap_or_else(data::rng::random_seed),
            height: inv.get_i64("height").unwrap_or(1024).max(16) as u32,
            width: inv.get_i64("width").unwrap_or(1024).max(16) as u32,
            start_step: inv.get_i64("start_step").unwrap_or(0).max(0) as usize,
            true_cfg: (true_cfg_scale > 0.0)
                .then_some(TrueCfg { scale: true_cfg_scale, start_step: inv.get_i64("cfg_start_step").unwrap_or(0).max(0) as usize }),
        },
        max_len: inv.get_i64("max_len").unwrap_or(DEFAULT_MAX_LEN as i64).max(1) as usize,
        id_weight: inv.get_f64("id_weight").unwrap_or(1.0) as f32,
        precision: flux1::Precision::from_name(&inv.get_str("precision").unwrap_or_else(|| "int8".into())).unwrap_or(flux1::Precision::Int8),
    }
}

/// `PulidCa`'s device footprint - resident for the whole DiT lifetime (unlike
/// ArcFace/EVA-CLIP/IDFormer, which only run once per identity) - so it is
/// folded into the "dit" placement `Need` via
/// [`flux1::pipeline::plan_flux1`]'s `dit_extra_bytes`, never costed
/// separately. Always fp32 (no kernel here is quantized, per the module docs).
fn pulid_ca_bytes(cfg: &PulidConfig, n_ca: usize) -> u64 {
    cfg.ca_manifest(n_ca).iter().map(|(_, shape)| shape.iter().product::<usize>() as u64 * 4).sum()
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
    bisenet: bisenet::BiSeNet,
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
    #[allow(clippy::too_many_arguments)]
    fn load(flux1_root: &str, pulid_root: &str, arcface_root: &str, clip_root: &str, bisenet_root: &str, variant: &str, h: u32, w: u32, precision: flux1::Precision) -> Result<Bundle, String> {
        let fcfg = Flux1Config::from_name(variant)?;
        let n_gen = ((h / 16) * (w / 16)) as usize;
        let pulid_cfg = PulidConfig::v0_9_1();
        let w_pulid = crate::import::read(pulid_root, &pulid_cfg)?;

        // The DiT and PulidCa MUST share one `Gpu` built from
        // `pulid::joint_kernels()` - a `Step` is only meaningful to the handle
        // that created it (`flux1::inject`'s documented contract), so a
        // `PulidCa` built on a device with a DIFFERENT kernel list resolves
        // its injected steps against the wrong pipeline indices. Placed via
        // the SAME plan `flux1::pipeline::load_with` would use for a bare
        // FLUX.1, with PulidCa's own resident bytes folded into the "dit" part
        // so the plan prices what actually gets built.
        let dit_extra = pulid_ca_bytes(&pulid_cfg, w_pulid.num_ca);
        let homes = flux1::pipeline::plan_flux1(&fcfg, precision, n_gen as u64, dit_extra)?;
        eprintln!("pulid: placement {}", homes.describe());
        let dit_gpu = homes.run("dit", || Gpu::new(crate::model::joint_kernels()))?;
        let flux1 = Flux1::load_shared(flux1_root, variant, h, w, dit_gpu.share(), precision, homes.clone())?;
        let ca = PulidCa::new_on(dit_gpu, pulid_cfg.clone(), crate::model::joint_kernels(), w_pulid.num_ca, n_gen, w_pulid.ca);

        // ArcFace/EVA-CLIP/IDFormer/BiSeNet each run once per identity, not
        // once per step - they keep their own independent kernel sets (never
        // pushed into the DiT's dispatch list, unlike `PulidCa`), so they are
        // not part of the "dit"/"te" `Need`s `plan_flux1` sizes. They still
        // need an EXPLICIT home, though: left unscoped, `Gpu::new` lands on
        // whatever the ambient default device is, which is not guaranteed to
        // avoid "te" - on a 2-card box with a ~21 GiB fp32 T5-XXL resident
        // there, an unpinned identity stack landing on the SAME card OOMs
        // (confirmed: forcing the ambient device to the DiT's own card with
        // `BRAIN_GPU_INDEX` removes the OOM). Pinning to "dit" instead is
        // safe because this stack is modest - ArcFace + BiSeNet + EVA-CLIP-L
        // + IDFormer together are well under the DiT part's own remaining
        // headroom on a 24 GiB card.
        let gpu = homes.run("dit", || Gpu::new(crate::model::KERNELS))?;
        let arcface = ArcFaceSession::load(arcface_root, gpu.new_like(&arcface::caps::SERVING_PIPELINES))?;

        let bisenet_path = std::path::Path::new(bisenet_root).join(BISENET_FILE);
        let bisenet_p = bisenet_path.to_str().ok_or("pulid: non-UTF8 BiSeNet checkpoint path")?;
        let bisenet_weights = bisenet::import::read(bisenet_p).map_err(|e| format!("pulid: reading {bisenet_p}: {e}"))?;
        let bisenet = bisenet::BiSeNet::new(gpu.new_like(bisenet::model::PIPELINES), bisenet::BiSeNetConfig::bisenet(), &bisenet_weights);

        let eva_cfg = EvaVisionConfig::eva02_l336();
        let eva_path = std::path::Path::new(clip_root).join(clip::caps::EVA_FILE);
        let eva_p = eva_path.to_str().ok_or("pulid: non-UTF8 EVA checkpoint path")?;
        let eva_tensors = checkpoint::torchpt::read(eva_p).map_err(|e| format!("pulid: reading {eva_p}: {e}"))?;
        let (eva_init, _report) = clip::import::import_eva_visual(eva_tensors, &eva_cfg)?;
        let eva_map: std::collections::HashMap<String, Vec<f32>> =
            eva_init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        // The EVA tower's kernels, plus `imaging`'s: `face_embeds` builds an
        // `imaging::Ctx` on THIS handle to resize the parsed face to 336px, and
        // both sets resolve by name, so one flat union is enough. The text
        // tower's list has no place here - PuLID never runs it (FLUX.1's own
        // T5/CLIP encoders live on the `te` device).
        let eva_kernels: Vec<(&str, &str)> =
            clip::model::VISION_PIPELINES.iter().chain(imaging::PIPELINES.iter()).copied().collect();
        let eva = EvaVision::new_on(gpu.new_like(&eva_kernels), eva_cfg, 1, &eva_map);

        let idformer = IdFormer::new(gpu.new_like(crate::model::KERNELS), pulid_cfg.clone(), w_pulid.encoder);
        // `id_weight` given here is a placeholder; every request overwrites it
        // via `set_id_weight` before use (read at step-build time, so this is
        // a field write, not a graph rebuild - see `PulidAdapter`'s docs).
        let adapter = PulidAdapter::new(ca, &pulid_cfg, fcfg.depth_double, fcfg.depth_single, 1.0);

        Ok(Bundle { flux1, arcface, bisenet, eva, idformer, adapter, pulid_cfg })
    }

    /// One face photo (HWC->CHW RGB `[0,1]`, as `id_tokens` always fed it) ->
    /// its raw ArcFace embedding and its EVA-CLIP L2-normalized CLS embedding
    /// + 5 tapped hidden states - the per-image half of identity extraction,
    /// split out so [`Bundle::id_tokens`] can average it over 1-4 references
    /// (upstream PuLID v1.1's own "one primary + up to three auxiliary"
    /// shape) before the ONE `idcond::compose` + `IdFormer` call every
    /// request already made.
    fn face_embeds(&self, chw: &[f32], w: u32, h: u32) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>), String> {
        let (arc_raw, face) = self.arcface.embed_raw_chw(chw, w, h, true, true)?;
        // `align=true` only returns `Ok` once a face was found, and always
        // sets `Some` in that case - see `ArcFaceSession::embed_raw_chw`.
        let face = face.expect("arcface: align=true returned Ok with no detected face");
        let kps: Vec<f32> = face.kps.iter().flat_map(|p| [p[0], p[1]]).collect();

        // The reference chain, reproduced exactly (see `crates/bisenet`'s own
        // module docs and its `tests/parity.rs`, gated at cosine 1.0 against
        // real facexlib output): the SAME landmarks ArcFace just aligned to
        // 112px, aligned again to BiSeNet's 512px FFHQ template, parsed, and
        // the face region background-whitened/grayscaled - THEN resized to
        // EVA-CLIP-L/336. This replaces the plain resize the module docs
        // used to describe as the one documented preprocessing gap.
        let aligned = bisenet::align::norm_crop_512(self.bisenet.gpu(), chw, 3, h, w, &kps)?;
        let logits = self.bisenet.forward(&bisenet::imagenet_normalize(&aligned));
        let masked = bisenet::mask::whiten_and_gray(&logits, &aligned, 512, 512);

        let side = EvaVisionConfig::eva02_l336().image_size;
        let ctx = imaging::Ctx::new(&self.eva.gpu);
        let src = ctx.upload("pulid.face", &masked);
        let (dst, _) = ctx.resize(&src, imaging::Shape::new(1, 3, 512, 512), side, side, imaging::Filter::Bicubic, imaging::AlignCorners::HalfPixel);
        let resized = ctx.download(&dst, 3 * side * side);
        self.eva.set_pixels(&resized);
        self.eva.forward();
        let eva_cls = self.eva.read_cls_embed_l2norm();
        let taps: Vec<Vec<f32>> = EvaVisionConfig::PULID_TAPS.iter().map(|&l| self.eva.read_x((l + 1) as usize)).collect();
        Ok((arc_raw, eva_cls, taps))
    }

    /// 1-4 face photos (HWC RGB `[0,1]` each, primary first) -> the 32
    /// projected ID tokens. With one photo this is bit-identical to the
    /// original single-image path (mean of one vector is the vector).
    ///
    /// **Fusion is mean-pooling each per-image representation (raw ArcFace
    /// embedding, L2-normalized EVA-CLIP CLS embedding, each of the 5 tapped
    /// hidden states) elementwise across references, before the single
    /// `idcond::compose` + `IdFormer` call.** This is the standard
    /// multi-reference approach several community PuLID implementations use,
    /// NOT a transcription of upstream PuLID v1.1's own SDXL-branch fusion
    /// code (which this workspace has neither the source for nor a reference
    /// dump to gate against) - so treat this as a real, useful capability,
    /// not a claim of upstream-equivalence for the >1-image case. The
    /// single-image case IS the parity-gated path (see `crate`'s own module
    /// docs), unaffected by this generalization.
    fn id_tokens(&self, faces: &[(Vec<f32>, u32, u32)]) -> Result<Vec<f32>, String> {
        assert!(!faces.is_empty(), "pulid: id_tokens needs at least one face image");
        let per_image: Vec<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> =
            faces.iter().map(|(chw, w, h)| self.face_embeds(chw, *w, *h)).collect::<Result<_, _>>()?;

        let n = per_image.len() as f32;
        let mean = |vs: Vec<&Vec<f32>>| -> Vec<f32> {
            let len = vs[0].len();
            (0..len).map(|i| vs.iter().map(|v| v[i]).sum::<f32>() / n).collect()
        };
        let arc_raw = mean(per_image.iter().map(|(a, _, _)| a).collect());
        let eva_cls = mean(per_image.iter().map(|(_, e, _)| e).collect());
        let n_taps = per_image[0].2.len();
        let taps: Vec<Vec<f32>> =
            (0..n_taps).map(|t| mean(per_image.iter().map(|(_, _, taps)| &taps[t]).collect())).collect();

        let cond = crate::idcond::compose(&self.pulid_cfg, &arc_raw, &eva_cls)?;
        self.idformer.set_inputs(&cond, &taps);
        self.idformer.forward();
        Ok(self.idformer.read_id_embedding())
    }

    fn generate(&self, req: &Req, id: &[f32]) -> Result<Vec<f32>, String> {
        self.adapter.set_id(id);
        self.adapter.set_id_weight(req.id_weight);
        self.flux1.generate_injected(&req.prompt, req.negative_prompt.as_deref(), &req.opts, req.max_len, Some(&self.adapter))
    }
}

// ===================== the shared work =====================

pub struct Session {
    flux1_root: String,
    pulid_root: String,
    arcface_root: String,
    clip_root: String,
    bisenet_root: String,
    built: Mutex<std::collections::HashMap<(String, u32, u32, &'static str), Bundle>>,
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        flux1_root: impl Into<String>,
        pulid_root: impl Into<String>,
        arcface_root: impl Into<String>,
        clip_root: impl Into<String>,
        bisenet_root: impl Into<String>,
    ) -> Session {
        Session {
            flux1_root: flux1_root.into(),
            pulid_root: pulid_root.into(),
            arcface_root: arcface_root.into(),
            clip_root: clip_root.into(),
            bisenet_root: bisenet_root.into(),
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
        let mut faces = Vec::new();
        for name in std::iter::once("face_image").chain(EXTRA_FACE_REFS) {
            if name != "face_image" && inv.get_blob(name).is_none() {
                continue;
            }
            let (hwc, fw, fh, c) = capability::blob::decode_hwc(inv, name)?;
            if c != 3 {
                return Err(format!("pulid: {name} must be RGB (3 channels), got {c}"));
            }
            faces.push((imaging::pixels::hwc_to_chw(&hwc, 3, fh as usize, fw as usize), fw, fh));
        }

        let key = (req.variant.clone(), h, w, req.precision.name());
        let mut guard = self.built.lock().map_err(|_| "pulid: pipeline lock poisoned")?;
        let b = match guard.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(Bundle::load(
                &self.flux1_root,
                &self.pulid_root,
                &self.arcface_root,
                &self.clip_root,
                &self.bisenet_root,
                &req.variant,
                h,
                w,
                req.precision,
            )?),
        };
        let id = b.id_tokens(&faces)?;
        let hwc_out = b.generate(&req, &id)?;
        Ok(Outcome::new().blob("image", capability::blob::image_blob(&hwc_out, w, h, 3)))
    }
}

// ===================== the provider =====================

type Roots = (String, String, String, String, String);
type HotSession = Arc<Mutex<Option<(Roots, Arc<Session>)>>>;

pub struct PulidProvider {
    roots: Roots,
    hot: HotSession,
}

impl PulidProvider {
    pub fn new(
        flux1_root: impl Into<String>,
        pulid_root: impl Into<String>,
        arcface_root: impl Into<String>,
        clip_root: impl Into<String>,
        bisenet_root: impl Into<String>,
    ) -> PulidProvider {
        PulidProvider { roots: (flux1_root.into(), pulid_root.into(), arcface_root.into(), clip_root.into(), bisenet_root.into()), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_FLUX1_DIR` + `BRAIN_PULID_DIR` (a `pulid_flux_v0.9.1.safetensors`
    /// file or its directory) + `BRAIN_ARCFACE_DIR` + `BRAIN_CLIP_DIR` (for the
    /// EVA-CLIP-L/336 file, the same variable `clip::caps` uses) +
    /// `BRAIN_BISENET_DIR` (a directory holding `parsing_bisenet.safetensors`
    /// - see `BISENET_FILE`'s doc for how to produce one) - `None` unless
    /// every one of the five is set and the FLUX.1 directory holds a
    /// released `transformer/`.
    pub fn from_env() -> Option<PulidProvider> {
        let get = |k: &str| std::env::var(k).ok().filter(|p| !p.is_empty());
        let (flux1_root, pulid_root, arcface_root, clip_root, bisenet_root) = (
            get("BRAIN_FLUX1_DIR")?,
            get("BRAIN_PULID_DIR")?,
            get("BRAIN_ARCFACE_DIR")?,
            get("BRAIN_CLIP_DIR")?,
            get("BRAIN_BISENET_DIR")?,
        );
        std::path::Path::new(&flux1_root)
            .join("transformer")
            .exists()
            .then(|| PulidProvider::new(flux1_root, pulid_root, arcface_root, clip_root, bisenet_root))
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
                let (a, b, c, d, e) = self.roots.clone();
                *guard = Some((self.roots.clone(), Arc::new(Session::new(a, b, c, d, e))));
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
        let p = PulidProvider::new("/nonexistent", "/nonexistent", "/nonexistent", "/nonexistent", "/nonexistent");
        assert!(p.action("edit").is_none());
    }

    #[test]
    fn from_env_declines_unless_all_five_directories_are_set() {
        assert!(
            PulidProvider::from_env().is_none()
                || [
                    std::env::var("BRAIN_FLUX1_DIR").is_ok(),
                    std::env::var("BRAIN_PULID_DIR").is_ok(),
                    std::env::var("BRAIN_ARCFACE_DIR").is_ok(),
                    std::env::var("BRAIN_CLIP_DIR").is_ok(),
                    std::env::var("BRAIN_BISENET_DIR").is_ok(),
                ]
                .into_iter()
                .all(|x| x)
        );
    }

    #[test]
    fn id_weight_carries_ui_range() {
        let spec = text2image_spec();
        let p = spec.params.iter().find(|p| p.name == "id_weight").expect("id_weight param");
        // Matches upstream PuLID-FLUX's own default (1.0) and UI range (0..3),
        // not an arbitrary Brain-specific choice.
        assert_eq!(p.default, Some(json!(1.0)));
        assert_eq!(p.min, Some(0.0));
        assert_eq!(p.max, Some(3.0));
    }

    #[test]
    fn start_step_defaults_to_zero_every_step() {
        let spec = text2image_spec();
        let p = spec.params.iter().find(|p| p.name == "start_step").expect("start_step param");
        assert_eq!(p.default, Some(json!(0)));
    }
}
