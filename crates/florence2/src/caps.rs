// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! UI-element visual grounding behind the generalized [`capability`]
//! interface - what makes `brain do florence2 ground …` work, per this
//! crate's own reason for existing: a screenshot-in, bounding-box-out
//! oracle a UI-testing workflow can call to drive taps deterministically.
//!
//! One action:
//!
//! * **`ground`** - image + a short text `target` phrase in, normalized
//!   `[x0,y0,x1,y1]` bounding boxes out (`{found, boxes: [{phrase, bbox}]}`,
//!   no output blob - `scrfd::caps`'s `detect` shape, not `qwen3vl`'s
//!   free-text `generate`). Internally: `"Locate {target} in the image."`
//!   (the reference processor's own OPEN_VOCABULARY_DETECTION template,
//!   read from `processing_florence2.py::task_prompts_with_input` rather
//!   than guessed - the literal task-marker token never appears in the
//!   tokenized input, it is purely a client-side template-selection key) ->
//!   [`crate::text::Florence2Lm::generate`] (greedy, EOS-terminated) ->
//!   [`crate::grounding::parse_boxes`].
//!
//! Preprocessing matches the checkpoint's own `preprocessor_config.json`
//! (`CLIPImageProcessor`, `do_center_crop=false`): a direct (non-aspect-
//! preserving) resize to `768x768`, bicubic (`resample: 3` is PIL's
//! `BICUBIC`), then per-channel `(x-mean)/std`.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Ctx, Filter, Normalization, Shape};
use serde_json::json;

use crate::grounding::parse_boxes;
use crate::import::{all_tensor_names, build_param_source, VISION_PROJECTOR_PREFIX};
use crate::text::{BartConfig, Florence2Lm, Florence2LmKernelIds};
use crate::tokenizer;
use crate::vision::{pipelines::PIPELINES, Davit, DavitConfig, DavitKernelIds, ImageProject, ImageProjectKernelIds};
use data::tokenizer::Tokenizer;

/// The model id used on the CLI (`brain do florence2 …`), over D-Bus and in
/// the residency manifest.
pub const MODEL: &str = "brain/florence2";

pub const IMAGE_SIZE: u32 = 768;
/// `preprocessor_config.json`'s `image_mean`/`image_std` - `CLIPImageProcessor`
/// defaults, but read from the checkpoint's own file rather than assumed.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A generous ceiling on generated tokens: a grounding answer is a handful
/// of `<loc_N>` tokens plus a short echoed phrase, never anywhere near this.
const MAX_NEW_TOKENS: u32 = 64;

/// [`crate::vision::pipelines::PIPELINES`] plus the two kernels only the
/// *serving* path dispatches (the CLIPImageProcessor-shaped preprocessing:
/// `imaging::Ctx::resize`'s bicubic filter, `imaging::Ctx::normalize`'s
/// `film_chan`). Appended, never reordered - see `scrfd::caps::
/// SERVING_PIPELINES`'s own doc for why (every `*KernelIds::resolve` in this
/// crate indexes [`PIPELINES`] positionally).
const N_MODEL: usize = PIPELINES.len();
pub const SERVING_PIPELINES: [(&str, &str); N_MODEL + 2] = serving_set();

const fn serving_set() -> [(&'static str, &'static str); N_MODEL + 2] {
    let mut k = [("", ""); N_MODEL + 2];
    let mut i = 0;
    while i < N_MODEL {
        k[i] = PIPELINES[i];
        i += 1;
    }
    k[N_MODEL] = ("resize_bicubic", kernels::RESIZE_BICUBIC);
    k[N_MODEL + 1] = ("film_chan", kernels::FILM_CHAN);
    k
}

pub fn ground_spec() -> ActionSpec {
    ActionSpec::new("ground", "locate a UI element or phrase in an image: normalized bounding box(es)")
        .param(ParamSpec::new("target", ParamType::Str, "the phrase or UI element to locate, e.g. \"the Login button\"").required())
        .param(ParamSpec::new("max_new_tokens", ParamType::Int, "cap on generated tokens (0 = default)").default(json!(0)))
        .input(BlobSpec::new("image", Media::Image, "the screenshot to search").required())
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "Florence-2 visual grounding: locate a phrase or UI element in an image as a normalized bounding box.", vec![ground_spec()])
}

// ===================== the shared work =====================

/// Florence-2 on one device - the single implementation of `ground`, shared
/// by [`Florence2Provider`] and the CLI's residency adapter.
pub struct FlorenceSession {
    gpu: Gpu,
    ps: paramstore::ParamStore,
    davit: Davit,
    davit_k: DavitKernelIds,
    proj: ImageProject,
    proj_k: ImageProjectKernelIds,
    lm_k: Florence2LmKernelIds,
    bart_cfg: BartConfig,
    t_vision: u32,
    tok: data::qwen_tokenizer::QwenBpe,
}

impl FlorenceSession {
    /// Import `<dir>/model.safetensors` + `<dir>/tokenizer.json` and build
    /// the vision tower on `gpu` (fixed `768x768` input shape, so this part
    /// builds once and is reused across every `ground` call on this
    /// session; [`crate::text::Florence2Lm`] is rebuilt per call instead,
    /// since its shape depends on that call's tokenized prompt length - see
    /// [`FlorenceSession::ground`]).
    pub fn load(dir: &str, gpu: Gpu) -> Result<FlorenceSession, String> {
        let hf_dir = std::path::Path::new(dir);
        let source = build_param_source(hf_dir)?;
        let davit_cfg = DavitConfig::florence2_base();
        let bart_cfg = BartConfig::florence2_base();

        let roles: Vec<_> = all_tensor_names(&davit_cfg, &bart_cfg)
            .into_iter()
            .map(|name| {
                let numel = source.get(&name).ok_or_else(|| format!("florence2: missing tensor {name}"))?.1.len();
                Ok((name, numel, paramstore::Role::Frozen))
            })
            .collect::<Result<_, String>>()?;
        let ps = paramstore::ParamStore::new_with_roles_src(&gpu, roles, &source);

        let davit_k = DavitKernelIds::resolve(&SERVING_PIPELINES);
        let davit = Davit::new(&gpu, &davit_k, "vision_tower", &davit_cfg, 4, 1e-5, false);
        let (davit_tokens, davit_dim) = davit_cfg.final_tokens_and_dim();
        let proj_k = ImageProjectKernelIds::resolve(&SERVING_PIPELINES);
        let proj = ImageProject::new(&gpu, VISION_PROJECTOR_PREFIX, 24, 24, davit_dim, bart_cfg.d_model, 1e-5);
        let lm_k = Florence2LmKernelIds::resolve(&SERVING_PIPELINES);

        let tok = tokenizer::load(dir)?;

        Ok(FlorenceSession { gpu, ps, davit, davit_k, proj, proj_k, lm_k, bart_cfg, t_vision: davit_tokens + 1, tok })
    }

    /// Resize+normalize an RGB `hwc` image to the checkpoint's fixed
    /// `768x768` input and run the vision tower, returning the
    /// `[t_vision,d_model]` projected vision tokens - `self.proj`'s own
    /// scratch buffer, valid until the next call that touches `self.proj`
    /// (this crate never runs two `ground` calls on the same session
    /// concurrently - `Florence2Provider`'s `hot` mutex serializes them).
    fn vision_tokens(&self, hwc: &[f32], w: u32, h: u32) -> &gpu_core::DeviceBuffer {
        let ctx = Ctx::new(&self.gpu);
        let chw = imaging::pixels::hwc_to_chw(hwc, 3, h as usize, w as usize);
        let src = ctx.upload("florence2.caps.src", &chw);
        let (resized, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), IMAGE_SIZE, IMAGE_SIZE, Filter::Bicubic, AlignCorners::HalfPixel);
        let normed = ctx.normalize(&resized, shape, &Normalization { mean: MEAN, std: STD });

        let unpooled = self.davit.forward_features_unpool(&self.gpu, &self.davit_k, &self.ps, &normed);
        self.proj.forward(&self.gpu, &self.proj_k, &self.ps, unpooled)
    }

    /// Run one `ground` invocation.
    pub fn ground(&self, inv: &Invocation) -> ActionResult {
        let target = inv.get_str("target").ok_or("florence2 ground: missing required param 'target'")?;
        let max_new_tokens = match inv.get_i64("max_new_tokens").unwrap_or(0) {
            0 => MAX_NEW_TOKENS,
            n if n > 0 => n as u32,
            n => return Err(format!("florence2 ground: max_new_tokens must be >= 0, got {n}")),
        };
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        let image_features = self.vision_tokens(&hwc, w, h);
        let prompt = format!("Locate {target} in the image.");
        let prompt_ids = self.tok.encode(&prompt);

        let lm = Florence2Lm::new(&self.gpu, self.bart_cfg, self.t_vision, prompt_ids.len() as u32, max_new_tokens + 1);
        let enc = lm.encode(&self.gpu, &self.lm_k, &self.ps, image_features, &prompt_ids);
        let ids = lm.generate(&self.gpu, &self.lm_k, &self.ps, enc, max_new_tokens);

        let text = self.tok.decode(&ids);
        let boxes = parse_boxes(&text);
        let list: Vec<serde_json::Value> = boxes.iter().map(|b| json!({"phrase": b.phrase, "bbox": b.bbox})).collect();
        Ok(Outcome::new().set("found", json!(!list.is_empty())).set("boxes", json!(list)))
    }

    /// Dispatch by action name - the seam the residency `Instance` uses.
    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "ground" => self.ground(inv),
            other => Err(format!("florence2: unknown action '{other}'")),
        }
    }
}

// ===================== the provider =====================

/// The executable grounder behind the manifest. Construction is free - the
/// checkpoint imports lazily on the first call and stays resident.
pub struct Florence2Provider {
    dir: String,
    hot: Arc<Mutex<Option<(String, FlorenceSession)>>>,
}

impl Florence2Provider {
    /// `dir` holds `config.json`, `model.safetensors` and `tokenizer.json`
    /// (the `microsoft/Florence-2-base` release layout).
    pub fn new(dir: impl Into<String>) -> Florence2Provider {
        Florence2Provider { dir: dir.into(), hot: Arc::new(Mutex::new(None)) }
    }

    pub const RELEASE_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];

    /// `BRAIN_FLORENCE2_DIR` - `None` when unset or when the directory does
    /// not hold the released checkpoint.
    pub fn from_env() -> Option<Florence2Provider> {
        let dir = std::env::var("BRAIN_FLORENCE2_DIR").ok().filter(|p| !p.is_empty())?;
        let d = std::path::Path::new(&dir);
        Florence2Provider::RELEASE_FILES.iter().all(|f| d.join(f).exists()).then(|| Florence2Provider::new(dir))
    }
}

impl Provider for Florence2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "ground").then(|| Arc::new(GroundAction { dir: self.dir.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct GroundAction {
    dir: String,
    hot: Arc<Mutex<Option<(String, FlorenceSession)>>>,
}

impl Action for GroundAction {
    fn spec(&self) -> ActionSpec {
        ground_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut guard = self.hot.lock().map_err(|_| "florence2: hot model lock poisoned")?;
        if !matches!(&*guard, Some((d, _)) if *d == self.dir) {
            *guard = None; // free the old build before importing another directory
            let gpu = Gpu::new(&SERVING_PIPELINES);
            *guard = Some((self.dir.clone(), FlorenceSession::load(&self.dir, gpu)?));
        }
        guard.as_ref().expect("built above").1.run("ground", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::{Blob, Registry};

    #[test]
    fn manifest_declares_ground_only() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1, "florence2 serves exactly one action so far");
        let g = m.actions.iter().find(|a| a.name == "ground").expect("ground");
        assert!(g.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert!(g.params.iter().any(|p| p.name == "target" && p.required));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = g.validate(Invocation::new().blob("image", img.clone()).set("target", json!("the login button"))).unwrap();
        assert_eq!(inv.get_i64("max_new_tokens"), Some(0));
        assert!(g.validate(Invocation::new().blob("image", img)).is_err(), "target is required");
    }

    /// The checkpoint is not on every box: a missing one must surface as a
    /// clean `ActionResult` error, not a panic.
    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(Florence2Provider::new("/nonexistent/florence2")));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = Invocation::new().blob("image", img).set("target", json!("x"));
        let err = reg.run(MODEL, "ground", inv, &mut |_| {}).unwrap_err();
        assert!(!err.is_empty(), "expected a descriptive error, got: {err}");
    }
}
