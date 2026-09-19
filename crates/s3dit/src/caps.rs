// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image's capabilities, declared through the generalized [`capability`]
//! interface. This is what makes `brain caps z-image` and `brain do z-image
//! <action> …` — and the equivalent `ActionRequest` over the event API — work
//! without a line of Z-Image-specific plumbing in the CLI or runtime.
//!
//! The manifest is **static** (no weights needed) so capability *discovery* is
//! free; only [`ZImageProvider`] (execution) loads the model. Actions mirror the
//! released Z-Image-Turbo surface: text-to-image, image-to-image, masked
//! inpainting, outpainting, and LoRA personalisation.

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

/// The model id used on the CLI (`brain do z-image …`) and the event API.
pub const MODEL: &str = "brain/s3dit";

/// The caption-capacity param, shared by the two actions that build a pipeline
/// for a fixed one. Mirrors `qwen3vl`'s `max_pixels`: a resident CAPACITY the
/// caller may raise, whose overflow is an error by name and never a silent
/// crop. See [`crate::pipeline::DEFAULT_CAP_LEN`] for why the default is what
/// it is.
fn cap_len_param() -> ParamSpec {
    ParamSpec::new(
        "cap_len",
        ParamType::Int,
        "resident capacity: caption tokens the pipeline's graphs are BUILT for; a longer prompt errors -- it is never truncated. Raising it rebuilds the pipeline.",
    )
    .default(json!(crate::pipeline::DEFAULT_CAP_LEN))
    .min(1.0)
    .max(crate::pipeline::max_cap_len(&crate::ZImageConfig::turbo(), 1) as f64)
    .step(1.0)
}

/// Shared generation params (steps / guidance / seed / size).
fn gen_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("steps", ParamType::Int, "denoising steps (Turbo≈8)").default(json!(8)).min(1.0).max(crate::pipeline::MAX_STEPS as f64).step(1.0))
        .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance scale; 0 disables (Turbo default)").default(json!(0.0)).min(0.0).max(30.0).step(0.1))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"))
        .param(ParamSpec::new("width", ParamType::Int, "output width, px").default(json!(1024)).min(64.0).max(2048.0).step(8.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px").default(json!(1024)).min(64.0).max(2048.0).step(8.0))
        .param(ParamSpec::new("precision", ParamType::Enum(vec!["int8".into(), "fp32".into()]), "DiT precision: int8 (1 GPU, fast) or fp32 (2 GPUs, higher fidelity)").default(json!("int8")))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let prompt = || ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required();
    let neg = || ParamSpec::new("negative_prompt", ParamType::Str, "what to avoid (only used when guidance>0)");
    let image_out = || BlobSpec::new("image", Media::Image, "the generated image");

    let text2image = gen_params(ActionSpec::new("text2image", "generate an image from a text prompt (posters, photos, art; strong at English/Chinese typography)").streaming())
        .param(prompt())
        .param(neg())
        .param(cap_len_param())
        .param(ParamSpec::new("adapter", ParamType::Str, "path to a trained LoRA adapter (from lora_train) to apply"))
        .output(image_out());

    let image2image = gen_params(ActionSpec::new("image2image", "regenerate an input image toward a prompt (style/lighting/weather changes, sketch→image)").streaming())
        .param(prompt())
        .param(neg())
        .param(ParamSpec::new("strength", ParamType::Float, "0=keep input, 1=ignore it; how much to change").default(json!(0.55)).min(0.0).max(1.0).step(0.01))
        .input(BlobSpec::new("image", Media::Image, "the starting image").required())
        .output(image_out());

    let inpaint = gen_params(ActionSpec::new("inpaint", "regenerate only the masked region of an image (object removal/replacement, sign-text change)").streaming())
        .param(prompt())
        .param(neg())
        .param(ParamSpec::new("strength", ParamType::Float, "how strongly to regenerate the masked region").default(json!(0.85)).min(0.0).max(1.0).step(0.01))
        .param(ParamSpec::new("feather", ParamType::Int, "mask-edge feather radius in latent cells (0 = hard edge)").default(json!(2)).min(0.0).max(64.0).step(1.0))
        .input(BlobSpec::new("image", Media::Image, "the image to edit").required())
        .input(BlobSpec::new("mask", Media::Mask, "white = regenerate, black = keep").required())
        .output(image_out());

    let outpaint = gen_params(ActionSpec::new("outpaint", "extend an image beyond its borders (canvas expand + inpaint of the new area)").streaming())
        .param(prompt())
        .param(ParamSpec::new("left", ParamType::Int, "pixels to add on the left").default(json!(0)))
        .param(ParamSpec::new("right", ParamType::Int, "pixels to add on the right").default(json!(0)))
        .param(ParamSpec::new("top", ParamType::Int, "pixels to add on top").default(json!(0)))
        .param(ParamSpec::new("bottom", ParamType::Int, "pixels to add on the bottom").default(json!(0)))
        .param(ParamSpec::new("feather", ParamType::Int, "seam feather radius in latent cells (0 = hard edge)").default(json!(3)).min(0.0).max(64.0).step(1.0))
        .input(BlobSpec::new("image", Media::Image, "the image to extend").required())
        .output(image_out());

    let lora_train = ActionSpec::new("lora_train", "fine-tune a LoRA adapter on a folder of captioned images (personalise a person/object/style)")
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "folder with images + a captions.yaml (`filename: prompt`) and/or captions.jsonl").required())
        .param(ParamSpec::new("save", ParamType::Str, "output path for the trained adapter").required())
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").default(json!(16)))
        .param(ParamSpec::new("steps", ParamType::Int, "training steps").default(json!(500)))
        .param(ParamSpec::new("size", ParamType::Int, "training square size, px").default(json!(512)))
        .param(ParamSpec::new("lr", ParamType::Float, "learning rate").default(json!(1e-4)))
        .param(cap_len_param())
        .param(ParamSpec::new("one_gpu", ParamType::Bool, "train on a single GPU (default: shard the 6B across both)").default(json!(false)))
        .output(BlobSpec::new("adapter", Media::Bytes, "the trained LoRA adapter checkpoint"));

    Manifest::new(
        MODEL,
        "Z-Image (Tongyi) — an efficient image-generation model: text-to-image, image-to-image, masked inpainting, outpainting, and LoRA personalisation.",
        vec![text2image, image2image, inpaint, outpaint, lora_train],
    )
}

// ===================== execution =====================

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, Invocation, Outcome, Progress, Provider};

/// Cache key for a resident text-to-image pipeline: everything that fixes the
/// built graphs. The caption CAPACITY is one of them - prompts are masked-padded
/// up to it (and refused past it, never truncated), so any prompt that fits
/// reuses the same hot weights, but a caller asking for a different capacity is
/// asking for different graphs.
type HotKey = (u32, u32, bool, Option<String>, u32); // (width, height, hifi, adapter path, cap_len)

/// The executable Z-Image model behind the manifest. Holds a **hot pipeline
/// cache** so a long-lived process (`brain run` / the event server) loads the
/// ~20 GB of weights once and reuses them across `ActionRequest`s — subsequent
/// generations are fast. Weight paths are BOUND at construction (see
/// [`ZImageProvider::from_paths`]) - never re-resolved per request.
pub struct ZImageProvider {
    hot: Arc<Mutex<Option<(HotKey, crate::pipeline::HotPipeline)>>>,
    paths: crate::pipeline::Paths,
}

impl ZImageProvider {
    /// Weights from the environment (`BRAIN_S3DIT_DIT` / `_VAE` / `_QWEN` /
    /// `_TOKENIZER`) - the historical construction path; still used wherever
    /// no resolved [`capability::Assembly`] is in hand.
    pub fn load() -> Result<ZImageProvider, String> {
        Ok(ZImageProvider::from_paths(crate::pipeline::Paths::from_env()?))
    }

    pub fn from_paths(paths: crate::pipeline::Paths) -> ZImageProvider {
        ZImageProvider { hot: Arc::new(Mutex::new(None)), paths }
    }
}

impl Provider for ZImageProvider {
    fn manifest(&self) -> capability::Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest()
            .actions
            .iter()
            .any(|a| a.name == name)
            .then(|| Arc::new(ZAction { name: name.to_string(), hot: self.hot.clone(), paths: self.paths.clone() }) as Arc<dyn Action>)
    }
}

/// One Z-Image action. Discovery, argument validation and dispatch are fully
/// wired through the generalized interface; the numeric execution runs the
/// assembled generation pipeline (encoder → flow-match sampler over the DiT → VAE
/// decode). Until that pipeline is assembled end-to-end (the DiT forward, VAE
/// decode, scheduler and Qwen encoder are each validated; the multi-step sampling
/// loop + tokenizer glue is the remaining piece), `run` reports precisely what is
/// pending rather than fabricating an image.
struct ZAction {
    name: String,
    hot: Arc<Mutex<Option<(HotKey, crate::pipeline::HotPipeline)>>>,
    paths: crate::pipeline::Paths,
}

impl Action for ZAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let paths = &self.paths;
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let mut on = |step, total, message: &str| progress(Progress::step(step, total, message.to_string()));
        match self.name.as_str() {
            "text2image" => {
                // Hot path: build the resident pipeline once per build shape,
                // reuse across calls so a long-lived server generates fast.
                //
                // The shape is settled and VALIDATED before the cache lock is
                // taken: `ensure_hot` must free the old resident before it can
                // build a replacement, so a request this DiT cannot serve has
                // to be refused here or it destroys ~20 GB of working weights
                // for every caller after it.
                let key = text2image_key(paths, inv)?;
                let seed = inv.get_i64("seed").unwrap_or(42).max(0) as u64;
                let steps = inv.get_i64("steps").unwrap_or(8).max(1) as u32;

                let mut guard = self.hot.lock().map_err(|_| "hot pipeline lock poisoned")?;
                let (width, height, hifi, adapter, cap_len) = key.clone();
                ensure_hot(&mut guard, key, || {
                    on(0, 1, "loading weights (first call for this size)");
                    // A fixed caption CAPACITY so any prompt that fits reuses the
                    // built graphs; one that does not is refused, not cropped.
                    crate::pipeline::HotPipeline::build_adapted(paths, width, height, cap_len, hifi, adapter.as_deref(), |m| on(0, 1, m))
                })?;
                let pipe = &guard.as_ref().expect("ensure_hot leaves the slot filled on success").1;
                emit(pipe.generate(&prompt, seed, steps, &inv.cancel, &mut on)?)
            }
            "image2image" => {
                let (image, w, h) = capability::blob::decode_image(inv, "image")?;
                let opts = opts_from(inv, w, h); // output matches the input image
                let init = crate::pipeline::Init { image: &image, strength: inv.get_f64("strength").unwrap_or(0.55) as f32, mask: None, feather: 0 };
                emit(crate::pipeline::generate_img(&prompt, &opts, paths, init, &mut on)?)
            }
            "inpaint" => {
                let (image, w, h) = capability::blob::decode_image(inv, "image")?;
                let (mask, mw, mh) = capability::blob::decode_plane(inv, "mask")?;
                if (mw, mh) != (w, h) {
                    return Err(format!("mask is {mw}×{mh} but image is {w}×{h}; they must match"));
                }
                let opts = opts_from(inv, w, h);
                let init = crate::pipeline::Init { image: &image, strength: inv.get_f64("strength").unwrap_or(0.85) as f32, mask: Some(&mask), feather: inv.get_i64("feather").unwrap_or(2).max(0) as u32 };
                emit(crate::pipeline::generate_img(&prompt, &opts, paths, init, &mut on)?)
            }
            "outpaint" => {
                let (image, w, h) = capability::blob::decode_image(inv, "image")?;
                let g = |k: &str| inv.get_i64(k).unwrap_or(0).max(0) as usize;
                let (canvas, mask, nw, nh) = build_outpaint_canvas(&image, w as usize, h as usize, g("left"), g("right"), g("top"), g("bottom"));
                let opts = opts_from(inv, nw as u32, nh as u32);
                // The new border regenerates from scratch (strength 1); the mask
                // re-anchors the original region every step so it is preserved.
                let init = crate::pipeline::Init { image: &canvas, strength: 1.0, mask: Some(&mask), feather: inv.get_i64("feather").unwrap_or(3).max(0) as u32 };
                emit(crate::pipeline::generate_img(&prompt, &opts, paths, init, &mut on)?)
            }
            "lora_train" => {
                let dir = inv.get_str("data").ok_or("lora_train: 'data' folder is required")?;
                let save = inv.get_str("save").ok_or("lora_train: 'save' path is required")?;
                let opts = crate::finetune::TrainOpts {
                    steps: inv.get_i64("steps").unwrap_or(500).max(1) as u32,
                    rank: inv.get_i64("rank").unwrap_or(16).max(1) as usize,
                    lr: inv.get_f64("lr").unwrap_or(1e-4) as f32,
                    // Checked conversions, not `as u32`: both fix the shape
                    // every graph in the run is built for, and `finetune::run`
                    // refuses an unbuildable one by name at its own entry.
                    size: u32_param(inv, "size", 512)?,
                    cap_len: u32_param(inv, "cap_len", crate::pipeline::DEFAULT_CAP_LEN)?,
                    seed: inv.get_i64("seed").map(|s| s.max(0) as u64).unwrap_or_else(data::rng::random_seed),
                    two_gpu: !inv.get_bool("one_gpu").unwrap_or(false),
                    save_path: save.clone(),
                    ckpt_every: 100,
                };
                let mut prog = |step: u32, total: u32, message: String| progress(Progress::step(step, total, message));
                let tensors = crate::finetune::run(paths, std::path::Path::new(&dir), &opts, &inv.cancel, &mut prog)?;
                // Return the trained artifact itself, not just its server-side path —
                // a remote client has no filesystem access to `save`.
                use capability::Blob;
                let bytes = std::fs::read(&save).map_err(|e| format!("read trained adapter '{save}': {e}"))?;
                Ok(Outcome::new()
                    .set("adapter", json!(save))
                    .set("steps", json!(opts.steps))
                    .set("tensors", json!(tensors.len()))
                    .blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"path": save}))))
            }
            other => Err(format!("z-image '{other}': unknown action")),
        }
    }
}

/// An invocation's integer param as a `u32`, REFUSED rather than wrapped when
/// it does not fit.
///
/// `Invocation` carries JSON numbers as `i64` and every build-shape param this
/// model takes is a `u32`, so each one crosses that boundary. `as u32`
/// truncates: `cap_len = 2^32 + 500` is positive, so it survived a `.max(1)`
/// floor and then became a perfectly ordinary-looking `500` - a malformed
/// request silently served at a capacity nobody asked for. The capability
/// layer cannot catch this for us: `ParamSpec::min`/`max` are advisory and
/// `ActionSpec::validate` never enforces them.
fn u32_param(inv: &Invocation, name: &str, default: u32) -> Result<u32, String> {
    match inv.get_i64(name) {
        None => Ok(default),
        Some(v) => u32::try_from(v).map_err(|_| format!("z-image: '{name}' must be between 0 and {} (got {v})", u32::MAX)),
    }
}

/// The hot-cache key a `text2image` invocation asks for, with its build shape
/// validated.
///
/// Called BEFORE the cache lock is taken, on purpose: [`ensure_hot`] drops the
/// resident pipeline before it can build a replacement, so every reason a
/// build can be refused up front has to be settled here - otherwise a single
/// unbuildable request evicts ~20 GB of working weights and every caller after
/// it pays the reload.
fn text2image_key(paths: &crate::pipeline::Paths, inv: &Invocation) -> Result<HotKey, String> {
    let width = u32_param(inv, "width", 1024)?;
    let height = u32_param(inv, "height", 1024)?;
    let cap_len = u32_param(inv, "cap_len", crate::pipeline::DEFAULT_CAP_LEN)?;
    let hifi = inv.get_str("precision").as_deref() == Some("fp32");
    let adapter = inv.get_str("adapter").filter(|s| !s.is_empty());
    crate::pipeline::check_build_shape(&paths.dit, width, height, cap_len)?;
    Ok((width, height, hifi, adapter, cap_len))
}

/// Make `slot` hold a pipeline built for `key`, building one only when the
/// cached entry is for a different key.
///
/// The cached pipeline is dropped BEFORE `build` runs, because two full
/// Z-Image residents (~20 GB each) do not fit one card. That is only safe
/// because [`text2image_key`] has already settled every up-front reason the
/// build could be refused, so `build` is reached only for a shape this DiT can
/// actually be built for.
fn ensure_hot<K: PartialEq, P>(slot: &mut Option<(K, P)>, key: K, build: impl FnOnce() -> Result<P, String>) -> Result<(), String> {
    if matches!(slot, Some((k, _)) if *k == key) {
        return Ok(());
    }
    *slot = None; // free the old resident weights before building new
    *slot = Some((key, build()?));
    Ok(())
}

/// Build [`crate::pipeline::Opts`] from an invocation, with an explicit output size.
fn opts_from(inv: &Invocation, width: u32, height: u32) -> crate::pipeline::Opts {
    crate::pipeline::Opts {
        steps: inv.get_i64("steps").unwrap_or(8).max(1) as u32,
        guidance: inv.get_f64("guidance").unwrap_or(0.0) as f32,
        seed: inv.get_i64("seed").unwrap_or(42).max(0) as u64,
        width,
        height,
        hifi: inv.get_str("precision").as_deref() == Some("fp32"),
    }
}

/// Wrap a generated [`crate::pipeline::Image`] as an image-output [`Outcome`]
/// (the shared `capability::blob` wire format).
fn emit(img: crate::pipeline::Image) -> ActionResult {
    Ok(Outcome::new()
        .set("width", json!(img.w))
        .set("height", json!(img.h))
        .blob("image", capability::blob::image_blob(&img.hwc, img.w as u32, img.h as u32, 3)))
}

/// Assemble an outpaint canvas: the input placed with `l/r/t/b` px borders
/// (edge-replicated so the VAE encode sees plausible content), the total size
/// rounded up to a multiple of 16 (extra added to right/bottom). Returns the
/// canvas (HWC `[0,1]`), a mask (`1` = new border to regenerate, `0` = keep the
/// original), and the canvas `nw,nh`.
fn build_outpaint_canvas(img: &[f32], w: usize, h: usize, l: usize, r: usize, t: usize, b: usize) -> (Vec<f32>, Vec<f32>, usize, usize) {
    // Round the total up to a multiple of 16 (VAE/patch constraint); the extra
    // falls on the right/bottom and is treated as more border to regenerate.
    let nw = (w + l + r).next_multiple_of(16);
    let nh = (h + t + b).next_multiple_of(16);
    let mut canvas = vec![0f32; nw * nh * 3];
    let mut mask = vec![1f32; nw * nh]; // default: regenerate
    for y in 0..nh {
        for x in 0..nw {
            let sx = (x as i64 - l as i64).clamp(0, w as i64 - 1) as usize;
            let sy = (y as i64 - t as i64).clamp(0, h as i64 - 1) as usize;
            for c in 0..3 {
                canvas[(y * nw + x) * 3 + c] = img[(sy * w + sx) * 3 + c];
            }
            if x >= l && x < l + w && y >= t && y < t + h {
                mask[y * nw + x] = 0.0; // original region: keep
            }
        }
    }
    (canvas, mask, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["text2image", "image2image", "inpaint", "outpaint", "lora_train"]);
        // text2image: prompt required, steps defaulted to 8, produces an image.
        let t2i = &m.actions[0];
        assert!(t2i.params.iter().any(|p| p.name == "prompt" && p.required));
        assert_eq!(t2i.params.iter().find(|p| p.name == "steps").unwrap().default, Some(json!(8)));
        assert_eq!(t2i.outputs[0].media, Media::Image);
        // inpaint requires both image and mask.
        let inp = m.actions.iter().find(|a| a.name == "inpaint").unwrap();
        assert!(inp.inputs.iter().any(|b| b.name == "mask" && b.media == Media::Mask && b.required));
        // lora_train declares the trained adapter as a retrievable output blob.
        let lt = m.actions.iter().find(|a| a.name == "lora_train").unwrap();
        assert!(lt.outputs.iter().any(|b| b.name == "adapter" && b.media == Media::Bytes));
        // the whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["actions"].as_array().unwrap().len(), 5);
    }

    /// The caption capacity is a declared, caller-settable resident capacity -
    /// on the two actions that BUILD a pipeline for a fixed one - and its
    /// description says overflow errors, because a caller reading the manifest
    /// is exactly who needs to know the prompt is not silently cropped.
    #[test]
    fn the_caption_capacity_is_a_declared_settable_param() {
        let m = manifest();
        for action in ["text2image", "lora_train"] {
            let a = m.actions.iter().find(|a| a.name == action).unwrap();
            let p = a.params.iter().find(|p| p.name == "cap_len").unwrap_or_else(|| panic!("{action} declares no cap_len"));
            assert_eq!(p.default, Some(json!(crate::pipeline::DEFAULT_CAP_LEN)));
            assert_eq!(p.min, Some(1.0));
            assert_eq!(p.max, Some(crate::pipeline::max_cap_len(&crate::ZImageConfig::turbo(), 1) as f64));
            assert!(p.help.contains("errors"), "{}", p.help);
        }
        // The per-call edit actions build at the prompt's own length, so they
        // have no capacity to declare.
        for action in ["image2image", "inpaint", "outpaint"] {
            let a = m.actions.iter().find(|a| a.name == action).unwrap();
            assert!(!a.params.iter().any(|p| p.name == "cap_len"), "{action} builds per call and needs no cap_len");
        }
    }

    fn unresolvable_paths() -> crate::pipeline::Paths {
        let p = |role: &str| format!("no-such-z-image-{role}");
        crate::pipeline::Paths { dit: p("dit"), vae: p("vae"), qwen: p("qwen"), tokenizer: p("tokenizer") }
    }

    /// `cap_len` crosses an `i64 → u32` boundary here. `as u32` TRUNCATES:
    /// `2^32 + 500` is positive, so it survived the `.max(1)` floor and then
    /// became a perfectly plausible-looking `500` - a malformed request
    /// silently served at a capacity nobody asked for. A checked conversion
    /// refuses it instead, for every param that makes the same crossing.
    #[test]
    fn an_out_of_range_integer_param_is_refused_not_truncated() {
        let paths = unresolvable_paths();
        for name in ["cap_len", "width", "height"] {
            let inv = Invocation::new().set(name, json!(4_294_967_796i64));
            let e = text2image_key(&paths, &inv).unwrap_err();
            assert!(e.contains(name), "the refusal must name the param: {e}");
            assert!(e.contains("4294967796"), "and the value it was given: {e}");
        }
        // A negative one is out of range the other way, and used to be clamped.
        let e = text2image_key(&paths, &Invocation::new().set("cap_len", json!(-1))).unwrap_err();
        assert!(e.contains("cap_len"), "{e}");
    }

    /// The build shape is validated BEFORE the hot cache is consulted. The
    /// eviction that precedes a rebuild is unconditional (two ~20 GB residents
    /// do not fit one card), so a request this pipeline can never serve has to
    /// be refused before it reaches that point - otherwise one bad `cap_len`
    /// dropped a working resident and every caller after it paid a full
    /// ~20 GB reload.
    #[test]
    fn an_unbuildable_request_is_refused_before_the_hot_cache_is_touched() {
        let paths = unresolvable_paths();
        let over = crate::pipeline::max_cap_len(&crate::ZImageConfig::turbo(), 1) + 1;
        let e = text2image_key(&paths, &Invocation::new().set("cap_len", json!(over))).unwrap_err();
        assert!(e.contains("cap_len"), "{e}");
        let e = text2image_key(&paths, &Invocation::new().set("width", json!(8208))).unwrap_err();
        assert!(e.contains("8208") || e.contains("8192"), "{e}");
        // A buildable shape yields the key it will be cached under.
        let k = text2image_key(&paths, &Invocation::new().set("cap_len", json!(1024))).unwrap();
        assert_eq!(k, (1024, 1024, false, None, 1024));
    }

    /// The cache slot itself: a matching key must not rebuild (that is the
    /// whole point of a resident), and a rebuild must replace what it evicted.
    #[test]
    fn a_matching_key_reuses_the_resident_instead_of_rebuilding() {
        let mut builds = 0;
        let mut slot: Option<(u32, &str)> = None;
        let build = |slot: &mut Option<(u32, &'static str)>, key: u32, builds: &mut u32| {
            ensure_hot(slot, key, || {
                *builds += 1;
                Ok("weights")
            })
        };
        build(&mut slot, 1, &mut builds).unwrap();
        build(&mut slot, 1, &mut builds).unwrap();
        assert_eq!(builds, 1, "a matching key must reuse the resident, not reload it");
        build(&mut slot, 2, &mut builds).unwrap();
        assert_eq!(builds, 2);
        assert_eq!(slot, Some((2, "weights")));
    }

    #[test]
    fn generation_params_carry_ui_ranges() {
        let t2i = &manifest().actions[0];
        let p = |name: &str| t2i.params.iter().find(|p| p.name == name).unwrap();
        assert_eq!((p("steps").min, p("steps").max, p("steps").step), (Some(1.0), Some(150.0), Some(1.0)));
        assert_eq!((p("guidance").min, p("guidance").max), (Some(0.0), Some(30.0)));
        assert_eq!((p("width").min, p("width").max, p("width").step), (Some(64.0), Some(2048.0), Some(8.0)));
        assert_eq!((p("height").min, p("height").max, p("height").step), (Some(64.0), Some(2048.0), Some(8.0)));

        let i2i = manifest().actions.iter().find(|a| a.name == "image2image").cloned().unwrap();
        let strength = i2i.params.iter().find(|p| p.name == "strength").unwrap().clone();
        assert_eq!((strength.min, strength.max, strength.step), (Some(0.0), Some(1.0), Some(0.01)));

        let inp = manifest().actions.iter().find(|a| a.name == "inpaint").cloned().unwrap();
        let feather = inp.params.iter().find(|p| p.name == "feather").unwrap().clone();
        assert_eq!((feather.min, feather.max, feather.step), (Some(0.0), Some(64.0), Some(1.0)));
    }
}
