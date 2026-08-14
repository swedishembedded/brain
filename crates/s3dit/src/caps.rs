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

/// Shared generation params (steps / guidance / seed / size).
fn gen_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("steps", ParamType::Int, "denoising steps (Turbo≈8)").default(json!(8)))
        .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance scale; 0 disables (Turbo default)").default(json!(0.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"))
        .param(ParamSpec::new("width", ParamType::Int, "output width, px").default(json!(1024)))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px").default(json!(1024)))
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
        .param(ParamSpec::new("adapter", ParamType::Str, "path to a trained LoRA adapter (from lora_train) to apply"))
        .output(image_out());

    let image2image = gen_params(ActionSpec::new("image2image", "regenerate an input image toward a prompt (style/lighting/weather changes, sketch→image)").streaming())
        .param(prompt())
        .param(neg())
        .param(ParamSpec::new("strength", ParamType::Float, "0=keep input, 1=ignore it; how much to change").default(json!(0.55)))
        .input(BlobSpec::new("image", Media::Image, "the starting image").required())
        .output(image_out());

    let inpaint = gen_params(ActionSpec::new("inpaint", "regenerate only the masked region of an image (object removal/replacement, sign-text change)").streaming())
        .param(prompt())
        .param(neg())
        .param(ParamSpec::new("strength", ParamType::Float, "how strongly to regenerate the masked region").default(json!(0.85)))
        .param(ParamSpec::new("feather", ParamType::Int, "mask-edge feather radius in latent cells (0 = hard edge)").default(json!(2)))
        .input(BlobSpec::new("image", Media::Image, "the image to edit").required())
        .input(BlobSpec::new("mask", Media::Mask, "white = regenerate, black = keep").required())
        .output(image_out());

    let outpaint = gen_params(ActionSpec::new("outpaint", "extend an image beyond its borders (canvas expand + inpaint of the new area)").streaming())
        .param(prompt())
        .param(ParamSpec::new("left", ParamType::Int, "pixels to add on the left").default(json!(0)))
        .param(ParamSpec::new("right", ParamType::Int, "pixels to add on the right").default(json!(0)))
        .param(ParamSpec::new("top", ParamType::Int, "pixels to add on top").default(json!(0)))
        .param(ParamSpec::new("bottom", ParamType::Int, "pixels to add on the bottom").default(json!(0)))
        .param(ParamSpec::new("feather", ParamType::Int, "seam feather radius in latent cells (0 = hard edge)").default(json!(3)))
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
/// built graphs. The caption length is a *property* of the built pipeline (not the
/// key) — prompts are padded/truncated to it, so any prompt reuses the same hot
/// weights.
type HotKey = (u32, u32, bool, Option<String>); // (width, height, hifi, adapter path)

/// The executable Z-Image model behind the manifest. Holds a **hot pipeline
/// cache** so a long-lived process (`brain run` / the event server) loads the
/// ~20 GB of weights once and reuses them across `ActionRequest`s — subsequent
/// generations are fast. Weight paths come from the environment
/// (`BRAIN_S3DIT_DIT` / `_VAE` / `_QWEN` / `_TOKENIZER`).
pub struct ZImageProvider {
    hot: Arc<Mutex<Option<(HotKey, crate::pipeline::HotPipeline)>>>,
}

impl ZImageProvider {
    pub fn load() -> Result<ZImageProvider, String> {
        Ok(ZImageProvider { hot: Arc::new(Mutex::new(None)) })
    }
}

impl Provider for ZImageProvider {
    fn manifest(&self) -> capability::Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| Arc::new(ZAction { name: name.to_string(), hot: self.hot.clone() }) as Arc<dyn Action>)
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
}

impl Action for ZAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let paths = crate::pipeline::Paths::from_env()?;
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let mut on = |step, total, message: &str| progress(Progress::step(step, total, message.to_string()));
        match self.name.as_str() {
            "text2image" => {
                // Hot path: build the resident pipeline once per (size, precision),
                // reuse across calls so a long-lived server generates fast.
                let width = inv.get_i64("width").unwrap_or(1024) as u32;
                let height = inv.get_i64("height").unwrap_or(1024) as u32;
                let hifi = inv.get_str("precision").as_deref() == Some("fp32");
                let seed = inv.get_i64("seed").unwrap_or(42).max(0) as u64;
                let steps = inv.get_i64("steps").unwrap_or(8).max(1) as u32;
                let adapter = inv.get_str("adapter").filter(|s| !s.is_empty());
                let key: HotKey = (width, height, hifi, adapter.clone());

                let mut guard = self.hot.lock().map_err(|_| "hot pipeline lock poisoned")?;
                let rebuild = !matches!(&*guard, Some((k, _)) if *k == key);
                if rebuild {
                    *guard = None; // free the old resident weights before building new
                    on(0, 1, "loading weights (first call for this size)");
                    // A fixed caption length so any prompt reuses the built graphs.
                    let pipe = crate::pipeline::HotPipeline::build_adapted(&paths, width, height, 64, hifi, adapter.as_deref(), |m| on(0, 1, m))?;
                    *guard = Some((key, pipe));
                }
                let pipe = &guard.as_ref().unwrap().1;
                emit(pipe.generate(&prompt, seed, steps, &inv.cancel, &mut on)?)
            }
            "image2image" => {
                let (image, w, h) = capability::blob::decode_image(inv, "image")?;
                let opts = opts_from(inv, w, h); // output matches the input image
                let init = crate::pipeline::Init { image: &image, strength: inv.get_f64("strength").unwrap_or(0.55) as f32, mask: None, feather: 0 };
                emit(crate::pipeline::generate_img(&prompt, &opts, &paths, init, &mut on)?)
            }
            "inpaint" => {
                let (image, w, h) = capability::blob::decode_image(inv, "image")?;
                let (mask, mw, mh) = capability::blob::decode_plane(inv, "mask")?;
                if (mw, mh) != (w, h) {
                    return Err(format!("mask is {mw}×{mh} but image is {w}×{h}; they must match"));
                }
                let opts = opts_from(inv, w, h);
                let init = crate::pipeline::Init { image: &image, strength: inv.get_f64("strength").unwrap_or(0.85) as f32, mask: Some(&mask), feather: inv.get_i64("feather").unwrap_or(2).max(0) as u32 };
                emit(crate::pipeline::generate_img(&prompt, &opts, &paths, init, &mut on)?)
            }
            "outpaint" => {
                let (image, w, h) = capability::blob::decode_image(inv, "image")?;
                let g = |k: &str| inv.get_i64(k).unwrap_or(0).max(0) as usize;
                let (canvas, mask, nw, nh) = build_outpaint_canvas(&image, w as usize, h as usize, g("left"), g("right"), g("top"), g("bottom"));
                let opts = opts_from(inv, nw as u32, nh as u32);
                // The new border regenerates from scratch (strength 1); the mask
                // re-anchors the original region every step so it is preserved.
                let init = crate::pipeline::Init { image: &canvas, strength: 1.0, mask: Some(&mask), feather: inv.get_i64("feather").unwrap_or(3).max(0) as u32 };
                emit(crate::pipeline::generate_img(&prompt, &opts, &paths, init, &mut on)?)
            }
            "lora_train" => {
                let dir = inv.get_str("data").ok_or("lora_train: 'data' folder is required")?;
                let save = inv.get_str("save").ok_or("lora_train: 'save' path is required")?;
                let opts = crate::finetune::TrainOpts {
                    steps: inv.get_i64("steps").unwrap_or(500).max(1) as u32,
                    rank: inv.get_i64("rank").unwrap_or(16).max(1) as usize,
                    lr: inv.get_f64("lr").unwrap_or(1e-4) as f32,
                    size: inv.get_i64("size").unwrap_or(512).max(16) as u32,
                    cap_len: 64,
                    seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
                    two_gpu: !inv.get_bool("one_gpu").unwrap_or(false),
                    save_path: save.clone(),
                    ckpt_every: 100,
                };
                let mut prog = |step: u32, total: u32, message: String| progress(Progress::step(step, total, message));
                let tensors = crate::finetune::run(&paths, std::path::Path::new(&dir), &opts, &inv.cancel, &mut prog)?;
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
}
