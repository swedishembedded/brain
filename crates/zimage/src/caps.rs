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
pub const MODEL: &str = "z-image";

/// Shared generation params (steps / guidance / seed / size).
fn gen_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("steps", ParamType::Int, "denoising steps (Turbo≈8)").default(json!(8)))
        .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance scale; 0 disables (Turbo default)").default(json!(0.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"))
        .param(ParamSpec::new("width", ParamType::Int, "output width, px").default(json!(1024)))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px").default(json!(1024)))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let prompt = || ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required();
    let neg = || ParamSpec::new("negative_prompt", ParamType::Str, "what to avoid (only used when guidance>0)");
    let image_out = || BlobSpec::new("image", Media::Image, "the generated image");

    let text2image = gen_params(ActionSpec::new("text2image", "generate an image from a text prompt (posters, photos, art; strong at English/Chinese typography)").streaming())
        .param(prompt())
        .param(neg())
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
        .input(BlobSpec::new("image", Media::Image, "the image to edit").required())
        .input(BlobSpec::new("mask", Media::Mask, "white = regenerate, black = keep").required())
        .output(image_out());

    let outpaint = gen_params(ActionSpec::new("outpaint", "extend an image beyond its borders (canvas expand + inpaint of the new area)").streaming())
        .param(prompt())
        .param(ParamSpec::new("left", ParamType::Int, "pixels to add on the left").default(json!(0)))
        .param(ParamSpec::new("right", ParamType::Int, "pixels to add on the right").default(json!(0)))
        .param(ParamSpec::new("top", ParamType::Int, "pixels to add on top").default(json!(0)))
        .param(ParamSpec::new("bottom", ParamType::Int, "pixels to add on the bottom").default(json!(0)))
        .input(BlobSpec::new("image", Media::Image, "the image to extend").required())
        .output(image_out());

    let lora_train = ActionSpec::new("lora_train", "fine-tune a LoRA adapter on a few images to personalise a person/object/style")
        .streaming()
        .param(ParamSpec::new("instance_prompt", ParamType::Str, "prompt naming the subject, e.g. 'a photo of sks dog'").required())
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").default(json!(16)))
        .param(ParamSpec::new("steps", ParamType::Int, "training steps").default(json!(1000)))
        .param(ParamSpec::new("lr", ParamType::Float, "learning rate").default(json!(1e-4)))
        .input(BlobSpec::new("images", Media::Bytes, "a zip/tar of training images").required())
        .output(BlobSpec::new("adapter", Media::Bytes, "the trained LoRA safetensors"));

    Manifest::new(
        MODEL,
        "Z-Image (Tongyi) — an efficient image-generation model: text-to-image, image-to-image, masked inpainting, outpainting, and LoRA personalisation.",
        vec![text2image, image2image, inpaint, outpaint, lora_train],
    )
}

// ===================== execution =====================

use std::sync::Arc;

use capability::{Action, ActionResult, Invocation, Outcome, Progress, Provider};

/// The executable Z-Image model behind the manifest. Constructed lazily (`load`),
/// it resolves each action to a runnable [`Action`]. Weight paths come from the
/// environment (`BRAIN_ZIMAGE_DIT` / `_VAE` / `QWEN3_4B`), mirroring the crate's
/// tests, so `brain do z-image …` needs no extra flags once they are set.
pub struct ZImageProvider {
    _priv: (),
}

impl ZImageProvider {
    pub fn load() -> Result<ZImageProvider, String> {
        Ok(ZImageProvider { _priv: () })
    }
}

impl Provider for ZImageProvider {
    fn manifest(&self) -> capability::Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| Arc::new(ZAction { name: name.to_string() }) as Arc<dyn Action>)
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
}

impl Action for ZAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let paths = crate::pipeline::Paths::from_env()?;
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let mut on = |step, total, message: &str| progress(Progress { step, total, message: message.to_string() });
        match self.name.as_str() {
            "text2image" => {
                let opts = opts_from(inv, inv.get_i64("width").unwrap_or(1024) as u32, inv.get_i64("height").unwrap_or(1024) as u32);
                emit(crate::pipeline::generate(&prompt, &opts, &paths, &mut on)?)
            }
            "image2image" => {
                let (image, w, h) = blob_image(inv, "image")?;
                let opts = opts_from(inv, w, h); // output matches the input image
                let init = crate::pipeline::Init { image: &image, strength: inv.get_f64("strength").unwrap_or(0.55) as f32, mask: None };
                emit(crate::pipeline::generate_img(&prompt, &opts, &paths, init, &mut on)?)
            }
            "inpaint" => {
                let (image, w, h) = blob_image(inv, "image")?;
                let (mask, mw, mh) = blob_plane(inv, "mask")?;
                if (mw, mh) != (w, h) {
                    return Err(format!("mask is {mw}×{mh} but image is {w}×{h}; they must match"));
                }
                let opts = opts_from(inv, w, h);
                let init = crate::pipeline::Init { image: &image, strength: inv.get_f64("strength").unwrap_or(0.85) as f32, mask: Some(&mask) };
                emit(crate::pipeline::generate_img(&prompt, &opts, &paths, init, &mut on)?)
            }
            "outpaint" => {
                let (image, w, h) = blob_image(inv, "image")?;
                let g = |k: &str| inv.get_i64(k).unwrap_or(0).max(0) as usize;
                let (canvas, mask, nw, nh) = build_outpaint_canvas(&image, w as usize, h as usize, g("left"), g("right"), g("top"), g("bottom"));
                let opts = opts_from(inv, nw as u32, nh as u32);
                // The new border regenerates from scratch (strength 1); the mask
                // re-anchors the original region every step so it is preserved.
                let init = crate::pipeline::Init { image: &canvas, strength: 1.0, mask: Some(&mask) };
                emit(crate::pipeline::generate_img(&prompt, &opts, &paths, init, &mut on)?)
            }
            other => Err(format!("z-image '{other}': not yet wired (lora_train reuses the training stack)")),
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
    }
}

/// Wrap a generated [`crate::pipeline::Image`] as an image-output [`Outcome`].
fn emit(img: crate::pipeline::Image) -> ActionResult {
    use capability::{Blob, Media, Outcome};
    let bytes: Vec<u8> = img.hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
    Ok(Outcome::new()
        .set("width", json!(img.w))
        .set("height", json!(img.h))
        .blob("image", Blob::new(Media::Image, bytes).with_meta(json!({"w": img.w, "h": img.h, "c": 3}))))
}

/// Decode an RGB image blob to HWC f32 `[0,1]` + its `w,h`. Errors if `c ≠ 3`.
fn blob_image(inv: &Invocation, name: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let (hwc, w, h, c) = blob_hwc(inv, name)?;
    if c != 3 {
        return Err(format!("'{name}' must be a 3-channel RGB image (got {c} channels)"));
    }
    Ok((hwc, w, h))
}

/// Decode a mask blob to a single-channel plane `[h·w]` in `[0,1]` (channel 0).
fn blob_plane(inv: &Invocation, name: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let (hwc, w, h, c) = blob_hwc(inv, name)?;
    let plane: Vec<f32> = (0..(w as usize * h as usize)).map(|i| hwc[i * c]).collect();
    Ok((plane, w, h))
}

/// Read a named blob's raw HWC f32 payload + `{w,h,c}` metadata.
fn blob_hwc(inv: &Invocation, name: &str) -> Result<(Vec<f32>, u32, u32, usize), String> {
    let b = inv.get_blob(name).ok_or_else(|| format!("missing required input '{name}'"))?;
    let w = b.meta["w"].as_u64().ok_or_else(|| format!("'{name}' blob missing w"))? as u32;
    let h = b.meta["h"].as_u64().ok_or_else(|| format!("'{name}' blob missing h"))? as u32;
    let hwc: Vec<f32> = b.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    let c = hwc.len() / (w as usize * h as usize);
    Ok((hwc, w, h, c))
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
        // the whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["actions"].as_array().unwrap().len(), 5);
    }
}
