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
        let steps = inv.get_i64("steps").unwrap_or(8).max(1) as u32;
        progress(Progress { step: 0, total: steps, message: "preparing".into() });
        // The generalized interface delivered a validated invocation; the sampling
        // pipeline is the remaining assembly (tracked separately). Report it rather
        // than emit a fake result.
        let _ = steps;
        Err(format!(
            "z-image '{}' is discoverable and its arguments validated, but the end-to-end generation pipeline \
             (tokenizer → Qwen encode → {}-step flow-match sampling over the DiT → VAE decode) is not yet assembled. \
             Set BRAIN_ZIMAGE_DIT/_VAE/QWEN3_4B and wire zimage::pipeline to enable execution.",
            self.name, steps
        ))
    }
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
