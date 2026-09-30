// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro's capability surface: `generate` (images and a conversation in,
//! text out: `brain-deepseekvl`'s chat action on Janus-Pro's understanding
//! composite) and `text2image` (a prompt in, a 384x384 image out, the shape
//! `/v1/images/generations` dispatches).
//!
//! The two actions load different builds of the same checkpoint (the
//! single-sequence understanding decoder, and the batched engine plus the
//! generation heads and the VQ decoder), each about 16 GB at bf16, so a
//! host keeps one of them resident at a time.

use std::sync::Mutex;

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::t2i::{Request, TextToImage};

/// The catalog id: the resolver accepts any directory holding a Janus-Pro
/// checkpoint, so the id names the family rather than a release.
pub const MODEL: &str = "brain/januspro";

/// The one image size Janus-Pro generates.
pub const IMAGE_SIZE: u32 = 384;
pub const DEFAULT_CFG_WEIGHT: f64 = 5.0;
pub const DEFAULT_TEMPERATURE: f64 = 1.0;

pub fn text2image_spec() -> ActionSpec {
    ActionSpec::new("text2image", "Janus-Pro: a prompt in, a 384x384 image out (classifier-free-guided sampling of 576 VQ-16 tokens)")
        .streaming()
        .param(ParamSpec::new("prompt", ParamType::Str, "what the image shows").required())
        .param(ParamSpec::new("width", ParamType::Int, "must be 384, the only size Janus-Pro generates").default(json!(IMAGE_SIZE)).min(IMAGE_SIZE as f64).max(IMAGE_SIZE as f64))
        .param(ParamSpec::new("height", ParamType::Int, "must be 384, the only size Janus-Pro generates").default(json!(IMAGE_SIZE)).min(IMAGE_SIZE as f64).max(IMAGE_SIZE as f64))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"))
        .param(ParamSpec::new("cfg_weight", ParamType::Float, "classifier-free guidance weight").default(json!(DEFAULT_CFG_WEIGHT)).min(0.0).max(20.0).step(0.1))
        .param(ParamSpec::new("temperature", ParamType::Float, "sampling temperature (0 = most likely token)").default(json!(DEFAULT_TEMPERATURE)).min(0.0).max(2.0).step(0.01))
        .param(ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY; overrides the model-store resolver's own pick when set").host_resolved())
        .output(BlobSpec::new("image", Media::Image, "the image: raw HWC f32 pixels in [0,1], meta {w,h,c} (capability::blob's wire convention)"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Janus-Pro-7B -- one Llama decoder that reads images (SigLIP-L + aligner) and draws them (VQ-16 tokens under \
         classifier-free guidance), at the checkpoint's own bf16.",
        vec![
            deepseekvl::caps::generate_spec("Janus-Pro: up to 8 images + a conversation in, greedy text out (streamed per token)"),
            text2image_spec(),
        ],
    )
    .with_max_context_tokens(deepseekvl::caps::MAX_CONTEXT as u64)
}

/// The manifest for the scheduled service (`weights` stripped).
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// Janus-Pro's understanding session, for serving, as `placement` puts it.
pub fn load_understanding(dir: &str, placement: deepseekvl::model::Placement) -> Result<deepseekvl::caps::Session, String> {
    let vlm = crate::model::load_understanding_placed(std::path::Path::new(dir), qwen3::Dtype::BF16, placement)?;
    // Janus-Pro's processor writes each image as `<image_placeholder>\n`.
    Ok(deepseekvl::caps::Session::new(vlm, "\n"))
}

/// Janus-Pro's generation path, for serving one image per request, on
/// `card` with `context` tokens per sequence.
pub fn load_t2i(dir: &str, card: u32, context: u32) -> Result<TextToImage, String> {
    gpu_core::devices::with_gpu(card, || TextToImage::load(std::path::Path::new(dir), 1, qwen3::Dtype::BF16, context))?
}

/// Where the understanding build goes over the cards' free memory now.
pub fn place_understanding_now(dir: &str) -> Result<deepseekvl::model::Placement, String> {
    deepseekvl::model::place(&crate::model::understanding_footprint(std::path::Path::new(dir))?, &gpu_core::capacity::available_gpus())
}

/// Where the generation build goes (card, context) over the cards' free
/// memory now.
pub fn place_t2i_now(dir: &str) -> Result<(u32, u32), String> {
    let fp = deepseekvl::model::Footprint::of(&crate::model::decoder_config(std::path::Path::new(dir))?, 0);
    crate::t2i::place(&fp, 1, &gpu_core::capacity::available_gpus())
}

/// Run one `text2image` invocation on a loaded generation path.
pub fn text2image(t2i: &mut TextToImage, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").filter(|p| !p.trim().is_empty()).ok_or("januspro text2image: 'prompt' is required")?;
    let (w, h) = (inv.get_i64("width").unwrap_or(IMAGE_SIZE as i64), inv.get_i64("height").unwrap_or(IMAGE_SIZE as i64));
    if (w, h) != (IMAGE_SIZE as i64, IMAGE_SIZE as i64) {
        return Err(format!("januspro text2image: Janus-Pro generates {IMAGE_SIZE}x{IMAGE_SIZE} images only, not {w}x{h} (request size \"{IMAGE_SIZE}x{IMAGE_SIZE}\")"));
    }
    let seed = inv.get_i64("seed").map(|s| s as u64).unwrap_or_else(rand_seed);
    let req = Request {
        prompt: &prompt,
        cfg_weight: inv.get_f64("cfg_weight").unwrap_or(DEFAULT_CFG_WEIGHT) as f32,
        temperature: inv.get_f64("temperature").unwrap_or(DEFAULT_TEMPERATURE) as f32,
        seed,
    };
    let mut images = t2i.generate(&req, &|| false, &mut |step, total| progress(Progress::step(step as u32, total as u32, "")))?;
    let img = images.pop().ok_or("januspro text2image: no image generated")?;
    let hwc: Vec<f32> = img.px.iter().map(|&v| v as f32 / 255.0).collect();
    Ok(Outcome::new().set("seed", json!(seed)).blob("image", capability::blob::image_blob(&hwc, img.w, img.h, 3)))
}

fn rand_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    (t as u64) ^ ((t >> 64) as u64) ^ (std::process::id() as u64).rotate_left(32)
}

/// The build a process keeps loaded: one of the two, never both.
enum Loaded {
    Understanding(deepseekvl::caps::Session),
    Generation(TextToImage),
}

static RESIDENT: Mutex<Option<(String, Loaded)>> = Mutex::new(None);

/// Direct provider: loads the build each action needs on first use.
#[derive(Default)]
pub struct JanusProProvider {
    default_dir: Option<String>,
}

impl JanusProProvider {
    pub fn new(default_dir: Option<String>) -> JanusProProvider {
        JanusProProvider { default_dir }
    }
}

impl Provider for JanusProProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        matches!(name, "generate" | "text2image").then(|| std::sync::Arc::new(JanusAction { name: name.to_string(), default_dir: self.default_dir.clone() }) as std::sync::Arc<dyn Action>)
    }
}

struct JanusAction {
    name: String,
    default_dir: Option<String>,
}

impl Action for JanusAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("the provider serves only declared actions")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv
            .get_str("weights")
            .filter(|s| !s.is_empty())
            .or_else(|| self.default_dir.clone())
            .ok_or("januspro: no checkpoint (pass 'weights', or configure one through the models directory)")?;
        let mut guard = RESIDENT.lock().map_err(|_| "januspro: resident lock poisoned")?;
        let want_generation = self.name == "text2image";
        let loaded = matches!(&*guard, Some((d, Loaded::Generation(_))) if *d == dir && want_generation) || matches!(&*guard, Some((d, Loaded::Understanding(_))) if *d == dir && !want_generation);
        if !loaded {
            *guard = None; // drop the other build before this one allocates
            let build = if want_generation {
                let (card, context) = place_t2i_now(&dir)?;
                Loaded::Generation(load_t2i(&dir, card, context)?)
            } else {
                Loaded::Understanding(load_understanding(&dir, place_understanding_now(&dir)?)?)
            };
            *guard = Some((dir, build));
        }
        match &mut guard.as_mut().expect("just loaded").1 {
            Loaded::Understanding(session) => session.generate(inv, progress),
            Loaded::Generation(t2i) => text2image(t2i, inv, progress),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_actions_have_the_shapes_the_api_serves() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let chat = m.actions.iter().find(|a| a.name == "generate").unwrap();
        assert!(chat.streaming && chat.params.iter().any(|p| p.name == "messages") && chat.outputs.iter().any(|o| o.media == Media::Text));
        let t2i = m.actions.iter().find(|a| a.name == "text2image").unwrap();
        assert!(t2i.outputs.iter().any(|o| o.name == "image" && o.media == Media::Image));
        assert!(t2i.params.iter().any(|p| p.name == "prompt") && !t2i.inputs.iter().any(|b| b.required), "a pure text-to-image action");
    }
}
