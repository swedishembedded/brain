// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the vision-language models its clients serve. If your team needs
// expertise in turning a chat request with pictures into bounded, validated decoder rows then you can procure our
// services by sending an email to info@swedishembedded.com.

//! How an Inkling 2 chat request with images becomes a scheduler request ([`Inkling2Images`], the builder of the shared
//! [`llmserve::image_frontend::ImageFrontend`]):
//!
//! 1. the images arrive as the request's image blobs (`image`, `image1`, ...: what the HTTP surfaces decode from inline
//!    `data:` URLs and D-Bus callers attach), one per image part of its messages, in order;
//! 2. each is held to the processor's bounds from its size alone (no side over [`inkling2::vision::MAX_SIDE`] pixels,
//!    and all of them together fewer 40x40-pixel regions than the context holds) before any pixel is processed;
//! 3. the checkpoint's own template renders the messages, client text escaped, writing ONE `<|unused_200054|>` per
//!    image part (a raw prompt with `chat: false` cannot place an image and is refused);
//! 4. the placeholder becomes one token per region of its image ([`inkling2::vision::expand_placeholders`], which also
//!    refuses image parts and images that do not match one to one);
//! 5. the tower encodes each image (a picture seen lately is served from [`EncodedImages`]), and the features stand at
//!    the placeholder positions as [`PromptEmbeds`], which the engine splices after its embedding norm.
//!
//! A request without image blobs takes the text front end, which refuses image parts left without their images
//! ([`refuse_unplaced_image_parts`]).

use std::sync::Arc;

use capability::{failure::tag, FailureClass, Invocation};
use data::qwen_tokenizer::QwenBpe;
use gpu_core::Gpu;
use imaging::{Decoded, RegionTiling};
use inkling2::vision::{expand_placeholders, splice_embeds, ImageFeatures, VisionTower};
use llmserve::image_frontend::{request_images_with_alpha, EncodedImages, ImageRequests};
use llmserve::ParsedRequest;
use model::serve::PromptEmbeds;
use qwen3::chat::ChatFormat;
use serde_json::{json, Value};

/// The content-part types the checkpoint's template renders as an image.
const IMAGE_PARTS: [&str; 3] = ["image", "input_image", "image_url"];

/// Images whose features the front end keeps: a conversation's pictures and a few concurrent ones.
const REMEMBERED_IMAGES: usize = 8;

fn invalid(message: impl AsRef<str>) -> String {
    tag(FailureClass::InvalidRequest, message)
}

/// How many image parts the messages of `inv` place (0 when it has none or they are not a JSON array; the template
/// reports a malformed list itself).
fn image_parts(inv: &Invocation) -> usize {
    let Some(messages) = inv.get_str("messages") else { return 0 };
    let Ok(Value::Array(list)) = serde_json::from_str::<Value>(&messages) else { return 0 };
    let is_image = |p: &Value| p.get("type").and_then(Value::as_str).is_some_and(|t| IMAGE_PARTS.contains(&t));
    list.iter().filter_map(|m| m.get("content").and_then(Value::as_array)).flatten().filter(|p| is_image(p)).count()
}

/// The text path's check: a request without image blobs must not place an image part (the template would render its
/// placeholder token with no features behind it), and one with image blobs reaches the text path only when the model
/// reads no images.
pub fn refuse_unplaced_image_parts(inv: &Invocation, reads_images: bool) -> Result<(), String> {
    if llmserve::image_frontend::has_images(inv) {
        return Err(invalid("inkling2: this checkpoint has no vision tower; send text only"));
    }
    let parts = image_parts(inv);
    if parts > 0 {
        let why = if reads_images { format!("the messages place {parts} image(s) but the request carries none; attach each image to its part") } else { "this checkpoint has no vision tower; send text only".to_string() };
        return Err(invalid(format!("inkling2: {why}")));
    }
    Ok(())
}

/// What building an image request needs: the chat path's tokenizer and template, the tower on the decoder's card, the
/// processor and the scheduler's context.
pub struct Inkling2Images {
    tok: QwenBpe,
    format: Arc<ChatFormat>,
    gpu: Gpu,
    tower: VisionTower,
    tiling: RegionTiling,
    image_token: u32,
    /// The most prompt tokens the scheduler holds for one sequence.
    context: usize,
    seen: EncodedImages<ImageFeatures, Decoded>,
}

impl Inkling2Images {
    /// `image_token` is the id the tokenizer gives the processor's placeholder text; `gpu` the device `tower` was
    /// uploaded to.
    pub fn new(tok: QwenBpe, format: Arc<ChatFormat>, gpu: Gpu, tower: VisionTower, tiling: RegionTiling, image_token: u32, context: usize) -> Inkling2Images {
        Inkling2Images { tok, format, gpu, tower, tiling, image_token, context, seen: EncodedImages::new(REMEMBERED_IMAGES) }
    }

    /// The tower, for callers that measure it.
    pub fn tower(&self) -> &VisionTower {
        &self.tower
    }

    /// How many images went through the tower (the rest were remembered).
    pub fn encodes(&self) -> u64 {
        self.seen.encodes()
    }

    /// Regions (decoder tokens) of an image of `w x h` pixels, refused by name past the processor's bounds.
    fn regions(&self, i: usize, w: u32, h: u32) -> Result<usize, String> {
        self.tiling.check_size(w, h).map_err(|e| invalid(format!("inkling2: image {i}: {e}")))?;
        Ok(w.div_ceil(self.tiling.region) as usize * h.div_ceil(self.tiling.region) as usize)
    }
}

impl ImageRequests for Inkling2Images {
    fn build(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        inv.cancel.check()?;
        let started = std::time::Instant::now();
        // The processor composites transparency onto its checkerboard, so the images keep their alpha (the image inputs
        // declare it, `capability::BlobSpec::alpha`).
        let images = request_images_with_alpha(inv)?;
        let regions = images.iter().enumerate().map(|(i, img)| {
            let (w, h) = img.size();
            self.regions(i, w, h)
        }).collect::<Result<Vec<usize>, String>>()?;
        let total: usize = regions.iter().sum();
        if total >= self.context {
            return Err(invalid(format!(
                "inkling2: the images need {total} tokens (one per {r}x{r}-pixel region), more than the context of {} tokens holds; send fewer or smaller images",
                self.context,
                r = self.tiling.region
            )));
        }
        if inv.get_bool("chat") == Some(false) {
            return Err(invalid("inkling2: images are read in chat messages; a raw prompt (chat: false) cannot place one"));
        }
        let text = inv.without_blobs();
        let text = match inv.get_str("messages").filter(|m| !m.trim().is_empty()) {
            Some(_) => text,
            // A bare prompt with images: one user turn, the images first.
            None => {
                let mut parts: Vec<Value> = images.iter().map(|_| json!({"type": "image_url", "image_url": {}})).collect();
                parts.push(json!({"type": "text", "text": inv.get_str("prompt").unwrap_or_default()}));
                text.set("messages", json!(json!([{"role": "user", "content": parts}]).to_string()))
            }
        };
        let mut req = qwen3::chat::parse_request_as(&self.tok, &self.format, &text)?;
        let expanded = expand_placeholders(&req.ids, self.image_token, &regions).map_err(|e| invalid(format!("inkling2: {e}")))?;
        let prepared = started.elapsed();
        let mut rows = Vec::with_capacity(total * self.tower.out_width());
        for (i, image) in images.iter().enumerate() {
            inv.cancel.check()?;
            let features = self.seen.try_rows_of(image, |img| {
                let tiled = self.tiling.preprocess(img).map_err(|e| invalid(format!("inkling2: image {i}: {e}")))?;
                self.tower.encode(&self.gpu, &[tiled])
            })?;
            rows.extend_from_slice(features.rows());
        }
        let features = ImageFeatures::new(self.tower.out_width(), regions, rows)?;
        let embeds = splice_embeds(&expanded, self.image_token, &features)?;
        residency::log::info(&format!(
            "inkling2: {} image(s), {total} tokens: decode and render {:.0} ms, tower {:.0} ms",
            images.len(),
            prepared.as_secs_f64() * 1e3,
            (started.elapsed() - prepared).as_secs_f64() * 1e3
        ));
        req.ids = expanded;
        Ok((req, Some(embeds)))
    }
}
