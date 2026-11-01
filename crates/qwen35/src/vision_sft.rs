// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language agents on a client's own verified trajectories, with
// screenshots and photographs in the conversation. If your team needs expertise in training a vision-reading model on
// the images it will be served, then you can procure our services by sending an email to info@swedishembedded.com.

//! Supervised fine-tuning on conversations whose turns hold images.
//!
//! A record is `generic-messages-v2` whose message `content` may be a list of parts, an OpenAI `image_url` part
//! holding an inline `data:` URL among them. The trainer reads it as serving reads the same request:
//!
//! * the **geometry** is the checkpoint's `preprocessor_config.json` ([`crate::vision_tower::VisionSpec`]), capped
//!   at the pixels serving caps at, so an image becomes the same number of decoder rows in both;
//! * the **text** is the chat template's render of the messages with each image expanded to its run of placeholder
//!   tokens ([`crate::vision_frontend::placeholder_run`]), the form serving renders;
//! * the **rows** and the **three-axis rotary positions** are [`crate::vision_frontend::prompt_embeds`], the function
//!   serving builds a request's [`model::serve::PromptEmbeds`] with, run on the training device through the vision
//!   tower; the decoder splices them where it splices them for serving ([`crate::model::Qwen35::attach_vision`]).
//!
//! The tower is frozen: its rows are inputs, no gradient is computed for them and nothing in it trains, and the
//! loss is the token loss of the supervised assistant turns, which never include a placeholder row (an image part
//! in a supervised turn is refused). The extra cost of an image is its rows in the decoder's sequence, its
//! forward through the tower per draw of the record, and the tower's weights and activations on the device.

use std::path::Path;
use std::sync::Arc;

use data::chat::{ImageBytes, ImageReader};
use data::chat_media::ExampleImages;
use imaging::pixels::Rgb8;
use model::serve::PromptEmbeds;
use model::Split;
use qwen3vl::preprocess::ImageGeometry;

use crate::vision_frontend::{holds_reserved_text, placeholder_run, placeholder_runs, prompt_embeds, reserved_text_refusal};
use crate::vision_tower::{read_spec, VisionSpec, VisionTokens, VisionTower, SERVING_MAX_IMAGE_PIXELS};

pub use capability::blob::{MAX_IMAGES, MAX_IMAGE_PIXELS};

/// The reader of a record's images for a checkpoint with a vision tower: the geometry and placeholder tokens serving
/// reads the same checkpoint with.
pub struct VisionReader {
    geometry: ImageGeometry,
    tokens: VisionTokens,
}

impl VisionReader {
    pub fn new(spec: &VisionSpec) -> VisionReader {
        VisionReader { geometry: spec.geometry.clone().capped_at(SERVING_MAX_IMAGE_PIXELS), tokens: spec.tokens }
    }

    /// The reader of the checkpoint in `dir`; `None` when it has no vision tower, which reads text only.
    pub fn open(dir: &Path) -> Result<Option<VisionReader>, String> {
        Ok(read_spec(dir)?.map(|spec| VisionReader::new(&spec)))
    }
}

impl ImageReader for VisionReader {
    fn image(&self, url: &str) -> Result<(ImageBytes, String), String> {
        let payload = capability::blob::data_url_payload(url).ok_or("image_url: only inline 'data:...;base64,...' URLs are read (no external fetch)")?;
        let bytes = data::base64::decode(payload)?;
        let image = imaging::codec::decode_bounded(&bytes, MAX_IMAGE_PIXELS).map_err(|e| format!("the image cannot be decoded: {e}"))?;
        let (rows, _) = self.geometry.tokens(image.h, image.w);
        Ok((Arc::from(bytes), placeholder_run(rows)))
    }

    fn check_text(&self, text: &str) -> Result<(), String> {
        if holds_reserved_text(text) {
            return Err(reserved_text_refusal());
        }
        Ok(())
    }

    fn placeholder_token(&self) -> Option<u32> {
        Some(self.tokens.image)
    }

    fn check_record(&self, media: &[ImageBytes]) -> Result<(), String> {
        if media.len() > MAX_IMAGES {
            return Err(format!("the record carries {} images, at most {MAX_IMAGES} are read", media.len()));
        }
        Ok(())
    }
}

/// The images of a prepared dataset and the tower that turns them into rows: what a [`crate::model::Qwen35`] reads a
/// batch of examples through.
pub struct VisionFeed {
    tower: VisionTower,
    train: Option<ExampleImages>,
    val: Option<ExampleImages>,
}

impl VisionFeed {
    /// The feed of the prepared dataset in `dir` over `tower`, or `None` when neither split holds an image.
    pub fn open(dir: &Path, tower: VisionTower) -> Result<Option<VisionFeed>, String> {
        let open = |split: &str| ExampleImages::open(dir, split).map_err(|e| format!("{}: {e}", dir.display()));
        let (train, val) = (open("train")?, open("val")?);
        Ok((train.is_some() || val.is_some()).then_some(VisionFeed { tower, train, val }))
    }

    /// The rows and rotary positions of example `example` of `split`, whose row of the batch holds `ids` (the
    /// example's tokens, then padding); `None` for an example without images.
    pub fn embeds(&self, split: Split, example: usize, ids: &[u32]) -> Result<Option<PromptEmbeds>, String> {
        let images = match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
        };
        let Some(images) = images.as_ref().filter(|i| i.count(example) > 0) else { return Ok(None) };
        let files = images.images(example).map_err(|e| format!("{split:?} example {example}: {e}"))?;
        let decoded = decode_all(&files).map_err(|e| format!("{split:?} example {example}: {e}"))?;
        prompt_embeds(&self.tower, ids, &decoded, &|| false)
            .map(Some)
            .map_err(|refusal| format!("{split:?} example {example}: {refusal:?}"))
    }
}

fn decode_all(files: &[Vec<u8>]) -> Result<Vec<Rgb8>, String> {
    files.iter().enumerate().map(|(i, bytes)| imaging::codec::decode_bounded(bytes, MAX_IMAGE_PIXELS).map_err(|e| format!("image {i} cannot be decoded: {e}"))).collect()
}

/// Check a prepared dataset's images against the checkpoint that will train on it, before any weight is loaded:
/// every example's images decode, and its token stream holds one run of placeholder tokens per image, as long as the
/// image's geometry says. The first example that does not is named.
pub fn check_prepared(dir: &Path, spec: &VisionSpec) -> Result<(), String> {
    let reader = VisionReader::new(spec);
    for split in ["train", "val"] {
        let Some(images) = ExampleImages::open(dir, split).map_err(|e| format!("{}: {e}", dir.display()))? else { continue };
        let ids = data::binio::read_tokens_u32(&dir.join(split)).map_err(|e| format!("{}: {e}", dir.display()))?;
        let starts = data::binio::read_u64_bin(&dir.join(format!("{split}.ex.bin"))).map_err(|e| format!("{}: {e}", dir.display()))?;
        if images.examples() != starts.len() {
            return Err(format!("{split}: images are indexed for {} examples, the dataset has {}", images.examples(), starts.len()));
        }
        for example in 0..starts.len() {
            let files = images.images(example).map_err(|e| format!("{split} example {example}: {e}"))?;
            let end = starts.get(example + 1).map_or(ids.len(), |&s| s as usize);
            let runs = placeholder_runs(&ids[starts[example] as usize..end], reader.tokens.image);
            let wanted: Vec<usize> = decode_all(&files).map_err(|e| format!("{split} example {example}: {e}"))?.iter().map(|i| reader.geometry.tokens(i.h, i.w).0 as usize).collect();
            if runs != wanted {
                return Err(format!("{split} example {example}: its text places image runs of {runs:?} tokens, its images need {wanted:?}"));
            }
        }
    }
    Ok(())
}

/// The directory holding `base`'s vision release: the checkpoint directory itself, or for a single-file checkpoint
/// (brain's own format) the directory the file sits in.
fn vision_dir(base: &Path) -> &Path {
    if base.is_dir() {
        base
    } else {
        base.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
    }
}

/// Attach the vision tower of `base` and the images of the prepared dataset in `dir` to `model`, when the dataset
/// has any. A dataset with images and a base without a vision tower is refused, naming both.
pub fn attach_for_dataset(model: &crate::model::Qwen35, base: &Path, dir: &Path) -> std::io::Result<()> {
    let has_images = ["train", "val"].iter().any(|split| dir.join(format!("{split}.img.bin")).exists());
    if !has_images {
        return Ok(());
    }
    let other = std::io::Error::other;
    let release = vision_dir(base);
    let spec = read_spec(release).map_err(other)?.ok_or_else(|| other(format!("the dataset {} holds images, and the base {} has no vision tower to read them with", dir.display(), base.display())))?;
    check_prepared(dir, &spec).map_err(other)?;
    let tower = VisionTower::open(release, spec, &model.gpu, model.cfg.d_model, SERVING_MAX_IMAGE_PIXELS).map_err(other)?;
    let feed = VisionFeed::open(dir, tower).map_err(other)?.expect("a dataset with an image sidecar has a feed");
    model.attach_vision(feed);
    Ok(())
}
