// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A `/videos` request's frame images and reference media, delivered to the
//! action that takes them.
//!
//! A model takes stills and references as host-resolved path params (a CLI
//! user names files on the machine); a remote caller can never name a path.
//! The action's [`VideoContract`] says which params a request's
//! `first_frame`/`last_frame` images and its `input_references` fill, and this
//! module is the host answering them: it decodes each inline `data:` URL at
//! the boundary (bounded, refused when malformed), writes it in a format the
//! model reads by extension (images as PNG, sound as WAV, video in its own
//! container) to a private directory of the job's own, and hands back the
//! params to set. The directory is removed when its [`JobInputs`] is dropped,
//! which the job does when it ends, however it ends.
//!
//! Swedish Embedded AB implements solutions for serving generative video
//! behind standard APIs for its clients. If your team needs expertise in
//! media-serving infrastructure then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::video::{reference_type, VideoContract, FRAME_TYPES};
use capability::Media;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use uuid::Uuid;

/// One `frame_images` entry: an `image_url` content part with its frame type.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameImage {
    #[serde(rename = "type")]
    kind: String,
    image_url: Url,
    frame_type: String,
}

/// One `input_references` entry: an image, video or audio content part.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reference {
    ImageUrl { image_url: Url },
    VideoUrl { video_url: Url },
    AudioUrl { audio_url: Url },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Url {
    url: String,
}

impl Reference {
    fn media(&self) -> Media {
        match self {
            Reference::ImageUrl { .. } => Media::Image,
            Reference::VideoUrl { .. } => Media::Video,
            Reference::AudioUrl { .. } => Media::Audio,
        }
    }
    fn url(&self) -> &str {
        match self {
            Reference::ImageUrl { image_url: u } | Reference::VideoUrl { video_url: u } | Reference::AudioUrl { audio_url: u } => &u.url,
        }
    }
}

/// What a request asks an action to take besides its params.
#[derive(Default)]
pub struct Wanted {
    /// `(frame_type, data URL)`, each frame type at most once.
    frames: Vec<(String, String)>,
    references: Vec<Reference>,
}

impl Wanted {
    /// The request's frame images and references, checked for shape: known,
    /// unrepeated frame types and `image_url` parts.
    pub fn new(frames: Vec<FrameImage>, references: Vec<Reference>) -> Result<Wanted, String> {
        let mut out = Wanted { frames: Vec::new(), references };
        for f in frames {
            if f.kind != "image_url" {
                return Err(format!("frame_images: type must be \"image_url\", not \"{}\"", f.kind));
            }
            if !FRAME_TYPES.contains(&f.frame_type.as_str()) {
                return Err(format!("frame_images: frame_type \"{}\" is not one of {}", f.frame_type, FRAME_TYPES.join(", ")));
            }
            if out.frames.iter().any(|(t, _)| *t == f.frame_type) {
                return Err(format!("frame_images: {} is given twice", f.frame_type));
            }
            out.frames.push((f.frame_type, f.image_url.url));
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty() && self.references.is_empty()
    }

    /// Why `contract` cannot take this request's inputs, or `None` when it can.
    pub fn unmet(&self, contract: &VideoContract) -> Option<String> {
        if let Some((t, _)) = self.frames.iter().find(|(t, _)| contract.frame_param(t).is_none()) {
            return Some(format!("takes no {t} image"));
        }
        if self.references.is_empty() {
            return None;
        }
        let Some(refs) = &contract.references else { return Some("takes no input_references".into()) };
        if let Some(r) = self.references.iter().find(|r| !refs.media.contains(&r.media())) {
            return Some(format!("takes no {} references", reference_type(r.media())));
        }
        (self.references.len() > refs.max).then(|| format!("takes at most {} references", refs.max))
    }

    /// Decode every input, write it under a new private directory of the
    /// job's own and return that directory with the params naming the files.
    /// Blocking: decoding and writing run on the caller's thread.
    pub fn materialize(&self, root: &Path, contract: &VideoContract) -> Result<(JobInputs, Map<String, Value>), String> {
        let inputs = JobInputs::create(root)?;
        let mut params = Map::new();
        for (frame_type, url) in &self.frames {
            let param = contract.frame_param(frame_type).ok_or_else(|| format!("takes no {frame_type} image"))?;
            let path = write_image(&inputs.dir, frame_type, url).map_err(|e| format!("frame_images ({frame_type}): {e}"))?;
            params.insert(param.into(), json!(path_str(&path)?));
        }
        if let (Some(refs), false) = (&contract.references, self.references.is_empty()) {
            let mut paths = Vec::with_capacity(self.references.len());
            for (i, r) in self.references.iter().enumerate() {
                let name = format!("reference-{i}");
                let path = match r.media() {
                    Media::Image => write_image(&inputs.dir, &name, r.url()),
                    Media::Audio => write_audio(&inputs.dir, &name, r.url()),
                    _ => write_video(&inputs.dir, &name, r.url()),
                }
                .map_err(|e| format!("input_references[{i}]: {e}"))?;
                paths.push(path_str(&path)?.to_owned());
            }
            params.insert(refs.param.clone(), json!(paths.join(&refs.separator.to_string())));
        }
        Ok((inputs, params))
    }
}

/// A job's private copies of its inputs, removed with it.
pub struct JobInputs {
    dir: PathBuf,
}

impl JobInputs {
    fn create(root: &Path) -> Result<JobInputs, String> {
        let dir = root.join(Uuid::new_v4().to_string());
        crate::surface::private_dir(&dir).map_err(|e| format!("the job's input directory: {e}"))?;
        Ok(JobInputs { dir })
    }
}

impl Drop for JobInputs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The path as the model's param takes it. The directory is the server's own,
/// so it never holds a separator a reference list splits on.
fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str().ok_or_else(|| format!("{} is not valid UTF-8", path.display()))
}

/// The bytes of an inline `data:<mime>;base64,` URL whose mime starts with
/// `kind/`; a remote URL is refused, never fetched.
fn data_url_bytes<'u>(url: &'u str, kind: &str) -> Result<(&'u str, Vec<u8>), String> {
    let mime = url.strip_prefix("data:").ok_or("only inline data: URLs are accepted; remote URLs are not fetched")?.split([';', ',']).next().unwrap_or_default();
    if !mime.starts_with(&format!("{kind}/")) {
        return Err(format!("expected a data:{kind}/... URL, got data:{mime}"));
    }
    let payload = capability::blob::data_url_payload(url).ok_or("not a base64 data: URL")?;
    Ok((mime, crate::b64::decode(payload)?))
}

/// An image, decoded within the per-image pixel bound and written as PNG.
fn write_image(dir: &Path, name: &str, url: &str) -> Result<PathBuf, String> {
    let (_, bytes) = data_url_bytes(url, "image")?;
    let image = imaging::codec::decode_bounded(&bytes, capability::blob::MAX_IMAGE_PIXELS)?;
    let path = dir.join(format!("{name}.png"));
    imaging::save_png(&path, &image)?;
    Ok(path)
}

/// A WAV soundtrack, parsed and written again so the model reads only what
/// brain's own parser accepted.
fn write_audio(dir: &Path, name: &str, url: &str) -> Result<PathBuf, String> {
    let (_, bytes) = data_url_bytes(url, "audio")?;
    let (rate, channels) = audio::wav::parse_planar(&bytes).map_err(|e| format!("audio references must be WAV: {e}"))?;
    let path = dir.join(format!("{name}.wav"));
    let planes: Vec<&[f32]> = channels.iter().map(Vec::as_slice).collect();
    audio::wav::write_multi(&path, &planes, rate).map_err(|e| e.to_string())?;
    Ok(path)
}

/// An MP4 or WebM clip, kept in its own container once it is known to decode.
fn write_video(dir: &Path, name: &str, url: &str) -> Result<PathBuf, String> {
    let (mime, bytes) = data_url_bytes(url, "video")?;
    let ext = match mime {
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        other => return Err(format!("video references must be video/mp4 or video/webm, not {other}")),
    };
    let path = dir.join(format!("{name}.{ext}"));
    std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
    imaging::video::probe_fps(&path).ok_or("the clip does not decode")?;
    Ok(path)
}
