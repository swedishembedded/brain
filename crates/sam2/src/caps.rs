// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2.1's capabilities behind the generalized [`capability`] interface —
//! what makes `brain caps sam2` / `brain do sam2 segment …`, the D-Bus `Run`
//! method and `brain perf`'s `CapabilityTarget` work with no SAM-specific
//! plumbing anywhere in the CLI or the transports.
//!
//! Two actions:
//!
//! - `segment`: an image plus point / box prompts in **source-image pixels**
//!   in, one mask out (the shared `capability::blob` wire format, re-tagged
//!   [`Media::Mask`]).
//! - `track`: a clip plus clicks on one of its frames in, the object's mask on
//!   every frame out, as an `imaging::maskseq` sequence ([`crate::track`]).
//!
//! # The encoder cache is the whole design
//!
//! SAM 2 is an *encode-once, prompt-many* model: the Hiera trunk is ~99 % of the
//! work and depends only on the image, while the two-way mask decoder is tiny
//! and depends on the prompt. [`Session`] therefore keeps the last [`Encoded`]
//! keyed by a hash of the image blob, so a second prompt on the same image skips
//! the trunk entirely. That cache is also what makes real batching possible —
//! see `resident_sam2::Sam2Instance::run_batch`, which groups a batch of
//! invocations by image so N prompts on one image cost ONE trunk pass.
//!
//! # Two honest approximations, both forced by a missing kernel
//!
//! 1. The reference resizes the source image to `image_size²` with
//!    `torchvision.Resize(antialias=True)`; brain has no antialiased resize
//!    kernel (`resize_bilinear` is the plain one — the same gap the crate docs
//!    already record for the mask prompt), so downscaling a large photo here is
//!    plain bilinear and will differ slightly from the reference on
//!    high-frequency content. Parity is gated on the *normalized model input*
//!    (`tests/parity.rs` rung 0), which this path does not change.
//! 2. The mask comes back as `sigmoid(logits)` resampled to the source grid, not
//!    a hard `logits > 0` — thresholding needs `bsq_quantize`
//!    (`imaging::mask::threshold`), which is not in [`crate::model::PIPELINES`].
//!    `prob > 0.5` is exactly `logit > 0`, so the client can threshold for free.

use std::sync::Arc;

use capability::presentation::maturity;
use capability::ModelPresentation;
use capability::{
    hot::Hot, Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Ctx, Filter, Shape};
use serde_json::json;

use crate::config::Sam2Config;
use crate::model::{Encoded, Prompt, Sam2, PIPELINES};

/// The model id used on the CLI (`brain do sam2 …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/sam2";

/// The released variants, by the name the `variant` param takes.
pub fn variant_config(name: &str) -> Result<Sam2Config, String> {
    match name {
        "tiny" => Ok(Sam2Config::hiera_tiny()),
        "large" => Ok(Sam2Config::hiera_large()),
        other => Err(format!("sam2: unknown variant '{other}' (tiny|large)")),
    }
}

/// The `segment` schema. Kept as one function so the [`Provider`], the residency
/// adapter and the static manifest cannot drift apart.
pub fn segment_spec() -> ActionSpec {
    ActionSpec::new("segment", "promptable segmentation: point/box prompts on an image -> a mask (SAM 2.1)")
        .param(ParamSpec::new("variant", ParamType::Enum(vec!["tiny".into(), "large".into()]), "released checkpoint variant").default(json!("tiny")))
        .param(ParamSpec::new(
            "points",
            ParamType::Str,
            "foreground/background points in SOURCE-image pixels, 'x,y;x,y;…' (empty = none)",
        ).default(json!("")))
        .param(ParamSpec::new(
            "labels",
            ParamType::Str,
            "one label per point, '1;0;…' (1 = foreground, 0 = background); empty = all foreground",
        ).default(json!("")))
        .param(ParamSpec::new("box", ParamType::Str, "a box prompt in source-image pixels, 'x1,y1,x2,y2' (empty = none)").default(json!("")))
        .param(ParamSpec::new("multimask", ParamType::Bool, "return the 3-way ambiguity head and pick the highest-IoU mask").default(json!(true)))
        .input(BlobSpec::new("image", Media::Image, "the image to segment").required())
        .output(BlobSpec::new("mask", Media::Mask, "sigmoid mask probability on the SOURCE image grid (single channel; threshold at 0.5)"))
}

/// The `track` schema, shared by the [`Provider`] and the residency adapter.
pub fn track_spec() -> ActionSpec {
    ActionSpec::new("track", "follow one object through a clip: clicks on one frame -> the object's mask on every frame (SAM 2.1 memory bank)")
        .streaming()
        .param(ParamSpec::new("variant", ParamType::Enum(vec!["tiny".into(), "large".into()]), "released checkpoint variant").default(json!("tiny")))
        .param(ParamSpec::new("points", ParamType::Str, "clicks ON the object in source pixels, 'x,y;x,y' - several pick one object together (head, coat and legs of a person)").required())
        .param(ParamSpec::new("exclude", ParamType::Str, "clicks on things that are NOT the object, 'x,y;x,y'").default(json!("")))
        .param(ParamSpec::new("prompt_frame", ParamType::Int, "the frame the clicks sit on; frames before it take its mask").min(0.0).max(capability::video::MAX_EDIT_CLIP_FRAMES as f64).default(json!(0)))
        .param(ParamSpec::new("fps", ParamType::Float, "resample the clip to this frame rate first; for a directory of frames, the rate they were taken at (default: the clip's own)").min(0.01).max(capability::bounds::FPS))
        .param(ParamSpec::new("max_frames", ParamType::Int, "track at most this many frames (0: the whole clip)").min(0.0).max(capability::video::MAX_EDIT_CLIP_FRAMES as f64).default(json!(0)))
        .param(ParamSpec::new("object_id", ParamType::Int, "the id the sequence records, for work with several objects").min(0.0).max(u32::MAX as f64).default(json!(0)))
        .param(ParamSpec::new("polarity", ParamType::Enum(vec!["object=255".into(), "object=0".into()]), "which value marks the object: object=255 (white object), or object=0 for an editor whose white means 'keep'").default(json!("object=255")))
        .param(ParamSpec::new("soft", ParamType::Bool, "keep SAM 2's probability ramp instead of hard 0/1 masks").default(json!(false)))
        .input(BlobSpec::new("video", Media::Video, "the clip, decoded whole at its own frame rate").required().clip())
        .output(BlobSpec::new("masks", Media::Video, "the object's mask on every frame: a mask sequence (one channel, its manifest in meta.maskseq)"))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "SAM 2.1 promptable segmentation: points and boxes -> a mask on an image, or one object's mask on every frame of a clip.", vec![segment_spec(), track_spec()])
        .with_presentation(
            ModelPresentation::default()
                .display_name("SAM 2.1")
                .vendor("Meta")
                .family("SAM")
                .version("2.1")
                .task_tags(&["segmentation"])
                .maturity(maturity::STABLE)
                .default_action("segment")
                .license("apache-2.0"),
        )
}

/// One checkpoint tensor: its name, shape and values.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

/// The tensors of a SAM 2.1 release checkpoint (`sam2.1_hiera_*.pt`, or an
/// equivalent `.safetensors`), named as the importer expects.
///
/// The archive root is `{"model": state_dict}` and `checkpoint::torchpt`
/// flattens with `'.'`, so the `model.` prefix is stripped here exactly as
/// `tests/parity.rs` does — one convention, two call sites, no third spelling.
pub fn read_checkpoint(weights: &str) -> Result<Vec<NamedTensor>, String> {
    let raw: Vec<NamedTensor> = if weights.ends_with(".safetensors") {
        checkpoint::safetensors::read(weights)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    } else {
        checkpoint::torchpt::read(weights)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    };
    if raw.is_empty() {
        return Err(format!("sam2: {weights} contains no tensors"));
    }
    Ok(raw.into_iter().map(|(n, s, d)| (n.strip_prefix("model.").unwrap_or(&n).to_string(), s, d)).collect())
}

/// Read a SAM 2.1 release checkpoint and build the image model on `gpu`.
pub fn load(weights: &str, cfg: Sam2Config, gpu: Gpu) -> Result<Sam2, String> {
    let (weights, _report) = crate::import::import(read_checkpoint(weights)?, &cfg)?;
    Ok(Sam2::new(gpu, cfg, &weights))
}

// ===================== the shared work =====================

/// A built SAM 2 plus its one-entry image-encoder cache — the single
/// implementation of `segment`, shared by the [`Sam2Provider`] and the residency
/// adapter (`crates/catalog/src/resident_sam2.rs`). Neither owns a second copy of
/// the preprocessing, the prompt parsing or the mask emission.
pub struct Session {
    model: Sam2,
    /// `(image-blob hash, source w, source h, encoding)` of the last image.
    cache: Option<(u64, u32, u32, Encoded)>,
}

impl Session {
    pub fn new(model: Sam2) -> Session {
        Session { model, cache: None }
    }

    pub fn model(&self) -> &Sam2 {
        &self.model
    }

    /// A cheap FNV-1a over the image payload — the cache key. A collision would
    /// segment a *different* image, so this is deliberately over the full bytes
    /// rather than over a sample of them.
    pub fn image_key(inv: &Invocation) -> u64 {
        let Some(b) = inv.get_blob("image") else { return 0 };
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &byte in &b.bytes {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Encode `inv`'s image if it is not the one already cached. Returns the
    /// source `(w, h)`.
    fn ensure_encoded(&mut self, inv: &Invocation) -> Result<(u32, u32), String> {
        let key = Session::image_key(inv);
        if let Some((k, w, h, _)) = &self.cache {
            if *k == key {
                return Ok((*w, *h));
            }
        }
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let enc = self.encode_pixels(&hwc, w, h);
        self.cache = Some((key, w, h, enc));
        Ok((w, h))
    }

    /// Run one `segment` invocation (already validated against
    /// [`segment_spec`]).
    pub fn segment(&mut self, inv: &Invocation) -> ActionResult {
        let (src_coords, labels) = parse_prompt(inv)?;
        let (w, h) = self.ensure_encoded(inv)?;
        let multimask = inv.get_bool("multimask").unwrap_or(true);
        let out = self.run_prompt(w, h, &src_coords, &labels, multimask)?;
        Ok(Outcome::new()
            .set("width", json!(out.width))
            .set("height", json!(out.height))
            .set("masks", json!(out.masks))
            .set("best", json!(out.best))
            .set("iou", json!(out.iou))
            .set("ious", json!(out.ious))
            .set("area", json!(out.area))
            .set("object_score", json!(out.object_score))
            .blob("mask", capability::blob::image_blob(&out.mask, out.width, out.height, 1).with_media(Media::Mask)))
    }

    /// [`Session::segment`]'s typed core, over raw pixels/dimensions rather
    /// than a wire-format [`Invocation`] - the same decode-then-typed-core
    /// split `codeformer::caps::Session::restore`/`rrdbnet::caps::Upscaler`
    /// already established, so a caller with pixels already in hand
    /// (`crates/sdk`'s `SegmentPipeline`) does not have to round-trip
    /// through a `Blob` to reach it. Caches the trunk encoding by a hash of
    /// `hwc` itself - a DIFFERENT cache key than [`Session::ensure_encoded`]'s
    /// (a hash of the wire-format blob bytes), since a typed caller has no
    /// blob to hash cheaply before deciding whether to decode; equally
    /// content-addressed, just over the representation each caller actually
    /// has first.
    pub fn segment_typed(&mut self, hwc: &[f32], w: u32, h: u32, coords_src: &[(f32, f32)], labels: &[f32], multimask: bool) -> Result<SegmentOutput, String> {
        self.ensure_encoded_pixels(hwc, w, h);
        self.run_prompt(w, h, coords_src, labels, multimask)
    }

    /// Encode `hwc` (source-image HWC `f32` `[0,1]`, `w*h*3`) if it is not
    /// already the one cached.
    fn ensure_encoded_pixels(&mut self, hwc: &[f32], w: u32, h: u32) {
        let key = pixel_key(hwc);
        if let Some((k, cw, ch, _)) = &self.cache {
            if *k == key && *cw == w && *ch == h {
                return;
            }
        }
        let enc = self.encode_pixels(hwc, w, h);
        self.cache = Some((key, w, h, enc));
    }

    /// The device-side resize/normalize/trunk-encode sequence both
    /// [`Session::ensure_encoded`] and [`Session::ensure_encoded_pixels`]
    /// need on a cache miss - the one real implementation, cached by two
    /// different keys for the two different callers above.
    fn encode_pixels(&self, hwc: &[f32], w: u32, h: u32) -> Encoded {
        let s = self.model.cfg.image_size;
        // Resize to the model's square input ON THE DEVICE (`resize_bilinear`)
        // — a host loop over a 4K frame would be invisible to `--device`. The
        // layout permutation around it is host glue by the `crates/imaging`
        // rule. `Sam2::preprocess` then applies the model's own pixel_mean/std,
        // so this path adds no second normalisation.
        let chw = imaging::pixels::hwc_to_chw(hwc, 3, h as usize, w as usize);
        let ctx = Ctx::new(&self.model.gpu);
        let src = ctx.upload("sam2.caps.src", &chw);
        let (dev, out_shape) = ctx.resize(&src, Shape::new(1, 3, h, w), s, s, Filter::Bilinear, AlignCorners::HalfPixel);
        let resized = ctx.download(&dev, out_shape.numel());
        let img = self.model.preprocess(&resized);
        self.model.encode(&img)
    }

    /// One prompt against whichever image is ALREADY cached at `(w, h)` - the
    /// shared "prompt in model space, decode, resample+squash the mask" math
    /// [`Session::segment`]/[`Session::segment_typed`] both need, factored out
    /// so encoding (two different cache-key strategies above) and prompting
    /// (one implementation) do not have to be the same function.
    fn run_prompt(&self, w: u32, h: u32, coords_src: &[(f32, f32)], labels: &[f32], multimask: bool) -> Result<SegmentOutput, String> {
        if coords_src.is_empty() {
            return Err("sam2 segment: give at least one point or a box prompt".into());
        }
        if coords_src.len() != labels.len() {
            return Err(format!("sam2 segment: {} coords but {} labels", coords_src.len(), labels.len()));
        }
        let s = self.model.cfg.image_size as f32;
        // Source pixels -> the model's square frame. The resize above is a plain
        // (non-aspect-preserving) stretch, exactly like the reference's
        // `Resize((1024, 1024))`, so each axis scales independently.
        let (sx, sy) = (s / w as f32, s / h as f32);
        let coords: Vec<(f32, f32)> = coords_src.iter().map(|&(x, y)| (x * sx, y * sy)).collect();

        let prompt = Prompt {
            coords,
            labels: labels.to_vec(),
            // The reference downsamples a full-resolution mask prompt with
            // `interpolate(antialias=True)`, which brain has no kernel for (see
            // the crate docs), so no mask prompt is exposed over the wire.
            mask_lowres: None,
            multimask_output: multimask,
        };
        let enc = &self.cache.as_ref().expect("encoded above").3;
        let dec = self.model.decode(enc, &prompt);

        // Masks are logits at `image_size²`; resample every candidate to the
        // source grid and squash — both on the device.
        let sz = self.model.cfg.image_size;
        let n = dec.n_masks;
        let ctx = Ctx::new(&self.model.gpu);
        let (resized, rshape) =
            ctx.resize(&dec.high_res_multimasks, Shape::new(1, n, sz, sz), h, w, Filter::Bilinear, AlignCorners::HalfPixel);
        let probs = ctx.buf(rshape.numel());
        let sig = self
            .model
            .gpu
            .kernel_index("sigmoid")
            .ok_or("sam2: the 'sigmoid' kernel is not registered on this device")?;
        self.model.gpu.submit(&[], &[self.model.gpu.step(sig, &[&resized, &probs], &[rshape.numel()], rshape.numel())]);
        let all = ctx.download(&probs, rshape.numel());

        let px = (w * h) as usize;
        let best = dec.best_iou_index.min(n as usize - 1);
        let mask = all[best * px..(best + 1) * px].to_vec();
        // A reduction to one scalar — host by the `crates/imaging` rule (the
        // readback dominates, and the per-pixel work already ran on the device).
        let area = mask.iter().filter(|&&p| p > 0.5).count();
        let object_score = self.model.gpu.read(&dec.object_score_logits, 1)[0];

        Ok(SegmentOutput { mask, width: w, height: h, masks: n, best, iou: dec.ious.get(best).copied().unwrap_or(0.0), ious: dec.ious.clone(), area, object_score })
    }
}

/// [`Session::segment_typed`]'s output: the winning mask (sigmoid
/// probability, source grid, `width*height`) plus the same metadata
/// [`Session::segment`]'s `Outcome` carries over the wire.
pub struct SegmentOutput {
    pub mask: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub masks: u32,
    pub best: usize,
    pub iou: f32,
    pub ious: Vec<f32>,
    pub area: usize,
    pub object_score: f32,
}

/// A cheap FNV-1a over raw pixel floats' bit patterns - [`Session::
/// segment_typed`]'s cache key, the pixel-based counterpart to
/// [`Session::image_key`]'s wire-blob-based one.
fn pixel_key(hwc: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in hwc {
        for byte in v.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A parsed prompt in **source-image pixels**: the `(x, y)` clicks and their
/// labels (1 foreground, 0 background, 2/3 box corners) — the two vectors
/// [`Prompt`] takes.
pub type PromptPoints = (Vec<(f32, f32)>, Vec<f32>);

/// Parse the `box` / `points` / `labels` params into [`PromptPoints`] in
/// **source-image pixels**. Pure string work: callable before a checkpoint is
/// read, which is why a malformed prompt costs no weight load.
///
/// A box is two labelled points, and the reference concatenates it BEFORE the
/// click points (`SAM2ImagePredictor._prep_prompts`). Order matters — the prompt
/// tokens are positional.
pub fn parse_prompt(inv: &Invocation) -> Result<PromptPoints, String> {
    let mut coords: Vec<(f32, f32)> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    let bx = inv.get_str("box").unwrap_or_default();
    if !bx.trim().is_empty() {
        let v = parse_floats(&bx)?;
        if v.len() != 4 {
            return Err(format!("sam2: box wants 'x1,y1,x2,y2', got {} numbers", v.len()));
        }
        coords.push((v[0], v[1]));
        coords.push((v[2], v[3]));
        labels.push(2.0);
        labels.push(3.0);
    }
    let mut clicks: Vec<(f32, f32)> = Vec::new();
    for group in inv.get_str("points").unwrap_or_default().split(';').map(str::trim).filter(|g| !g.is_empty()) {
        let v = parse_floats(group)?;
        if v.len() != 2 {
            return Err(format!("sam2: point '{group}' wants 'x,y'"));
        }
        clicks.push((v[0], v[1]));
    }
    let lbl = inv.get_str("labels").unwrap_or_default();
    let click_labels: Vec<f32> = if lbl.trim().is_empty() {
        vec![1.0; clicks.len()]
    } else {
        let mut v = Vec::new();
        for t in lbl.split(';').map(str::trim).filter(|t| !t.is_empty()) {
            v.push(t.parse::<f32>().map_err(|_| format!("sam2: label '{t}' is not a number"))?);
        }
        if v.len() != clicks.len() {
            return Err(format!("sam2: {} labels for {} points", v.len(), clicks.len()));
        }
        v
    };
    coords.extend(clicks);
    labels.extend(click_labels);
    if coords.is_empty() {
        return Err("sam2 segment: give at least one point or a box prompt".into());
    }
    Ok((coords, labels))
}

/// `"1.5, 2"` -> `[1.5, 2.0]`.
fn parse_floats(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f32>().map_err(|_| format!("sam2: '{t}' is not a number")))
        .collect()
}

// ===================== tracking =====================

/// A `track` call checked and decoded, before any weight is read: the
/// frames to track, their size and rate, and the request.
pub struct TrackCall {
    pub frames: Vec<Vec<f32>>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub request: crate::track::TrackRequest,
    /// The caller's cancellation, polled between frames.
    pub cancel: capability::CancelToken,
}

/// Decode and check a `track` invocation. Every refusal (no clicks, a click
/// off the frame, a clip with no frame rate) happens here, so it costs no
/// checkpoint read.
pub fn track_call(inv: &Invocation) -> Result<TrackCall, String> {
    let blob = inv.get_blob("video").ok_or("sam2 track: 'video' is required")?;
    let own_fps = blob.meta.get("fps").and_then(|v| v.as_f64()).filter(|f| *f > 0.0);
    let want_fps = inv.get_f64("fps").filter(|f| *f > 0.0);
    let mut decoded = capability::blob::decode_video(inv, "video")?;
    let fps = match (own_fps, want_fps) {
        (Some(own), Some(want)) => {
            let keep = crate::track::resample_indices(decoded.len(), own, want);
            decoded = keep.into_iter().map(|i| decoded[i].clone()).collect();
            want
        }
        (Some(own), None) => own,
        (None, Some(want)) => want,
        (None, None) => return Err("sam2 track: the clip states no frame rate (a directory of frames); give it with fps".into()),
    };
    if let Some(max) = inv.get_i64("max_frames").filter(|&m| m > 0) {
        decoded.truncate(max as usize);
    }
    let (width, height) = (decoded[0].1, decoded[0].2);
    let mut points = clicks(&inv.get_str("points").unwrap_or_default(), 1.0)?;
    points.extend(clicks(&inv.get_str("exclude").unwrap_or_default(), 0.0)?);
    let request = crate::track::TrackRequest {
        points,
        prompt_frame: inv.get_i64("prompt_frame").unwrap_or(0).max(0) as usize,
        object_id: inv.get_i64("object_id").unwrap_or(0).max(0) as u32,
        polarity: imaging::maskseq::Polarity::parse(&inv.get_str("polarity").unwrap_or_else(|| "object=255".into()))?,
        binary: !inv.get_bool("soft").unwrap_or(false),
    };
    request.check(decoded.len(), width, height)?;
    Ok(TrackCall { frames: decoded.into_iter().map(|f| f.0).collect(), width, height, fps, request, cancel: inv.cancel.clone() })
}

/// `'x,y;x,y'` as `(x, y, label)` clicks.
fn clicks(s: &str, label: f32) -> Result<Vec<(f32, f32, f32)>, String> {
    s.split(';')
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .map(|g| match parse_floats(g)?[..] {
            [x, y] => Ok((x, y, label)),
            _ => Err(format!("sam2: click '{g}' wants 'x,y'")),
        })
        .collect()
}

/// Run a checked `track` call on a tracking model ([`crate::track::load_video`]).
pub fn run_track(model: &Sam2, call: &TrackCall, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let masks = crate::track::track_clip(model, &call.frames, call.width, call.height, call.fps, &call.request, &mut |done, total| {
        progress(Progress::step(done as u32, total as u32, "tracking"));
        !call.cancel.is_cancelled()
    })?;
    let occluded: Vec<usize> = masks.occluded_frames().into_keys().collect();
    Ok(Outcome::new().set("frames", json!(masks.frames.len())).set("occluded_frames", json!(occluded)).blob("masks", masks.to_blob()))
}

// ===================== the provider =====================

/// The executable SAM 2 model behind the manifest. Construction is free — the
/// checkpoint imports lazily on the first `segment` and stays resident until a
/// call names a different checkpoint.
pub struct Sam2Provider {
    weights: String,
    hot: Arc<Hot<String, Session>>,
    /// The tracking model (the whole checkpoint, memory bank included), kept
    /// apart from the image session so a `segment` never pays for it.
    tracker: Arc<Hot<String, Sam2>>,
}

impl Sam2Provider {
    /// `weights` is the checkpoint path - resolved by `brain_modelstore::resolve`
    /// (see [`crate::spec::Sam2Spec`]) from the models directory, never a
    /// baked-in path or an env var read directly here.
    pub fn new(weights: impl Into<String>) -> Sam2Provider {
        Sam2Provider { weights: weights.into(), hot: Arc::new(Hot::new()), tracker: Arc::new(Hot::new()) }
    }
}

/// The provider over a resolved [`capability::Assembly`] (its `weights` role).
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(Sam2Provider::new(assembly.role_path("weights")?)))
}

impl Provider for Sam2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "segment" => Some(Arc::new(SegmentAction { weights: self.weights.clone(), hot: self.hot.clone() })),
            "track" => Some(Arc::new(TrackAction { weights: self.weights.clone(), hot: self.tracker.clone() })),
            _ => None,
        }
    }
}

struct SegmentAction {
    weights: String,
    hot: Arc<Hot<String, Session>>,
}

impl Action for SegmentAction {
    fn spec(&self) -> ActionSpec {
        segment_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Reject a malformed/absent prompt BEFORE a 150 MB checkpoint read.
        parse_prompt(inv)?;
        let variant = inv.get_str("variant").unwrap_or_else(|| "tiny".into());
        let key = format!("{}:{variant}", self.weights);
        self.hot.try_get_mut(
            key,
            || Ok(Session::new(load(&self.weights, variant_config(&variant)?, Gpu::new(PIPELINES))?)),
            |s| s.segment(inv),
        )
    }
}

struct TrackAction {
    weights: String,
    hot: Arc<Hot<String, Sam2>>,
}

impl Action for TrackAction {
    fn spec(&self) -> ActionSpec {
        track_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let call = track_call(inv)?;
        let variant = inv.get_str("variant").unwrap_or_else(|| "tiny".into());
        let key = format!("{}:{variant}", self.weights);
        self.hot.try_get_mut(key, || crate::track::load_video(&self.weights, variant_config(&variant)?, Gpu::new(PIPELINES)), |m| run_track(m, &call, progress))
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::{Blob, Invocation, Registry};

    #[test]
    fn manifest_declares_segment() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["segment", "track"]);
        let a = &m.actions[0];
        assert_eq!(a.name, "segment");
        assert!(!a.streaming, "segment is one-shot (the mask is the single artifact)");
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert_eq!(a.outputs[0].name, "mask");
        assert_eq!(a.outputs[0].media, Media::Mask, "a mask must be tagged as one so a client can route it");
        // defaults fill; a missing image is rejected.
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = a.validate(Invocation::new().blob("image", img)).unwrap();
        assert_eq!(inv.get_str("variant").as_deref(), Some("tiny"));
        assert_eq!(inv.get_bool("multimask"), Some(true));
        assert!(a.validate(Invocation::new()).is_err());
        assert_eq!(manifest().to_json()["actions"][0]["name"], "segment");
    }

    #[test]
    fn variants_are_the_two_released_checkpoints() {
        assert_eq!(variant_config("tiny").unwrap().embed_dim, Sam2Config::hiera_tiny().embed_dim);
        assert_eq!(variant_config("large").unwrap().embed_dim, Sam2Config::hiera_large().embed_dim);
        assert!(variant_config("huge").is_err());
    }

    /// Prompt-free calls and unparseable prompts must be errors, and a missing
    /// checkpoint a clean `ActionResult` error — never a panic.
    #[test]
    fn bad_prompts_and_missing_weights_are_clean_errors() {
        let mut reg = Registry::new();
        reg.register(Arc::new(Sam2Provider::new("/nonexistent/sam2.pt")));
        let img = || Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        // no prompt at all: rejected before any weight is touched
        let err = reg.run(MODEL, "segment", Invocation::new().blob("image", img()), &mut |_| {}).unwrap_err();
        assert!(err.contains("at least one point"), "{err}");
        // a malformed box is likewise a message, not a panic
        let inv = Invocation::new().blob("image", img()).set("box", json!("1,2,3"));
        assert!(!reg.run(MODEL, "segment", inv, &mut |_| {}).unwrap_err().is_empty());
    }

    /// A `track` call that cannot run is refused before the checkpoint is
    /// read: no clicks, a clip with no frame rate, a click off the frame.
    #[test]
    fn a_track_call_is_checked_before_any_weight_loads() {
        let mut reg = Registry::new();
        reg.register(Arc::new(Sam2Provider::new("/nonexistent/sam2.pt")));
        let clip = |fps: Option<f64>| {
            let mut b = capability::blob::video_blob(&[(vec![0.0; 4 * 4 * 3], 4, 4), (vec![0.0; 4 * 4 * 3], 4, 4)]).unwrap();
            if let Some(f) = fps {
                b.meta["fps"] = json!(f);
            }
            b
        };
        let run = |inv: Invocation| reg.run(MODEL, "track", inv, &mut |_| {}).unwrap_err();
        assert!(run(Invocation::new().blob("video", clip(Some(24.0)))).contains("points"));
        assert!(run(Invocation::new().blob("video", clip(None)).set("points", json!("1,1"))).contains("frame rate"));
        assert!(run(Invocation::new().blob("video", clip(Some(24.0))).set("points", json!("9,1"))).contains("outside"));
        let call = track_call(&track_spec().validate(Invocation::new().blob("video", clip(None)).set("points", json!("1,1")).set("exclude", json!("2,2")).set("fps", json!(12.0)).set("polarity", json!("object=0"))).unwrap()).unwrap();
        assert_eq!((call.frames.len(), call.fps, call.request.points.len()), (2, 12.0, 2));
        assert_eq!(call.request.polarity, imaging::maskseq::Polarity::ObjectBlack);
    }

    #[test]
    fn floats_parse_with_spaces_and_reject_garbage() {
        assert_eq!(parse_floats(" 1.5, 2 ").unwrap(), vec![1.5, 2.0]);
        assert!(parse_floats("1,x").is_err());
    }
}
