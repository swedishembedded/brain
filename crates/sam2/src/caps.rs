// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2.1's capabilities behind the generalized [`capability`] interface —
//! what makes `brain caps sam2` / `brain do sam2 segment …`, the D-Bus `Run`
//! method and `brain perf`'s `CapabilityTarget` work with no SAM-specific
//! plumbing anywhere in the CLI or the transports.
//!
//! One action, `segment`: an image plus point / box prompts in **source-image
//! pixels** in, one mask out (the shared `capability::blob` wire format,
//! re-tagged [`Media::Mask`]).
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

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Ctx, Filter, Shape};
use serde_json::json;

use crate::config::Sam2Config;
use crate::model::{Encoded, Prompt, Sam2, PIPELINES};

/// The model id used on the CLI (`brain do sam2 …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "sam2";

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

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "SAM 2.1 promptable segmentation (image path): points and boxes -> masks.", vec![segment_spec()])
}

/// Read a SAM 2.1 release checkpoint (`sam2.1_hiera_*.pt`, or an equivalent
/// `.safetensors`) and build the model on `gpu`.
///
/// The archive root is `{"model": state_dict}` and `checkpoint::torchpt`
/// flattens with `'.'`, so the `model.` prefix is stripped here exactly as
/// `tests/parity.rs` does — one convention, two call sites, no third spelling.
pub fn load(weights: &str, cfg: Sam2Config, gpu: Gpu) -> Result<Sam2, String> {
    let raw: Vec<(String, Vec<usize>, Vec<f32>)> = if weights.ends_with(".safetensors") {
        checkpoint::safetensors::read(weights)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    } else {
        checkpoint::torchpt::read(weights)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    };
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> =
        raw.into_iter().map(|(n, s, d)| (n.strip_prefix("model.").unwrap_or(&n).to_string(), s, d)).collect();
    if tensors.is_empty() {
        return Err(format!("sam2: {weights} contains no tensors"));
    }
    let (weights, _report) = crate::import::import(tensors, &cfg)?;
    Ok(Sam2::new(gpu, cfg, &weights))
}

// ===================== the shared work =====================

/// A built SAM 2 plus its one-entry image-encoder cache — the single
/// implementation of `segment`, shared by the [`Sam2Provider`] and the residency
/// adapter (`crates/cli/src/resident_sam2.rs`). Neither owns a second copy of
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
        let s = self.model.cfg.image_size;

        // Resize to the model's square input ON THE DEVICE (`resize_bilinear`)
        // — a host loop over a 4K frame would be invisible to `--device`. The
        // layout permutation around it is host glue by the `crates/imaging`
        // rule. `Sam2::preprocess` then applies the model's own pixel_mean/std,
        // so this path adds no second normalisation.
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);
        let ctx = Ctx::new(&self.model.gpu);
        let src = ctx.upload("sam2.caps.src", &chw);
        let (dev, out_shape) = ctx.resize(&src, Shape::new(1, 3, h, w), s, s, Filter::Bilinear, AlignCorners::HalfPixel);
        let resized = ctx.download(&dev, out_shape.numel());

        let img = self.model.preprocess(&resized);
        let enc = self.model.encode(&img);
        self.cache = Some((key, w, h, enc));
        Ok((w, h))
    }

    /// Run one `segment` invocation (already validated against
    /// [`segment_spec`]).
    pub fn segment(&mut self, inv: &Invocation) -> ActionResult {
        let (src_coords, labels) = parse_prompt(inv)?;
        let (w, h) = self.ensure_encoded(inv)?;
        let s = self.model.cfg.image_size as f32;
        // Source pixels -> the model's square frame. The resize above is a plain
        // (non-aspect-preserving) stretch, exactly like the reference's
        // `Resize((1024, 1024))`, so each axis scales independently.
        let (sx, sy) = (s / w as f32, s / h as f32);
        let coords: Vec<(f32, f32)> = src_coords.into_iter().map(|(x, y)| (x * sx, y * sy)).collect();

        let prompt = Prompt {
            coords,
            labels,
            // The reference downsamples a full-resolution mask prompt with
            // `interpolate(antialias=True)`, which brain has no kernel for (see
            // the crate docs), so no mask prompt is exposed over the wire.
            mask_lowres: None,
            multimask_output: inv.get_bool("multimask").unwrap_or(true),
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
        let mask = &all[best * px..(best + 1) * px];
        // A reduction to one scalar — host by the `crates/imaging` rule (the
        // readback dominates, and the per-pixel work already ran on the device).
        let area = mask.iter().filter(|&&p| p > 0.5).count();

        Ok(Outcome::new()
            .set("width", json!(w))
            .set("height", json!(h))
            .set("masks", json!(n))
            .set("best", json!(best))
            .set("iou", json!(dec.ious.get(best).copied().unwrap_or(0.0)))
            .set("ious", json!(dec.ious))
            .set("area", json!(area))
            .set("object_score", json!(self.model.gpu.read(&dec.object_score_logits, 1)[0]))
            .blob("mask", capability::blob::image_blob(mask, w, h, 1).with_media(Media::Mask)))
    }
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

// ===================== the provider =====================

/// The executable SAM 2 model behind the manifest. Construction is free — the
/// checkpoint imports lazily on the first `segment` and stays resident until a
/// call names a different checkpoint.
pub struct Sam2Provider {
    weights: String,
    hot: Arc<Mutex<Option<(String, Session)>>>,
}

impl Sam2Provider {
    /// `weights` is the checkpoint path; it comes from a CLI flag or
    /// `BRAIN_SAM2_WEIGHTS`, never a baked-in path.
    pub fn new(weights: impl Into<String>) -> Sam2Provider {
        Sam2Provider { weights: weights.into(), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_SAM2_WEIGHTS` — `None` when unset or missing on disk, so the
    /// caller can skip registration rather than serve a model that cannot load.
    pub fn from_env() -> Option<Sam2Provider> {
        let path = std::env::var("BRAIN_SAM2_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        std::path::Path::new(&path).exists().then(|| Sam2Provider::new(path))
    }
}

impl Provider for Sam2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "segment").then(|| {
            Arc::new(SegmentAction { weights: self.weights.clone(), hot: self.hot.clone() }) as Arc<dyn Action>
        })
    }
}

struct SegmentAction {
    weights: String,
    hot: Arc<Mutex<Option<(String, Session)>>>,
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
        let mut guard = self.hot.lock().map_err(|_| "sam2: hot model lock poisoned")?;
        if !matches!(&*guard, Some((k, _)) if *k == key) {
            *guard = None; // free the old build before importing the new one
            let cfg = variant_config(&variant)?;
            let gpu = Gpu::new(PIPELINES);
            *guard = Some((key, Session::new(load(&self.weights, cfg, gpu)?)));
        }
        guard.as_mut().expect("built above").1.segment(inv)
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
        assert_eq!(m.actions.len(), 1);
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

    #[test]
    fn floats_parse_with_spaces_and_reject_garbage() {
        assert_eq!(parse_floats(" 1.5, 2 ").unwrap(), vec![1.5, 2.0]);
        assert!(parse_floats("1,x").is_err());
    }
}
