// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One `text2image` / `edit` call, planned and checked before any weight is
//! read: the references at the size they will be encoded at, the canvas, the
//! tile plan, the mask, the adapters, and the token ceilings the pipeline is
//! built for.
//!
//! The rules here are what decide the cost and the geometry of a run:
//!
//! - **The init reference is the canvas.** When the first reference seeds the
//!   starting latent (`strength` is given, or a `mask`), an unset
//!   width/height takes its size (cropped to /16), so an edit does not have
//!   to echo the picture's dimensions.
//! - **Other references are bounded.** A reference costs `(w/16)*(h/16)`
//!   tokens and attention is quadratic in the joint sequence, so each is
//!   encoded at most [`DEFAULT_REF_EDGE`] on its long edge unless `ref_size`
//!   says otherwise (`0`: unbounded). The init reference, and a windowed
//!   run's anchor, are never bounded: the bound would decide the canvas's own
//!   resolution.
//! - **A tiled canvas is built per window**: the DiT is sized for the largest
//!   window, the VAE for the whole canvas.

use capability::Invocation;

use crate::pipeline::GenOpts;
use crate::{AdapterSpec, Mask, Precision, Tiling};

/// Long edge, in pixels, a reference is encoded at when `ref_size` is unset.
pub const DEFAULT_REF_EDGE: u32 = 512;

/// The extra reference inputs `edit` takes beside `image`, in order.
pub const EXTRA_REFS: [&str; 7] = ["image0", "image1", "image2", "image3", "image4", "image5", "image6"];

/// A generation call planned from its params and the SIZES of its images
/// alone - everything the pipeline is built for, without reading a pixel.
/// The scheduler keys an instance on it; [`Plan::decode`] then reads the
/// images.
pub struct Plan {
    pub prompt: String,
    /// The variant asked for; the checkpoint's own size binds it later.
    pub variant: String,
    pub precision: Precision,
    /// Whether `precision` was given, rather than defaulted.
    pub precision_explicit: bool,
    pub opts: GenOpts,
    pub adapters: Vec<AdapterSpec>,
    /// How the pipeline is built beyond its placement plan.
    pub tuning: crate::pipeline::Tuning,
    /// Each reference: its input name and the long-edge bound it is encoded
    /// at (`None`: its own size).
    pub inputs: Vec<(String, Option<u32>)>,
    /// `(empty, h, w)` per reference at the size it conditions at - the shape
    /// the pipeline's token arithmetic reads.
    sizes: Vec<(Vec<f32>, u32, u32)>,
    /// What a person running it should know about its geometry: which
    /// reference was bounded, kept, or resampled, and how the canvas is tiled.
    pub notes: Vec<String>,
}

impl Plan {
    /// Generated plus reference tokens one forward sees: what the DiT is
    /// built for. Tiled, that is the largest window and its share of the
    /// references, not the whole canvas.
    pub fn forward_tokens(&self) -> u32 {
        crate::pipeline::gen_tokens_per_forward(&self.opts) + crate::pipeline::ref_tokens_per_forward(&self.sizes, &self.opts)
    }

    /// Latent tokens of the whole canvas: what the VAE decodes.
    pub fn canvas_tokens(&self) -> u32 {
        (self.opts.height / 16) * (self.opts.width / 16)
    }

    /// The references as the pipeline takes them: `[-1,1]` CHW at the
    /// planned size, read from `inv`'s image inputs.
    pub fn decode(&self, inv: &Invocation) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
        self.inputs
            .iter()
            .map(|(name, bound)| {
                let (hwc, w, h) = capability::blob::decode_image(inv, name)?;
                crate::pipeline::ref_from_hwc_bounded(&hwc, w, h, *bound)
            })
            .collect()
    }
}

/// An image input's `(w, h)`, from its meta.
fn image_size(inv: &Invocation, name: &str) -> Result<Option<(u32, u32)>, String> {
    let Some(b) = inv.get_blob(name) else { return Ok(None) };
    let dim = |k: &str| b.meta.get(k).and_then(|v| v.as_u64()).filter(|&v| v > 0 && v <= u32::MAX as u64).map(|v| v as u32);
    match (dim("w"), dim("h")) {
        (Some(w), Some(h)) => Ok(Some((w, h))),
        _ => Err(format!("input '{name}' carries no image size (meta w/h)")),
    }
}

/// Plan `inv` (validated against its action's spec). `editing` is the `edit`
/// action: it requires `image` and reads `strength` and `mask`.
pub fn plan(inv: &Invocation, editing: bool) -> Result<Plan, String> {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let variant = inv.get_str("variant").unwrap_or_else(|| "klein-4b".into());
    let precision_explicit = inv.params.get("precision").is_some();
    let precision = Precision::from_name(&inv.get_str("precision").unwrap_or_else(|| "fp32".into()))?;

    let mut given: Vec<(String, u32, u32)> = Vec::new();
    match image_size(inv, "image")? {
        Some((w, h)) => given.push(("image".into(), w, h)),
        None if editing => return Err("edit: input 'image' is required".into()),
        None => {}
    }
    for name in EXTRA_REFS {
        if let Some((w, h)) = image_size(inv, name)? {
            given.push((name.into(), w, h));
        }
    }

    // 0 is a real setting: the source back through the VAE round trip.
    let strength = editing.then(|| inv.get_f64("strength")).flatten().map(|s| s as f32);
    let mask = match inv.get_blob("mask").filter(|_| editing) {
        Some(_) => {
            let (hwc, w, h) = capability::blob::decode_image(inv, "mask")?;
            Some(Mask::from_hwc(&hwc, w, h)?)
        }
        None => None,
    };
    // `strength` of exactly 1.0 still anchors: it is a full redraw FROM the
    // reference, not "no reference" (`GenOpts::strength`'s range is (0, 1]).
    let anchored = strength.is_some() || mask.is_some();
    let anchor = anchored.then(|| given.first().map(|r| crate::pipeline::ref_crop_size(r.1, r.2))).flatten();
    let (dw, dh) = anchor.unwrap_or((512, 512));
    let width = inv.get_i64("width").map(|v| v.max(16) as u32).unwrap_or(dw);
    let height = inv.get_i64("height").map(|v| v.max(16) as u32).unwrap_or(dh);
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(format!("width/height must be multiples of 16 (got {width}x{height})"));
    }
    let mut notes = Vec::new();
    if anchor.is_some() && (inv.get_i64("width").is_none() || inv.get_i64("height").is_none()) {
        notes.push(format!("output {width}x{height}, taken from the init image"));
    }
    let steps = inv.get_i64("steps").unwrap_or(0).max(0) as u32;
    let opts = GenOpts {
        width,
        height,
        strength,
        steps: (steps > 0).then_some(steps),
        guidance: inv.get_f64("guidance").unwrap_or(4.0) as f32,
        seed: inv.get_i64("seed").ok_or("'seed' is required")?.max(0) as u64,
        mask,
        experimental_steps: inv.get_bool("experimental_steps").unwrap_or(false),
        ref_resolution_scale: inv.get_f64("ref_resolution_scale").map_or(crate::pipeline::DEFAULT_REF_RESOLUTION_SCALE, |v| v as f32),
        tile: tiling_from(inv.get_i64("tile_size"), inv.get_i64("tile_overlap"))?,
    };
    let windowed = crate::pipeline::runs_in_windows(&opts);
    let ref_size = inv.get_i64("ref_size").map(|v| v.max(0) as u32);
    if let Some(n) = ref_size.filter(|&n| n != 0 && n < 16) {
        return Err(format!("ref_size must be 0 (unbounded) or at least 16 (got {n})"));
    }
    let mut inputs = Vec::with_capacity(given.len());
    let mut sizes = Vec::with_capacity(given.len());
    for (i, (name, w, h)) in given.into_iter().enumerate() {
        let pinned = i == 0 && (anchored || windowed);
        let bound = match ref_size {
            _ if pinned => None,
            Some(0) => None,
            Some(m) => Some(m),
            None => Some(DEFAULT_REF_EDGE),
        };
        let (tw, th) = bound.map_or((w, h), |m| crate::pipeline::fit_long_edge(w, h, m));
        if (tw, th) != (w, h) {
            notes.push(format!("reference '{name}' {w}x{h} conditions at {tw}x{th} (ref_size {})", bound.unwrap_or(0)));
        } else if pinned {
            notes.push(format!("reference '{name}' {w}x{h} kept at full size - {}", if anchored { "it seeds the init latent" } else { "it anchors every window" }));
        }
        let (cw, ch) = crate::pipeline::ref_crop_size(tw, th);
        sizes.push((Vec::new(), ch, cw));
        inputs.push((name, bound));
    }
    if let Some(t) = opts.tile {
        let tiles = crate::pipeline::plan_tiles((height / 16) as usize, (width / 16) as usize, Some(t));
        notes.push(format!("tiled: {} window(s) of {}x{} px, overlap {}", tiles.len(), tiles[0].tw * 16, tiles[0].th * 16, t.overlap));
    }
    let adapters = AdapterSpec::parse_list(&inv.get_str("adapter").unwrap_or_default(), inv.get_f64("lora_scale").unwrap_or(1.0) as f32)?;
    Ok(Plan { prompt, variant, precision, precision_explicit, opts, adapters, tuning: crate::caps::tuning_of(inv)?, inputs, sizes, notes })
}

/// The tile plan `tile_size`/`tile_overlap` ask for; `None` (or a size of 0)
/// is the untiled path, and an overlap without a size is refused rather than
/// ignored.
fn tiling_from(size: Option<i64>, overlap: Option<i64>) -> Result<Option<Tiling>, String> {
    match (size.filter(|&n| n > 0), overlap) {
        (None, Some(_)) => Err("tile_overlap needs tile_size (tiling is off without it)".into()),
        (None, None) => Ok(None),
        (Some(n), ov) => {
            let n = n as u32;
            let t = Tiling { overlap: ov.map_or(Tiling::new(n).overlap, |o| o.max(0) as u32), size: n };
            t.check()?;
            Ok(Some(t))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn image(w: u32, h: u32) -> capability::Blob {
        capability::blob::image_blob(&vec![0.5; (w * h * 3) as usize], w, h, 3)
    }

    fn base() -> Invocation {
        Invocation::new().set("prompt", json!("a cat")).set("seed", json!(1))
    }

    /// An edit that seeds from its image takes the image's size; a plain
    /// reference is bounded to the default edge, and `ref_size 0` lifts it.
    #[test]
    fn the_init_image_sets_the_canvas_and_other_references_are_bounded() {
        let inv = base().blob("image", image(640, 480)).blob("image0", image(1024, 768)).set("strength", json!(0.6));
        let p = plan(&inv, true).unwrap();
        assert_eq!((p.opts.width, p.opts.height), (640, 480));
        let refs = p.decode(&inv).unwrap();
        assert_eq!((refs[0].2, refs[0].1), (640, 480), "the init image is never bounded");
        assert_eq!((refs[1].2, refs[1].1), (512, 384), "another reference is bounded to 512 on its long edge");
        assert_eq!(p.sizes.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>(), refs.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>(), "the plan predicts the decoded sizes");
        let refs = plan(&inv.clone().set("ref_size", json!(0)), true).unwrap().decode(&inv).unwrap();
        assert_eq!((refs[1].2, refs[1].1), (1024, 768));
        assert!(plan(&inv.set("ref_size", json!(8)), true).err().unwrap().contains("ref_size"));
    }

    /// `strength` 0 anchors on the image (the VAE round trip), it is not "no
    /// strength".
    #[test]
    fn a_zero_strength_still_starts_from_the_image() {
        let p = plan(&base().blob("image", image(640, 480)).set("strength", json!(0.0)), true).unwrap();
        assert_eq!((p.opts.strength, p.opts.width, p.opts.height), (Some(0.0), 640, 480));
    }

    /// A precision the caller did not give stays unstated after the spec
    /// fills its defaults, so a GGUF DiT can take its own tier.
    #[test]
    fn an_unset_precision_is_not_an_explicit_one() {
        let t2i = crate::caps::manifest().actions.into_iter().find(|a| a.name == "text2image").unwrap();
        assert!(!plan(&t2i.validate(base()).unwrap(), false).unwrap().precision_explicit);
        assert!(plan(&t2i.validate(base().set("precision", json!("fp32"))).unwrap(), false).unwrap().precision_explicit);
    }

    /// Without an init image the documented 512 default stands; an edit
    /// without an image, an off-grid size and a lone tile overlap are refused.
    #[test]
    fn the_defaults_and_refusals() {
        let r = plan(&base(), false).unwrap();
        assert_eq!((r.opts.width, r.opts.height, r.inputs.len()), (512, 512, 0));
        assert!(plan(&base(), true).err().unwrap().contains("'image'"));
        assert!(plan(&base().set("width", json!(500)), false).err().unwrap().contains("multiples of 16"));
        assert!(plan(&base().set("tile_overlap", json!(64)), false).err().unwrap().contains("tile_size"));
        let r = plan(&base().set("adapter", json!("a.safetensors@0.5,b.brain")).set("lora_scale", json!(0.8)), false).unwrap();
        assert_eq!(r.adapters.iter().map(|a| a.scale).collect::<Vec<_>>(), [0.5, 0.8]);
    }
}
