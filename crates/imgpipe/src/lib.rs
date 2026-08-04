// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composed imaging pipeline: *"change only X, keep the rest exactly as it
//! was"* as one operation.
//!
//! This is what the imaging workstream was for. Segmentation gives a
//! pixel-accurate region, a model edits or restores it, and the result is
//! composited back — and the contract is **exact**, not perceptual: every pixel
//! outside the mask is bit-identical to the input.
//!
//! # One `Run`, not four
//!
//! The pipeline is a single [`capability::Action`] over a JSON stage list rather
//! than four D-Bus round-trips. Composing it client-side would move a
//! full-resolution image across the bus after every stage; here the image stays
//! put and only the final result is returned. `capability::Media` already has
//! `Image` and `Mask` and `Run` already passes blobs by fd, so this needs no new
//! D-Bus surface — see `docs/imaging/plan.md` §3.5.
//!
//! # Model stages go through the registry
//!
//! [`Pipeline::run`] dispatches `segment` and `restore` through a
//! [`capability::Registry`], so this crate depends on no model. That keeps the
//! composition testable without a single checkpoint — the mask algebra and the
//! bit-exactness contract are what can actually go wrong here, and both are
//! checkable with a stub provider — and it makes a new stage a registry entry
//! rather than a new dependency edge.
//!
//! # The exactness contract, and how it is kept
//!
//! `imaging::mask::composite(new, old, mask)` computes `old + mask*(new - old)`.
//! Where the mask is exactly `0.0` that is `old + 0.0`, which in fp32 returns
//! `old`'s exact bits. So "unchanged outside the mask" is a property of the
//! arithmetic rather than of a tolerance, and the test asserts the BITS.
//!
//! **Feathering deliberately breaks it, and that is what it is for.** A
//! feathered mask is non-zero in a band outside the hard region, so those pixels
//! do move. [`Outcome::mask`] is therefore returned alongside the image: it is
//! the authoritative record of which pixels were authorised to change.

pub mod caps;

use capability::{Blob, Invocation, Media, Registry};
use gpu_core::Gpu;
use imaging::{Ctx, Shape};
use serde_json::Value;

/// One step of the pipeline.
///
/// Deliberately a linear sequence rather than a general DAG: every editing
/// recipe this workstream targets — segment, refine the mask, edit, composite —
/// is a chain, and a DAG would add a scheduler with no user.
///
/// Parsed by hand from `serde_json::Value` rather than with a derive: this
/// workspace declares `serde_json` but deliberately not `serde`'s derive macro,
/// and one stage enum is not worth widening the dependency closure for. The
/// hand-written parser also lets an unknown `op` or a misspelled parameter be an
/// error naming the offender, which is what the tests pin.
#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    /// Promptable segmentation (`crates/sam2`). Replaces the working mask.
    Segment {
        /// Point prompts as `[x, y]` in source pixels.
        points: Vec<[f32; 2]>,
        /// Box prompts as `[x0, y0, x1, y1]` in source pixels.
        boxes: Vec<[f32; 4]>,
    },
    /// Grow the mask by `radius` pixels.
    Dilate { radius: u32 },
    /// Shrink the mask by `radius` pixels.
    Erode { radius: u32 },
    /// Soften the mask edge. NOTE: this makes pixels OUTSIDE the hard region
    /// change — see the module docs.
    Feather { radius: u32 },
    /// Select everything the mask does not.
    Invert,
    /// Blind face restoration (`crates/restore`); `w` is the fidelity dial.
    Restore { w: f32 },
}

/// Default fidelity dial when `restore` omits `w`.
pub const DEFAULT_W: f32 = 0.5;

/// The parameter names each `op` accepts. Anything else is an error rather than
/// a silent default — a misspelled `radius` that quietly became 0 would produce
/// a plausible image that did not do what was asked.
fn allowed(op: &str) -> &'static [&'static str] {
    match op {
        "segment" => &["op", "points", "boxes"],
        "dilate" | "erode" | "feather" => &["op", "radius"],
        "invert" => &["op"],
        "restore" => &["op", "w"],
        _ => &[],
    }
}

fn u32_field(o: &serde_json::Map<String, Value>, op: &str, k: &str) -> Result<u32, String> {
    o.get(k)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("imgpipe: '{op}' needs an integer '{k}'"))
        .map(|v| v as u32)
}

fn pairs<const N: usize>(o: &serde_json::Map<String, Value>, k: &str) -> Result<Vec<[f32; N]>, String> {
    let Some(v) = o.get(k) else { return Ok(Vec::new()) };
    let arr = v.as_array().ok_or_else(|| format!("imgpipe: '{k}' must be an array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for e in arr {
        let row = e.as_array().ok_or_else(|| format!("imgpipe: each '{k}' entry must be an array of {N}"))?;
        if row.len() != N {
            return Err(format!("imgpipe: each '{k}' entry must have {N} numbers, got {}", row.len()));
        }
        let mut a = [0f32; N];
        for (i, x) in row.iter().enumerate() {
            a[i] = x.as_f64().ok_or_else(|| format!("imgpipe: '{k}' entries must be numbers"))? as f32;
        }
        out.push(a);
    }
    Ok(out)
}

impl Stage {
    fn from_value(v: &Value) -> Result<Stage, String> {
        let o = v.as_object().ok_or("imgpipe: each stage must be an object")?;
        let op = o.get("op").and_then(|x| x.as_str()).ok_or("imgpipe: each stage needs a string 'op'")?;
        let ok = allowed(op);
        if ok.is_empty() {
            return Err(format!("imgpipe: unknown stage op '{op}'"));
        }
        if let Some(bad) = o.keys().find(|k| !ok.contains(&k.as_str())) {
            return Err(format!("imgpipe: '{op}' has no parameter '{bad}' (accepts {ok:?})"));
        }
        Ok(match op {
            "segment" => Stage::Segment { points: pairs::<2>(o, "points")?, boxes: pairs::<4>(o, "boxes")? },
            "dilate" => Stage::Dilate { radius: u32_field(o, op, "radius")? },
            "erode" => Stage::Erode { radius: u32_field(o, op, "radius")? },
            "feather" => Stage::Feather { radius: u32_field(o, op, "radius")? },
            "invert" => Stage::Invert,
            "restore" => Stage::Restore {
                w: o.get("w").and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(DEFAULT_W),
            },
            _ => unreachable!("guarded by `allowed`"),
        })
    }
}

/// A parsed pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub stages: Vec<Stage>,
}

impl Spec {
    pub fn parse(json: &str) -> Result<Spec, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("imgpipe: bad stage list: {e}"))?;
        let o = v.as_object().ok_or("imgpipe: the stage list must be an object with 'stages'")?;
        if let Some(bad) = o.keys().find(|k| k.as_str() != "stages") {
            return Err(format!("imgpipe: unknown top-level key '{bad}' (expected 'stages')"));
        }
        let arr = o.get("stages").and_then(|s| s.as_array()).ok_or("imgpipe: 'stages' must be an array")?;
        Ok(Spec { stages: arr.iter().map(Stage::from_value).collect::<Result<_, _>>()? })
    }

    /// How many stages replace image content. Zero means no composite is needed
    /// and the input is returned untouched.
    pub fn edits(&self) -> usize {
        self.stages.iter().filter(|s| matches!(s, Stage::Restore { .. })).count()
    }
}

/// What a pipeline run produced.
pub struct Outcome {
    /// The composited image, CHW RGB in `[0,1]`.
    pub image: Vec<f32>,
    /// The mask actually composited with (1 channel) — the authoritative record
    /// of which pixels were allowed to move.
    pub mask: Vec<f32>,
    pub w: u32,
    pub h: u32,
}

/// Executes a [`Spec`] against a registry of models.
pub struct Pipeline<'a> {
    pub gpu: &'a Gpu,
    pub registry: &'a Registry,
}

impl<'a> Pipeline<'a> {
    pub fn new(gpu: &'a Gpu, registry: &'a Registry) -> Pipeline<'a> {
        Pipeline { gpu, registry }
    }

    /// Run `spec` over a source image (CHW RGB in `[0,1]`).
    ///
    /// The working mask starts all-ones: with nothing segmented yet, an edit
    /// applies everywhere, which is the least surprising reading of a stage list
    /// that never segments.
    pub fn run(&self, spec: &Spec, chw: &[f32], w: u32, h: u32) -> Result<Outcome, String> {
        let img_shape = Shape::new(1, 3, h, w);
        let m_shape = Shape::new(1, 1, h, w);
        if chw.len() != img_shape.numel() as usize {
            return Err(format!("imgpipe: image is {} floats, expected {}", chw.len(), img_shape.numel()));
        }
        let ctx = Ctx::new(self.gpu);
        let mut mask_host = vec![1.0f32; m_shape.numel() as usize];
        let mut cur_host = chw.to_vec();

        for (i, st) in spec.stages.iter().enumerate() {
            match st {
                Stage::Segment { points, boxes } => {
                    mask_host = self.segment(&cur_host, w, h, points, boxes).map_err(|e| format!("stage {i}: {e}"))?;
                    if mask_host.len() != m_shape.numel() as usize {
                        return Err(format!(
                            "stage {i}: segment returned {} floats, expected {}",
                            mask_host.len(),
                            m_shape.numel()
                        ));
                    }
                }
                Stage::Dilate { radius } | Stage::Erode { radius } | Stage::Feather { radius } => {
                    let m = ctx.upload("imgpipe.mask", &mask_host);
                    let out = match st {
                        Stage::Dilate { .. } => imaging::mask::dilate(&ctx, &m, m_shape, *radius),
                        Stage::Erode { .. } => imaging::mask::erode(&ctx, &m, m_shape, *radius),
                        _ => imaging::mask::feather(&ctx, &m, m_shape, *radius),
                    };
                    mask_host = ctx.download(&out, m_shape.numel());
                }
                Stage::Invert => {
                    let m = ctx.upload("imgpipe.mask", &mask_host);
                    let out = imaging::mask::invert(&ctx, &m, m_shape);
                    mask_host = ctx.download(&out, m_shape.numel());
                }
                Stage::Restore { w: dial } => {
                    cur_host = self.restore(&cur_host, w, h, *dial).map_err(|e| format!("stage {i}: {e}"))?;
                    if cur_host.len() != img_shape.numel() as usize {
                        return Err(format!(
                            "stage {i}: restore returned {} floats, expected {}",
                            cur_host.len(),
                            img_shape.numel()
                        ));
                    }
                }
            }
        }

        // Composite ONCE, at the end: `old + mask*(new - old)`. Where the mask is
        // exactly 0 that returns `old`'s exact bits — the contract.
        let image = if spec.edits() == 0 {
            chw.to_vec()
        } else {
            let orig = ctx.upload("imgpipe.src", chw);
            let edited = ctx.upload("imgpipe.edited", &cur_host);
            let m = ctx.upload("imgpipe.mask", &mask_host);
            let (m3, _) = imaging::mask::broadcast_channels(&ctx, &m, m_shape, 3);
            let out = imaging::mask::composite(&ctx, &edited, &orig, &m3, img_shape);
            ctx.download(&out, img_shape.numel())
        };
        Ok(Outcome { image, mask: mask_host, w, h })
    }

    fn segment(&self, chw: &[f32], w: u32, h: u32, points: &[[f32; 2]], boxes: &[[f32; 4]]) -> Result<Vec<f32>, String> {
        let hwc = imaging::pixels::chw_to_hwc(chw, 3, h as usize, w as usize);
        let mut inv = Invocation::new().blob("image", capability::blob::image_blob(&hwc, w, h, 3));
        if !points.is_empty() {
            inv = inv.set("points", serde_json::json!(points));
        }
        if !boxes.is_empty() {
            inv = inv.set("boxes", serde_json::json!(boxes));
        }
        let out = self.registry.run("sam2", "segment", inv, &mut |_| {})?;
        let b = out.blobs.get("mask").ok_or("sam2 segment returned no 'mask' blob")?;
        decode_plane(b, w, h)
    }

    fn restore(&self, chw: &[f32], w: u32, h: u32, dial: f32) -> Result<Vec<f32>, String> {
        let hwc = imaging::pixels::chw_to_hwc(chw, 3, h as usize, w as usize);
        let inv = Invocation::new()
            .set("w", serde_json::json!(dial))
            .blob("image", capability::blob::image_blob(&hwc, w, h, 3));
        let out = self.registry.run("restore", "restore_face", inv, &mut |_| {})?;
        let b = out.blobs.get("image").ok_or("restore returned no 'image' blob")?;
        let hwc = decode_rgb(b, w, h)?;
        Ok(imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize))
    }
}

fn as_f32(b: &Blob) -> Vec<f32> {
    b.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn decode_plane(b: &Blob, w: u32, h: u32) -> Result<Vec<f32>, String> {
    let v = as_f32(b);
    let want = (w * h) as usize;
    if v.len() != want {
        return Err(format!("mask blob is {} floats, expected {want} ({w}x{h})", v.len()));
    }
    Ok(v)
}

fn decode_rgb(b: &Blob, w: u32, h: u32) -> Result<Vec<f32>, String> {
    let v = as_f32(b);
    let want = (w * h * 3) as usize;
    if v.len() != want {
        return Err(format!("image blob is {} floats, expected {want} ({w}x{h}x3)", v.len()));
    }
    Ok(v)
}

/// The media kinds this crate exchanges — named here so a manifest cannot drift
/// from what [`Pipeline::run`] actually decodes.
pub const IMAGE_MEDIA: Media = Media::Image;
pub const MASK_MEDIA: Media = Media::Mask;

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Action, ActionResult, ActionSpec, BlobSpec, Manifest, Outcome as CapOutcome, Progress, Provider};
    use std::sync::Arc;

    /// A segmenter selecting a fixed rectangle, and a "restorer" that paints the
    /// WHOLE image a constant. Between them the contract is checkable without a
    /// checkpoint: every pixel the restorer destroyed outside the rectangle must
    /// be restored by the composite.
    struct StubSeg {
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    }
    struct StubRestore;

    fn blob_f32(v: &[f32], media: Media) -> Blob {
        Blob::new(media, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    impl Action for StubSeg {
        fn spec(&self) -> ActionSpec {
            ActionSpec::new("segment", "stub")
                .param(capability::ParamSpec::new("points", capability::ParamType::Str, "points"))
                .param(capability::ParamSpec::new("boxes", capability::ParamType::Str, "boxes"))
                .input(BlobSpec::new("image", Media::Image, "i"))
                .output(BlobSpec::new("mask", Media::Mask, "m"))
        }
        fn run(&self, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            let (_, w, h) = capability::blob::decode_image(inv, "image")?;
            let mut m = vec![0.0f32; (w * h) as usize];
            for y in self.y0..self.y1.min(h) {
                for x in self.x0..self.x1.min(w) {
                    m[(y * w + x) as usize] = 1.0;
                }
            }
            Ok(CapOutcome::new().blob("mask", blob_f32(&m, Media::Mask)))
        }
    }
    impl Action for StubRestore {
        fn spec(&self) -> ActionSpec {
            // `w` must be declared: Registry::run validates the invocation
            // against the spec, so an undeclared param is rejected — which is
            // the capability layer doing its job, and caught this stub.
            ActionSpec::new("restore_face", "stub")
                .param(capability::ParamSpec::new("w", capability::ParamType::Float, "fidelity dial"))
                .input(BlobSpec::new("image", Media::Image, "i"))
                .output(BlobSpec::new("image", Media::Image, "o"))
        }
        fn run(&self, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            let (hwc, _, _) = capability::blob::decode_image(inv, "image")?;
            // Deliberately destroy EVERY pixel: anything surviving outside the
            // mask proves the composite protected it.
            let painted = vec![0.123_456_79f32; hwc.len()];
            Ok(CapOutcome::new().blob("image", blob_f32(&painted, Media::Image)))
        }
    }

    struct StubProvider(&'static str, Arc<dyn Action>);
    impl Provider for StubProvider {
        fn manifest(&self) -> Manifest {
            Manifest::new(self.0, "stub", vec![self.1.spec()])
        }
        fn action(&self, _n: &str) -> Option<Arc<dyn Action>> {
            Some(self.1.clone())
        }
    }

    fn registry(seg: StubSeg) -> Registry {
        let mut r = Registry::new();
        r.register(Arc::new(StubProvider("sam2", Arc::new(seg))));
        r.register(Arc::new(StubProvider("restore", Arc::new(StubRestore))));
        r
    }

    fn ramp(w: u32, h: u32) -> Vec<f32> {
        (0..(3 * w * h) as usize).map(|i| (i % 251) as f32 / 251.0).collect()
    }

    /// THE CONTRACT. Not a tolerance — the bits.
    #[test]
    fn unselected_pixels_are_bit_identical() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (w, h) = (32u32, 24u32);
        let src = ramp(w, h);
        let gpu = gpu_core::testgpu::dev(imaging::PIPELINES);
        let reg = registry(StubSeg { x0: 8, y0: 6, x1: 20, y1: 18 });
        let spec = Spec { stages: vec![Stage::Segment { points: vec![], boxes: vec![] }, Stage::Restore { w: 0.5 }] };
        let out = Pipeline::new(&gpu, &reg).run(&spec, &src, w, h).expect("run");

        let mut changed = 0usize;
        for c in 0..3usize {
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = (c * h as usize + y) * w as usize + x;
                    let inside = (8..20).contains(&x) && (6..18).contains(&y);
                    if inside {
                        changed += (out.image[i] != src[i]) as usize;
                    } else {
                        assert_eq!(
                            out.image[i].to_bits(),
                            src[i].to_bits(),
                            "pixel ({x},{y}) ch{c} is OUTSIDE the mask and moved: {} -> {}",
                            src[i],
                            out.image[i]
                        );
                    }
                }
            }
        }
        assert!(changed > 0, "the edit must actually change the selected region");
    }

    /// A stage list with no edit returns the input bit for bit — segmenting and
    /// refining a mask is not an edit.
    #[test]
    fn a_mask_only_pipeline_returns_the_input_untouched() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (w, h) = (16u32, 16u32);
        let src = ramp(w, h);
        let gpu = gpu_core::testgpu::dev(imaging::PIPELINES);
        let reg = registry(StubSeg { x0: 2, y0: 2, x1: 10, y1: 10 });
        let spec = Spec { stages: vec![Stage::Segment { points: vec![], boxes: vec![] }, Stage::Dilate { radius: 2 }] };
        let out = Pipeline::new(&gpu, &reg).run(&spec, &src, w, h).expect("run");
        assert!(out.image.iter().zip(&src).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn the_stage_list_parses_from_json() {
        let s = Spec::parse(
            r#"{"stages":[{"op":"segment","points":[[10.0,20.0]]},{"op":"feather","radius":3},{"op":"restore","w":0.7}]}"#,
        )
        .expect("parse");
        assert_eq!(s.stages.len(), 3);
        assert_eq!(s.edits(), 1);
        assert!(matches!(s.stages[2], Stage::Restore { w } if (w - 0.7).abs() < 1e-6));
    }

    #[test]
    fn an_unknown_op_is_an_error_not_a_skip() {
        // A silently ignored stage would produce a plausible image that did not
        // do what was asked.
        assert!(Spec::parse(r#"{"stages":[{"op":"enhance"}]}"#).is_err());
        // ... and so would a misspelled parameter.
        assert!(Spec::parse(r#"{"stages":[{"op":"dilate","radius_px":3}]}"#).is_err());
    }
}
