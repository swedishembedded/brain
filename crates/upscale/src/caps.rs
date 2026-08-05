// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN behind the generalized [`capability`] interface — what makes
//! `brain caps`, `brain do … upscale`, the D-Bus `Run` method and `brain perf`
//! work with no upscaler-specific plumbing in the CLI or the transports.
//!
//! One action, `upscale`: an image in, a `scale`x image out.
//!
//! **No `run_batch` override, deliberately.** RRDBNet is a dense conv net whose
//! cost is linear in pixels and whose peak VRAM is linear in them too, so
//! grouping N images saves no work and multiplies the high-water mark by N. The
//! serial default is the right answer here, and saying so is the point —
//! `docs/serving-contract.md` asks for a genuine batching decision, not
//! necessarily a genuine batch.
//!
//! **Value range and geometry.** The reference feeds RGB in `[0,1]` (unlike the
//! VQGAN stack's `[-1,1]`), which is also brain's wire format, so there is no
//! affine here — only the HWC-blob to CHW-model layout permutation. The graph is
//! recorded for one input size, so `w`/`h` are part of the residency instance
//! key.

use std::sync::Arc;

use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec,
    ParamType, Progress, Provider,
};
use serde_json::json;

/// The canonical served id — the upstream repo, exactly
/// (`docs/models/naming.md`).
pub const MODEL: &str = "ai-forever/Real-ESRGAN";

pub fn upscale_spec() -> ActionSpec {
    ActionSpec::new("upscale", "super-resolve an image (Real-ESRGAN RRDBNet generator)")
        .param(
            ParamSpec::new(
                "tile",
                ParamType::Int,
                "process in tiles of this many input pixels a side (0 = whole image); \
                 peak VRAM is quadratic in the tile, so a large image needs this",
            )
            .default(json!(0)),
        )
        .input(BlobSpec::new("image", Media::Image, "the image to upscale, RGB in [0,1]").required())
        .output(BlobSpec::new("image", Media::Image, "the upscaled image, RGB in [0,1]"))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Real-ESRGAN x4 super-resolution (the RRDBNet generator; the discriminator is training-only).",
        vec![upscale_spec()],
    )
}

/// What a host must implement to serve the action — the seam the residency
/// adapter and the in-process provider share, so neither owns a copy of the
/// blob/layout handling below.
pub trait Upscaler: Send + Sync {
    /// `chw` is `[3,h,w]` in `[0,1]`; return `([3,oh,ow], ow, oh)` in `[0,1]`.
    fn upscale(&self, chw: &[f32], w: u32, h: u32, tile: u32) -> Result<(Vec<f32>, u32, u32), String>;
}

/// The single implementation of the action, over any [`Upscaler`].
pub fn run_upscale(up: &dyn Upscaler, inv: &Invocation) -> ActionResult {
    let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
    let tile = inv.get_i64("tile").unwrap_or(0).max(0) as u32;
    let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);
    let (out, ow, oh) = up.upscale(&chw, w, h, tile)?;
    let want = 3 * (ow as usize) * (oh as usize);
    if out.len() != want {
        return Err(format!("upscale: model returned {} floats, expected {want} for {ow}x{oh}", out.len()));
    }
    let hwc = imaging::pixels::chw_to_hwc(&out, 3, oh as usize, ow as usize);
    Ok(Outcome::new()
        .set("w", json!(ow))
        .set("h", json!(oh))
        .set("scale", json!(ow as f64 / w.max(1) as f64))
        .blob("image", capability::blob::image_blob(&hwc, ow, oh, 3)))
}

struct UpscaleAction<T: Upscaler>(Arc<T>);

impl<T: Upscaler + 'static> Action for UpscaleAction<T> {
    fn spec(&self) -> ActionSpec {
        upscale_spec()
    }

    fn run(&self, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        run_upscale(self.0.as_ref(), inv)
    }
}

/// The provider a residency adapter registers.
pub struct UpscaleProvider<T: Upscaler> {
    inner: Arc<T>,
}

impl<T: Upscaler + 'static> UpscaleProvider<T> {
    pub fn new(inner: T) -> UpscaleProvider<T> {
        UpscaleProvider { inner: Arc::new(inner) }
    }
}

impl<T: Upscaler + 'static> Provider for UpscaleProvider<T> {
    fn manifest(&self) -> Manifest {
        manifest()
    }

    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "upscale").then(|| Arc::new(UpscaleAction(self.inner.clone())) as Arc<dyn Action>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nearest-2x, so these tests exercise the PLUMBING (sizes, layout
    /// round-trip, rejection) rather than the model.
    struct Stub;
    impl Upscaler for Stub {
        fn upscale(&self, chw: &[f32], w: u32, h: u32, _t: u32) -> Result<(Vec<f32>, u32, u32), String> {
            let (ow, oh) = (w * 2, h * 2);
            let mut out = vec![0.0f32; 3 * (ow * oh) as usize];
            for c in 0..3usize {
                for y in 0..oh as usize {
                    for x in 0..ow as usize {
                        out[(c * oh as usize + y) * ow as usize + x] =
                            chw[(c * h as usize + y / 2) * w as usize + x / 2];
                    }
                }
            }
            Ok((out, ow, oh))
        }
    }

    #[test]
    fn the_manifest_declares_one_action_and_only_that_one_resolves() {
        let p = UpscaleProvider::new(Stub);
        let m = p.manifest();
        assert_eq!(m.actions.len(), 1);
        assert_eq!(m.actions[0].name, "upscale");
        assert!(p.action("upscale").is_some());
        assert!(p.action("segment").is_none(), "an undeclared action must not resolve");
    }

    /// The layout round-trip is what silently corrupts: the blob is HWC and the
    /// model is CHW, so a missing permutation is a scrambled image with exactly
    /// the right size and range — nothing structural catches it.
    #[test]
    fn an_hwc_blob_round_trips_through_the_chw_model() {
        let (w, h) = (2u32, 2u32);
        let chw: Vec<f32> = (0..12).map(|i| i as f32 / 12.0).collect();
        let hwc = imaging::pixels::chw_to_hwc(&chw, 3, h as usize, w as usize);
        let inv = Invocation::new().blob("image", capability::blob::image_blob(&hwc, w, h, 3));

        let p = UpscaleProvider::new(Stub);
        let out = p.action("upscale").unwrap().run(&inv, &mut |_| {}).expect("run");
        let b = out.blobs.get("image").expect("image out");
        let flat: Vec<f32> =
            b.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let got = imaging::pixels::hwc_to_chw(&flat, 3, 4, 4);
        for c in 0..3usize {
            for y in 0..4usize {
                for x in 0..4usize {
                    assert_eq!(got[(c * 4 + y) * 4 + x], chw[(c * 2 + y / 2) * 2 + x / 2], "c{c} y{y} x{x}");
                }
            }
        }
    }

    /// A model that returns the wrong number of floats must be caught here,
    /// not reshaped into a plausible picture downstream.
    #[test]
    fn a_short_model_output_is_rejected() {
        struct Short;
        impl Upscaler for Short {
            fn upscale(&self, _c: &[f32], w: u32, h: u32, _t: u32) -> Result<(Vec<f32>, u32, u32), String> {
                Ok((vec![0.0; 3], w * 4, h * 4))
            }
        }
        let hwc = vec![0.0f32; 3 * 4];
        let inv = Invocation::new().blob("image", capability::blob::image_blob(&hwc, 2, 2, 3));
        let e = run_upscale(&Short, &inv).unwrap_err();
        assert!(e.contains("expected"), "{e}");
    }
}
