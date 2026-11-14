// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN behind the generalized [`capability`] interface - what makes
//! `brain caps`, `brain do … upscale`, the D-Bus `Run` method and `brain perf`
//! work with no upscaler-specific plumbing in the CLI or the transports.
//!
//! One action, `upscale`: an image in, a `scale`x image out.
//!
//! **No `run_batch` override, deliberately.** RRDBNet is a dense conv net whose
//! cost is linear in pixels and whose peak VRAM is linear in them too, so
//! grouping N images saves no work and multiplies the high-water mark by N. The
//! serial default is the right answer here, and saying so is the point -
//! the serving contract asks for a genuine batching decision, not
//! necessarily a genuine batch.
//!
//! **Tiling is exact.** `tile > 0` sweeps ONE fixed-size graph over windows of
//! `tile + 2R` input pixels a side, where `R` is the network's receptive radius
//! ([`RrdbConfig::receptive_radius`], 349 for `x4plus`'s 23 blocks). Each window
//! lies entirely inside the image (it is shifted inward at a border, never
//! padded past it) and keeps only the `tile`-pixel output core it owns; the
//! cores partition the output. Every kept pixel therefore sees its whole
//! receptive field, or a window edge that is the image edge, where the convs
//! zero-pad exactly as the whole-image pass does. The result equals the
//! whole-image output (`tile = 0`) bit for bit, which the parity tests assert.
//! An image no longer than a window in one dimension runs whole in that
//! dimension.
//!
//! The price is memory and work, not quality. A window holds `(tile + 2R)^2`
//! input pixels, so on `x4plus` a tile is worth cutting only for images larger
//! than about `2R + tile` per side (698 + tile); a smaller image is one window
//! anyway. The work grows by the overlap factor `((tile + 2R) / tile)^2`
//! relative to the whole image: several times the pixels at tile 512 on
//! `x4plus` and two orders of magnitude more at tile 64. Peak memory is set by `tile + 2R`, so shrinking
//! `tile` below `2R` saves almost nothing and multiplies the work.
//!
//! **Value range and geometry.** The reference feeds RGB in `[0,1]` (unlike the
//! VQGAN stack's `[-1,1]`), which is also brain's wire format, so there is no
//! affine here - only the HWC-blob to CHW-model layout permutation. The graph is
//! recorded for one input size, so `w`/`h` are part of the residency instance
//! key.

use std::sync::{Arc, Mutex};

use capability::presentation::maturity;
use capability::ModelPresentation;
use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec,
    ParamType, Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;
use vae::blocks::Tensors;

use crate::config::RrdbConfig;
use crate::model::Rrdb;

/// The served id.
///
/// A BARE name, like its siblings in the imaging stack (`sam2`, `scrfd`,
/// `arcface`, `vqgan`, `restore`, `clip`, `imgpipe`) - which resolve under the reserved
/// `brain` vendor. `crates/cli/tests/it/model_ids.rs` requires every static
/// catalog id to sit under a reserved vendor, so the upstream-repo spelling
/// (`ai-forever/Real-ESRGAN`) is NOT usable here: that vendor is fetchable, and
/// this model is served from a locally-configured checkpoint, not fetched.
pub const MODEL: &str = "brain/rrdbnet";

pub fn upscale_spec() -> ActionSpec {
    ActionSpec::new("upscale", "super-resolve an image (Real-ESRGAN RRDBNet generator)")
        .param(
            ParamSpec::new(
                "tile",
                ParamType::Int,
                "process in windows of this many input pixels a side plus the network's \
                 receptive radius on each side (0 = whole image); output is identical to \
                 the whole-image pass, peak VRAM follows the window and work grows by \
                 ((tile + 2R) / tile)^2, so it pays only for images larger than 2R + tile",
            )
            .min(0.0)
            .max(capability::bounds::SIZE_PX)
            .default(json!(0)),
        )
        .input(BlobSpec::new("image", Media::Image, "the image to upscale, RGB in [0,1]").required())
        .output(BlobSpec::new("image", Media::Image, "the upscaled image, RGB in [0,1]"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Real-ESRGAN x4 super-resolution (the RRDBNet generator; the discriminator is training-only).",
        vec![upscale_spec()],
    )
    .with_presentation(
        ModelPresentation::default()
            .display_name("Real-ESRGAN")
            .vendor("Tencent ARC")
            .family("Real-ESRGAN")
            .version("x4")
            .task_tags(&["upscale", "restoration"])
            .maturity(maturity::STABLE)
            .default_action("upscale")
            .license("bsd-3-clause"),
    )
}

/// What a host must implement to serve the action - the seam the residency
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

// ===================== the built model =====================

/// A built RRDBNet, rebuilt when the requested input size changes.
///
/// The graph is recorded for one `(h, w)`, so serving arbitrary sizes means
/// either a build per size or tiling onto one fixed build. Both are here: `tile
/// = 0` builds for the whole image (fast, memory grows with the image), any
/// other value builds ONE graph at the tile size and sweeps it.
pub struct Session {
    gpu: Gpu,
    cfg: RrdbConfig,
    weights: Tensors,
    built: Mutex<Option<(u32, u32, Rrdb)>>,
}

impl Session {
    pub fn new(gpu: Gpu, cfg: RrdbConfig, weights: Tensors) -> Session {
        Session { gpu, cfg, weights, built: Mutex::new(None) }
    }

    pub fn config(&self) -> &RrdbConfig {
        &self.cfg
    }

    /// Run one `[3,h,w]` CHW tile through a graph built for exactly that size,
    /// reusing the previous build when the size is unchanged.
    fn run_exact(&self, chw: &[f32], w: u32, h: u32) -> Vec<f32> {
        let mut b = self.built.lock().expect("session lock");
        if !matches!(&*b, Some((bw, bh, _)) if *bw == w && *bh == h) {
            // Drop the old graph BEFORE building the new one: holding both
            // doubles peak device memory for no reason.
            *b = None;
            *b = Some((w, h, Rrdb::new(self.gpu.share(), self.cfg.clone(), &self.weights, h, w, false)));
        }
        b.as_ref().expect("just built").2.run(chw)
    }
}

/// One axis of a tiled sweep: the window cut from the input and the output core
/// it owns.
#[derive(Debug, PartialEq, Eq)]
struct Span {
    /// First input pixel of the window.
    start: u32,
    /// The part of the axis this window is responsible for, `[core_start, core_end)`.
    core_start: u32,
    core_end: u32,
}

/// Plan one axis: cores of `tile` pixels partition `[0, extent)`, and each core
/// gets a window of `len = min(extent, tile + 2*radius)` pixels that lies
/// inside `[0, extent)`.
///
/// The window starts `radius` before its core, shifted inward at the borders
/// instead of padded past them. So either the core has `radius` pixels of
/// context on a side, or that side of the window IS the image edge, where the
/// network's convs zero-pad exactly as they do on the whole image. When the
/// image is not longer than a window (`len == extent`) there is one window and
/// the axis runs whole.
fn plan_axis(extent: u32, tile: u32, radius: u32) -> (u32, Vec<Span>) {
    let len = extent.min(tile.saturating_add(radius.saturating_mul(2)));
    let spans = (0..extent)
        .step_by(tile as usize)
        .map(|core_start| Span {
            start: core_start.saturating_sub(radius).min(extent - len),
            core_start,
            core_end: (core_start + tile).min(extent),
        })
        .collect();
    (len, spans)
}

impl Session {
    /// Copy window `(wx, wy)` of size `(ww, wh)` out of a `[3,h,w]` image.
    fn cut(chw: &[f32], w: u32, h: u32, (wx, wy): (u32, u32), (ww, wh): (u32, u32)) -> Vec<f32> {
        let (w, h, wx, wy, ww, wh) = (w as usize, h as usize, wx as usize, wy as usize, ww as usize, wh as usize);
        let mut cut = Vec::with_capacity(3 * ww * wh);
        for c in 0..3 {
            for y in 0..wh {
                let row = (c * h + wy + y) * w + wx;
                cut.extend_from_slice(&chw[row..row + ww]);
            }
        }
        cut
    }

    /// Exact tiled upscale: see the module docs. One fixed-size graph is swept
    /// over windows of `tile + 2R` pixels a side (`R` =
    /// [`RrdbConfig::receptive_radius`]), and each window contributes only the
    /// output core it owns.
    fn upscale_tiled(&self, chw: &[f32], w: u32, h: u32, tile: u32) -> Vec<f32> {
        let s = self.cfg.scale as usize;
        let r = self.cfg.receptive_radius();
        let (win_w, xs) = plan_axis(w, tile, r);
        let (win_h, ys) = plan_axis(h, tile, r);
        let (ow, oh) = (w as usize * s, h as usize * s);
        let (uw, uh) = (win_w as usize * s, win_h as usize * s);
        let mut out = vec![0.0f32; 3 * ow * oh];
        for y in &ys {
            for x in &xs {
                let up = self.run_exact(&Self::cut(chw, w, h, (x.start, y.start), (win_w, win_h)), win_w, win_h);
                for c in 0..3 {
                    for oy in y.core_start as usize * s..y.core_end as usize * s {
                        let src_y = oy - y.start as usize * s;
                        let src = (c * uh + src_y) * uw + (x.core_start - x.start) as usize * s;
                        let dst = (c * oh + oy) * ow + x.core_start as usize * s;
                        let n = (x.core_end - x.core_start) as usize * s;
                        out[dst..dst + n].copy_from_slice(&up[src..src + n]);
                    }
                }
            }
        }
        out
    }
}

impl Upscaler for Session {
    fn upscale(&self, chw: &[f32], w: u32, h: u32, tile: u32) -> Result<(Vec<f32>, u32, u32), String> {
        let s = self.cfg.scale;
        if chw.len() != 3 * (w as usize) * (h as usize) {
            return Err(format!("upscale: input is {} floats, expected {}", chw.len(), 3 * w * h));
        }
        if w == 0 || h == 0 {
            return Err(format!("upscale: empty image {w}x{h}"));
        }
        let out = if tile == 0 { self.run_exact(chw, w, h) } else { self.upscale_tiled(chw, w, h, tile) };
        Ok((out, w * s, h * s))
    }
}

/// Import a released checkpoint and build a session.
pub fn load(path: &str, gpu: Gpu) -> Result<Session, String> {
    let (tensors, shapes, src) = crate::import::read(path)?;
    let cfg = RrdbConfig::from_tensors(&shapes)?;
    eprintln!(
        "upscale: {path} ({src:?}) feat={} grow={} blocks={} scale={}x",
        cfg.num_feat, cfg.num_grow_ch, cfg.num_block, cfg.scale
    );
    let tensors = crate::import::validate(tensors, &cfg)?;
    Ok(Session::new(gpu, cfg, tensors))
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
    /// the right size and range - nothing structural catches it.
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
            b.bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
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

    /// The plan is a partition of the axis into cores, every window has the
    /// same length, lies inside the axis, and holds its core with `radius`
    /// pixels of context on each side unless that side is the axis edge.
    #[test]
    fn an_axis_plan_partitions_with_full_context_or_the_image_edge() {
        for (extent, tile, radius) in [(91u32, 8u32, 34u32), (85, 8, 34), (300, 64, 10), (50, 7, 3), (10, 4, 0)] {
            let (len, spans) = plan_axis(extent, tile, radius);
            assert_eq!(len, extent.min(tile + 2 * radius));
            let mut next = 0;
            for sp in &spans {
                assert_eq!(sp.core_start, next, "cores must tile the axis without gaps or overlap");
                next = sp.core_end;
                assert!(sp.start + len <= extent, "window must lie inside the image: {sp:?}");
                assert!(sp.core_start >= sp.start && sp.core_end <= sp.start + len, "core inside window: {sp:?}");
                assert!(sp.core_start - sp.start >= radius || sp.start == 0, "left context: {sp:?}");
                assert!(sp.start + len - sp.core_end >= radius || sp.start + len == extent, "right context: {sp:?}");
            }
            assert_eq!(next, extent);
        }
    }

    /// An axis not longer than one window is a single whole-axis window.
    #[test]
    fn a_short_axis_runs_whole() {
        let (len, spans) = plan_axis(40, 8, 34);
        assert_eq!((len, spans.len(), spans[0].start), (40, 5, 0));
        assert!(spans.iter().all(|s| s.start == 0), "every window is the whole axis");
    }
}
