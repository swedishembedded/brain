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
//! **Value range and geometry.** The reference feeds RGB in `[0,1]` (unlike the
//! VQGAN stack's `[-1,1]`), which is also brain's wire format, so there is no
//! affine here - only the HWC-blob to CHW-model layout permutation. The graph is
//! recorded for one input size, so `w`/`h` are part of the residency instance
//! key.

use std::sync::{Arc, Mutex};

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
/// `brain` vendor. `crates/cli/tests/model_ids.rs` requires every static
/// catalog id to sit under a reserved vendor, so the upstream-repo spelling
/// (`ai-forever/Real-ESRGAN`) is NOT usable here: that vendor is fetchable, and
/// this model is served from a locally-configured checkpoint, not fetched.
pub const MODEL: &str = "brain/upscale";

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

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Real-ESRGAN x4 super-resolution (the RRDBNet generator; the discriminator is training-only).",
        vec![upscale_spec()],
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

/// Halo, in INPUT pixels, added around every tile and cropped from its output.
///
/// **Tiling is an APPROXIMATION, and on the released net a visible one.**
/// RRDBNet is fully convolutional with 3x3 convs, so a pixel depends on a
/// neighbourhood whose radius grows with depth: roughly `1 + 15*num_block + 1`
/// input pixels, which is ~347 at `x4plus`'s 23 blocks. A halo that made hard-
/// cropped tiling exact would be larger than any tile worth cutting, so the
/// halo trades seam against memory rather than removing the seam.
///
/// Measured (`the_tile_seam_shrinks_with_the_halo`,
/// `the_tile_seam_on_the_released_net`), tile 12, max |delta| against a single
/// tile covering the whole image at the same halo:
///
/// | halo | tiny, 2 blocks | x4plus, 23 blocks |
/// |---|---|---|
/// | 4  | 1.0e0   | - |
/// | 8  | 2.0e-1  | - |
/// | 16 | 9.2e-4  | **7.3e-1** |
/// | 32 | 3.3e-6  | 1.6e-1 |
/// | 48 | -       | 5.9e-2 |
///
/// The first draft of this constant was 16, justified on the 2-block toy where
/// it is 4x below an 8-bit step. On the model anyone actually runs it is off by
/// nearly three orders of magnitude. 32 is the current
/// cost/quality point, and `tile` defaults to 0 so callers who can afford the
/// memory never meet the trade-off at all.
///
/// **A blended tiler was tried and is WORSE - do not re-try it without new
/// evidence.** Feathering the overlap instead of hard-cropping it measured
/// 2.1e-2 on the tiny config where cropping measures 3.3e-6, and 2.0e-1 vs
/// 1.6e-1 on x4plus. The reason is structural: blending mixes each tile's HALO
/// pixels - the least accurate ones it computed, precisely because they had the
/// least context - back into the output, while cropping discards them and keeps
/// only the well-conditioned interior. Blending buys continuity of the error at
/// the cost of its magnitude, and here the magnitude is what was wrong. The
/// real fix is a bigger halo, i.e. memory.
pub const TILE_HALO: u32 = 32;

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

/// Replicate-pad `[3,h,w]` by `pad` on every side.
///
/// Replication, not zeros: a zero border is a hard black edge the net would
/// happily sharpen into the output.
fn pad_replicate(chw: &[f32], w: u32, h: u32, pad: u32) -> (Vec<f32>, u32, u32) {
    let (pw, ph) = (w + 2 * pad, h + 2 * pad);
    let mut out = vec![0.0f32; 3 * (pw * ph) as usize];
    for c in 0..3usize {
        for y in 0..ph as usize {
            let sy = (y as i64 - pad as i64).clamp(0, h as i64 - 1) as usize;
            for x in 0..pw as usize {
                let sx = (x as i64 - pad as i64).clamp(0, w as i64 - 1) as usize;
                out[(c * ph as usize + y) * pw as usize + x] = chw[(c * h as usize + sy) * w as usize + sx];
            }
        }
    }
    (out, pw, ph)
}

impl Session {
    /// [`Upscaler::upscale`] with the halo as a parameter, so a test can measure
    /// how the seam responds to it instead of trusting [`TILE_HALO`].
    pub fn upscale_with_halo(
        &self,
        chw: &[f32],
        w: u32,
        h: u32,
        tile: u32,
        halo: u32,
    ) -> Result<(Vec<f32>, u32, u32), String> {
        let s = self.cfg.scale;
        let (ow, oh) = (w * s, h * s);
        if chw.len() != 3 * (w as usize) * (h as usize) {
            return Err(format!("upscale: input is {} floats, expected {}", chw.len(), 3 * w * h));
        }

        if tile == 0 {
            return Ok((self.run_exact(chw, w, h), ow, oh));
        }

        // One graph at (tile + 2*halo) square, swept over the image. Every tile
        // is the SAME size - including the edge ones, which are replicate-padded
        // - so the graph is built once.
        let side = tile + 2 * halo;
        let mut out = vec![0.0f32; 3 * (ow as usize) * (oh as usize)];
        let (padded, pw, ph) = pad_replicate(chw, w, h, halo);
        let mut ty = 0u32;
        while ty < h {
            let mut tx = 0u32;
            while tx < w {
                // Cut `side` pixels starting `halo` before the tile origin.
                let mut cut = vec![0.0f32; 3 * (side * side) as usize];
                for c in 0..3usize {
                    for y in 0..side as usize {
                        let sy = ((ty + y as u32) as usize).min(ph as usize - 1);
                        for x in 0..side as usize {
                            let sx = ((tx + x as u32) as usize).min(pw as usize - 1);
                            cut[(c * side as usize + y) * side as usize + x] =
                                padded[(c * ph as usize + sy) * pw as usize + sx];
                        }
                    }
                }
                let up = self.run_exact(&cut, side, side);
                // Crop the scaled halo and write the interior.
                let (hs, us) = ((halo * s) as usize, (side * s) as usize);
                for c in 0..3usize {
                    for y in 0..(tile * s) as usize {
                        let dy = (ty * s) as usize + y;
                        if dy >= oh as usize {
                            break;
                        }
                        for x in 0..(tile * s) as usize {
                            let dx = (tx * s) as usize + x;
                            if dx >= ow as usize {
                                break;
                            }
                            out[(c * oh as usize + dy) * ow as usize + dx] =
                                up[(c * us + hs + y) * us + hs + x];
                        }
                    }
                }
                tx += tile;
            }
            ty += tile;
        }
        Ok((out, ow, oh))
    }
}

impl Upscaler for Session {
    fn upscale(&self, chw: &[f32], w: u32, h: u32, tile: u32) -> Result<(Vec<f32>, u32, u32), String> {
        self.upscale_with_halo(chw, w, h, tile, TILE_HALO)
    }
}

impl UpscaleProvider<Session> {
    /// Build from `BRAIN_ESRGAN_WEIGHTS`, on a device of this crate's kernel
    /// set. `None` when the var is unset or names nothing that exists - the
    /// caller turns that into its own "set BRAIN_..." message.
    pub fn from_env() -> Option<UpscaleProvider<Session>> {
        let path = std::env::var("BRAIN_ESRGAN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        if !std::path::Path::new(&path).exists() {
            return None;
        }
        let gpu = Gpu::new(&crate::model::KERNELS);
        load(&path, gpu).ok().map(UpscaleProvider::new)
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
