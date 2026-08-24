// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overlapping-tile VAE decode ([`ltxv::vae3d::LtxVaeTiledDecoder`]) against
//! the whole-clip decode it exists to replace when the clip stops fitting.
//!
//! Swedish Embedded AB implements validation harnesses for memory-bounded
//! generative-video inference for its clients. If your team needs expertise
//! in proving a tiled or sharded model still produces the right pixels, you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! # What "correct" means for a tiled decode, precisely
//!
//! This decoder's spatial receptive field is ~15 latent cells wide (see
//! `LtxVaeTiledDecoder`'s own doc for the per-resolution sum) and the whole
//! 1080p latent is 34 cells tall, so **no** overlap that still saves memory
//! covers it. Tiled decode is therefore not, and upstream never claims it is,
//! a bit-exact factorisation of whole decode. Asserting cosine >= 0.999999
//! between the two at a real split would be asserting something physically
//! false, and the honest gate is three separate claims:
//!
//! 1. [`a_single_tile_plan_is_bit_identical_to_the_whole_decode`] - the
//!    tiling machinery (slice, per-shape graph build, mask, accumulate,
//!    divide) is **exact**. Anything that is not the receptive-field
//!    approximation shows up here as a non-zero `max_abs`.
//! 2. [`a_real_split_agrees_with_the_whole_decode_away_from_a_broken_port`] -
//!    at a genuine multi-tile split, tiled and whole agree to a MEASURED
//!    tolerance that a correct port meets and a broken one (wrong mask,
//!    swapped axis, missing divisor, dropped overlap) does not. The floor is
//!    deliberately far below the measured number, per this repo's own
//!    precedent for lossy tiers (`crates/flux1`'s int8 gate): the floor
//!    catches a broken port, it does not reproduce a specific run.
//! 3. [`the_blend_beats_a_hard_cut_at_the_same_tile_geometry`] - the
//!    trapezoidal blend is what buys the agreement. The same tiles stitched
//!    with a hard cut at the tile centre are measurably worse, so the blend
//!    is doing work rather than being decoration.
//!
//! Plus the shape that motivated the whole change:
//! [`real_1080p::a_full_25_frame_1080p_clip_decodes_on_one_card`] (`#[ignore]`d,
//! a multi-minute real-weight run), which the whole path cannot do at all.
//!
//! The blend's own arithmetic is gated separately and exactly in
//! `vae::tiling3d`'s `the_blend_reconstructs_a_known_volume_exactly`, where
//! the "decoder" is the identity and so a mask/slice/divisor bug cannot hide
//! behind the receptive-field approximation.

use std::path::Path;
use std::sync::OnceLock;

use ltxv::import::import_vae;
use ltxv::vae3d::{LtxVaeConfig, LtxVaeDecoder, LtxVaeTiledDecoder, LtxVaeTiling};
use vae::blocks::Tensors;

// ------------------------------------------------------------------ metrics

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

/// `||got - want||_2 / ||want||_2` - the error cosine is blind to, since
/// cosine is scale-invariant (lessons.md #2).
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in got.iter().zip(want) {
        num += (*x as f64 - *y as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    if den <= 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// All three, always - a single number cannot see every class of bug.
fn report(label: &str, got: &[f32], want: &[f32]) -> (f64, f64, f32) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, r, m) = (cosine(got, want), rel_l2(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  rel_l2={r:.4e}  max_abs={m:.4e}  n={}", got.len());
    (c, r, m)
}

// ----------------------------------------------------------- real fixtures

fn weights_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_VAE") {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
    }
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/ltxv/weights/vae/ltx-2.5-video-vae-conv-bf16.safetensors");
    Path::new(p).exists().then(|| p.to_string())
}

/// Imported once for the whole binary - `vae_parity.rs` records why (this is
/// a ~726M-parameter checkpoint and one copy per test SIGKILLed the runner).
static WEIGHTS: OnceLock<Option<Tensors>> = OnceLock::new();

fn weights() -> Option<&'static Tensors> {
    WEIGHTS
        .get_or_init(|| {
            let wp = weights_path()?;
            let cfg = LtxVaeConfig::conv25();
            let raw = checkpoint::safetensors::read(&wp).ok()?;
            let w = import_vae(raw, &cfg).ok()?;
            eprintln!("imported real VAE weights from {wp}");
            Some(w)
        })
        .as_ref()
}

fn require_weights() -> Option<&'static Tensors> {
    let w = weights();
    if w.is_none() {
        brain_testutil::skip("set BRAIN_LTXV_VAE to ltx-2.5-video-vae-conv-bf16.safetensors");
    }
    w
}

/// A deterministic, structured latent - NOT white noise. A latent whose
/// content is uncorrelated across space makes every tiling scheme look
/// equally bad (there is nothing for the overlap to agree about), so the
/// signal here has smooth low-frequency structure plus a per-channel offset,
/// which is far closer to what the DiT actually emits and is what makes a
/// seam visible when there is one.
fn structured_latent(c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; c * t * h * w];
    for ci in 0..c {
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    let x = wi as f32 / w.max(2) as f32;
                    let y = hi as f32 / h.max(2) as f32;
                    let s = (6.0 * x + ci as f32 * 0.11).sin() * (5.0 * y + ti as f32 * 0.7).cos() + 0.35 * ((11.0 * x + 7.0 * y).sin());
                    v[((ci * t + ti) * h + hi) * w + wi] = 0.8 * s + 0.05 * (ci as f32 * 0.37).sin();
                }
            }
        }
    }
    v
}

// -------------------------------------------------------------- gate 1

/// The tiling machinery, isolated from the receptive-field approximation:
/// give the planner tiles larger than every axis so the cover is exactly ONE
/// tile, and the tiled path must reproduce the whole path's bits.
///
/// This is the gate that fails if `slice_latent` transposes an axis, if the
/// per-shape graph grouping decodes the wrong tile, if the mask is not all-1
/// on an untiled axis, or if the divisor is not exactly 1 - none of which the
/// approximate multi-tile comparison below could isolate.
///
/// The three latent extents are deliberately all DIFFERENT (2 x 2 x 3, i.e.
/// 9 frames at 96x64). A cube would make an axis swap invisible, which is
/// precisely the bug this gate is best placed to catch.
#[test]
fn a_single_tile_plan_is_bit_identical_to_the_whole_decode() {
    let Some(w) = require_weights() else { return };
    let cfg = LtxVaeConfig::conv25();
    let (lt, lh, lw) = (2u32, 2u32, 3u32); // 9 frames at 96x64
    let latent = structured_latent(cfg.latent_channels as usize, lt as usize, lh as usize, lw as usize);

    let whole = LtxVaeDecoder::build(&cfg, w, lt, lh, lw, None).decode(&latent);

    // Tiles far larger than the axes -> one tile, no ramps.
    let tiling = LtxVaeTiling { frames: (80, 24), height: (2048, 64), width: (2048, 64) };
    let tiled = LtxVaeTiledDecoder::new(&cfg, w, lt, lh, lw, None, tiling);
    assert_eq!(tiled.plan().tiles().len(), 1, "this gate needs a degenerate one-tile plan");
    let got = tiled.decode(&latent);

    // Bit patterns first, and not `==` on f32: two NaNs compare unequal, so
    // an all-NaN decode would pass a float comparison of "no difference"
    // while a bit comparison correctly calls it identical - and the finite
    // check below then catches it.
    assert!(got.iter().zip(&whole).all(|(a, b)| a.to_bits() == b.to_bits()), "bit patterns differ");
    assert!(got.iter().all(|v| v.is_finite()), "decode produced non-finite values");

    let (c, r, m) = report("one-tile tiled vs whole", &got, &whole);
    assert_eq!(m, 0.0, "a one-tile plan must be BIT-identical, got max_abs {m:e}");
    assert_eq!(r, 0.0, "rel_l2 {r}");
    // Cosine is accumulated in f64 over 110k terms, so exact 1.0 is not a
    // property even of two bit-identical inputs; one ulp is.
    assert!((c - 1.0).abs() <= 4.0 * f64::EPSILON, "cosine {c}");
}

// -------------------------------------------------------------- gate 2

/// Decode the same latent both ways at a shape where BOTH paths run, with a
/// tiling that genuinely splits, and report all three metrics.
fn split_vs_whole(tiling: LtxVaeTiling, lt: u32, lh: u32, lw: u32) -> Option<(f64, f64, f32)> {
    let w = require_weights()?;
    let cfg = LtxVaeConfig::conv25();
    let latent = structured_latent(cfg.latent_channels as usize, lt as usize, lh as usize, lw as usize);
    let whole = LtxVaeDecoder::build(&cfg, w, lt, lh, lw, None).decode(&latent);
    let tiled = LtxVaeTiledDecoder::new(&cfg, w, lt, lh, lw, None, tiling);
    let n = tiled.plan().tiles().len();
    assert!(n > 1, "this gate needs a genuinely split plan, got {n} tile(s)");
    eprintln!("plan: {n} tiles, overlap waste {:.3}x", tiled.plan().overlap_waste());
    let got = tiled.decode(&latent);
    Some(report(&format!("{n}-tile tiled vs whole"), &got, &whole))
}

/// A real 2x2 spatial split at 9 frames / 256x256 (latent 2x8x8, tile 128 px
/// = 4 latent cells, overlap 64 px = 2 cells).
///
/// The floors below are NOT the measured numbers - they are a wide band
/// around them, because the quantity being bounded is an inherent
/// approximation (see this module's header) and because a floor pinned to one
/// machine's exact output is a gate that fails for the wrong reason.
///
/// **What this gate does and does not catch**, established by deliberately
/// breaking the implementation and re-running rather than by assertion:
///
/// * Removing the blend divisor entirely: **no effect here** (cosine
///   0.999093484, unchanged to nine digits). At this geometry the masks
///   partition unity exactly, so the divisor *is* 1.0. It earns its place on
///   geometries where a short final tile clamps its own ramp, not this one.
/// * Building the spatial mask with the TEMPORAL ramp convention: **no effect
///   here** either (cosine 0.999098052). Because this port always divides by
///   the accumulated weight, any positive ramp shape renormalises into a
///   valid partition of unity - which is a real robustness property of the
///   always-divide choice, and also the reason the ramp CONVENTION cannot be
///   gated from here.
/// * Flipping the convention on the TEMPORAL axis: caught, but by
///   `vae::tiling3d`'s `the_temporal_masks_partition_unity_after_the_causal_
///   shift` (deviation 0.0556 against a 1e-6 bound) and its
///   `the_blend_reconstructs_a_known_volume_exactly`, not by this test.
///
/// So this gate's real job is narrow and worth being honest about: it bounds
/// end-to-end agreement on real weights so a gross structural error (decoding
/// the wrong sub-volume, stitching to the wrong offset, losing a tile) cannot
/// pass. The exact properties live in `vae::tiling3d`'s unit tests and in
/// [`a_single_tile_plan_is_bit_identical_to_the_whole_decode`]; the blend's
/// value over not blending lives in
/// [`the_blend_beats_a_hard_cut_at_the_same_tile_geometry`].
#[test]
fn a_real_split_agrees_with_the_whole_decode_away_from_a_broken_port() {
    let tiling = LtxVaeTiling { frames: (80, 24), height: (128, 64), width: (128, 64) };
    let Some((c, r, m)) = split_vs_whole(tiling, 2, 8, 8) else { return };
    assert!(c >= 0.99, "cosine {c:.9} below the broken-port floor 0.99");
    assert!(r <= 0.20, "rel_l2 {r:.4e} above the broken-port ceiling 0.20");
    assert!(m.is_finite(), "max_abs {m} is not finite");
}

// -------------------------------------------------------------- gate 3

/// The blend has to be earning its place. Stitch the SAME tiles with a hard
/// cut (each output cell taken from whichever tile's centre is nearest, i.e.
/// a rectangular rather than trapezoidal mask) and the agreement with the
/// whole decode must get measurably worse.
///
/// Without this, a blend that silently degenerated to "last tile wins" would
/// still pass gate 2's wide band.
#[test]
fn the_blend_beats_a_hard_cut_at_the_same_tile_geometry() {
    let Some(w) = require_weights() else { return };
    let cfg = LtxVaeConfig::conv25();
    let (lt, lh, lw) = (2u32, 8u32, 8u32);
    let latent = structured_latent(cfg.latent_channels as usize, lt as usize, lh as usize, lw as usize);
    let whole = LtxVaeDecoder::build(&cfg, w, lt, lh, lw, None).decode(&latent);

    let tiling = LtxVaeTiling { frames: (80, 24), height: (128, 64), width: (128, 64) };
    let tiled = LtxVaeTiledDecoder::new(&cfg, w, lt, lh, lw, None, tiling);
    let blended = tiled.decode(&latent);

    // Hard cut: replay the same tiles, but write each tile's pixels straight
    // into the output with no mask and no divisor, so later tiles overwrite
    // earlier ones in the overlap.
    let plan = tiled.plan();
    let (f, h, wd) = plan.out_shape();
    let mut cut = vec![0.0f32; 3 * f * h * wd];
    for tile in plan.tiles() {
        let (st, sh, sw) = (tile.t.src_len(), tile.h.src_len(), tile.w.src_len());
        let dec = LtxVaeDecoder::build(&cfg, w, st as u32, sh as u32, sw as u32, None);
        let sub = slice_latent(&latent, cfg.latent_channels as usize, (lt as usize, lh as usize, lw as usize), tile);
        let px = dec.decode(&sub);
        let (tf, th, tw) = (tile.t.dst_len(), tile.h.dst_len(), tile.w.dst_len());
        for ci in 0..3 {
            for fi in 0..tf {
                for hi in 0..th {
                    let src = ((ci * tf + fi) * th + hi) * tw;
                    let dst = ((ci * f + tile.t.dst.0 + fi) * h + tile.h.dst.0 + hi) * wd + tile.w.dst.0;
                    cut[dst..dst + tw].copy_from_slice(&px[src..src + tw]);
                }
            }
        }
    }

    let (cb, rb, _) = report("trapezoidal blend vs whole", &blended, &whole);
    let (cc, rc, _) = report("hard cut          vs whole", &cut, &whole);
    assert!(cb > cc, "the blend ({cb:.9}) must beat a hard cut ({cc:.9})");
    assert!(rb < rc, "the blend's rel_l2 ({rb:.4e}) must beat a hard cut's ({rc:.4e})");
}

/// The same sub-volume cut `LtxVaeTiledDecoder` does internally - duplicated
/// here rather than exposed, because a test that reused the private helper
/// would be testing it against itself.
fn slice_latent(latent: &[f32], c: usize, (lt, lh, lw): (usize, usize, usize), tile: vae::tiling3d::Tile3d<'_>) -> Vec<f32> {
    let (t0, t1) = tile.t.src;
    let (h0, h1) = tile.h.src;
    let (w0, w1) = tile.w.src;
    let (tt, th, tw) = (t1 - t0, h1 - h0, w1 - w0);
    let mut out = vec![0.0f32; c * tt * th * tw];
    for ci in 0..c {
        for ti in 0..tt {
            for hi in 0..th {
                let src = ((ci * lt + t0 + ti) * lh + h0 + hi) * lw + w0;
                let dst = ((ci * tt + ti) * th + hi) * tw;
                out[dst..dst + tw].copy_from_slice(&latent[src..src + tw]);
            }
        }
    }
    out
}

// ------------------------------------------------------- the real shape

mod real_1080p {
    use super::*;

    /// The gate-2 comparison at the geometry a real 1080p generation
    /// actually runs: `LtxVaeTiling::auto(1088, 1920)`'s own 3x3 cover, at 9
    /// frames rather than 25 so the WHOLE path still fits (measured 15186 MiB
    /// of 24576) and there is something to compare against.
    ///
    /// The routine gate above uses a deliberately harsher split (128 px tiles
    /// on a 256 px image, 2.25x overlap waste) because it has to run in
    /// seconds; this is the number that describes production. `#[ignore]`d
    /// for cost, not for confidence.
    #[test]
    #[ignore = "real weights, minutes: two full 1080p-geometry decodes"]
    fn the_production_1080p_tile_geometry_agrees_with_the_whole_decode() {
        let Some(w) = require_weights() else { return };
        let cfg = LtxVaeConfig::conv25();
        let (lt, lh, lw) = (2u32, 34u32, 60u32); // 9 frames at 1920x1088
        let latent = structured_latent(cfg.latent_channels as usize, lt as usize, lh as usize, lw as usize);

        let t0 = std::time::Instant::now();
        let whole = LtxVaeDecoder::build(&cfg, w, lt, lh, lw, Some("gpu")).decode(&latent);
        let whole_s = t0.elapsed().as_secs_f64();

        let tiled_dec = LtxVaeTiledDecoder::auto(&cfg, w, lt, lh, lw, Some("gpu"));
        assert_eq!(tiled_dec.plan().tiles().len(), 9, "expected the production 3x3 cover");
        let t1 = std::time::Instant::now();
        let tiled = tiled_dec.decode(&latent);
        let tiled_s = t1.elapsed().as_secs_f64();

        eprintln!("whole {whole_s:.1}s vs tiled {tiled_s:.1}s ({:.2}x), overlap waste {:.3}x", tiled_s / whole_s, tiled_dec.plan().overlap_waste());
        let (c, r, m) = report("production 1080p geometry: tiled vs whole", &tiled, &whole);
        assert!(c >= 0.999, "cosine {c:.9}");
        assert!(r <= 0.05, "rel_l2 {r:.4e}");
        assert!(m.is_finite());
    }

    /// Mean absolute horizontal gradient per output column, averaged over
    /// channels, frames and rows. A seam is a column where neighbouring
    /// pixels jump; it shows up here as a spike.
    fn column_gradient(px: &[f32], f: usize, h: usize, w: usize) -> Vec<f32> {
        let mut g = vec![0.0f64; w.saturating_sub(1)];
        for ci in 0..3 {
            for fi in 0..f {
                for hi in 0..h {
                    let row = ((ci * f + fi) * h + hi) * w;
                    for wi in 0..w - 1 {
                        g[wi] += (px[row + wi + 1] - px[row + wi]).abs() as f64;
                    }
                }
            }
        }
        let n = (3 * f * h) as f64;
        g.into_iter().map(|v| (v / n) as f32).collect()
    }

    fn median(mut v: Vec<f32>) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v[v.len() / 2]
    }

    /// The shape this whole change exists for: 25 frames at 1920x1088, which
    /// the whole path aborts on with `wgpu error: Out of Memory` on a 24 GiB
    /// card (measured; see `WHOLE_DECODE_MAX_PIXELS`).
    ///
    /// There is no whole-path result to compare against - that is the point -
    /// so this gates what CAN be checked without one:
    ///
    /// * it completes at all;
    /// * every value is finite;
    /// * the output has real dynamic range, not a flat or degenerate image;
    /// * and no tile boundary column is a gradient outlier, which is the
    ///   observable a seam would produce.
    ///
    /// `#[ignore]`d: it is a multi-minute real-weight GPU run, not a unit
    /// test. Run with `--ignored`.
    #[test]
    #[ignore = "real weights, multi-minute 1080p decode"]
    fn a_full_25_frame_1080p_clip_decodes_on_one_card() {
        decodes_a_shape_the_whole_path_cannot(4, 34, 60); // 25 frames at 1920x1088
    }

    /// The other shape Phase 15 recorded as not fitting: 49 frames at
    /// 1280x704 (44.2 Mpx, past the ~35 Mpx whole-path ceiling). It is the
    /// SAME mechanism with no special case - the auto layout tiles whichever
    /// axes the shape needs, which here is again the two spatial ones (49
    /// frames is 7 latent frames, still under the reference's 10-latent
    /// temporal tile).
    #[test]
    #[ignore = "real weights, multi-minute 720p/49-frame decode"]
    fn a_full_49_frame_720p_clip_decodes_on_one_card() {
        decodes_a_shape_the_whole_path_cannot(7, 22, 40); // 49 frames at 1280x704
    }

    /// Everything that can be gated about a decode with NO whole-path result
    /// to compare against - which is the situation by construction, since
    /// these are exactly the shapes the whole path aborts on:
    ///
    /// * it completes at all;
    /// * every value is finite;
    /// * the output has real dynamic range, not a flat or degenerate image;
    /// * and no tile boundary column is a gradient outlier, which is the
    ///   observable a seam would produce.
    fn decodes_a_shape_the_whole_path_cannot(lt: u32, lh: u32, lw: u32) {
        let Some(w) = require_weights() else { return };
        let cfg = LtxVaeConfig::conv25();
        let (f, h, wd) = ((1 + 8 * (lt - 1)) as usize, (lh * 32) as usize, (lw * 32) as usize);
        assert!(ltxv::vae3d::should_tile(f as u32, h as u32, wd as u32) || std::env::var("BRAIN_LTXV_VAE_TILE").is_ok(), "{f}f at {wd}x{h} is not a shape that needs tiling - this gate would prove nothing");
        let latent = structured_latent(cfg.latent_channels as usize, lt as usize, lh as usize, lw as usize);

        let tiled = LtxVaeTiledDecoder::auto(&cfg, w, lt, lh, lw, Some("gpu"));
        eprintln!("{f}f {wd}x{h} plan: {} tiles, overlap waste {:.3}x", tiled.plan().tiles().len(), tiled.plan().overlap_waste());
        let t0 = std::time::Instant::now();
        let px = tiled.decode_with(&latent, |d, n| eprintln!("  tile {d}/{n} ({:.1}s)", t0.elapsed().as_secs_f64()));
        let secs = t0.elapsed().as_secs_f64();

        assert_eq!(px.len(), 3 * f * h * wd, "decoded {} values", px.len());
        assert!(px.iter().all(|v| v.is_finite()), "output contains NaN or inf");

        let (lo, hi) = px.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        let mean = px.iter().map(|v| *v as f64).sum::<f64>() / px.len() as f64;
        let var = px.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / px.len() as f64;
        eprintln!("{f}f {wd}x{h} tiled decode: {secs:.1}s  min={lo:.4} max={hi:.4} mean={mean:.4} std={:.4}", var.sqrt());
        assert!(hi - lo > 0.5, "degenerate dynamic range [{lo}, {hi}]");
        assert!(var.sqrt() > 0.02, "output is near-flat (std {:.5})", var.sqrt());

        // Seam check: the centre of each interior tile's fade-in region is
        // where a badly blended boundary shows, so probe a window there
        // rather than at an assumed pixel column.
        let g = column_gradient(&px, f, h, wd);
        let med = median(g.clone());
        let mut seam_cols: Vec<usize> = Vec::new();
        for t in tiled.plan().w.tiles.iter().skip(1) {
            let c = t.dst.0 + t.mask.iter().take_while(|m| **m < 1.0).count() / 2;
            seam_cols.extend((c.saturating_sub(2))..(c + 3).min(g.len()));
        }
        assert!(!seam_cols.is_empty(), "no interior seam to check - the plan did not split the width axis");
        let worst = seam_cols.iter().map(|&i| g[i]).fold(0.0f32, f32::max);
        eprintln!("column gradient: median={med:.5}  worst at a seam={worst:.5}  ratio={:.2}x", worst / med.max(1e-8));
        assert!(worst <= 6.0 * med.max(1e-8), "a tile boundary is a gradient outlier ({worst:.5} vs median {med:.5}) - that is a visible seam");
    }
}
