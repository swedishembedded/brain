// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Preparing the conditioning IMAGE — the one input a ControlNet has that its
//! backbone does not.
//!
//! What the reference does, and what this reproduces: diffusers builds the
//! control image with a `VaeImageProcessor(do_convert_rgb=True,
//! **do_normalize=False**)`, so the tensor the ControlNet sees is plain
//! **`[0, 1]` CHW fp32 at PIXEL resolution** — *not* the `[-1, 1]` the VAE's
//! own image input uses. Feeding it `[-1, 1]` runs, produces finite residuals,
//! and is wrong everywhere; that asymmetry between two adjacent inputs of the
//! same pipeline is why this lives in a named module with a test rather than
//! inline at a call site.
//!
//! ## The ZipDepth synergy, and what it costs
//! brain already produces depth maps (`crates/depth`), so a depth-conditioned
//! ControlNet needs no external preprocessor: [`from_depth`] is the whole
//! adapter — normalise to `[0, 1]` and replicate to three channels, which is
//! exactly what `diffusers`' depth examples do to a MiDaS/DPT output before
//! `prepare_image`.
//!
//! It is deliberately a **function over an already-computed map, not a
//! dependency on `crates/depth`**. The cost of wiring the crate itself is not
//! the code (a `zipdepth::Predictor` call is a few lines); it is that there is no
//! depth-conditioned SDXL ControlNet checkpoint on this machine, so such a path
//! could not be parity-gated and would ship as untested plumbing behind a
//! `depth` → `vision` → `model` dependency edge. This module is the part that
//! *is* testable without one.

/// `[0, 1]` CHW fp32 from an interleaved RGB8 buffer — the conditioning tensor
/// a `ControlNetModel` expects.
///
/// The cast and the permutation are **`imaging::pixels`**, not a local loop:
/// that module exists precisely because `chw_to_hwc`/`hwc_to_chw` had been
/// written five times in this workspace and the `/255` twin a sixth, and a
/// ControlNet's conditioning image is the same host-side layout work every other
/// imaging crate hands to it. All that is genuinely local here is the channel
/// order.
///
/// `swap_rb` implements `controlnet_conditioning_channel_order == "bgr"`, which
/// a few released ControlNets ship; every SDXL one in scope is `"rgb"`.
pub fn from_rgb8(px: &[u8], h: u32, w: u32, swap_rb: bool) -> Vec<f32> {
    let (h, w) = (h as usize, w as usize);
    assert_eq!(px.len(), h * w * 3, "from_rgb8: {} bytes for {h}x{w}x3", px.len());
    let mut out = imaging::pixels::hwc_to_chw(&imaging::pixels::u8_to_unit(px), 3, h, w);
    if swap_rb {
        let n = h * w;
        let (r, rest) = out.split_at_mut(n);
        r.swap_with_slice(&mut rest[n..]);
    }
    out
}

/// `[0, 1]` CHW fp32 from a single-channel map, replicated across three
/// channels.
///
/// `range` is `Some((lo, hi))` for a fixed normalisation, or `None` to use the
/// map's own min/max. Per-image min/max is what the depth examples do; a fixed
/// range is what a video pipeline needs so consecutive frames do not flicker.
/// A constant map normalises to all zeros rather than dividing by zero.
pub fn from_map(map: &[f32], h: u32, w: u32, range: Option<(f32, f32)>) -> Vec<f32> {
    let n = (h as usize) * (w as usize);
    assert_eq!(map.len(), n, "from_map: {} values for {h}x{w}", map.len());
    let (lo, hi) = range.unwrap_or_else(|| {
        map.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)))
    });
    let span = hi - lo;
    let mut out = vec![0.0f32; 3 * n];
    for i in 0..n {
        let v = if span.abs() < f32::EPSILON { 0.0 } else { ((map[i] - lo) / span).clamp(0.0, 1.0) };
        out[i] = v;
        out[n + i] = v;
        out[2 * n + i] = v;
    }
    out
}

/// A ZipDepth (`crates/depth`) prediction as a control image: per-image
/// min/max normalisation, replicated to three channels.
///
/// Takes the map rather than a `zipdepth::Predictor` on purpose - see the module
/// header.
pub fn from_depth(depth: &[f32], h: u32, w: u32) -> Vec<f32> {
    from_map(depth, h, w, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_is_zero_to_one_chw_not_minus_one_to_one() {
        let px = [0u8, 128, 255, 255, 0, 0];
        let v = from_rgb8(&px, 1, 2, false);
        assert_eq!(v, vec![0.0, 1.0, 128.0 / 255.0, 0.0, 255.0 / 255.0, 0.0]);
        assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn bgr_swaps_only_r_and_b() {
        let px = [10u8, 20, 30];
        let rgb = from_rgb8(&px, 1, 1, false);
        let bgr = from_rgb8(&px, 1, 1, true);
        assert_eq!(bgr, vec![rgb[2], rgb[1], rgb[0]]);
    }

    #[test]
    fn depth_normalises_and_replicates() {
        let d = [1.0f32, 3.0, 5.0, 2.0];
        let v = from_depth(&d, 2, 2);
        assert_eq!(v.len(), 12);
        assert_eq!(&v[..4], &[0.0, 0.5, 1.0, 0.25]);
        assert_eq!(&v[4..8], &v[..4]);
        assert_eq!(&v[8..], &v[..4]);
    }

    /// A flat map must not become NaN. This is the shape of input a real depth
    /// net produces on a blank wall.
    #[test]
    fn a_constant_map_does_not_divide_by_zero() {
        let v = from_depth(&[2.0f32; 4], 2, 2);
        assert!(v.iter().all(|x| *x == 0.0), "{v:?}");
    }
}
