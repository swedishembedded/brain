// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measures for "is this render any good", so a test can answer that instead
//! of a person looking at a picture.
//!
//! # Why accuracy alone is not enough
//!
//! A reconstruction can be wrong in two directions and PSNR only sees one of
//! them well. Blur is the dangerous one: a blurred render stays close to its
//! reference in mean-squared terms - low-frequency content is most of the
//! energy in a photograph - so a scene can lose every edge it had and still
//! score respectably. Worse, blur has no reference at all when the render is
//! at a resolution nothing was captured at, which is exactly when it happens.
//!
//! [`hf_energy`] is therefore the second measure: how much high-frequency
//! content an image contains, on its own, with nothing to compare against.
//! Ratioed against a reference's it catches both failures with one number -
//! below the band is blur, above it is speckle - and neither is visible in
//! PSNR until it is severe.
//!
//! Swedish Embedded AB implements automated image-quality gating for rendering
//! and reconstruction pipelines. If your team needs regressions in visual
//! quality caught by a test rather than by a customer, you can procure our
//! services by sending an email to info@swedishembedded.com.

/// Rec. 601 luma of interleaved RGB in `[0,1]`.
fn luma(rgb: &[f32], i: usize) -> f32 {
    0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
}

/// Peak signal-to-noise ratio in dB between two interleaved-RGB images in
/// `[0,1]`. `INFINITY` when they are identical.
pub fn psnr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr: images differ in size");
    let mse: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    -10.0 * mse.log10()
}

/// Mean squared 4-neighbour Laplacian of the luma channel: how much
/// high-frequency content an image carries, measured WITHOUT a reference.
///
/// A sharp image of a textured scene scores high; the same image blurred
/// scores lower in proportion to how much edge energy the blur removed. The
/// border ring is skipped rather than clamped, so a frame's edge does not
/// register as an edge.
pub fn hf_energy(rgb: &[f32], w: usize, h: usize) -> f64 {
    assert_eq!(rgb.len(), w * h * 3, "hf_energy: {w}x{h} does not match {} values", rgb.len());
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let lap = 4.0 * luma(rgb, i)
                - luma(rgb, i - 1)
                - luma(rgb, i + 1)
                - luma(rgb, i - w)
                - luma(rgb, i + w);
            acc += (lap as f64).powi(2);
        }
    }
    acc / ((w - 2) * (h - 2)) as f64
}

/// `hf_energy(render) / hf_energy(reference)`: 1.0 is "as sharp as the thing
/// it is supposed to look like".
///
/// Below 1 the render lost detail the reference has (blur, over-smoothing, a
/// scene rendered above the resolution it was reconstructed at). Above 1 it
/// has detail the reference does not (speckle, ringing, splat aliasing from
/// an over-fitted scene). A gate wants a BAND, not a floor.
pub fn sharpness_ratio(render: &[f32], reference: &[f32], w: usize, h: usize) -> f64 {
    let r = hf_energy(reference, w, h);
    assert!(r > 0.0, "sharpness_ratio: the reference has no high-frequency content to compare against");
    hf_energy(render, w, h) / r
}

/// PSNR in dB over the pixels `mask` weights (`None` = all), interleaved
/// RGB in `[0,1]`.
pub fn psnr_masked(a: &[f32], b: &[f32], mask: Option<&[f32]>) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr_masked: images differ in size");
    let (mut se, mut n) = (0.0f64, 0.0f64);
    for p in 0..a.len() / 3 {
        let w = mask.map_or(1.0, |m| m[p] as f64);
        if w <= 0.0 {
            continue;
        }
        for c in 0..3 {
            se += w * ((a[p * 3 + c] - b[p * 3 + c]) as f64).powi(2);
        }
        n += 3.0 * w;
    }
    if se <= 0.0 || n <= 0.0 {
        return f64::INFINITY;
    }
    -10.0 * (se / n).log10()
}

/// Structural similarity (Wang et al., IEEE TIP 2004): the mean SSIM map of
/// two interleaved-RGB images in `[0,1]`, per channel, over an 11x11 gaussian
/// window of sigma 1.5 with `C1 = 0.01²`, `C2 = 0.03²` - the constants every
/// splatting paper reports SSIM with - averaged over the pixels `mask`
/// weights (`None` = all) and the channels. The window is renormalized at the
/// frame's border rather than zero-padded, so an edge pixel is scored on the
/// image it has.
pub fn ssim(a: &[f32], b: &[f32], w: usize, h: usize, mask: Option<&[f32]>) -> f64 {
    assert_eq!(a.len(), w * h * 3);
    assert_eq!(b.len(), w * h * 3);
    const R: isize = 5;
    let g: Vec<f64> = (-R..=R).map(|k| (-((k * k) as f64) / (2.0 * 1.5 * 1.5)).exp()).collect();
    let (c1, c2) = (0.01f64 * 0.01, 0.03f64 * 0.03);
    // five moments per channel, blurred separably
    let blur = |f: &(dyn Fn(usize) -> f64 + Sync)| -> Vec<f64> {
        let rows = backend_cpu::par::map(h * w, |i| {
            let (y, x) = (i / w, i % w);
            let (mut s, mut n) = (0.0, 0.0);
            for (k, gk) in g.iter().enumerate() {
                let xx = x as isize + k as isize - R;
                if xx >= 0 && (xx as usize) < w {
                    s += gk * f(y * w + xx as usize);
                    n += gk;
                }
            }
            s / n
        });
        backend_cpu::par::map(h * w, |i| {
            let (y, x) = (i / w, i % w);
            let (mut s, mut n) = (0.0, 0.0);
            for (k, gk) in g.iter().enumerate() {
                let yy = y as isize + k as isize - R;
                if yy >= 0 && (yy as usize) < h {
                    s += gk * rows[yy as usize * w + x];
                    n += gk;
                }
            }
            s / n
        })
    };
    let (mut total, mut n) = (0.0f64, 0.0f64);
    for c in 0..3 {
        let pa = |i: usize| a[i * 3 + c] as f64;
        let pb = |i: usize| b[i * 3 + c] as f64;
        let ma = blur(&pa);
        let mb = blur(&pb);
        let saa = blur(&|i| pa(i) * pa(i));
        let sbb = blur(&|i| pb(i) * pb(i));
        let sab = blur(&|i| pa(i) * pb(i));
        for i in 0..w * h {
            let wgt = mask.map_or(1.0, |m| m[i] as f64);
            if wgt <= 0.0 {
                continue;
            }
            let (va, vb, cov) = (saa[i] - ma[i] * ma[i], sbb[i] - mb[i] * mb[i], sab[i] - ma[i] * mb[i]);
            let s = ((2.0 * ma[i] * mb[i] + c1) * (2.0 * cov + c2)) / ((ma[i] * ma[i] + mb[i] * mb[i] + c1) * (va + vb + c2));
            total += wgt * s;
            n += wgt;
        }
    }
    if n > 0.0 { total / n } else { 1.0 }
}

/// How much a rendered camera path flickers: the mean absolute SECOND
/// difference `|I(t+1) - 2 I(t) + I(t-1)|` over consecutive frames of a path
/// sampled finely enough that the true image changes smoothly. Smooth motion
/// has a near-zero second difference; a splat popping in front of another as
/// the sort order flips, or a sub-pixel gaussian aliasing, is a spike in it.
/// Interleaved RGB frames of one size; 0 for fewer than three.
pub fn temporal_instability(frames: &[Vec<f32>]) -> f64 {
    if frames.len() < 3 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    let mut n = 0usize;
    for t in 1..frames.len() - 1 {
        for ((a, b), c) in frames[t - 1].iter().zip(&frames[t]).zip(&frames[t + 1]) {
            acc += (*c as f64 - 2.0 * *b as f64 + *a as f64).abs();
            n += 1;
        }
    }
    acc / n.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3x3 box blur, as a stand-in for every way a render loses detail.
    fn blur(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut out = rgb.to_vec();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                for c in 0..3 {
                    let mut s = 0.0;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            s += rgb[((y + dy - 1) * w + (x + dx - 1)) * 3 + c];
                        }
                    }
                    out[(y * w + x) * 3 + c] = s / 9.0;
                }
            }
        }
        out
    }

    /// Something shaped like a photograph rather than a test card: most of
    /// the energy in smooth low-frequency content, with fine texture on top.
    /// A pure checkerboard would make the point too easily - blur destroys
    /// nearly all of its energy, so PSNR catches it too, and the argument for
    /// measuring sharpness separately collapses.
    fn photographic(w: usize, h: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let base = 0.25 + 0.3 * (x as f32 / w as f32) + 0.09 * (y as f32 / 18.0).sin();
                let tex = 0.06 * (((x / 3) + (y / 3)) % 2) as f32;
                for c in 0..3 {
                    v[(y * w + x) * 3 + c] = (base + tex).clamp(0.0, 1.0);
                }
            }
        }
        v
    }

    /// The gate is only worth having if it fires, so the detector is itself
    /// under test. A 3x3 box blur over a photograph-like image removes ~90% of
    /// its high-frequency content while still scoring over 30 dB - a figure
    /// that passes any accuracy gate anyone would write. That gap IS the
    /// argument for measuring sharpness separately, so both halves are
    /// asserted: sharpness must catch it, and PSNR must not.
    #[test]
    fn blur_is_caught_by_sharpness_and_waved_through_by_psnr() {
        let (w, h) = (96, 96);
        let sharp = photographic(w, h);
        let soft = blur(&sharp, w, h);

        let ratio = sharpness_ratio(&soft, &sharp, w, h);
        assert!(ratio < 0.25, "a 3x3 box blur only dropped sharpness to {ratio:.3}; too dull to gate with");

        let db = psnr(&soft, &sharp);
        assert!(
            db > 30.0,
            "the blurred copy scores {db:.1} dB. If PSNR alone catches this, the case for a \
             separate sharpness gate is weaker than this module claims - re-check the claim."
        );
        assert_eq!(sharpness_ratio(&sharp, &sharp, w, h), 1.0);
    }

    /// And the other direction: noise ADDS high-frequency energy, so one
    /// number catches the over-fitted, aliasing or speckled render that a
    /// floor-only check would wave through. Same image, same PSNR range,
    /// opposite side of the band.
    #[test]
    fn speckle_pushes_the_same_ratio_the_other_way() {
        let (w, h) = (96, 96);
        let sharp = photographic(w, h);
        let mut speckled = sharp.clone();
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for v in speckled.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (*v + ((state >> 40) as f32 / 16777216.0 - 0.5) * 0.1).clamp(0.0, 1.0);
        }
        let ratio = sharpness_ratio(&speckled, &sharp, w, h);
        let db = psnr(&speckled, &sharp);
        assert!(ratio > 1.8, "speckle only raised sharpness to {ratio:.2}; a band gate would not see it");
        assert!(db > 25.0, "the speckled copy scores {db:.1} dB, low enough that PSNR would have caught it");
    }

    #[test]
    fn psnr_is_infinite_for_an_exact_match_and_finite_otherwise() {
        let a = photographic(16, 16);
        assert!(psnr(&a, &a).is_infinite());
        let mut b = a.clone();
        b[0] += 0.5;
        assert!(psnr(&a, &b).is_finite());
    }
}
