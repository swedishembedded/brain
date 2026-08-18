// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side space-to-depth / depth-to-space at the video VAE's outer
//! boundary - LTX's `patchify`/`unpatchify` (`ltx_core.model.video_vae.ops`),
//! run once on the raw pixel tensor **before** it is uploaded to the device
//! (encoder input) and once **after** it is read back (decoder output).
//!
//! It is done here as a plain host loop rather than a device kernel because
//! it runs exactly ONCE per encode/decode call, at the very edge of the
//! graph, on a tensor that is about to be uploaded/was just read back anyway
//! - a device kernel would only add a submit for no benefit.
//!
//! **Not the same channel sub-order as
//! [`vae::blocks3d::Builder3d::space_to_depth`]/`::depth_to_space`** (the
//! internal down/up-sample resamples) - a real bug found the hard way (real-
//! weight decoder cosine 0.982, "structurally right, numerically off", every
//! per-block tap up to and including `conv_out` bit-exact against the golden,
//! only the FINAL `unpatchify`'d `recon` off) before this doc comment existed.
//! `ops.py`'s `patchify`/`unpatchify` literally write their target axis as
//! `'... -> b (c p r q) f h w'` - `r` is tied to `(w r)` on the LHS (the
//! WIDTH sub-offset) and `q` to `(h q)` (the HEIGHT sub-offset), so the
//! order is c, **then width-offset, then height-offset (innermost)**.
//! `sampling.py`'s `SpaceToDepthDownsample`/`DepthToSpaceUpsample` instead
//! write `'... -> b (c p1 p2 p3) d h w'` with `p2` tied to `(h p2)` and `p3`
//! to `(w p3)` - **height before width**, the opposite order, and that one
//! IS what the two `Builder3d` methods implement (confirmed correct: every
//! internal resample block's tap matched the golden bit-exactly). Two
//! different upstream functions, two different conventions - do not unify
//! them into one "the LTX convention".

/// `[C,T,H,W] -> [C*ph*pw, T, H/ph, W/pw]`, `ops.py`'s `patchify` restricted
/// to `patch_size_t=1` (LTX never patches time): `einops`
/// `'c (h q)(w r) -> (c r q) h w'` - width offset `r` OUTER, height offset
/// `q` INNER: `y[(c*pw+iw)*ph+ih, t, ho, wo] = x[c, t, ho*ph+ih, wo*pw+iw]`.
pub fn patchify(x: &[f32], c: usize, t: usize, h: usize, w: usize, ph: usize, pw: usize) -> Vec<f32> {
    assert_eq!(x.len(), c * t * h * w, "patchify: {} values, expected {}", x.len(), c * t * h * w);
    assert!(h.is_multiple_of(ph) && w.is_multiple_of(pw), "patchify: {h}x{w} not divisible by ({ph},{pw})");
    let (ho, wo) = (h / ph, w / pw);
    let mut out = vec![0f32; c * ph * pw * t * ho * wo];
    for ci in 0..c {
        for iw in 0..pw {
            for ih in 0..ph {
                let co = (ci * pw + iw) * ph + ih;
                for ti in 0..t {
                    for hoi in 0..ho {
                        for woi in 0..wo {
                            let src = ((ci * t + ti) * h + (hoi * ph + ih)) * w + (woi * pw + iw);
                            let dst = ((co * t + ti) * ho + hoi) * wo + woi;
                            out[dst] = x[src];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`]: `[C*ph*pw, T, H/ph, W/pw] -> [C,T,H,W]`.
pub fn unpatchify(x: &[f32], cout: usize, t: usize, ho: usize, wo: usize, ph: usize, pw: usize) -> Vec<f32> {
    let cin = cout * ph * pw;
    assert_eq!(x.len(), cin * t * ho * wo, "unpatchify: {} values, expected {}", x.len(), cin * t * ho * wo);
    let (h, w) = (ho * ph, wo * pw);
    let mut out = vec![0f32; cout * t * h * w];
    for ci in 0..cout {
        for iw in 0..pw {
            for ih in 0..ph {
                let co = (ci * pw + iw) * ph + ih;
                for ti in 0..t {
                    for hoi in 0..ho {
                        for woi in 0..wo {
                            let src = ((co * t + ti) * ho + hoi) * wo + woi;
                            let dst = ((ci * t + ti) * h + (hoi * ph + ih)) * w + (woi * pw + iw);
                            out[dst] = x[src];
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weight-free, bit-exact: `unpatchify(patchify(x)) == x` on synthetic
    /// data at the real `patch_size=4`. The cheapest possible regression
    /// guard on the channel-ordering convention - getting it wrong silently
    /// permutes every pixel while still producing a plausible-looking (high
    /// but not perfect cosine) image, exactly the bug this test would have
    /// caught had it pinned the CORRECT order from the start (see this
    /// module's header).
    #[test]
    fn patchify_and_unpatchify_round_trip_bit_exactly() {
        let (c, t, h, w, p) = (3usize, 9usize, 64usize, 64usize, 4usize);
        let x: Vec<f32> = (0..(c * t * h * w)).map(|i| (i as f32) * 0.01 - 3.0).collect();
        let patched = patchify(&x, c, t, h, w, p, p);
        assert_eq!(patched.len(), c * p * p * t * (h / p) * (w / p));
        let back = unpatchify(&patched, c, t, h / p, w / p, p, p);
        assert_eq!(back, x);
    }

    /// The channel grouping order pinned by hand on a tiny case: patch size 2
    /// over a single 4x4 frame, one channel - `co = iw*2+ih` (WIDTH offset
    /// outer, HEIGHT offset inner - `ops.py`'s `(c p r q)`, `r`=width before
    /// `q`=height), the opposite of `Builder3d::space_to_depth`'s `(c p1 p2
    /// p3)` (height before width).
    #[test]
    fn channel_grouping_is_c_outer_then_width_then_height() {
        // 1x1x4x4, values = row-major index so the source pixel is obvious.
        #[rustfmt::skip]
        let x: Vec<f32> = vec![
             0.0,  1.0,  2.0,  3.0,
             4.0,  5.0,  6.0,  7.0,
             8.0,  9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0,
        ];
        let y = patchify(&x, 1, 1, 4, 4, 2, 2);
        // c*p*p = 4 channels out, t = 1, (h/p)*(w/p) = 2*2.
        assert_eq!(y.len(), 4 * 2 * 2);
        // co=0 (iw=0,ih=0): top-left of each 2x2 block -> [0, 2, 8, 10]
        assert_eq!(&y[0..4], &[0.0, 2.0, 8.0, 10.0]);
        // co=1 (iw=0,ih=1): bottom-left -> [4, 6, 12, 14]
        assert_eq!(&y[4..8], &[4.0, 6.0, 12.0, 14.0]);
        // co=2 (iw=1,ih=0): top-right -> [1, 3, 9, 11]
        assert_eq!(&y[8..12], &[1.0, 3.0, 9.0, 11.0]);
        // co=3 (iw=1,ih=1): bottom-right -> [5, 7, 13, 15]
        assert_eq!(&y[12..16], &[5.0, 7.0, 13.0, 15.0]);
    }
}
