// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Colour normalisation as **data**.
//!
//! `IMAGENET_MEAN` / `IMAGENET_STD` were declared byte-identically in
//! `zipdepth::init` and `worldmirror2::preprocess`. Hoisting the constants is safe; what
//! is *not* safe is unifying where they are applied, and that is worth stating
//! at the point where the constants live:
//!
//! * `depth` folds mean/std into the **first BatchNorm's** parameters at import
//!   (`zipdepth::init`), so the model consumes `[0, 1]` directly and its predictor
//!   never normalises. Nothing to migrate there but the two arrays.
//! * `mirror` applies them **per frame** inside `worldmirror2::model::forward`, for
//!   reference parity. `worldmirror2::preprocess::preprocess` also applies them and
//!   has zero callers, while `cli::mirror_cli` re-implements that function
//!   *without* them — so naively merging the CLI onto `preprocess` normalises
//!   twice.
//!
//! Hence: constants and a value type here; application stays put until a
//! migrator moves each site deliberately.
//!
//! ## Colour *conversion*
//!
//! The workspace's only colour-space conversion, [`yuyv_to_rgb`], now lives
//! here. It was `capture::convert`, which made `crates/capture` — a hand-rolled
//! V4L2 ioctl binding — also the home of a pixel-format converter. `capture` is
//! V4L2 only again; the one caller (`cli::depth_cli`) calls this.
//!
//! It stays on the host deliberately: it consumes the *packed 4:2:2 byte* stream
//! straight out of the V4L2 mmap, upstream of any device buffer, and the frame is
//! uploaded exactly once — after this — rather than twice.

/// ImageNet channel means, RGB, for inputs already scaled to `[0, 1]`.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet channel standard deviations, RGB.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A per-channel `(x - mean) / std`.
///
/// One type covers every normalisation in the workspace, because they differ
/// only in their numbers: ImageNet ([`Normalization::IMAGENET`]),
/// `qwen3vl::preprocess::normalize_unit`'s `(x - 0.5) / 0.5`
/// ([`Normalization::HALF`], the `[0,1] -> [-1,1]` map), and the pass-through
/// used by models that normalise internally ([`Normalization::IDENTITY`]).
///
/// It carries no channel *order* assumption beyond "three channels, in the
/// buffer's own order". Callers that hand it a BGR buffer get BGR statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Normalization {
    /// Leaves values untouched — for models that normalise inside their own
    /// forward (`mirror`) or that folded the constants into a BatchNorm
    /// (`depth`).
    pub const IDENTITY: Normalization = Normalization { mean: [0.0; 3], std: [1.0; 3] };
    /// ImageNet statistics.
    pub const IMAGENET: Normalization =
        Normalization { mean: IMAGENET_MEAN, std: IMAGENET_STD };
    /// `(x - 0.5) / 0.5`, i.e. the `[0, 1] -> [-1, 1]` value-range map that
    /// `flux2::finetune`, `s3dit::finetune` and `qwen3vl::preprocess` each write
    /// out by hand. It is a normalisation, not a separate concept.
    pub const HALF: Normalization = Normalization { mean: [0.5; 3], std: [0.5; 3] };

    /// The forward map as `(scale, shift)` with `y = x * scale + shift`.
    ///
    /// This is the form the device path wants: `film_chan` is brain's
    /// per-channel affine and it is the ONLY kernel needed to normalise,
    /// denormalise, negate, invert a mask or rescale a value range. Returning
    /// scale/shift rather than mean/std keeps the algebra in one place instead
    /// of at every dispatch site.
    pub fn scale_shift(&self) -> ([f32; 3], [f32; 3]) {
        let mut scale = [0.0f32; 3];
        let mut shift = [0.0f32; 3];
        for c in 0..3 {
            assert!(self.std[c] != 0.0, "Normalization: std[{c}] is zero");
            scale[c] = 1.0 / self.std[c];
            shift[c] = -self.mean[c] / self.std[c];
        }
        (scale, shift)
    }

    /// The inverse map as `(scale, shift)`: `x = y * std + mean`.
    pub fn inverse_scale_shift(&self) -> ([f32; 3], [f32; 3]) {
        (self.std, self.mean)
    }
}

// ---- YUYV 4:2:2 -> interleaved RGB8 ----
//
// YUYV (a.k.a. YUY2) packs two pixels in four bytes: `Y0 U Y1 V`, so the two
// pixels share one chroma pair. brain forces this format at V4L2 negotiation
// (MJPEG would need a JPEG decoder in the capture path), so this is the only
// conversion a webcam frame needs. The BT.601 full-range coefficients match
// what UVC webcams emit.

/// Convert a `YUYV` buffer (`w*h*2` bytes) to interleaved `RGB8` (`w*h*3` bytes).
///
/// `w` must be even (YUYV pairs pixels horizontally); an odd width is a caller
/// bug, not something to paper over, so it panics.
pub fn yuyv_to_rgb(yuyv: &[u8], w: u32, h: u32) -> Vec<u8> {
    assert_eq!(w % 2, 0, "YUYV width must be even (pixels are paired), got {w}");
    assert_eq!(yuyv.len(), (w * h * 2) as usize, "YUYV buffer must be w*h*2 bytes");
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut si = 0usize;
    let mut di = 0usize;
    let npairs = (w * h / 2) as usize;
    for _ in 0..npairs {
        let y0 = yuyv[si] as f32;
        let u = yuyv[si + 1] as f32 - 128.0;
        let y1 = yuyv[si + 2] as f32;
        let v = yuyv[si + 3] as f32 - 128.0;
        si += 4;
        // BT.601. Same u/v for both pixels (4:2:2 shares chroma).
        let r_off = 1.402 * v;
        let g_off = -0.344136 * u - 0.714136 * v;
        let b_off = 1.772 * u;
        for y in [y0, y1] {
            rgb[di] = clamp8(y + r_off);
            rgb[di + 1] = clamp8(y + g_off);
            rgb[di + 2] = clamp8(y + b_off);
            di += 3;
        }
    }
    rgb
}

fn clamp8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_shift_reproduces_x_minus_mean_over_std() {
        let n = Normalization::IMAGENET;
        let (s, b) = n.scale_shift();
        for c in 0..3 {
            let x = 0.7f32;
            let direct = (x - n.mean[c]) / n.std[c];
            assert!((x * s[c] + b[c] - direct).abs() < 1e-6);
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        for n in [Normalization::IMAGENET, Normalization::HALF, Normalization::IDENTITY] {
            let (s, b) = n.scale_shift();
            let (is, ib) = n.inverse_scale_shift();
            for c in 0..3 {
                let x = 0.3f32;
                let y = x * s[c] + b[c];
                assert!((y * is[c] + ib[c] - x).abs() < 1e-6, "{n:?} channel {c}");
            }
        }
    }

    #[test]
    fn half_is_the_unit_to_signed_value_range_map() {
        let (s, b) = Normalization::HALF.scale_shift();
        // 0 -> -1, 1 -> +1: the map written inline four times in the workspace.
        assert!((0.0 * s[0] + b[0] + 1.0).abs() < 1e-6);
        assert!((1.0 * s[0] + b[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn identity_is_a_no_op() {
        let (s, b) = Normalization::IDENTITY.scale_shift();
        assert_eq!((s, b), ([1.0; 3], [0.0; 3]));
    }
}