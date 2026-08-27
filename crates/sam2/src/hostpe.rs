// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two POSITION TABLES SAM 2 builds from constants, plus the point-prompt
//! encoding — the only host arithmetic in this crate.
//!
//! Why these are host code and not kernels: every function here produces a
//! CONSTANT table (or encodes at most a handful of prompt points). They run once
//! per model build / once per prompt, never per pixel of a feature map, and they
//! are the same "build the table on the host, upload it, dispatch against it"
//! move `flux2` makes for its RoPE tables. Nothing here is a shared primitive
//! that belongs in `model::hostmath`: `PositionEmbeddingSine` and
//! `PositionEmbeddingRandom` are SAM 2's own definitions, and getting either
//! subtly wrong is a parity failure, not a style question — hence the goldens
//! (`possine_level*`, `dense_pe`, `sparse_embeddings`) that pin all three.
//!
//! Anything that touches a whole feature map (the interpolated `pos_embed`, the
//! dense mask embedding, every add) is a kernel dispatch in [`crate::model`].

use std::f32::consts::PI;

/// `PositionEmbeddingSine(num_pos_feats, temperature, normalize=True)` over an
/// `h x w` grid, returned NCHW-flat as `[num_pos_feats, h, w]`.
///
/// Channel layout: the first half is the Y encoding, the second half X; within a
/// half, even channels are `sin`, odd are `cos`, and BOTH read the same
/// `dim_t[j] = temperature^(2j / half)` because torch's `dim_t` uses integer
/// `dim_t // 2`.
pub fn sine(num_pos_feats: u32, temperature: f32, h: u32, w: u32) -> Vec<f32> {
    let half = (num_pos_feats / 2) as usize;
    let scale = 2.0 * PI;
    let eps = 1e-6f32;
    let dim_t: Vec<f32> = (0..half).map(|i| temperature.powf(2.0 * ((i / 2) as f32) / half as f32)).collect();
    let mut out = vec![0.0f32; num_pos_feats as usize * (h * w) as usize];
    for y in 0..h {
        // y_embed counts from 1 and is divided by its own last row.
        let ye = (y + 1) as f32 / (h as f32 + eps) * scale;
        for x in 0..w {
            let xe = (x + 1) as f32 / (w as f32 + eps) * scale;
            let hw = (y * w + x) as usize;
            for j in 0..half {
                let (py, px) = (ye / dim_t[j], xe / dim_t[j]);
                let v_y = if j % 2 == 0 { py.sin() } else { py.cos() };
                let v_x = if j % 2 == 0 { px.sin() } else { px.cos() };
                out[j * (h * w) as usize + hw] = v_y;
                out[(half + j) * (h * w) as usize + hw] = v_x;
            }
        }
    }
    out
}

/// `PositionEmbeddingRandom._pe_encoding` for one point already normalised to
/// `[0,1]^2`: `cat(sin, cos)` of `2π · ((2c-1) @ G)`, `G = [2, feats]`.
pub fn pe_encode(gauss: &[f32], feats: usize, x: f32, y: f32) -> Vec<f32> {
    let (cx, cy) = (2.0 * x - 1.0, 2.0 * y - 1.0);
    let mut out = vec![0.0f32; 2 * feats];
    for j in 0..feats {
        let v = 2.0 * PI * (cx * gauss[j] + cy * gauss[feats + j]);
        out[j] = v.sin();
        out[feats + j] = v.cos();
    }
    out
}

/// `PromptEncoder.get_dense_pe()` — `PositionEmbeddingRandom.forward((h, w))`,
/// returned NCHW-flat as `[2*feats, h, w]`. Grid coordinates are pixel CENTRES
/// (`cumsum - 0.5`), normalised by the grid extent.
pub fn dense_pe(gauss: &[f32], feats: usize, h: u32, w: u32) -> Vec<f32> {
    let hw = (h * w) as usize;
    let mut out = vec![0.0f32; 2 * feats * hw];
    for y in 0..h {
        for x in 0..w {
            let e = pe_encode(gauss, feats, (x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let p = (y * w + x) as usize;
            for (c, v) in e.iter().enumerate() {
                out[c * hw + p] = *v;
            }
        }
    }
    out
}

/// The learned embedding a point label selects, as an index into
/// `point_embeddings` — or `None` for the label `-1` padding point, which takes
/// `not_a_point_embed` INSTEAD of (not in addition to) the positional term.
///
/// Labels 2 and 3 are a box's top-left and bottom-right corners: the reference
/// never passes `boxes=` on this path, it passes the corners as points with
/// those labels, which is why a box and a point run the identical code.
/// Any other label is REJECTED rather than folded into the `-1` case: the
/// reference's chain of `torch.where`s leaves such a point carrying its bare
/// positional term with no learned embedding at all, which is a different
/// (and almost certainly unintended) vector — silently substituting
/// `not_a_point_embed` would be a parity divergence no golden covers.
pub fn label_slot(label: f32) -> Option<usize> {
    match label as i32 {
        0 => Some(0),
        1 => Some(1),
        2 => Some(2),
        3 => Some(3),
        -1 => None,
        other => panic!("sam2: point label {other} is not one of -1 (pad), 0/1 (bg/fg), 2/3 (box corners)"),
    }
}

/// `PromptEncoder._embed_points(points, labels, pad=True)`.
///
/// `coords` are absolute pixel coordinates in the `image_size` frame, `(x, y)`
/// per point. A padding point (label `-1`, coordinate `(0,0)`) is ALWAYS
/// appended, because the image path always calls the prompt encoder with
/// `boxes=None`. Returns `[(n+1), 2*feats]` row-major.
pub fn embed_points(
    gauss: &[f32],
    feats: usize,
    point_embeddings: &[Vec<f32>; 4],
    not_a_point: &[f32],
    coords: &[(f32, f32)],
    labels: &[f32],
    image_size: (u32, u32),
) -> Vec<f32> {
    assert_eq!(coords.len(), labels.len(), "one label per point");
    let d = 2 * feats;
    let mut out = Vec::with_capacity((coords.len() + 1) * d);
    let pts = coords.iter().map(|&(x, y)| (x, y)).chain(std::iter::once((-0.5f32, -0.5f32)));
    let labs = labels.iter().copied().chain(std::iter::once(-1.0f32));
    for ((x, y), l) in pts.zip(labs) {
        // `points + 0.5` (pixel centre) then normalise by (W, H). The padding
        // point is (0,0) BEFORE the shift, hence (-0.5,-0.5) above.
        let (px, py) = ((x + 0.5) / image_size.1 as f32, (y + 0.5) / image_size.0 as f32);
        let mut row = match label_slot(l) {
            Some(_) => pe_encode(gauss, feats, px, py),
            None => not_a_point.to_vec(),
        };
        if let Some(slot) = label_slot(l) {
            for (r, e) in row.iter_mut().zip(point_embeddings[slot].iter()) {
                *r += *e;
            }
        }
        out.extend_from_slice(&row);
    }
    out
}

/// `compute_axial_cis(dim, end_x, end_y, theta)` as the `(cos, sin)` pair the
/// `rope_interleave_table` kernel takes: two `[end_x*end_y, dim/2]` tables.
///
/// SAM 2's memory attention rotates the INTERLEAVED channel pairs `(2j, 2j+1)`
/// (`view_as_complex` on `reshape(..., -1, 2)`), and splits the `dim/2` pairs in
/// half: the first `dim/4` carry the X frequency, the second `dim/4` the Y one.
/// Token `t` sits at `x = t % end_x`, `y = t / end_x`.
pub fn axial_rope_tables(dim: u32, end_x: u32, end_y: u32, theta: f32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(dim % 4, 0, "axial RoPE wants dim divisible by 4, got {dim}");
    let quarter = (dim / 4) as usize;
    let half = (dim / 2) as usize;
    // `1 / theta^(arange(0, dim, 4)[:dim/4] / dim)`.
    let freqs: Vec<f32> = (0..quarter).map(|i| 1.0 / theta.powf((4 * i) as f32 / dim as f32)).collect();
    let n = (end_x * end_y) as usize;
    let (mut cos_t, mut sin_t) = (vec![0.0f32; n * half], vec![0.0f32; n * half]);
    for t in 0..n {
        let tx = (t as u32 % end_x) as f32;
        let ty = (t as u32 / end_x) as f32;
        for j in 0..quarter {
            let (ax, ay) = (tx * freqs[j], ty * freqs[j]);
            cos_t[t * half + j] = ax.cos();
            sin_t[t * half + j] = ax.sin();
            cos_t[t * half + quarter + j] = ay.cos();
            sin_t[t * half + quarter + j] = ay.sin();
        }
    }
    (cos_t, sin_t)
}

/// `sam2_utils.get_1d_sine_pe(pos, dim, temperature)` for one scalar position:
/// `cat(sin(pos / dim_t), cos(pos / dim_t))` over `dim/2` frequencies, where
/// `dim_t[i] = temperature^(2*(i/2) / (dim/2))` with INTEGER `i/2` - the two
/// halves are concatenated, NOT interleaved (unlike `sine`).
pub fn sine_1d(pos: f32, dim: u32, temperature: f32) -> Vec<f32> {
    let pe = (dim / 2) as usize;
    let mut out = vec![0.0f32; dim as usize];
    for i in 0..pe {
        let dt = temperature.powf(2.0 * ((i / 2) as f32) / pe as f32);
        let v = pos / dt;
        out[i] = v.sin();
        out[pe + i] = v.cos();
    }
    out
}

/// `y = W x + b` for a handful of rows, on the host.
///
/// The object-pointer temporal encoding is at most `max_obj_ptrs_in_encoder`
/// rows through one `[mem_dim, d_model]` projection - the same "a few rows, run
/// it on the host" case as the point prompts above, and keeping it here means
/// the memory-token assembly uploads ONE finished buffer instead of dispatching
/// a 16-row GEMM.
pub fn linear_rows(x: &[f32], w: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    assert_eq!(w.len(), n * k, "weight is [n, k]");
    assert_eq!(b.len(), n, "bias is [n]");
    assert_eq!(x.len() % k, 0, "x is [rows, k]");
    let rows = x.len() / k;
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut acc = b[j];
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

/// Tile `src` (`[c, sh, sw]`) over an `h x w` grid — `Tensor.tile` with integer
/// repeat factors, which is what Hiera does to its window position embedding
/// before adding it to the interpolated background embedding. Pure index
/// arithmetic, no math.
pub fn tile_chw(src: &[f32], c: u32, sh: u32, sw: u32, h: u32, w: u32) -> Vec<f32> {
    assert_eq!(h % sh, 0, "tile: {h} not a multiple of {sh}");
    assert_eq!(w % sw, 0, "tile: {w} not a multiple of {sw}");
    let mut out = vec![0.0f32; (c * h * w) as usize];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[((ch * h + y) * w + x) as usize] = src[((ch * sh + y % sh) * sw + x % sw) as usize];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_is_bounded_and_shaped() {
        let p = sine(256, 10000.0, 4, 6);
        assert_eq!(p.len(), 256 * 24);
        assert!(p.iter().all(|v| v.abs() <= 1.0 + 1e-6));
    }

    /// The X half of the axial table must vary along x and be CONSTANT along y
    /// (and the Y half the mirror image). Transposing the two halves is the
    /// classic 2D-RoPE bug and it is invisible on a square grid otherwise.
    #[test]
    fn axial_rope_splits_x_and_y_into_the_two_halves() {
        let (cos_t, sin_t) = axial_rope_tables(8, 4, 3, 10000.0);
        let half = 4usize;
        assert_eq!(cos_t.len(), 12 * half);
        // token (x=1,y=0) is index 1; token (x=1,y=1) is index 5.
        let (a, b) = (1usize, 5usize);
        // X half (cols 0..2) is the same for both: same x.
        for j in 0..2 {
            assert!((cos_t[a * half + j] - cos_t[b * half + j]).abs() < 1e-6, "x half moved with y");
        }
        // Y half (cols 2..4) must differ: different y.
        assert!((sin_t[a * half + 2] - sin_t[b * half + 2]).abs() > 1e-6, "y half did not move with y");
        // y = 0 leaves the Y half at angle 0.
        assert!(sin_t[a * half + 2].abs() < 1e-6);
    }

    #[test]
    fn sine_1d_concatenates_sin_then_cos() {
        let e = sine_1d(0.0, 8, 10000.0);
        assert_eq!(e.len(), 8);
        assert!(e[..4].iter().all(|v| v.abs() < 1e-6), "sin(0) half");
        assert!(e[4..].iter().all(|v| (*v - 1.0).abs() < 1e-6), "cos(0) half");
    }

    #[test]
    fn tile_repeats_periodically() {
        let src: Vec<f32> = (0..(2 * 2 * 2) as usize).map(|i| i as f32).collect();
        let t = tile_chw(&src, 2, 2, 2, 4, 4);
        // channel 0, row 0: [0,1,0,1]; row 2 repeats row 0.
        assert_eq!(&t[0..4], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&t[8..12], &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn a_padding_point_takes_not_a_point_and_no_positional_term() {
        let gauss = vec![0.25f32; 4]; // [2, 2]
        let pe = [vec![1.0; 4], vec![2.0; 4], vec![3.0; 4], vec![4.0; 4]];
        let nap = vec![9.0f32; 4];
        let e = embed_points(&gauss, 2, &pe, &nap, &[(10.0, 20.0)], &[1.0], (100, 100));
        assert_eq!(e.len(), 2 * 4);
        // row 1 is the appended padding point: exactly `not_a_point_embed`.
        assert_eq!(&e[4..8], &nap[..]);
        // row 0 carries the label-1 embedding on top of the positional term.
        assert!(e[..4].iter().all(|v| (*v - 9.0).abs() > 1e-6));
    }
}
