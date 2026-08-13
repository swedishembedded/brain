// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL ViT vision encoder (built on the shared `model::vit` block builder).
//!
//! This module grows the encoder incrementally. Today: the positional pieces —
//! per-patch 2-D positions in spatial-merge-block order and the 2-D vision RoPE
//! tables that feed the existing `rope2d` kernel.

/// Per-patch `(h, w)` grid positions for a `hp × wp` patch grid, emitted in
/// **spatial-merge-block order** (each `m × m` block's patches contiguous), so the
/// PatchMerger's `reshape(-1, m²·C)` groups the right patches. Ports HF
/// `rot_pos_emb` / `get_vision_position_ids`: reshape `(hp/m, m, wp/m, m)`,
/// permute to `(hp/m, wp/m, m, m)`, flatten. `hp` and `wp` must be multiples of `m`.
pub fn vision_position_ids(hp: u32, wp: u32, merge: u32) -> Vec<(u32, u32)> {
    assert!(hp.is_multiple_of(merge) && wp.is_multiple_of(merge), "grid must be a multiple of merge size");
    let mut out = Vec::with_capacity((hp * wp) as usize);
    for bh in 0..hp / merge {
        for bw in 0..wp / merge {
            for ih in 0..merge {
                for iw in 0..merge {
                    out.push((bh * merge + ih, bw * merge + iw));
                }
            }
        }
    }
    out
}

/// Build the 2-D vision-RoPE `(cos, sin)` tables `[seq, head_dim/2]` for the given
/// per-patch positions, ready to upload as the two `rope2d` table buffers.
///
/// Qwen3-VL convention (`Qwen3VLVisionRotaryEmbedding`): rotary dim = head_dim/2;
/// `inv_freq[j] = theta^(-2j/rotary_dim)` for `j` in `0..rotary_dim/2`; each token's
/// angle vector is `concat(h·inv_freq, w·inv_freq)` (length rotary_dim = head_dim/2),
/// then duplicated and rotated half-split over head_dim. So the first quarter of the
/// table's channels are driven by the h-position and the second quarter by w.
/// theta = 10000 for the ViT.
pub fn vision_rope_tables(positions: &[(u32, u32)], head_dim: u32, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = (head_dim / 2) as usize; // = rotary_dim; table width
    let quarter = half / 2; // freqs per axis
    let seq = positions.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (t, &(h, w)) in positions.iter().enumerate() {
        for j in 0..quarter {
            let inv = theta.powf(-2.0 * j as f32 / half as f32);
            let ah = h as f32 * inv;
            let aw = w as f32 * inv;
            cos[t * half + j] = ah.cos();
            sin[t * half + j] = ah.sin();
            cos[t * half + quarter + j] = aw.cos();
            sin[t * half + quarter + j] = aw.sin();
        }
    }
    (cos, sin)
}

/// Bilinear-interpolation corner indices + weights for resampling the learned
/// `side × side` pos-embed table onto a `grid_h × grid_w` patch grid, in
/// spatial-merge-block order (matching [`vision_position_ids`]). Ports HF
/// `get_vision_bilinear_indices_and_weights` exactly: sample coordinates
/// `linspace(0, side-1, n)` (endpoints hit the table corners), take the floor/ceil
/// neighbours (ceil clamped to `side-1`), and the four standard bilinear weights.
/// Returns one `[c00, c01, c10, c11]` index quad (flattened `row*side+col` into the
/// table) and one `[w00, w01, w10, w11]` weight quad per patch token; the pos-embed
/// for a patch is `Σ table[idx[k]] * wts[k]`. Single image grid (temporal `t=1`);
/// callers repeat per frame.
pub fn pos_embed_bilinear(grid_h: u32, grid_w: u32, merge: u32, side: u32) -> (Vec<[u32; 4]>, Vec<[f32; 4]>) {
    assert!(side >= 1 && grid_h.is_multiple_of(merge) && grid_w.is_multiple_of(merge));
    // linspace(0, side-1, n)[i]; a single sample sits at 0 (torch semantics).
    let lin = |i: u32, n: u32| -> f32 {
        if n <= 1 {
            0.0
        } else {
            i as f32 * (side as f32 - 1.0) / (n as f32 - 1.0)
        }
    };
    let mut idx = Vec::with_capacity((grid_h * grid_w) as usize);
    let mut wts = Vec::with_capacity((grid_h * grid_w) as usize);
    for bh in 0..grid_h / merge {
        for bw in 0..grid_w / merge {
            for ih in 0..merge {
                for iw in 0..merge {
                    let (hi, wi) = (bh * merge + ih, bw * merge + iw);
                    let (hg, wg) = (lin(hi, grid_h), lin(wi, grid_w));
                    let (hf, wf) = (hg.floor() as u32, wg.floor() as u32);
                    let (hc, wc) = ((hf + 1).min(side - 1), (wf + 1).min(side - 1));
                    let (hfr, wfr) = (hg - hf as f32, wg - wf as f32);
                    idx.push([hf * side + wf, hf * side + wc, hc * side + wf, hc * side + wc]);
                    wts.push([(1.0 - hfr) * (1.0 - wfr), (1.0 - hfr) * wfr, hfr * (1.0 - wfr), hfr * wfr]);
                }
            }
        }
    }
    (idx, wts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_ids_single_block() {
        // 2×2 grid, merge 2 → one block, row-major within the block.
        assert_eq!(vision_position_ids(2, 2, 2), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn position_ids_two_blocks_wide() {
        // 2×4 grid, merge 2 → two horizontally-adjacent blocks, each contiguous.
        assert_eq!(
            vision_position_ids(2, 4, 2),
            vec![
                (0, 0), (0, 1), (1, 0), (1, 1), // block (bh=0,bw=0)
                (0, 2), (0, 3), (1, 2), (1, 3), // block (bh=0,bw=1)
            ]
        );
    }

    #[test]
    fn position_ids_cover_grid_once() {
        let ids = vision_position_ids(4, 6, 2);
        assert_eq!(ids.len(), 24);
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 24, "every (h,w) appears exactly once");
    }

    #[test]
    fn rope_tables_split_h_and_w() {
        // head_dim 64 → half 32, quarter 16. First quarter uses h, second uses w.
        let (hd, theta) = (64u32, 10000.0f32);
        let (cos, _) = vision_rope_tables(&[(2, 3)], hd, theta);
        let half = (hd / 2) as usize;
        let quarter = half / 2;
        let inv = |j: usize| theta.powf(-2.0 * j as f32 / half as f32);
        // channel j<quarter: angle = h·inv(j)
        assert!((cos[0] - (2.0 * inv(0)).cos()).abs() < 1e-6);
        assert!((cos[5] - (2.0 * inv(5)).cos()).abs() < 1e-6);
        // channel quarter+j: angle = w·inv(j)
        assert!((cos[quarter] - (3.0 * inv(0)).cos()).abs() < 1e-6);
        assert!((cos[quarter + 5] - (3.0 * inv(5)).cos()).abs() < 1e-6);
    }

    #[test]
    fn rope_tables_shape() {
        let (cos, sin) = vision_rope_tables(&[(0, 0), (1, 1), (2, 0)], 64, 10000.0);
        assert_eq!(cos.len(), 3 * 32);
        assert_eq!(sin.len(), 3 * 32);
    }

    #[test]
    fn bilinear_weights_are_a_partition_of_unity() {
        let (_, wts) = pos_embed_bilinear(6, 10, 2, 48);
        for q in &wts {
            let s: f32 = q.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "weights must sum to 1, got {s}");
        }
    }

    #[test]
    fn bilinear_identity_when_grid_equals_table() {
        // grid == side: each patch lands exactly on a table cell (frac 0), so the
        // first corner carries all the weight and indexes that cell directly.
        let side = 4;
        let (idx, wts) = pos_embed_bilinear(4, 4, 1, side);
        for (k, (id, w)) in idx.iter().zip(&wts).enumerate() {
            let (hi, wi) = (k as u32 / 4, k as u32 % 4); // merge=1 → row-major
            assert_eq!(id[0], hi * side + wi);
            assert!((w[0] - 1.0).abs() < 1e-6, "exact hit → weight [1,0,0,0]");
            assert!(w[1] + w[2] + w[3] < 1e-6);
        }
    }

    #[test]
    fn bilinear_midpoint_split() {
        // side 2, grid 3 (merge 1): linspace(0,1,3) = [0, 0.5, 1]. The middle
        // sample (col 1) is halfway between table cols 0 and 1.
        let (idx, wts) = pos_embed_bilinear(1, 3, 1, 2);
        // patch (0,1): h_floor 0, h_ceil clamps to 1 (weight 0); w_floor 0,
        // w_ceil 1, w_frac 0.5 → corners [0,1,2,3], weights [0.5,0.5,0,0].
        assert_eq!(idx[1], [0, 1, 2, 3]);
        assert!((wts[1][0] - 0.5).abs() < 1e-6 && (wts[1][1] - 0.5).abs() < 1e-6);
        assert!(wts[1][2] < 1e-6 && wts[1][3] < 1e-6);
    }
}
