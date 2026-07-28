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
    assert!(hp % merge == 0 && wp % merge == 0, "grid must be a multiple of merge size");
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
}
