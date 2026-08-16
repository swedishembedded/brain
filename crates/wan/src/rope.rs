// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Three-axis rotary position embedding over the `(frame, height, width)`
//! token grid.
//!
//! Wan's `rope_apply` builds one `[1024, head_dim/2]` complex table per axis
//! and, for a token at grid position `(f, h, w)`, concatenates row `f` of the
//! frame table, row `h` of the height table and row `w` of the width table -
//! which is exactly [`dit::rope::RopeConfig`] at `n_axes = 3`. Nothing here is
//! Wan-specific except the axis split and the id order, so the tables and the
//! interleaved rotation itself are the shared ones.
//!
//! Two facts that are easy to get wrong and are pinned by tests below:
//!
//! * **The axis split biases toward time.** Upstream splits the `c = head_dim/2`
//!   complex pairs as `[c - 2*(c/3), c/3, c/3]`, so an indivisible `c` gives
//!   the frame axis the remainder rather than truncating. At `head_dim = 128`
//!   that is `[22, 21, 21]` pairs, i.e. `[44, 42, 42]` real components.
//! * **Each axis normalises by its OWN width.** `rope_params(1024, d_axis)`
//!   uses `theta^(-2k/d_axis)`, not `theta^(-2k/head_dim)`, so the three axes
//!   have different frequency ladders. A single shared ladder would look
//!   plausible and be wrong everywhere but the frame axis.

use dit::rope::{tables_for_ids, RopeConfig, RopeTables};

use crate::config::WanConfig;

/// Upstream's RoPE base (`rope_params(..., theta=10000)`).
pub const THETA: f64 = 10000.0;

/// Table length per axis (`rope_params(1024, ...)`). A grid extent past this is
/// a config error, not something to wrap: `dit::rope` asserts on it.
pub const MAX_SEQ_LEN: u32 = 1024;

/// The `RopeConfig` for a Wan variant: the three axis widths from
/// [`WanConfig::rope_axes_dims`], all at `MAX_SEQ_LEN`.
pub fn rope_config(cfg: &WanConfig) -> RopeConfig {
    RopeConfig {
        axes_dims: cfg.rope_axes_dims().iter().map(|&d| d as u32).collect(),
        axes_lens: vec![MAX_SEQ_LEN; 3],
        theta: THETA,
    }
}

/// Position ids `[tokens · 3]` for a `(f, h, w)` patch grid, in the token order
/// the patch embedding produces: `flatten(2)` over `[dim, f, h, w]` walks `w`
/// fastest, then `h`, then `f`.
pub fn grid_ids(f: u32, h: u32, w: u32) -> Vec<u32> {
    let mut ids = Vec::with_capacity((f * h * w) as usize * 3);
    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                ids.extend_from_slice(&[fi, hi, wi]);
            }
        }
    }
    ids
}

/// Per-token `(cos, sin)` rotation tables for a `(f, h, w)` grid, row-major
/// `[tokens · head_dim/2]` - the layout `rope_interleave_table` reads.
pub fn tables(cfg: &WanConfig, f: u32, h: u32, w: u32) -> RopeTables {
    tables_for_ids(&rope_config(cfg), &grid_ids(f, h, w), 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_widths_match_upstreams_split() {
        let c = WanConfig::t2v_1_3b();
        let r = rope_config(&c);
        assert_eq!(r.axes_dims, vec![44, 42, 42]);
        assert_eq!(r.head_dim(), 128);
        assert_eq!(r.half(), 64);
    }

    /// Token order is `w` fastest: a patch-embedding output is `[dim, f, h, w]`
    /// flattened over the last three axes, so a row-major `(f, h, w)` walk is
    /// the only order that lines the ids up with the tokens.
    #[test]
    fn ids_walk_width_fastest() {
        let ids = grid_ids(2, 2, 3);
        assert_eq!(&ids[..9], &[0, 0, 0, 0, 0, 1, 0, 0, 2]);
        assert_eq!(&ids[9..18], &[0, 1, 0, 0, 1, 1, 0, 1, 2]);
        assert_eq!(&ids[18..21], &[1, 0, 0]);
        assert_eq!(ids.len(), 2 * 2 * 3 * 3);
    }

    /// The three axes must NOT share one frequency ladder. Position 1 on the
    /// frame axis and position 1 on the width axis have different angles at the
    /// same within-axis index `k`, because the exponent's denominator is the
    /// axis width. Checked at `k = 1` of each axis's own block.
    #[test]
    fn each_axis_uses_its_own_width_in_the_exponent() {
        let cfg = WanConfig::t2v_1_3b();
        let t = tables(&cfg, 2, 2, 2);
        // Axis blocks inside a token's `half` row: frame [0,22), height
        // [22,43), width [43,64).
        let (fa, ha) = (0usize, 22usize);
        // Token 0 is (0,0,0) -> every angle 0. Token (1,0,0) is index h*w = 4.
        let frame1 = t.cos[4 * t.half + fa + 1];
        // Token (0,1,0) is index w = 2 -> its height-axis entry at k=1.
        let height1 = t.cos[2 * t.half + ha + 1];
        let angle_f = (1.0f64) * THETA.powf(-2.0 / 44.0);
        let angle_h = (1.0f64) * THETA.powf(-2.0 / 42.0);
        assert!((frame1 as f64 - angle_f.cos()).abs() < 1e-6, "frame axis ladder");
        assert!((height1 as f64 - angle_h.cos()).abs() < 1e-6, "height axis ladder");
        assert!((angle_f.cos() - angle_h.cos()).abs() > 1e-9, "the ladders must differ");
    }

    #[test]
    fn table_shape_is_one_row_per_token() {
        let cfg = WanConfig::t2v_1_3b();
        let t = tables(&cfg, 3, 4, 5);
        assert_eq!(t.seq, 60);
        assert_eq!(t.half, 64);
        assert_eq!(t.cos.len(), 60 * 64);
        assert_eq!(t.sin.len(), 60 * 64);
    }
}
