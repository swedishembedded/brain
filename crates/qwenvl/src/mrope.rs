// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Interleaved multi-axis RoPE (M-RoPE) for Qwen3-VL — a pure host table builder.
//!
//! Qwen3-VL's only positional novelty is that each rotary channel is driven by
//! one of three position axes (T, H, W) in an **interleaved** layout, rather than
//! the single sequential position of ordinary RoPE. Because the rotation itself
//! is the standard half-split `rotate_half`, this needs **no new device kernel**:
//! the host builds the per-token `(cos, sin)` tables with the right axis feeding
//! each channel, and the existing `rope2d.wgsl` (table-driven, half-split, pairs
//! `(d, d+half)`) applies them — exactly as `crates/dit` does for Z-Image, but
//! with the interleaved stride-3 slot→axis map instead of contiguous blocks.
//!
//! Layout (matching HF `apply_interleaved_mrope`, `mrope_section = [T,H,W]`, sum
//! = head_dim/2): start every channel on axis T, then for axis H overwrite
//! channels `1, 4, 7, …` and for axis W `2, 5, 8, …`, each for its section count.
//! Net: channel `d` is owned by axis `d % 3` until the H/W sections are exhausted,
//! after which the tail belongs to T. For 4B `[24,20,20]`: T = {0,3,…,57} ∪
//! {60,61,62,63} (24), H = {1,4,…,58} (20), W = {2,5,…,59} (20).

/// Per-channel axis owner (0 = T, 1 = H, 2 = W) over the `half = head_dim/2`
/// rotary channels, from the interleaved `mrope_section` layout.
pub fn axis_map(mrope_section: [u32; 3], half: usize) -> Vec<usize> {
    let mut m = vec![0usize; half]; // default: temporal
    for axis in 1..3usize {
        for i in 0..mrope_section[axis] as usize {
            let slot = axis + 3 * i;
            if slot < half {
                m[slot] = axis;
            }
        }
    }
    m
}

/// Build the interleaved-M-RoPE `(cos, sin)` tables for a sequence of 3-axis
/// position ids. `positions[t] = [t_pos, h_pos, w_pos]`; returns row-major
/// `[seq, half]` cos and sin (`half = head_dim/2`) ready to upload as the two
/// `rope2d` table buffers. `inv_freq[d] = theta^(-2d/head_dim)`, matching HF.
pub fn mrope_tables(positions: &[[u32; 3]], mrope_section: [u32; 3], head_dim: u32, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = (head_dim / 2) as usize;
    let amap = axis_map(mrope_section, half);
    let seq = positions.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (t, p) in positions.iter().enumerate() {
        for d in 0..half {
            let inv_freq = theta.powf(-2.0 * d as f32 / head_dim as f32);
            let angle = p[amap[d]] as f32 * inv_freq;
            cos[t * half + d] = angle.cos();
            sin[t * half + d] = angle.sin();
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_map_interleaved_4b() {
        let m = axis_map([24, 20, 20], 64);
        // Interleaved head: T,H,W,T,H,W…
        assert_eq!(&m[..6], &[0, 1, 2, 0, 1, 2]);
        // H section ends at channel 1+3·19 = 58; W at 2+3·19 = 59.
        assert_eq!(m[58], 1);
        assert_eq!(m[59], 2);
        // Tail [60,64) reverts to T (H/W sections exhausted).
        assert_eq!(&m[60..64], &[0, 0, 0, 0]);
        // Per-axis counts equal the section sizes.
        let count = |a: usize| m.iter().filter(|&&x| x == a).count();
        assert_eq!((count(0), count(1), count(2)), (24, 20, 20));
    }

    #[test]
    fn diagonal_positions_collapse_to_plain_rope() {
        // When all three axes share a position (text tokens), M-RoPE must equal
        // ordinary RoPE at that position regardless of the axis map.
        let (hd, theta) = (128u32, 5_000_000.0f32);
        let (cos, sin) = mrope_tables(&[[7, 7, 7]], [24, 20, 20], hd, theta);
        let half = (hd / 2) as usize;
        for d in 0..half {
            let inv = theta.powf(-2.0 * d as f32 / hd as f32);
            let ang = 7.0 * inv;
            assert!((cos[d] - ang.cos()).abs() < 1e-6);
            assert!((sin[d] - ang.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn image_positions_route_by_axis() {
        // Distinct axis positions: channel 0 uses T, 1 uses H, 2 uses W.
        let (hd, theta) = (128u32, 5_000_000.0f32);
        let (cos, _) = mrope_tables(&[[10, 20, 30]], [24, 20, 20], hd, theta);
        let inv = |d: u32| theta.powf(-2.0 * d as f32 / hd as f32);
        assert!((cos[0] - (10.0 * inv(0)).cos()).abs() < 1e-6); // T
        assert!((cos[1] - (20.0 * inv(1)).cos()).abs() < 1e-6); // H
        assert!((cos[2] - (30.0 * inv(2)).cos()).abs() < 1e-6); // W
        assert!((cos[3] - (10.0 * inv(3)).cos()).abs() < 1e-6); // back to T
    }

    #[test]
    fn table_shape() {
        let (cos, sin) = mrope_tables(&[[0, 0, 0], [1, 2, 3], [4, 5, 6]], [24, 20, 20], 128, 5e6);
        assert_eq!(cos.len(), 3 * 64);
        assert_eq!(sin.len(), 3 * 64);
    }
}
