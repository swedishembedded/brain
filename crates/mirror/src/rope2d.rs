// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side normalized 2D-RoPE tables (ports `norm_rope.py` +
//! `visual_transformer.py` position semantics exactly).
//!
//! Per-frame token layout `[special×patch_start, patch(y,x)…]`; patch grid
//! positions are shifted +1 and specials sit at (0,0), so the sincos grid is
//! `(hp+1)×(wp+1)` with per-axis "separate" normalization:
//!   coord(v, n) = 2*((v+0.5)/n) - 1
//!   angles(tok) = concat(2π·cy/periods, 2π·cx/periods)   -> [half = head_dim/2]
//! The reference then duplicates angles to head_dim and rotates half-split
//! pairs (d, d+half) — pairs share angle index d, so `[td, half]` tables are
//! exact (see `rope2d.wgsl`).

/// Build `[td, half]` cos/sin tables for one frame's tokens.
pub fn rope_tables(
    periods: &[f32],
    hp: usize,
    wp: usize,
    patch_start: usize,
) -> (Vec<f32>, Vec<f32>) {
    let quarter = periods.len();
    let half = 2 * quarter;
    let (gh, gw) = (hp + 1, wp + 1);
    let td = patch_start + hp * wp;
    let mut cos = vec![0.0f32; td * half];
    let mut sin = vec![0.0f32; td * half];
    let coord = |v: usize, n: usize| 2.0 * ((v as f32 + 0.5) / n as f32) - 1.0;
    let mut write = |row: usize, gy: usize, gx: usize| {
        let cy = coord(gy, gh);
        let cx = coord(gx, gw);
        for k in 0..quarter {
            let ay = 2.0 * std::f32::consts::PI * cy / periods[k];
            let ax = 2.0 * std::f32::consts::PI * cx / periods[k];
            cos[row * half + k] = ay.cos();
            sin[row * half + k] = ay.sin();
            cos[row * half + quarter + k] = ax.cos();
            sin[row * half + quarter + k] = ax.sin();
        }
    };
    for t in 0..patch_start {
        write(t, 0, 0);
    }
    for y in 0..hp {
        for x in 0..wp {
            write(patch_start + y * wp + x, y + 1, x + 1);
        }
    }
    (cos, sin)
}
