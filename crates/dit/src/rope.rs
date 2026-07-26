// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-axis rotary position embedding (RoPE) for DiT token grids.
//!
//! A faithful port of Z-Image's `RopeEmbedder` (diffusers `transformer_z_image
//! .py`). Each attention head's `head_dim` is split across N positional **axes**
//! (Z-Image: `[32,48,48]` → a sequence axis + image height/width); a token's
//! position is an N-tuple of ids. Per axis `i` with `d = axes_dims[i]`:
//!
//! ```text
//! freqs[k] = theta^(-2k/d),  k in 0..d/2
//! angle    = pos * freqs[k]
//! (cos,sin) = (cos angle, sin angle)
//! ```
//!
//! The per-token table concatenates each axis's `d/2` `(cos,sin)` pairs in axis
//! order, giving `head_dim/2` pairs total. The rotation is **interleaved**
//! (adjacent pairs `(x[2j], x[2j+1])`, matching diffusers' `view_as_complex` on
//! `reshape(..., -1, 2)`), applied identically to every head. Angles are
//! computed in f64 (as diffusers does) then stored as f32.

/// Axis partition of a head's `head_dim` for multi-axis RoPE.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeConfig {
    /// Per-axis dims; must sum to `head_dim`. Z-Image: `[32,48,48]`.
    pub axes_dims: Vec<u32>,
    /// Per-axis max position (table length). Z-Image: `[1024,512,512]`.
    pub axes_lens: Vec<u32>,
    /// RoPE base. Z-Image: `256.0`.
    pub theta: f64,
}

impl RopeConfig {
    /// `head_dim = Σ axes_dims`.
    pub fn head_dim(&self) -> usize {
        self.axes_dims.iter().map(|&d| d as usize).sum()
    }
    /// `head_dim/2` — the number of `(cos,sin)` rotation pairs per token.
    pub fn half(&self) -> usize {
        self.head_dim() / 2
    }

    /// Per-axis `(cos,sin)` tables: axis `i` → row-major `[axes_lens[i] ·
    /// axes_dims[i]/2]`, indexed `[pos·(d/2) + k]`.
    pub fn precompute(&self) -> Vec<Vec<(f32, f32)>> {
        self.axes_dims
            .iter()
            .zip(&self.axes_lens)
            .map(|(&d, &len)| {
                let half = (d / 2) as usize;
                let mut tbl = Vec::with_capacity(len as usize * half);
                for pos in 0..len as usize {
                    for k in 0..half {
                        // freqs[k] = theta^(-2k/d); angle = pos * freqs[k].
                        let freq = self.theta.powf(-((2 * k) as f64) / d as f64);
                        let angle = pos as f64 * freq;
                        tbl_push(&mut tbl, angle);
                    }
                }
                tbl
            })
            .collect()
    }
}

fn tbl_push(tbl: &mut Vec<(f32, f32)>, angle: f64) {
    tbl.push((angle.cos() as f32, angle.sin() as f32));
}

/// Per-token `(cos,sin)` rotation tables, row-major `[seq · half]`.
#[derive(Clone, Debug)]
pub struct RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub seq: usize,
    pub half: usize,
}

/// Build the per-token rotation tables from position ids `[seq · n_axes]`
/// (row-major; `ids[t·n_axes + i]` is token `t`'s position along axis `i`).
/// Concatenates each axis's `d/2` pairs in axis order → `half` pairs per token.
pub fn tables_for_ids(cfg: &RopeConfig, ids: &[u32], n_axes: usize) -> RopeTables {
    assert_eq!(n_axes, cfg.axes_dims.len(), "n_axes must match axes_dims");
    assert_eq!(ids.len() % n_axes, 0, "ids length {} not a multiple of {n_axes}", ids.len());
    let seq = ids.len() / n_axes;
    let half = cfg.half();
    let axis_tbls = cfg.precompute();
    let axis_half: Vec<usize> = cfg.axes_dims.iter().map(|&d| (d / 2) as usize).collect();

    let mut cos = Vec::with_capacity(seq * half);
    let mut sin = Vec::with_capacity(seq * half);
    for t in 0..seq {
        for a in 0..n_axes {
            let pos = ids[t * n_axes + a] as usize;
            let ah = axis_half[a];
            assert!(pos < cfg.axes_lens[a] as usize, "axis {a} pos {pos} >= len {}", cfg.axes_lens[a]);
            let row = &axis_tbls[a][pos * ah..pos * ah + ah];
            for &(c, s) in row {
                cos.push(c);
                sin.push(s);
            }
        }
    }
    RopeTables { cos, sin, seq, half }
}
