// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's 3-axis rotary position table builder.
//!
//! Ported directly from `MiniMaxH3RotaryPosEmbed.forward`
//! (`transformer_minimax_h3.py`): one shared `inv_freq` buffer of
//! `rope_freq_dim` frequencies, `inv_freq[k] = rope_theta^(-k/rope_freq_dim)`
//! for `k` in `0..rope_freq_dim`. For every packed-sequence row and every
//! axis `a` in `{t, h, w}`, `freqs[a] = position_ids[row][a] * inv_freq`
//! (`rope_freq_dim` values); the three axes concatenate in `(t, h, w)` order
//! to `3 * rope_freq_dim` angles per row.
//!
//! The reference then concatenates that block with ITSELF to
//! `2 * 3 * rope_freq_dim` before taking `cos`/`sin` (the `rotate_half`
//! convention: channel `m` and `m + half` share the same angle). This module
//! stops one step earlier and returns only the `half = 3 * rope_freq_dim`
//! wide table - exactly what `crates/kernels/wgsl/rope2d_partial.wgsl`
//! consumes (`cos_t`/`sin_t` are `[tmod, half]`, and the kernel itself derives
//! channel `m + half`'s rotation from row `m`'s own angle), so nothing here
//! duplicates the doubling the reference does purely to feed a
//! elementwise-over-the-full-`rotary_dim` kernel a same-shape `cos`/`sin`
//! pair.
//!
//! Swedish Embedded AB implements this rotary position embedding builder for
//! its clients. If your team needs expertise in porting diffusion
//! transformer positional encodings, you can procure our services by sending
//! an email to info@swedishembedded.com.

use crate::config::H3TransformerConfig;

/// Per-row `(cos, sin)` rotation tables, row-major `[seq_len, half]`.
#[derive(Clone, Debug)]
pub struct RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub seq_len: usize,
    /// `3 * rope_freq_dim` - half of [`H3TransformerConfig::rotary_dim`].
    pub half: usize,
}

/// Build the per-row `(cos, sin)` tables from `position_ids`, a row-major
/// `[seq_len, 3]` array of `(t, h, w)` rotary coordinates (the same tensor
/// the real transformer's `forward` takes as an explicit argument - this
/// port does not build it; see this crate's module doc for why).
pub fn build_tables(cfg: &H3TransformerConfig, position_ids: &[f32]) -> RopeTables {
    assert_eq!(position_ids.len() % 3, 0, "rope::build_tables: position_ids length {} is not a multiple of 3", position_ids.len());
    let seq_len = position_ids.len() / 3;
    let rfd = cfg.rope_freq_dim as usize;
    let half = 3 * rfd;

    // inv_freq[k] = rope_theta^(-k/rope_freq_dim), k in 0..rope_freq_dim.
    // Computed in f64 (as the reference computes it in fp32 arange/pow, but
    // f64 here loses nothing and keeps this host table build independent of
    // any device rounding).
    let inv_freq: Vec<f64> = (0..rfd).map(|k| (cfg.rope_theta as f64).powf(-(k as f64) / rfd as f64)).collect();

    let mut cos = vec![0f32; seq_len * half];
    let mut sin = vec![0f32; seq_len * half];
    for row in 0..seq_len {
        for (axis, &freq_axis) in [0usize, 1, 2].iter().enumerate() {
            let pos = position_ids[row * 3 + freq_axis] as f64;
            for (k, &freq) in inv_freq.iter().enumerate() {
                let angle = pos * freq;
                let col = axis * rfd + k;
                cos[row * half + col] = angle.cos() as f32;
                sin[row * half + col] = angle.sin() as f32;
            }
        }
    }
    RopeTables { cos, sin, seq_len, half }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_shape_matches_config() {
        let cfg = H3TransformerConfig::tiny();
        let position_ids = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let t = build_tables(&cfg, &position_ids);
        assert_eq!(t.seq_len, 2);
        assert_eq!(t.half, 3 * cfg.rope_freq_dim as usize);
        assert_eq!(t.cos.len(), t.seq_len * t.half);
        assert_eq!(t.sin.len(), t.seq_len * t.half);
    }

    /// Position (0,0,0) rotates every angle by zero - cos=1, sin=0 - the
    /// simplest oracle available without a Python reference.
    #[test]
    fn zero_position_is_the_identity_rotation() {
        let cfg = H3TransformerConfig::tiny();
        let t = build_tables(&cfg, &[0.0, 0.0, 0.0]);
        assert!(t.cos.iter().all(|&c| (c - 1.0).abs() < 1e-6), "cos must be 1 at position 0: {:?}", t.cos);
        assert!(t.sin.iter().all(|&s| s.abs() < 1e-6), "sin must be 0 at position 0: {:?}", t.sin);
    }

    /// Hand-computed against the reference formula at one nonzero point:
    /// inv_freq[0] = theta^0 = 1, so axis t's angle at k=0 is just the raw
    /// position value.
    #[test]
    fn first_frequency_is_the_raw_position() {
        let cfg = H3TransformerConfig::tiny();
        let t = build_tables(&cfg, &[2.5, 0.0, 0.0]);
        assert!((t.cos[0] - 2.5f32.cos()).abs() < 1e-6);
        assert!((t.sin[0] - 2.5f32.sin()).abs() < 1e-6);
    }
}
