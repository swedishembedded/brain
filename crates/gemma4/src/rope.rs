// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gemma-4's RoPE: table construction (this module, host math) + rotation
//! (device, `rope2d` - see this module's doc for why `rope2d`, NOT
//! `rope2d_partial`, is the correct kernel for BOTH layer types, including
//! the `full_attention`/`proportional` one, and why the first guess here was
//! wrong).
//!
//! ## Why this DOES reuse the existing rotate-half kernel, unlike LTX's own
//!
//! `crates/ltxv/src/rope.rs` needed a standalone construction because LTX's
//! per-token angle comes from a band/axis/fractional-position grid, and its
//! heads read DIFFERENT sub-tables. Gemma-4's RoPE is the OPPOSITE case:
//! `apply_rotary_pos_emb`/`rotate_half` in `gemma3n.modeling_gemma3n` is the
//! textbook GPT-NeoX split-half rotation (`y1=x1*cos-x2*sin; y2=x2*cos+
//! x1*sin`, pairing channel `d` with `d + head_dim/2`) driven by a PURELY
//! ANALYTIC angle (`inv_freq[i] = theta^(-2i/dim)`), and EVERY head within
//! one layer reads the exact SAME table. That is `crates/kernels/wgsl/
//! rope2d.wgsl`'s contract, native `heads` param included. It is also NOT
//! `crates/dit::rope::RopeConfig`'s contract (interleaved-pair rotation, a
//! genuinely different topology).
//!
//! ## `rope2d_partial` was the natural first guess for `full_attention` - and
//! ## is REFUTED: it uses a DIFFERENT pairing convention than Gemma-4's own
//!
//! `full_attention` layers use `rope_type="proportional"`: only `rope_angles
//! = floor(partial_rotary_factor * global_head_dim / 2)` of the `inv_freq`
//! entries are nonzero, the rest are exactly ZERO (`nope_angles`, real HF
//! source). `rope2d_partial.wgsl` ("rotate only the first `2*half` channels
//! of each head, the rest pass through") sounds like exactly this - but it
//! pairs channel `d` with `d + half` where `half = rot_dim/2` (built for
//! Qwen3.5/Moondream's OWN partial-rotary convention: the rotated sub-block
//! is treated as its own self-contained rotate-half pair, i.e. `d` pairs
//! with `d + rope_angles`, not `d + head_dim/2`). Gemma-4's ACTUAL
//! `rotate_half` NEVER changes its pairing distance for a partial rotation -
//! it *always* pairs `d` with `d + head_dim/2` (the FULL head's own
//! half-point), and gets its "partial" behavior purely from `inv_freq`
//! having ZERO entries past `rope_angles` (an algebraic identity:
//! `cos=1,sin=0` at those positions), not from a shorter pairing distance.
//! Dispatching `rope2d_partial` with `half=rope_angles` therefore rotates the
//! WRONG pairs entirely (e.g. real config `rope_angles=2`: it pairs `(0,2)`
//! and `(1,3)`, while the reference pairs `(0, head_dim/2)` and `(1,
//! head_dim/2+1)`) - caught empirically, not by inspection: the tiny-config
//! parity test's `full_attention` self-attention tap came back at cosine
//! 0.77 with this kernel, while the `sliding_attention` tap (unaffected,
//! `rope2d` with no partial anything) was already at cosine 1.0 - isolating
//! the defect to exactly this one call before any code was reread.
//!
//! **The fix needs no new kernel.** Since Gemma-4's "partial" rotation is
//! really "rotate the WHOLE head at `rope2d`'s own pairing distance, with a
//! table that happens to have zero entries past `rope_angles`", this
//! module's [`full_table`] builds a table at width `head_dim/2` (matching
//! [`sliding_table`]'s own width convention exactly) - the first
//! `rope_angles` columns are the genuine rotated frequencies, the rest are
//! `(cos=1, sin=0)` identity rows - and [`apply_rope_full`] (`rope2d`) is
//! reused UNCHANGED for both layer types. `rope2d_partial` is not used
//! anywhere in this crate.
//!
//! Computed in `f64` (angle accumulation, `cos`/`sin`), cast to `f32` only at
//! the end - the same precedent `ltxv::rope`'s own doc records (the
//! reference's own `torch` `f32` forward is reproduced to well under 1e-6 max
//! abs deviation this way, see the self-validated `cos^2+sin^2==1` invariant
//! in `tools/goldens/gemma4_dump_reference.py`).

/// One layer type's per-token `(cos, sin)` rotation table, row-major `[T,
/// half]`, `half = head_dim/2` for BOTH layer types (see this module's doc
/// for why `full_attention`'s table is NOT narrowed to `rope_angles` - the
/// zero-padded remainder is genuine table content, not an omitted range).
#[derive(Clone, Debug)]
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
    pub t: usize,
}

/// `sliding_attention`'s table: `rope_type="default"`, every one of
/// `head_dim`'s `head_dim/2` frequency slots rotates -
/// `Gemma3RotaryEmbedding.compute_default_rope_parameters`:
/// `inv_freq[i] = theta^(-2i/head_dim)`.
pub fn sliding_table(head_dim: u32, theta: f64, t: usize) -> RopeTable {
    let half = (head_dim / 2) as usize;
    let mut cos = vec![0f32; t * half];
    let mut sin = vec![0f32; t * half];
    for pos in 0..t {
        for k in 0..half {
            let inv_freq = theta.powf(-((2 * k) as f64) / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            cos[pos * half + k] = angle.cos() as f32;
            sin[pos * half + k] = angle.sin() as f32;
        }
    }
    RopeTable { cos, sin, half, t }
}

/// `full_attention`'s table: `rope_type="proportional"` -
/// `_compute_proportional_rope_parameters` in `transformers.
/// modeling_rope_utils`. `rope_angles = floor(partial_rotary_factor *
/// global_head_dim / 2)` columns carry the genuine rotated frequency (the
/// exponent is normalized by the FULL `global_head_dim`, not by
/// `2*rope_angles`); columns `[rope_angles, global_head_dim/2)` are
/// `(cos=1, sin=0)` identity rows - see this module's doc for why this
/// width (`global_head_dim/2`, matching [`sliding_table`]'s own convention),
/// not `rope_angles`, is the correct table shape for [`apply_rope_full`].
pub fn full_table(global_head_dim: u32, theta: f64, partial_rotary_factor: f64, t: usize) -> RopeTable {
    let half = (global_head_dim / 2) as usize;
    let rope_angles = ((partial_rotary_factor * global_head_dim as f64) / 2.0).floor() as usize;
    assert!(rope_angles <= half, "rope_angles {rope_angles} exceeds head_dim/2 {half}");
    let mut cos = vec![1f32; t * half];
    let mut sin = vec![0f32; t * half];
    for pos in 0..t {
        for k in 0..rope_angles {
            let inv_freq = theta.powf(-((2 * k) as f64) / global_head_dim as f64);
            let angle = pos as f64 * inv_freq;
            cos[pos * half + k] = angle.cos() as f32;
            sin[pos * half + k] = angle.sin() as f32;
        }
    }
    RopeTable { cos, sin, half, t }
}

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// A [`RopeTable`], uploaded to the device - `half` travels alongside the
/// buffers (rather than being recomputed at the call site) so there is
/// exactly ONE place (`sliding_table`/`full_table`) that computes it.
pub struct DeviceRope {
    pub cos: DeviceBuffer,
    pub sin: DeviceBuffer,
    pub half: u32,
}

pub fn upload_rope(gpu: &Gpu, tbl: &RopeTable) -> DeviceRope {
    let cos = gpu.storage(tbl.cos.len() as u64);
    gpu.write_f32(&cos, &tbl.cos);
    let sin = gpu.storage(tbl.sin.len() as u64);
    gpu.write_f32(&sin, &tbl.sin);
    DeviceRope { cos, sin, half: tbl.half as u32 }
}

/// `rope2d.wgsl` (`kernels::ROPE2D`): the WHOLE `head_dim` of every head in
/// `[rows, heads*head_dim]`-shaped `buf` rotates (channels past
/// `2*half=head_dim` do not exist for this layout), all heads reading the
/// SAME `[t, half]` table in ONE dispatch (native `heads` param - no
/// per-head loop, unlike LTX's own usage of this kernel). Used for BOTH
/// layer types - see this module's doc for why `full_attention`'s "partial"
/// behavior lives entirely in the TABLE ([`full_table`]'s zero-padded
/// columns), not in a different kernel.
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_full(gpu: &Gpu, kernel: usize, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, heads: u32, half: u32, row_stride: u32) -> Step {
    gpu.step(kernel, &[buf, cos, sin], &[rows, heads, half, row_stride, 0, rows, f(1.0)], rows * heads * half)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cos^2+sin^2==1` - the same structural invariant the golden dumper
    /// self-validates with, on BOTH tables (the full/global table's
    /// zero-padded identity columns trivially satisfy this too: `cos=1,
    /// sin=0`).
    #[test]
    fn both_tables_are_unit_rotations() {
        for tbl in [sliding_table(8, 10_000.0, 8), full_table(16, 1_000_000.0, 0.25, 8)] {
            for (c, s) in tbl.cos.iter().zip(&tbl.sin) {
                let dev = (*c as f64 * *c as f64 + *s as f64 * *s as f64 - 1.0).abs();
                assert!(dev < 1e-5, "cos^2+sin^2 deviates by {dev}");
            }
        }
    }

    #[test]
    fn full_table_is_zero_padded_past_rope_angles_not_narrowed() {
        // partial_rotary_factor=0.25, global_head_dim=16 -> rope_angles =
        // floor(0.25*16/2) = 2, so columns [2, 8) must be the (cos=1, sin=0)
        // identity, NOT absent from the table (see this module's doc for why
        // `rope2d_partial`'s narrower-table convention is the wrong fit).
        // `t=2`: row 0 (position 0) has angle 0 everywhere regardless of
        // frequency, so the "genuine rotation" check reads row 1 (position 1).
        let tbl = full_table(16, 1_000_000.0, 0.25, 2);
        assert_eq!(tbl.half, 8); // == global_head_dim/2, not rope_angles
        let row1 = &tbl.sin[8..16];
        assert_ne!(row1[0], 0.0, "column 0 must be a genuine rotated frequency");
        assert_ne!(row1[1], 0.0, "column 1 must be a genuine rotated frequency");
        for (k, &r1) in row1.iter().enumerate().skip(2) {
            assert_eq!(tbl.cos[k], 1.0, "row 0 column {k} must be the identity (cos=1)");
            assert_eq!(tbl.sin[k], 0.0, "row 0 column {k} must be the identity (sin=0)");
            assert_eq!(r1, 0.0, "row 1 column {k} must be the identity (sin=0)");
        }
    }
}
