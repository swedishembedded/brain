// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's RoPE: table construction (this module, host math) + rotation
//! (device, see [`apply_rope`]).
//!
//! ## Why this does NOT reuse `crates/dit::rope::RopeConfig`
//!
//! `dit::rope` (Z-Image/Wan's convention) builds ONE `(cos,sin)` table per
//! token that is shared, unmodified, across every attention head - each
//! axis contributes a FIXED sub-range of the table, the same for every head.
//! LTX's construction is different in two ways at once: the rotation is
//! **split/rotate-half** (GPT-NeoX style: pair `(j, j+head_dim/2)`), not
//! `dit::rope`'s interleaved `(2j, 2j+1)`; and, more fundamentally, the
//! per-token table is built ONCE at width `inner_dim/2` and then chunked
//! SEQUENTIALLY across heads (head `h` reads columns
//! `[h*head_dim/2, (h+1)*head_dim/2)` of that one table) - so different
//! heads see genuinely different frequency content, not the same table
//! replicated. Neither property fits `RopeConfig`'s per-axis-fixed-per-head
//! model, so this is a standalone construction, ported directly from
//! `ltx_core.model.transformer.rope` (`generate_freq_grid_pytorch`,
//! `generate_freqs`, `split_freqs_cis`, `precompute_freqs_cis`).
//!
//! ## Construction, precisely (verified against the real golden numbers)
//!
//! For `n_pos_dims` position axes (3 for video: frame, height, width) and
//! `inner_dim`:
//!
//! 1. `L = inner_dim / (2 * n_pos_dims)` **band** count (integer division -
//!    `inner_dim` need not be a multiple of `2*n_pos_dims`; the remainder is
//!    made up by front-padding, step 4).
//! 2. `indices[k] = theta^(k/(L-1)) * pi/2` for `k` in `0..L` (`L==1`:
//!    `indices[0] = pi/2`, matching `torch.linspace(0,1,1) == [0.0]`).
//! 3. Per token, per axis `a`, the **fractional position** is the MIDPOINT
//!    of that token's `[start,end)` patch bounds on axis `a`, divided by
//!    `max_pos[a]` (`use_middle_indices_grid=true`, the only mode this
//!    milestone implements). The per-token, per-band, per-axis angle is
//!    `indices[k] * (2*frac_pos[a] - 1)` - **band-major, axis-minor**:
//!    flattened as `[band0_axis0, band0_axis1, band0_axis2, band1_axis0,
//!    ...]`, width `L*n_pos_dims`.
//! 4. `cos`/`sin` of that flattened angle vector, then FRONT-padded with
//!    `inner_dim/2 - L*n_pos_dims` entries of `(cos=1, sin=0)` (identity
//!    rotation) to reach width `inner_dim/2` exactly.
//! 5. The resulting per-token `inner_dim/2`-wide vector is chunked
//!    SEQUENTIALLY into `num_heads` groups of `head_dim/2` each - head `h`
//!    gets columns `[h*head_dim/2, (h+1)*head_dim/2)`. This step is the one
//!    `dit::rope::RopeConfig` cannot express (see the module doc above).
//!
//! `positions` is read as `[n_pos_dims, T, 2]` row-major (axis outermost,
//! then token, then `[start,end)`) - exactly the shape the golden's
//! `positions` tensor and `Modality.positions[0]` (`B` sliced) carry.
//!
//! Computed in `f64` throughout (angle accumulation, `cos`/`sin`), cast to
//! `f32` only at the end - the reference's own `double_precision_rope=False`
//! path for this tiny config runs in `torch` `f32`, but `f64` host math
//! reproduces its output to within `~6e-8` max abs deviation (COSINE
//! `0.9999999999999996`, verified empirically against the real
//! `rope_cos`/`rope_sin` golden tensors before this module was written), so
//! there is no accuracy reason to match `f32` bit-for-bit and every other
//! reason (the roadmap's own "computed in fp64" note for the general LTX
//! case) to prefer the wider intermediate precision.

use std::f64::consts::FRAC_PI_2;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Per-token, per-head `(cos, sin)` rotation tables - `[heads, T, half]`
/// row-major, `half = head_dim/2`. This is the SAME layout the golden's
/// `rope_cos`/`rope_sin` tensors carry (post `swapaxes(1,2)`, `B` sliced).
#[derive(Clone, Debug)]
pub struct LtxRopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub heads: usize,
    pub t: usize,
    /// `head_dim / 2` - the per-head table width.
    pub half: usize,
}

impl LtxRopeTables {
    /// This head's `[T, half]` row-major `(cos, sin)` slice.
    pub fn head(&self, h: usize) -> (&[f32], &[f32]) {
        assert!(h < self.heads, "rope table has {} heads, asked for {h}", self.heads);
        let n = self.t * self.half;
        (&self.cos[h * n..h * n + n], &self.sin[h * n..h * n + n])
    }
}

/// Build [`LtxRopeTables`] for a `(frame, height, width)` video grid.
///
/// `positions`: `[3, t, 2]` row-major `[start, end)` patch bounds per axis
/// per token (see this module's doc, step 3). `max_pos`: the three
/// `positional_embedding_max_pos` normalizers.
pub fn ltx_rope_tables(inner_dim: u32, num_heads: u32, theta: f64, max_pos: [u32; 3], positions: &[f32], t: usize) -> LtxRopeTables {
    const N_POS: usize = 3;
    assert_eq!(positions.len(), N_POS * t * 2, "positions must be [3, t, 2] = {} values, got {}", N_POS * t * 2, positions.len());
    let total_half = (inner_dim / 2) as usize; // inner_dim/2, the full per-token table width
    let l = (inner_dim as usize) / (2 * N_POS); // band count
    assert!(l >= 1, "inner_dim {inner_dim} too small for {N_POS}-axis RoPE");
    let current_freqs = l * N_POS;
    assert!(current_freqs <= total_half, "band width {current_freqs} exceeds inner_dim/2 {total_half}");
    let pad = total_half - current_freqs;

    let heads = num_heads as usize;
    let half = total_half / heads;
    assert_eq!(half * heads, total_half, "inner_dim/2 ({total_half}) must be a whole multiple of num_heads ({heads})");

    // Step 2: theta^(k/(L-1)) * pi/2.
    let indices: Vec<f64> = (0..l)
        .map(|k| {
            let frac = if l > 1 { k as f64 / (l - 1) as f64 } else { 0.0 };
            theta.powf(frac) * FRAC_PI_2
        })
        .collect();

    // Steps 3-4: the full per-token [total_half]-wide (cos,sin) vectors.
    let mut cos_full = vec![0f32; t * total_half];
    let mut sin_full = vec![0f32; t * total_half];
    for ti in 0..t {
        for k in 0..pad {
            cos_full[ti * total_half + k] = 1.0;
            sin_full[ti * total_half + k] = 0.0;
        }
        for k in 0..l {
            for (a, &mp) in max_pos.iter().enumerate() {
                let start = positions[(a * t + ti) * 2] as f64;
                let end = positions[(a * t + ti) * 2 + 1] as f64;
                let mid = (start + end) / 2.0;
                let frac_pos = mid / mp as f64;
                let angle = indices[k] * (2.0 * frac_pos - 1.0);
                let idx = pad + k * N_POS + a;
                cos_full[ti * total_half + idx] = angle.cos() as f32;
                sin_full[ti * total_half + idx] = angle.sin() as f32;
            }
        }
    }

    // Step 5: chunk sequentially into [heads, T, half].
    let mut cos = vec![0f32; heads * t * half];
    let mut sin = vec![0f32; heads * t * half];
    for h in 0..heads {
        for ti in 0..t {
            for d in 0..half {
                cos[(h * t + ti) * half + d] = cos_full[ti * total_half + h * half + d];
                sin[(h * t + ti) * half + d] = sin_full[ti * total_half + h * half + d];
            }
        }
    }
    LtxRopeTables { cos, sin, heads, t, half }
}

/// Apply one head's rotation to `buf` **in place**, via `rope2d.wgsl`
/// (`kernels::ROPE2D`).
///
/// ## Which kernel, and why
///
/// `crates/kernels/wgsl/rope_neox.wgsl` uses the SAME split/rotate-half math
/// LTX needs (`y1=x1*cos-x2*sin; y2=x2*cos+x1*sin`), which made it the
/// natural first guess - but it computes its own angle ANALYTICALLY from a
/// single integer position and a scalar `theta` (`angle = t *
/// theta^(-2j/head_dim)`), with no table input at all. LTX's per-token angle
/// comes from the band/axis construction above, which is not of that
/// closed form (front-padding, band-major-axis-minor layout, per-head
/// sequential chunking), so `rope_neox` cannot serve it - **this guess is
/// REFUTED**. `rope2d.wgsl` (`kernels::ROPE2D`) implements the exact same
/// rotate-half math but reads the angle from a host-precomputed `[tmod,
/// half]` `cos`/`sin` table instead, which is exactly what an LTX table
/// needs.
///
/// The one contract mismatch: `rope2d`'s table is shared across every head
/// in one dispatch (`Params.heads` heads all read the same `[tmod, half]`
/// table), but LTX's heads read DIFFERENT sub-tables (see this module's
/// doc, step 5). The fix is not a new kernel - it is calling `rope2d` once
/// PER HEAD (`heads=1` in the dispatch), each with that head's own `[T,
/// half]` table (see [`LtxRopeTables::head`]) and `off = h*head_dim` selecting
/// that head's channel range within the `[T, inner_dim]` buffer. `sign=1.0`
/// (forward rotation); `tmod=T` (an exact per-token table, no frame-repeat).
pub fn apply_rope_step(gpu: &Gpu, kernel: usize, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, t: u32, head_dim: u32, row_stride: u32, off: u32) -> Step {
    let half = head_dim / 2;
    gpu.step(kernel, &[buf, cos, sin], &[t, 1, half, row_stride, off, t, f(1.0)], t * half)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cos^2 + sin^2 == 1` everywhere - the same structural invariant the
    /// golden dumper self-validates with (a genuine `(cos(theta),
    /// sin(theta))` table, not corrupted data).
    #[test]
    fn tables_are_unit_rotations() {
        let positions: Vec<f32> = {
            // A tiny 2x2x2 grid's [start,end) bounds, matching the golden's
            // own construction (integer grid coords, end = start+1).
            let mut v = vec![0f32; 3 * 8 * 2];
            let mut idx = 0;
            let mut coords = Vec::new();
            for f in 0..2 {
                for h in 0..2 {
                    for w in 0..2 {
                        coords.push((f, h, w));
                    }
                }
            }
            for (axis, get) in [
                (0usize, (|c: &(i32, i32, i32)| c.0) as fn(&(i32, i32, i32)) -> i32),
                (1usize, |c: &(i32, i32, i32)| c.1),
                (2usize, |c: &(i32, i32, i32)| c.2),
            ] {
                for (t, c) in coords.iter().enumerate() {
                    let start = get(c) as f32;
                    v[(axis * 8 + t) * 2] = start;
                    v[(axis * 8 + t) * 2 + 1] = start + 1.0;
                }
                let _ = idx;
                idx += 1;
            }
            v
        };
        let r = ltx_rope_tables(64, 4, 10000.0, [20, 2048, 2048], &positions, 8);
        assert_eq!(r.cos.len(), 4 * 8 * 8);
        for (c, s) in r.cos.iter().zip(&r.sin) {
            let dev = (*c as f64 * *c as f64 + *s as f64 * *s as f64 - 1.0).abs();
            assert!(dev < 1e-5, "cos^2+sin^2 deviates by {dev}");
        }
    }
}
