// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The table-driven half-split RoPE that several decoders share: a host-built
//! `(cos, sin)` table, its device upload, and the `rope2d` dispatch that
//! rotates a buffer in place.
//!
//! Swedish Embedded AB implements solutions for rotary position encodings in
//! long-context decoders for its clients. If your team needs expertise in
//! position encodings and their numerical parity with a reference then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! `rope2d.wgsl` pairs channel `d` with `d + half` of each head
//! (`y1 = x1 cos - x2 sin`, `y2 = x2 cos + x1 sin`) and reads the angle from a
//! `[rows, half]` table shared by every head of one dispatch, which is the
//! rotation `rotate_half` implements in the Hugging Face decoders. A model
//! chooses its table (the plain reciprocal-power one below, a partially
//! rotating one, a per-head one built from a position grid); the dispatch and
//! the upload are the same for all of them.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Row-major `[t, half]` cosine and sine tables for positions `0..t`.
#[derive(Clone, Debug)]
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
    pub t: usize,
}

/// The plain table for `head_dim`-wide heads at base `theta`: `inv_freq[i] =
/// 1 / theta^(2i/head_dim)` and `angle = position * inv_freq`, all in f32 as
/// the reference decoders compute them (`rope_scaling::base_inv_freq` is the
/// same reciprocal power).
pub fn half_split_table(head_dim: usize, theta: f32, t: usize) -> RopeTable {
    let inv_freq = crate::rope_scaling::base_inv_freq(head_dim as u32, theta);
    let half = inv_freq.len();
    let mut cos = Vec::with_capacity(t * half);
    let mut sin = Vec::with_capacity(t * half);
    for pos in 0..t {
        for inv in &inv_freq {
            let angle = pos as f32 * inv;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    RopeTable { cos, sin, half, t }
}

/// A [`RopeTable`] resident on the device. `half` travels with the buffers so
/// the dispatch cannot be given a width the table was not built for.
pub struct DeviceRope {
    pub cos: DeviceBuffer,
    pub sin: DeviceBuffer,
    pub half: u32,
}

impl DeviceRope {
    pub fn upload(gpu: &Gpu, table: &RopeTable) -> DeviceRope {
        let cos = gpu.storage(table.cos.len() as u64);
        gpu.write_f32(&cos, &table.cos);
        let sin = gpu.storage(table.sin.len() as u64);
        gpu.write_f32(&sin, &table.sin);
        DeviceRope { cos, sin, half: table.half as u32 }
    }

    /// Rotate `heads` heads of every row of `buf` (`[rows, row_stride]`) in
    /// place. `kernel` is the index of `rope2d` in the caller's kernel table.
    pub fn apply(&self, gpu: &Gpu, kernel: usize, buf: &DeviceBuffer, rows: u32, heads: u32, row_stride: u32) -> Step {
        rope2d_step(gpu, kernel, buf, &self.cos, &self.sin, rows, heads, self.half, row_stride, 0)
    }
}

/// One `rope2d` dispatch: rotate `heads` heads, `2 * half` channels each, of
/// the `rows` rows of `buf` (row stride `row_stride`, region starting at
/// channel `off`), row `r` reading table row `r`.
#[allow(clippy::too_many_arguments)]
pub fn rope2d_step(gpu: &Gpu, kernel: usize, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, heads: u32, half: u32, row_stride: u32, off: u32) -> Step {
    gpu.step(kernel, &[buf, cos, sin], &[rows, heads, half, row_stride, off, rows, f(1.0)], rows * heads * half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_the_identity_and_every_row_is_a_unit_rotation() {
        let t = half_split_table(16, 10_000.0, 5);
        assert_eq!((t.half, t.t), (8, 5));
        assert!(t.cos[..8].iter().all(|c| *c == 1.0) && t.sin[..8].iter().all(|s| *s == 0.0));
        for (c, s) in t.cos.iter().zip(&t.sin) {
            assert!((c * c + s * s - 1.0).abs() < 1e-6);
        }
        // Frequency 0 rotates by one radian per position; the slowest by far less.
        assert!((t.sin[8] - 1.0f32.sin()).abs() < 1e-6);
        assert!(t.sin[8 + 7].abs() < 1e-2);
    }
}
