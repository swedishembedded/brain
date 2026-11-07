// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Half-split RoPE for the sliding layers: a host-built `(cos, sin)` table and
//! the existing `rope2d` kernel.
//!
//! Swedish Embedded AB implements solutions for rotary position encodings in
//! long-context decoders for its clients. If your team needs expertise in
//! position encodings and their numerical parity with a reference then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! The reference rotates `q * cos + rotate_half(q) * sin` with
//! `rotate_half(x) = [-x2, x1]` over the two halves of the head, in f32, with
//! `inv_freq[i] = 1 / theta^(2i/dim)` and `angle = position * inv_freq`. That
//! is exactly `rope2d.wgsl`'s contract (channel `d` pairs with `d + half`,
//! every head reading the same `[t, half]` table), so no new kernel is needed.
//! Global layers apply no rotation and never build a table.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Row-major `[t, half]` cosine and sine tables, positions `0..t`.
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

/// The table for `head_dim`-wide heads at base `theta`, in f32 like the
/// reference (`model::rope_scaling::base_inv_freq` is the same reciprocal
/// power).
pub fn half_split_table(head_dim: usize, theta: f32, t: usize) -> RopeTable {
    let inv_freq = model::rope_scaling::base_inv_freq(head_dim as u32, theta);
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
    RopeTable { cos, sin, half }
}

/// A [`RopeTable`] resident on the device.
pub struct DeviceRope {
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    half: u32,
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
        gpu.step(kernel, &[buf, &self.cos, &self.sin], &[rows, heads, self.half, row_stride, 0, rows, f(1.0)], rows * heads * self.half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_the_identity_and_every_row_is_a_unit_rotation() {
        let t = half_split_table(16, 10_000.0, 5);
        assert_eq!(t.half, 8);
        assert!(t.cos[..8].iter().all(|c| *c == 1.0) && t.sin[..8].iter().all(|s| *s == 0.0));
        for (c, s) in t.cos.iter().zip(&t.sin) {
            assert!((c * c + s * s - 1.0).abs() < 1e-6);
        }
        // Frequency 0 rotates by one radian per position; the slowest by far less.
        assert!((t.sin[8] - 1.0f32.sin()).abs() < 1e-6);
        assert!(t.sin[8 + 7].abs() < 1e-2);
    }
}
