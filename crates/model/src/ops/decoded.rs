// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training adapters over
// quantized base models without ever expanding the base to fp32, for its
// clients. If your team needs expertise in magnitude-normalised adapters
// (DoRA) over 4-bit and 8-bit bases then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Reductions over the DECODED values of a weight of any tier: the squared norm of each row and the
//! product of the rows with a small dense matrix. A magnitude-normalised adapter (DoRA) rescales every
//! output channel by the norm of the adapted weight row, so it reads these two quantities of its frozen
//! base; both are computed per bounded row tile of the shared tile decode ([`Ops::decode_tile`]), never
//! from a full-size fp32 copy of the weight, so the scratch is [`Ops::tile_budget_bytes`] whatever the
//! tier and the tensor size.

use gpu_core::{f, DeviceBuffer, Step};

use super::*;

impl Ops {
    /// Visit the dense f32 values of `w` as row tiles: `visit(steps, tile, row0, rows)` gets a
    /// `[rows, k]` f32 buffer holding rows `row0 .. row0 + rows`. A dense weight is one tile (its own
    /// buffer, no copy); every other tier is decoded into one scratch of at most the tile budget.
    fn for_each_dense_tile(&self, s: &mut Vec<Step>, w: &Weight, mut visit: impl FnMut(&mut Vec<Step>, &DeviceBuffer, u32, u32)) {
        match w {
            Weight::F32 { w: dense, n, .. } => visit(s, dense, 0, *n),
            _ => self.for_each_tile_f32(s, w, (0, w.n()), |s, tile, row0, rows, _| match tile {
                Weight::F32 { w: buf, .. } => visit(s, buf, row0, rows),
                _ => unreachable!("the f32 tile loop hands out dense tiles"),
            }),
        }
    }

    /// `out[n] = sum_k W[n, k]^2` for the `n` rows of `w`, of whatever tier, over the decoded values.
    pub fn row_sq_norms(&self, s: &mut Vec<Step>, w: &Weight, out: &DeviceBuffer) {
        let (n, k) = (w.n(), w.k());
        let row_dot = self.tiled_kernel("row_dot");
        let mut part: Option<DeviceBuffer> = None;
        self.for_each_dense_tile(s, w, |s, tile, row0, rows| {
            if rows == n {
                return s.push(self.gpu.step(row_dot, &[tile, tile, out], &[rows, k, 0, 0, f(1.0)], rows));
            }
            let part = part.get_or_insert_with(|| self.gpu.storage_uninit(rows as u64)).clone();
            s.push(self.gpu.step(row_dot, &[tile, tile, &part], &[rows, k, 0, 0, f(1.0)], rows));
            self.copy_cols(s, &part, out, 1, rows, rows, 0, n, row0);
        });
    }

    /// `out[n, r] = sum_k W[n, k] X[r, k]` (`W X^t`) over the decoded values of `w`, for a dense
    /// `x` of `r` rows of width `k`.
    pub fn project_rows(&self, s: &mut Vec<Step>, w: &Weight, x: &DeviceBuffer, r: u32, out: &DeviceBuffer) {
        let (n, k) = (w.n(), w.k());
        let rhs = Weight::dense_f32(x.clone(), r, k);
        let mut part: Option<DeviceBuffer> = None;
        self.for_each_dense_tile(s, w, |s, tile, row0, rows| {
            let act = self.act_f32(tile, 0, rows, k);
            if rows == n {
                return self.matmul(s, &rhs, &act, out, 0);
            }
            let first_rows = rows;
            let part = part.get_or_insert_with(|| self.gpu.storage_uninit(first_rows as u64 * r as u64)).clone();
            self.matmul(s, &rhs, &act, &part, 0);
            self.copy_cols(s, &part, out, 1, rows * r, rows * r, 0, n * r, row0 * r);
        });
    }
}
