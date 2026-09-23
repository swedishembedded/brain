// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `pixels` arm's own trainable path: a small host-side linear
//! projector turning one grid cell's color into one row the encoder never
//! produced, and a plain Adam step to train it - the ONLY new trainable
//! weights this sample adds. Everything downstream of the projected row
//! (the head's cross-attention, its optimizer) is `crates/decide`'s existing
//! kept-features path, untouched.
//!
//! Deliberately host-side and deliberately tiny (19 inputs, one linear
//! layer): the question under test is whether a row from OUTSIDE the encoder
//! can drive the head at all, not how large a vision tower can be bolted on.
//! A GPU projector is the natural next step once this is proven, not before.

use crate::scene::{Rng, CELLS};

/// 3 (mean RGB) + 16 (one-hot cell identity, see the module doc on why the
/// identity is explicit rather than left for the projector to infer from a
/// continuous position).
pub const FEAT_DIM: usize = 3 + CELLS;

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    lr: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, lr }
    }

    fn step(&mut self, params: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= self.lr * mhat / (vhat.sqrt() + eps);
        }
    }
}

/// `Linear(FEAT_DIM, d_model)`, one row per grid cell, applied independently
/// per cell - not a shared spatial kernel, since sixteen cells is the whole
/// image.
pub struct Projector {
    d_model: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    adam_w: Adam,
    adam_b: Adam,
}

impl Projector {
    pub fn new(d_model: usize, seed: u64, lr: f32) -> Projector {
        let mut rng = Rng::new(seed);
        // Fan-in scaled uniform init - plain Xavier, nothing the encoder's
        // own init needs to match since this feeds a frozen encoder's
        // hidden-state SLOT, not its embedding table.
        let scale = (1.0 / FEAT_DIM as f32).sqrt();
        let w = (0..d_model * FEAT_DIM).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect();
        Projector {
            d_model,
            w,
            b: vec![0.0; d_model],
            gw: vec![0.0; d_model * FEAT_DIM],
            gb: vec![0.0; d_model],
            adam_w: Adam::new(d_model * FEAT_DIM, lr),
            adam_b: Adam::new(d_model, lr),
        }
    }

    fn feature(cell: usize, rgb: [f32; 3]) -> [f32; FEAT_DIM] {
        let mut f = [0.0f32; FEAT_DIM];
        f[0] = rgb[0];
        f[1] = rgb[1];
        f[2] = rgb[2];
        f[3 + cell] = 1.0;
        f
    }

    /// One row per cell, flattened `[CELLS * d_model]` - exactly the layout
    /// `Features::from_parts` expects for a contiguous run of state rows.
    pub fn forward_rows(&self, colors: &[[f32; 3]; CELLS]) -> Vec<f32> {
        let mut out = vec![0.0f32; CELLS * self.d_model];
        for cell in 0..CELLS {
            let feat = Self::feature(cell, colors[cell]);
            for h in 0..self.d_model {
                let mut acc = self.b[h];
                let row = h * FEAT_DIM;
                for (f, &x) in feat.iter().enumerate() {
                    acc += self.w[row + f] * x;
                }
                out[cell * self.d_model + h] = acc;
            }
        }
        out
    }

    /// Accumulate `dW`/`db` from the upstream gradient - `dL/d(forward_rows
    /// output)`, read back out of the encoder's own seed buffer after
    /// `Decide::accumulate_kept`'s reverse pass wrote it there. Call
    /// exactly once per `forward_rows` call it corresponds to; `step` clears
    /// the accumulator.
    pub fn accumulate(&mut self, colors: &[[f32; 3]; CELLS], d_rows: &[f32]) {
        assert_eq!(d_rows.len(), CELLS * self.d_model, "one gradient row per cell");
        for cell in 0..CELLS {
            let feat = Self::feature(cell, colors[cell]);
            for h in 0..self.d_model {
                let g = d_rows[cell * self.d_model + h];
                self.gb[h] += g;
                let row = h * FEAT_DIM;
                for (f, &x) in feat.iter().enumerate() {
                    self.gw[row + f] += g * x;
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.adam_w.step(&mut self.w, &self.gw);
        self.adam_b.step(&mut self.b, &self.gb);
        self.gw.iter_mut().for_each(|g| *g = 0.0);
        self.gb.iter_mut().for_each(|g| *g = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed-form check independent of `crates/decide` entirely: fit the
    /// projector against a FIXED target row for cell 0 and confirm the loss
    /// (squared error against that target) actually decreases. If the
    /// forward/accumulate/step arithmetic disagreed with each other this
    /// would diverge or stall, not just converge slowly.
    #[test]
    fn the_projector_reduces_a_fixed_target_mse_over_training() {
        let d_model = 8;
        let mut p = Projector::new(d_model, 7, 0.05);
        let colors = [[0.5f32; 3]; CELLS];
        let target: Vec<f32> = (0..d_model).map(|i| i as f32 * 0.1 - 0.3).collect();

        let mse = |rows: &[f32]| -> f32 {
            rows[..d_model].iter().zip(&target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / d_model as f32
        };

        let before = mse(&p.forward_rows(&colors));
        for _ in 0..200 {
            let rows = p.forward_rows(&colors);
            let mut d_rows = vec![0.0f32; CELLS * d_model];
            for h in 0..d_model {
                d_rows[h] = 2.0 * (rows[h] - target[h]) / d_model as f32;
            }
            p.accumulate(&colors, &d_rows);
            p.step();
        }
        let after = mse(&p.forward_rows(&colors));
        assert!(after < before * 0.05, "projector barely moved: {before} -> {after}");
    }
}
