// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder's image memory: one row per valid encoder cell (with a gated
//! sinusoidal position code) followed by one row per proposal slot.
//!
//! A slot row projects the head feature at the box centre together with a
//! sine/cosine code of the box, adds a projected mean of the encoder cells
//! under the box, normalises, and adds the scaled embedding of the slot's
//! reference token so the decoder can name the slot.

use super::ops::{self, Linear};
use super::weights::Weights;
use crate::config::{MirageConfig, BOX_ENCODING_DIM};

/// A proposal slot: its box and the reference token that names it.
#[derive(Debug, Clone, Copy)]
pub struct Slot {
    /// `[x0, y0, x1, y1]`, canvas-normalised.
    pub bbox: [f32; 4],
    /// Embedding row of the reference token.
    pub token: usize,
}

/// Builds decoder memory.
pub struct MemoryBuilder {
    pub(crate) dim: usize,
    pub(crate) grid: usize,
    pub(crate) side: usize,
    pub(crate) cell_px: f32,
    pub(crate) pe_gate: f32,
    pub(crate) slot_proj: Linear,
    pub(crate) slot_pool: Linear,
    pub(crate) slot_norm: Vec<f32>,
    pub(crate) embedding: Vec<f32>,}

/// Sine then cosine of `pos * exp(-ln(1e4) / (dim/2 - 1) * i)` for `i < dim/2`.
pub(crate) fn sinusoid(pos: f64, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = -(1e4f64.ln()) / (half as f64 - 1.0);
    let angles: Vec<f64> = (0..half).map(|i| pos * (step * i as f64).exp()).collect();
    angles.iter().map(|a| a.sin() as f32).chain(angles.iter().map(|a| a.cos() as f32)).collect()
}

/// Sine and cosine of the four rounded box coordinates at eight octave frequencies, `[BOX_ENCODING_DIM]`.
pub(crate) fn box_code(b: [f32; 4]) -> Vec<f32> {
    let mut code = Vec::with_capacity(BOX_ENCODING_DIM);
    for coord in b {
        let angles: Vec<f64> = (0..BOX_ENCODING_DIM / 8).map(|k| f64::from(coord) * std::f64::consts::PI * 2f64.powi(k as i32)).collect();
        code.extend(angles.iter().map(|a| a.sin() as f32).chain(angles.iter().map(|a| a.cos() as f32)));
    }
    code
}

/// The cells a slot pools: the valid cells of a `g x g` grid whose centre lies in the box (inclusive), else the
/// box-centre cell.
pub(crate) fn pooled_cells(g: usize, cell_px: f32, valid: &[bool], b: [f32; 4]) -> Vec<usize> {
    let canvas = (g as f32) * cell_px;
    let px = b.map(|v| v * canvas);
    let mut picked: Vec<usize> = (0..g * g)
        .filter(|&c| {
            let (cx, cy) = ((c % g) as f32 * cell_px + cell_px / 2.0, (c / g) as f32 * cell_px + cell_px / 2.0);
            valid[c] && px[0] <= cx && cx <= px[2] && px[1] <= cy && cy <= px[3]
        })
        .collect();
    if picked.is_empty() {
        let cell = |lo: f32, hi: f32| ((g as f32 * (lo + hi) / 2.0) as usize).min(g - 1);
        picked.push(cell(b[1], b[3]) * g + cell(b[0], b[2]));
    }
    picked
}

impl MemoryBuilder {
    /// Prepare from the checkpoint tensors.
    pub fn new(cfg: &MirageConfig, w: &Weights) -> MemoryBuilder {
        MemoryBuilder {
            dim: cfg.dim,
            grid: cfg.grid(),
            side: cfg.canvas / cfg.stride,
            cell_px: (cfg.canvas / cfg.grid()) as f32,
            pe_gate: w.get("pe_gate")[0],
            slot_proj: w.linear("slot_proj/kernel"),
            slot_pool: w.linear("slot_pool/kernel"),
            slot_norm: w.vec("slot_norm/scale"),
            embedding: w.vec("embedding/embedding"),
        }
    }

    /// The gated position code of every encoder cell, `[grid², dim]`.
    pub fn position_rows(&self) -> Vec<f32> {
        let (d, g) = (self.dim, self.grid);
        let half = d / 2;
        let gate = ops::sigmoid(self.pe_gate);
        (0..g * g)
            .flat_map(|cell| [sinusoid((cell / g) as f64, half), sinusoid((cell % g) as f64, half)].concat().into_iter().map(move |p| gate * p))
            .collect()
    }

    /// Memory rows for the valid cells of `encoder` (`[grid², dim]`) and the
    /// given slots; `head_in` is the dense head's input map.
    pub fn build(&self, encoder: &[f32], valid: &[bool], head_in: &[f32], slots: &[Slot]) -> Vec<f32> {
        let (d, g) = (self.dim, self.grid);
        let pos = self.position_rows();
        let mut rows = Vec::new();
        for cell in (0..g * g).filter(|&c| valid[c]) {
            rows.extend(encoder[cell * d..(cell + 1) * d].iter().zip(&pos[cell * d..(cell + 1) * d]).map(|(e, p)| e + p));
        }
        if slots.is_empty() {
            return rows;
        }
        let (proj_in, pooled) = self.slot_inputs(encoder, valid, head_in, slots);
        let mut u = self.slot_proj.apply(&proj_in, slots.len());
        ops::add_inplace(&mut u, &self.slot_pool.apply(&pooled, slots.len()));
        let mut m = ops::zc_norm(&u, d, &self.slot_norm);
        for (row, e) in m.chunks_exact_mut(d).zip(self.slot_embedding(slots).chunks_exact(d)) {
            row.iter_mut().zip(e).for_each(|(a, e)| *a += e);
        }
        rows.extend(m);
        rows
    }

    /// Scaled embeddings of the slots' reference tokens, `[slots, dim]`.
    pub fn slot_embedding(&self, slots: &[Slot]) -> Vec<f32> {
        let (d, root) = (self.dim, (self.dim as f32).sqrt());
        slots.iter().flat_map(|s| self.embedding[s.token * d..(s.token + 1) * d].iter().map(move |e| root * e)).collect()
    }

    /// The two inputs of the slot projections: the head feature at the box
    /// centre with the box code (`[slots, head_dim + 64]`) and the pooled
    /// encoder features (`[slots, dim]`).
    pub fn slot_inputs(&self, encoder: &[f32], valid: &[bool], head_in: &[f32], slots: &[Slot]) -> (Vec<f32>, Vec<f32>) {
        let d = self.dim;
        let head_dim = self.slot_proj.inn - BOX_ENCODING_DIM;
        let mut proj_in = Vec::with_capacity(slots.len() * self.slot_proj.inn);
        let mut pooled = Vec::with_capacity(slots.len() * d);
        for s in slots {
            let b = s.bbox.map(|v| (v.clamp(0.0, 1.0) * 1000.0).round() * 0.001);
            let cell = |lo: f32, hi: f32| ((self.side as f32 * (lo + hi) / 2.0) as usize).min(self.side - 1);
            let (cy, cx) = (cell(b[1], b[3]), cell(b[0], b[2]));
            proj_in.extend_from_slice(&head_in[(cy * self.side + cx) * head_dim..][..head_dim]);
            proj_in.extend(box_code(b));
            pooled.extend(self.pool_cells(encoder, valid, b));
        }
        (proj_in, pooled)
    }

    /// Mean encoder feature of the pooled cells of a box.
    fn pool_cells(&self, encoder: &[f32], valid: &[bool], b: [f32; 4]) -> Vec<f32> {
        let d = self.dim;
        let picked = pooled_cells(self.grid, self.cell_px, valid, b);
        let mut mean = vec![0f32; d];
        for &c in &picked {
            mean.iter_mut().zip(&encoder[c * d..(c + 1) * d]).for_each(|(a, e)| *a += e);
        }
        mean.iter_mut().for_each(|v| *v /= picked.len() as f32);
        mean
    }
}
