// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-parallel Z-Image training across two P40s, memory-safe by design.
//!
//! The main-layer stack is cut at `cut`: stage 0 (card 0) runs the wrapper front
//! (timestep, embedders, refiners) + main layers `[0, cut)`; stage 1 (card 1) runs
//! main layers `[cut, end)`; the final layer + loss run on the host. Only the flat
//! `[uni ‖ c]` residual crosses between cards (forward), and `[d_uni ‖ dc]` back —
//! host-staged (no NVLink needed). The gradients are bit-parity with the
//! single-device path (`grads_pipelined`, validated in `tests/shard_parity.rs`).
//!
//! Memory: the model's weights/grads/optimiser live ONCE in host RAM (~120 GB for
//! the fp32 6B, well under 172 GB). Each card's [`BlockDev`] STREAMS its stage's
//! layer slice one block at a time, so the GPU holds ~one block (hundreds of MB) +
//! activations — never a whole stage, never the whole model. VRAM per card stays
//! far under 24 GB regardless of model size; there is no configuration that OOMs
//! either the cards or RAM. Both engines come from a single `new_wgpu_multi`
//! enumeration, so they map to distinct cards without the collision the naive
//! two-`new_wgpu` path hits.

use crate::devgrad::BlockDev;
use crate::modelgrad::{Cfg, ModelGrads, ModelWeights};
use crate::train::{assemble, Batch, DeviceTrainer};

/// A two-card pipeline trainer for one Z-Image model (weights held by the caller
/// in host RAM). Stage 0 = front + main`[0,cut)` on card 0; stage 1 = main`[cut,end)`
/// on card 1; head (final layer + loss) on the host.
pub struct ShardTrainer {
    cfg: Cfg,
    cut: usize,
    stage0: DeviceTrainer, // card 0
    stage1: DeviceTrainer, // card 1
}

impl ShardTrainer {
    /// Build the pipeline over 2 GPUs (single enumeration). `cut` splits the
    /// `n_layers` main layers between the cards.
    pub fn new(cfg: Cfg, cut: usize) -> ShardTrainer {
        assert!(cut >= 1 && cut < cfg.n_layers, "cut must be in 1..n_layers");
        let mut engs = BlockDev::new_multi(2, cfg.ntot(), cfg.dim, cfg.nh);
        let e1 = engs.pop().unwrap();
        let e0 = engs.pop().unwrap();
        ShardTrainer { cfg, cut, stage0: DeviceTrainer::with_engine(cfg, e0), stage1: DeviceTrainer::with_engine(cfg, e1) }
    }

    /// One full pipelined forward+backward. Returns `(loss, grads)` — identical to
    /// the single-device `DeviceTrainer::grads` up to the f32 boundary rounding.
    pub fn grads(&self, w: &ModelWeights, b: &Batch) -> (f64, ModelGrads) {
        let cut = self.cut;

        // ---- forward ----
        // stage 0 (card 0): front + main[0, cut)
        let (uni, front) = self.stage0.front_fwd(w, b);
        let (uni, in0) = self.stage0.main_fwd_ctx(&w.main[..cut], uni, &front.c32(), &front.uni_cos(), &front.uni_sin());
        // boundary [uni ‖ c] host-staged to card 1 (uni + c32 are already f32).
        // stage 1 (card 1): main[cut, end)
        let (uni, in1) = self.stage1.main_fwd_ctx(&w.main[cut..], uni, &front.c32(), &front.uni_cos(), &front.uni_sin());
        // head (host): final layer + loss
        let (loss, dpred, back) = self.stage0.back_fwd(w, &uni, &front, b);

        // ---- backward ----
        let (d_uni, dc, bg) = self.stage0.back_bwd(w, &back, &dpred, &front);
        // stage 1 (card 1)
        let (d_uni, dc, mut mg1) = self.stage1.main_bwd_ctx(&w.main[cut..], &in1, &front.c32(), &front.uni_cos(), &front.uni_sin(), &d_uni, dc);
        // boundary [d_uni ‖ dc] host-staged back to card 0
        // stage 0 (card 0)
        let (d_uni, dc, mut mg0) = self.stage0.main_bwd_ctx(&w.main[..cut], &in0, &front.c32(), &front.uni_cos(), &front.uni_sin(), &d_uni, dc);
        let fg = self.stage0.front_bwd(w, b, &front, &d_uni, dc);

        mg0.append(&mut mg1);
        (loss, assemble(fg, bg, mg0))
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }
}
