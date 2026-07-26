// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-parallel Z-Image training across two P40s, memory-safe by design.
//!
//! The main-layer stack is cut at `cut`: stage 0 (card 0) runs the wrapper front
//! (timestep, embedders, refiners) + main layers `[0, cut)`; stage 1 (card 1) runs
//! main layers `[cut, end)` + the final layer + loss. Only the flat `[uni ‖ c]`
//! residual crosses between cards (forward), and `[d_uni ‖ dc]` back — host-staged
//! (no NVLink needed). Gradients are parity with the single-device path
//! (`tests/shard_parity.rs`, `tests/shard_2card.rs`).
//!
//! Memory: the model's weights/grads/optimiser live ONCE in host RAM (~120 GB for
//! the fp32 6B, well under 172 GB — a data-parallel replica would duplicate this
//! and OOM). Each card's [`BlockDev`] STREAMS its stage's layer slice one block at
//! a time, so the GPU holds ~one block + activations — never a whole stage. VRAM
//! per card stays far under 24 GB at any model size.
//!
//! Two schedules: [`ShardTrainer::grads`] (sequential — simple, correct, one card
//! at a time) and [`ShardTrainer::grads_microbatched`] (**GPipe** — the two cards
//! run concurrently, one thread each, connected by channels, so while card 0
//! forwards microbatch `k+1` card 1 forwards `k`; the efficient path).

use std::sync::mpsc::channel;

use crate::devgrad::BlockDev;
use crate::modelgrad::{Cfg, ModelGradsF32, ModelWeightsF32};
use crate::train::{assemble, back_grad_add, front_grad_add, main_grad_add, to64, uni_rope, Back, Batch, BackGrads, DeviceTrainer, Front, FrontGrads};
use crate::grad::GradsF32;

/// A two-card pipeline trainer for one Z-Image model (weights held by the caller
/// in host RAM). Stage 0 = front + main`[0,cut)` on card 0; stage 1 = main`[cut,end)`
/// + head on card 1.
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

    /// One full pipelined forward+backward, **sequential** (one card at a time).
    /// Returns `(loss, grads)`.
    pub fn grads(&self, w: &ModelWeightsF32, b: &Batch) -> (f64, ModelGradsF32) {
        let cut = self.cut;
        let (uni, front) = self.stage0.front_fwd(w, b);
        let (uni, in0) = self.stage0.main_fwd_ctx(&w.main[..cut], uni, front.c32(), front.uni_cos(), front.uni_sin());
        let (uni, in1) = self.stage1.main_fwd_ctx(&w.main[cut..], uni, front.c32(), front.uni_cos(), front.uni_sin());
        let (loss, dpred, back) = self.stage1.back_fwd(w, &uni, front.cvec(), b);
        let (d_uni, dc, bg) = self.stage1.back_bwd(w, &back, &dpred, front.cvec());
        let (d_uni, dc, mut mg1) = self.stage1.main_bwd_ctx(&w.main[cut..], &in1, front.c32(), front.uni_cos(), front.uni_sin(), &d_uni, dc);
        let (d_uni, dc, mut mg0) = self.stage0.main_bwd_ctx(&w.main[..cut], &in0, front.c32(), front.uni_cos(), front.uni_sin(), &d_uni, dc);
        let fg = self.stage0.front_bwd(w, b, &front, &d_uni, dc);
        mg0.append(&mut mg1);
        (loss, assemble(fg, bg, mg0))
    }

    /// **GPipe** micro-batched step: the two cards run concurrently (one thread
    /// each). Stage 0 forwards all microbatches (streaming card 0), sending each
    /// `[uni ‖ c]` boundary to stage 1, which forwards on card 1 and computes the
    /// loss; then the backward sweep runs in reverse, stage 1 sending `[d_uni ‖ dc]`
    /// back. While card 0 works on microbatch `k+1`, card 1 works on `k` — both
    /// cards busy. Gradients accumulate across microbatches. Returns
    /// `(summed_loss, summed_grads)`; average by `1/m` in the optimiser.
    pub fn grads_microbatched(&self, w: &ModelWeightsF32, mbs: &[Batch]) -> (f64, ModelGradsF32) {
        let (cut, m) = (self.cut, mbs.len());
        let (fwd_tx, fwd_rx) = channel::<(Vec<f32>, Vec<f32>)>(); // (uni, c)
        let (bwd_tx, bwd_rx) = channel::<(Vec<f32>, Vec<f64>)>(); // (d_uni, dc)
        let (s0, s1, wref) = (&self.stage0, &self.stage1, w);

        let (r0, r1) = std::thread::scope(|sc| {
            // ---- stage 0 (card 0): front + main[0,cut) ----
            let h0 = sc.spawn(move || {
                let mut fronts: Vec<Front> = Vec::with_capacity(m);
                let mut in0s: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
                for mb in mbs {
                    let (uni, front) = s0.front_fwd(wref, mb);
                    let (uni, in0) = s0.main_fwd_ctx(&wref.main[..cut], uni, front.c32(), front.uni_cos(), front.uni_sin());
                    fwd_tx.send((uni, front.c32().to_vec())).unwrap();
                    fronts.push(front);
                    in0s.push(in0);
                }
                let mut fg: Option<FrontGrads> = None;
                let mut mg0: Option<Vec<GradsF32>> = None;
                for k in 0..m {
                    let (d_uni, dc) = bwd_rx.recv().unwrap();
                    let fr = &fronts[k];
                    let (d_uni, dc, g0) = s0.main_bwd_ctx(&wref.main[..cut], &in0s[k], fr.c32(), fr.uni_cos(), fr.uni_sin(), &d_uni, dc);
                    let f = s0.front_bwd(wref, &mbs[k], fr, &d_uni, dc);
                    match &mut fg {
                        None => fg = Some(f),
                        Some(t) => front_grad_add(t, &f),
                    }
                    match &mut mg0 {
                        None => mg0 = Some(g0),
                        Some(t) => main_grad_add(t, &g0),
                    }
                }
                (fg.unwrap(), mg0.unwrap())
            });
            // ---- stage 1 (card 1): main[cut,end) + head ----
            let h1 = sc.spawn(move || {
                let mut in1s: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
                let mut backs: Vec<Back> = Vec::with_capacity(m);
                let mut dpreds: Vec<Vec<f64>> = Vec::with_capacity(m);
                let mut cvecs: Vec<Vec<f64>> = Vec::with_capacity(m);
                let mut ropes: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(m);
                let mut loss = 0.0;
                for mb in mbs {
                    let (uni, c32) = fwd_rx.recv().unwrap();
                    let (ucos, usin) = uni_rope(mb);
                    let (uni, in1) = s1.main_fwd_ctx(&wref.main[cut..], uni, &c32, &ucos, &usin);
                    let cvec = to64(&c32);
                    let (l, dpred, back) = s1.back_fwd(wref, &uni, &cvec, mb);
                    loss += l;
                    in1s.push(in1);
                    backs.push(back);
                    dpreds.push(dpred);
                    cvecs.push(cvec);
                    ropes.push((ucos, usin));
                }
                let mut bg: Option<BackGrads> = None;
                let mut mg1: Option<Vec<GradsF32>> = None;
                for k in 0..m {
                    let (d_uni, dc, b) = s1.back_bwd(wref, &backs[k], &dpreds[k], &cvecs[k]);
                    let (ucos, usin) = &ropes[k];
                    let c32: Vec<f32> = cvecs[k].iter().map(|&x| x as f32).collect();
                    let (d_uni, dc, g1) = s1.main_bwd_ctx(&wref.main[cut..], &in1s[k], &c32, ucos, usin, &d_uni, dc);
                    bwd_tx.send((d_uni, dc)).unwrap();
                    match &mut bg {
                        None => bg = Some(b),
                        Some(t) => back_grad_add(t, &b),
                    }
                    match &mut mg1 {
                        None => mg1 = Some(g1),
                        Some(t) => main_grad_add(t, &g1),
                    }
                }
                (loss, bg.unwrap(), mg1.unwrap())
            });
            (h0.join().unwrap(), h1.join().unwrap())
        });

        let (fg, mut mg0) = r0;
        let (loss, bg, mut mg1) = r1;
        mg0.append(&mut mg1);
        (loss, assemble(fg, bg, mg0))
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }
    pub fn cut(&self) -> usize {
        self.cut
    }
}
