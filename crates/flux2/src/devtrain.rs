// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) whole-model LoRA training step for FLUX.2 Klein - the device
//! counterpart of [`crate::modelgrad`], and what makes a real fine-tune
//! reachable at all.
//!
//! The split follows `s3dit::train`: the expensive block stack runs on the GPU
//! through the persistent [`crate::devgrad::BlockDev`] engine (a forward sweep
//! that saves each block's *input* slab on the device, then a reverse sweep
//! that recomputes each block's forward and backpropagates through it), while
//! the thin conditioning front - timestep sinusoid → `time_in` MLP → the three
//! global modulation linears + the final adaLN - stays on the host, where it is
//! four mat-VECs at `m = 1`.
//!
//! Two things are on the device here that `s3dit::train` keeps on the host, and
//! both are forced by FLUX.2's shapes rather than by preference:
//!
//! * **the embedders.** `txt_in` is `[hidden, 3·qwen_hidden]` over 512 text
//!   rows - tens of GFLOP, not a wrapper detail. They are frozen and their
//!   inputs are data, so they need no backward at all: a forward-only device
//!   pass with no gradient machinery attached.
//! * **the final layer.** Keeping it on the device means the only thing that
//!   crosses PCIe per step is the `[n_img, in_channels]` prediction and its
//!   gradient, instead of the `[n_img, hidden]` residual slab.
//!
//! **Only the adapter trains.** The frozen base is read, never differentiated:
//! no `dW` is ever formed (see [`crate::devgrad`] for the low-rank identity
//! that replaces it), no optimiser state exists for it, and the modulation /
//! embedder / head / QK-norm gradients this returns are there because they are
//! nearly free and because they widen the gate in `tests/device_train.rs` - the
//! LoRA step itself ignores them.
//!
//! Gated by `tests/device_train.rs` against the FD-gradchecked host reference.

use gpu_core::{DeviceBuffer, Gpu};

use crate::devgrad::{BlockDev, DoubleDev, SingleDev, N_SITES, SITE_FINAL, SITE_IMG1, SITE_IMG2, SITE_SGL, SITE_TXT1, SITE_TXT2};
use crate::grad::{linear, silu, Dims, Mod, ModGrad};
use crate::lora::LoraAdapter;
use crate::modelgrad::{timestep_embedding, Batch, Cfg, ModelWeights, TDIM};

/// Everything one device step produces. Only [`Self::lora`] feeds the
/// optimiser; the rest exists so the parity gate covers the whole chain rather
/// than only the parts the adapter happens to touch.
pub struct StepGrads {
    /// `(dA [r,in], dB [out,r])` per adapter pair, in [`LoraAdapter::pairs`] order.
    pub lora: Vec<(Vec<f32>, Vec<f32>)>,
    /// The six modulation sites' `(shift, scale, gate)` grads, accumulated over
    /// the whole block stack (the modulation is global).
    pub sites: Vec<ModGrad<f32>>,
    /// `(dnq, dnk)` per QK-RMSNorm site: each double block's image stream then
    /// its text stream, then each single block.
    pub qk: Vec<(Vec<f32>, Vec<f32>)>,
}

/// Persistent device trainer: owns the GPU engine(s), the frozen base weights
/// (device-resident) and the per-block activation slabs.
///
/// **Cards.** klein-4B's fp32 base fits one 24 GiB card; klein-9B's does not,
/// so the block stack can be **split across cards** ([`Self::new_multi`]).
/// The split is a clean cut in the stack - the residual crosses it as one
/// `[n, hidden]` slab forward and one backward, staged through the host - the
/// same boundary `s3dit::train::grads_pipelined` proves is complete. Nothing
/// else has to be shared: the modulation sites and the RoPE tables are
/// uploaded to every engine, and each engine reads back its own blocks' grads.
pub struct DeviceTrainer {
    /// One engine per card, in placement order. Block `i` runs on
    /// `engs[blk_eng[i]]`.
    engs: Vec<BlockDev>,
    blk_eng: Vec<usize>,
    cfg: Cfg,
    // Host-side frozen conditioning matrices - every one of these is used as a
    // single `m = 1` mat-vec per step.
    time_a: Vec<f32>,
    time_b: Vec<f32>,
    mod_img: Vec<f32>,
    mod_txt: Vec<f32>,
    mod_single: Vec<f32>,
    final_adaln: Vec<f32>,
    // Device-resident frozen wrapper linears (embedders on the first engine,
    // the head on the last - where their inputs already are).
    img_in: DeviceBuffer,
    txt_in: DeviceBuffer,
    final_w: DeviceBuffer,
    dbl: Vec<DoubleDev>,
    sgl: Vec<SingleDev>,
    // Per-step slabs: `xs[i]` is the input to block `i` and lives on that
    // block's engine; `xs[depth]` is the stack output on the last engine. Kept
    // on the device - they are what the backward recomputes from.
    xs: Vec<DeviceBuffer>,
    /// Per engine: the gradient ping-pong pair the reverse sweep swaps.
    dcur: Vec<DeviceBuffer>,
    dnext: Vec<DeviceBuffer>,
    tok: DeviceBuffer,
    ctx: DeviceBuffer,
    pred: DeviceBuffer,
    dpred: DeviceBuffer,
}

/// How many blocks each engine takes, balanced by **weight bytes** rather than
/// count: a double block holds two streams and is twice a single block's
/// parameters, so an even count would leave one card far heavier than the
/// other. Returns the engine index of every block, in stack order.
fn place_blocks(n_double: usize, n_single: usize, engines: usize) -> Vec<usize> {
    let depth = n_double + n_single;
    if engines <= 1 {
        return vec![0; depth];
    }
    // Relative weights: a double block is two streams, a single block one.
    let w: Vec<f64> = (0..depth).map(|i| if i < n_double { 2.0 } else { 1.0 }).collect();
    let total: f64 = w.iter().sum();
    let per = total / engines as f64;
    let mut out = Vec::with_capacity(depth);
    let (mut e, mut acc) = (0usize, 0.0f64);
    for (i, &wi) in w.iter().enumerate() {
        // Never start a new engine if the remaining blocks could not fill the
        // ones after it - every engine must get at least one block.
        let remaining = depth - i;
        if e + 1 < engines && acc + wi * 0.5 > per && remaining > engines - e - 1 {
            e += 1;
            acc = 0.0;
        }
        acc += wi;
        out.push(e);
    }
    out
}

impl DeviceTrainer {
    /// Build a trainer on a single fresh device. `w` is uploaded in full, so
    /// the caller should drop its host copy afterwards.
    pub fn new(cfg: Cfg, rank: usize, w: &ModelWeights<f32>) -> DeviceTrainer {
        DeviceTrainer::with_gpu(Gpu::new_wgpu(crate::devgrad::KERNELS), cfg, rank, w)
    }

    /// [`Self::new`] over an existing device (card selection, or sharing one
    /// device with the rest of a pipeline).
    pub fn with_gpu(gpu: Gpu, cfg: Cfg, rank: usize, w: &ModelWeights<f32>) -> DeviceTrainer {
        DeviceTrainer::over(vec![gpu], cfg, rank, w)
    }

    /// Build across `cards` physical GPUs through a SINGLE device enumeration
    /// (`Gpu::new_wgpu_multi`, the collision-free placement the inference
    /// sharding uses). This is what makes klein-9B trainable at all: its fp32
    /// frozen base is larger than one 24 GiB card.
    pub fn new_multi(cards: usize, cfg: Cfg, rank: usize, w: &ModelWeights<f32>) -> DeviceTrainer {
        assert!(cards >= 1, "need at least one card");
        if cards == 1 {
            // One card goes through the ordinary selection ladder, so
            // `--device` / `BRAIN_GPU_INDEX` still choose which card. The
            // multi-device enumeration deliberately ignores that ladder (it
            // matches cards by identity), which is right for a real split and
            // wrong for a single-card run.
            return DeviceTrainer::new(cfg, rank, w);
        }
        DeviceTrainer::over(Gpu::new_wgpu_multi(crate::devgrad::KERNELS, cards), cfg, rank, w)
    }

    fn over(gpus: Vec<Gpu>, cfg: Cfg, rank: usize, w: &ModelWeights<f32>) -> DeviceTrainer {
        let (d, mlp, cin) = (cfg.hidden, cfg.mlp, cfg.in_channels);
        let n = cfg.n();
        let engs: Vec<BlockDev> = gpus.into_iter().map(|g| BlockDev::from_gpu(g, n, d, cfg.n_heads, mlp, rank)).collect();
        let blk_eng = place_blocks(w.dbl.len(), w.sgl.len(), engs.len());
        let depth = blk_eng.len();
        assert_eq!(depth, w.dbl.len() + w.sgl.len(), "placement covers every block");
        let dbl: Vec<DoubleDev> = w
            .dbl
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let e = &engs[blk_eng[i]];
                DoubleDev {
                    img: e.stream(&b.img.wq, &b.img.wk, &b.img.wv, &b.img.wo, &b.img.w1, &b.img.w3, &b.img.w2, &b.img.nq, &b.img.nk),
                    txt: e.stream(&b.txt.wq, &b.txt.wk, &b.txt.wv, &b.txt.wo, &b.txt.w1, &b.txt.w3, &b.txt.w2, &b.txt.nq, &b.txt.nk),
                }
            })
            .collect();
        let nd = w.dbl.len();
        let sgl: Vec<SingleDev> = w
            .sgl
            .iter()
            .enumerate()
            .map(|(j, b)| engs[blk_eng[nd + j]].single(&b.wq, &b.wk, &b.wv, &b.w1, &b.w3, &b.wo_a, &b.wo_b, &b.nq, &b.nk))
            .collect();
        // `xs[i]` lives with block `i`; the stack output stays with the last.
        let xs: Vec<DeviceBuffer> = (0..=depth)
            .map(|i| {
                let e = &engs[blk_eng[i.min(depth - 1)]];
                e.slab(e.n_max())
            })
            .collect();
        let dcur: Vec<DeviceBuffer> = engs.iter().map(|e| e.slab(e.n_max())).collect();
        let dnext: Vec<DeviceBuffer> = engs.iter().map(|e| e.slab(e.n_max())).collect();
        let first = engs[0].gpu();
        let last = engs[blk_eng[depth - 1]].gpu();
        let img_in = first.storage_init("flux2 img_in", &w.img_in);
        let txt_in = first.storage_init("flux2 txt_in", &w.txt_in);
        let final_w = last.storage_init("flux2 final", &w.final_w);
        let tok = first.storage((cfg.n_img() * cin) as u64);
        let ctx = first.storage((cfg.txt_len * cfg.context_in_dim) as u64);
        let pred = last.storage((cfg.n_img() * cin) as u64);
        let dpred = last.storage((cfg.n_img() * cin) as u64);
        DeviceTrainer {
            engs,
            blk_eng,
            cfg,
            time_a: w.time_a.clone(),
            time_b: w.time_b.clone(),
            mod_img: w.mod_img.clone(),
            mod_txt: w.mod_txt.clone(),
            mod_single: w.mod_single.clone(),
            final_adaln: w.final_adaln.clone(),
            img_in,
            txt_in,
            final_w,
            dbl,
            sgl,
            xs,
            dcur,
            dnext,
            tok,
            ctx,
            pred,
            dpred,
        }
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }
    /// The first engine's device - what a single-card caller means by "the GPU".
    pub fn gpu(&self) -> &Gpu {
        self.engs[0].gpu()
    }
    /// How many cards the block stack is spread over.
    pub fn cards(&self) -> usize {
        self.engs.len()
    }

    /// Device bytes of the frozen base + adapter buffers, per engine - what a
    /// caller has to fit on each card before it can train at this size.
    pub fn weight_bytes_per_card(&self) -> Vec<u64> {
        let mut v = vec![0u64; self.engs.len()];
        let nd = self.dbl.len();
        for (i, b) in self.dbl.iter().enumerate() {
            v[self.blk_eng[i]] += b.img.bytes() + b.txt.bytes();
        }
        for (j, b) in self.sgl.iter().enumerate() {
            v[self.blk_eng[nd + j]] += b.bytes();
        }
        v
    }

    /// Total device bytes the frozen base and the adapter buffers occupy.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes_per_card().iter().sum()
    }

    /// The engine block `i` of the stack runs on.
    fn eng(&self, i: usize) -> &BlockDev {
        &self.engs[self.blk_eng[i]]
    }

    /// Stage one `[n, hidden]` slab from engine `from` to engine `to` through
    /// the host. This is the ONLY thing that crosses a card boundary - one
    /// slab forward and one backward per cut, per step.
    fn carry(&self, from: usize, src: &DeviceBuffer, to: usize, dst: &DeviceBuffer) {
        if from == to {
            return;
        }
        let v = self.engs[from].gpu().read(src, self.cfg.n() * self.cfg.hidden);
        self.engs[to].gpu().write_f32(dst, &v);
    }

    fn dims(&self) -> Dims {
        self.cfg.dims()
    }

    /// The conditioning front, on the host: `t → sinusoid → time_in → silu →
    /// the three modulation linears + the final adaLN`, sliced into the six
    /// `(shift, scale, gate)` sites in BFL's chunk order.
    fn sites(&self, t: f64) -> [Mod<f32>; N_SITES] {
        let d = self.cfg.hidden;
        let te = timestep_embedding::<f32>(t);
        let hpre = linear(&te, 1, TDIM, &self.time_a, d);
        let h: Vec<f32> = hpre.iter().map(|&v| silu(v)).collect();
        let vec_ = linear(&h, 1, d, &self.time_b, d);
        let sv: Vec<f32> = vec_.iter().map(|&v| silu(v)).collect();
        let m_img = linear(&sv, 1, d, &self.mod_img, 6 * d);
        let m_txt = linear(&sv, 1, d, &self.mod_txt, 6 * d);
        let m_sgl = linear(&sv, 1, d, &self.mod_single, 3 * d);
        let m_fin = linear(&sv, 1, d, &self.final_adaln, 2 * d);
        let chunk = |m: &[f32], c: usize| m[c * d..(c + 1) * d].to_vec();
        let site = |m: &[f32], c: usize| Mod { shift: chunk(m, 3 * c), scale: chunk(m, 3 * c + 1), gate: chunk(m, 3 * c + 2) };
        // Positional, in the order the engine indexes: the SITE_* constants say
        // which slot is which, and the asserts pin that this literal agrees.
        let out: [Mod<f32>; N_SITES] = [
            site(&m_img, 0),
            site(&m_img, 1),
            site(&m_txt, 0),
            site(&m_txt, 1),
            site(&m_sgl, 0),
            // final layer: shift then scale, no gate - the zero gate slot is
            // never read (the head has no gated residual).
            Mod { shift: chunk(&m_fin, 0), scale: chunk(&m_fin, 1), gate: vec![0.0; d] },
        ];
        debug_assert_eq!(SITE_IMG1, 0);
        debug_assert_eq!(SITE_IMG2, 1);
        debug_assert_eq!(SITE_TXT1, 2);
        debug_assert_eq!(SITE_TXT2, 3);
        debug_assert_eq!(SITE_SGL, 4);
        debug_assert_eq!(SITE_FINAL, 5);
        out
    }

    /// Push this step's adapter factors to the device (the base never moves).
    fn upload_adapter(&self, ad: &LoraAdapter) {
        let scale = ad.scale();
        let host = ad.pairs();
        let mut i = 0;
        for (bi, b) in self.dbl.iter().enumerate() {
            let e = self.eng(bi);
            for st in [&b.img, &b.txt] {
                for l in [&st.wq, &st.wk, &st.wv, &st.wo, &st.w1, &st.w3, &st.w2] {
                    e.upload_lora(l, &host[i].a, &host[i].b, scale);
                    i += 1;
                }
            }
        }
        let nd = self.dbl.len();
        for (j, sb) in self.sgl.iter().enumerate() {
            let e = self.eng(nd + j);
            for l in [&sb.wq, &sb.wk, &sb.wv, &sb.w1, &sb.w3, &sb.wo_a, &sb.wo_b] {
                e.upload_lora(l, &host[i].a, &host[i].b, scale);
                i += 1;
            }
        }
        assert_eq!(i, host.len(), "device pair count {i} != adapter pair count {}", host.len());
    }

    /// One full training evaluation on the device: forward, loss, backward.
    /// Returns `(loss, grads)`.
    pub fn grads(&self, ad: &LoraAdapter, b: &Batch<f32>) -> (f64, StepGrads) {
        let cfg = &self.cfg;
        let cin = cfg.in_channels;
        let ni = cfg.n_img();
        let dm = self.dims();
        let depth = self.blk_eng.len();
        let nd = self.dbl.len();
        let last = self.blk_eng[depth - 1];

        let sites = self.sites(b.t);
        for e in &self.engs {
            e.upload_mods(&sites);
            e.upload_rope(&b.cos, &b.sin);
        }
        self.engs[0].gpu().write_f32(&self.tok, &b.img);
        self.engs[0].gpu().write_f32(&self.ctx, &b.ctx);
        self.upload_adapter(ad);

        // ---- forward ----
        self.engs[0].embed(&self.txt_in, &self.ctx, cfg.context_in_dim, &self.img_in, &self.tok, cin, dm, &self.xs[0]);
        for i in 0..depth {
            let (ei, to) = (self.blk_eng[i], self.blk_eng[(i + 1).min(depth - 1)]);
            // Within a card the block writes STRAIGHT into the next block's
            // input slab: nothing crosses PCIe. Only a card boundary needs the
            // engine-local scratch and the one staged slab.
            let out = if to == ei { &self.xs[i + 1] } else { &self.dcur[ei] };
            let e = &self.engs[ei];
            if i < nd {
                e.double_forward(&self.dbl[i], dm, &self.xs[i], out);
            } else {
                e.single_forward(&self.sgl[i - nd], dm, &self.xs[i], out);
            }
            self.carry(ei, out, to, &self.xs[i + 1]);
        }
        self.engs[last].head_forward(&self.final_w, &self.xs[depth], dm, cin, &self.pred);

        // ---- loss (host: [n_img, in_channels] is the only slab that crosses) ----
        let pred = self.engs[last].gpu().read(&self.pred, ni * cin);
        let (loss, dpred) = crate::modelgrad::loss(&pred, &b.target);
        self.engs[last].gpu().write_f32(&self.dpred, &dpred);

        // ---- backward ----
        self.engs[last].head_backward(&self.final_w, &self.xs[depth], dm, cin, &self.dpred, &self.dcur[last]);
        // Ping-pong the incoming/outgoing grad slabs within a card; stage one
        // slab across when the stack crosses to the previous card.
        let (mut cur, mut alt) = (&self.dcur[last], &self.dnext[last]);
        let mut from = last;
        for i in (0..depth).rev() {
            let ei = self.blk_eng[i];
            if ei != from {
                self.carry(from, cur, ei, &self.dcur[ei]);
                cur = &self.dcur[ei];
                alt = &self.dnext[ei];
                from = ei;
            }
            let e = &self.engs[ei];
            if i < nd {
                e.double_backward(&self.dbl[i], dm, &self.xs[i], cur, alt);
            } else {
                e.single_backward(&self.sgl[i - nd], dm, &self.xs[i], cur, alt);
            }
            std::mem::swap(&mut cur, &mut alt);
        }

        // ---- read back ----
        let scale = ad.scale();
        let mut lora = Vec::new();
        let mut qk = Vec::new();
        for (i, blk) in self.dbl.iter().enumerate() {
            let e = self.eng(i);
            for st in [&blk.img, &blk.txt] {
                for l in [&st.wq, &st.wk, &st.wv, &st.wo, &st.w1, &st.w3, &st.w2] {
                    lora.push(e.lin_grads(l, scale));
                }
                qk.push(e.stream_norm_grads(st));
            }
        }
        for (j, sb) in self.sgl.iter().enumerate() {
            let e = self.eng(nd + j);
            for l in [&sb.wq, &sb.wk, &sb.wv, &sb.w1, &sb.w3, &sb.wo_a, &sb.wo_b] {
                lora.push(e.lin_grads(l, scale));
            }
            qk.push(e.single_norm_grads(sb));
        }
        // The modulation sites are global: every engine accumulated the
        // contribution of the blocks it ran, so the totals are their sum.
        let mut sites: Vec<ModGrad<f32>> = self.engs[0].mod_grads();
        for e in &self.engs[1..] {
            for (acc, g) in sites.iter_mut().zip(e.mod_grads()) {
                for (a, b2) in acc.shift.iter_mut().zip(&g.shift) {
                    *a += b2;
                }
                for (a, b2) in acc.scale.iter_mut().zip(&g.scale) {
                    *a += b2;
                }
                for (a, b2) in acc.gate.iter_mut().zip(&g.gate) {
                    *a += b2;
                }
            }
        }
        (loss, StepGrads { lora, sites, qk })
    }

    /// One optimiser step: device gradients → Adam on the adapter's `A,B`.
    pub fn step(&self, ad: &mut LoraAdapter, b: &Batch<f32>, lr: f32) -> f64 {
        let (loss, g) = self.grads(ad, b);
        ad.step_projected(&g.lora, lr);
        loss
    }
}
