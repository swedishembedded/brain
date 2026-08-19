// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) full-model training step for the Wan DiT.
//!
//! The expensive part - the block stack, which is essentially all of the
//! model's FLOPs - runs on the GPU through the persistent
//! [`crate::devgrad::BlockDev`] engine: a forward sweep that saves each block's
//! input, then a reverse backward sweep. The thin wrapper stays on the host and
//! calls the SAME helpers the reference does: `patch_embedding`, the text
//! embedding MLP, the whole timestep path, the modulated head, unpatchify and
//! the flow-matching loss. Those are a handful of small linears whose cost is a
//! rounding error next to one block.
//!
//! [`DeviceTrainer::grads`] is a drop-in replacement for
//! [`crate::modelgrad::grads::<f32>`]: same [`Batch`] in, same
//! `(loss, ModelGrads<f32>)` out, so [`crate::lora::LoraAdapter::step`] and
//! [`crate::finetune`] consume either without knowing which ran.
//!
//! [`DeviceTrainer::lora_grads`] is the same step specialised to a FROZEN base:
//! [`DeviceTrainer::begin_lora`] makes every block's base resident, and each
//! step then moves only the rank-sized `(A, B)` to the device and `(dA, dB)`
//! back, with the effective-weight fold and the gradient projection both
//! on-device (see [`crate::devgrad`]). It returns
//! [`crate::lora::LoraGrads`] instead of a full [`ModelGrads`] because the
//! frozen tensors' grads have no consumer.
//!
//! The three shared conditioning couplings are the same ones the host
//! reference is built around, and each is invisible in a forward: `e0` is one
//! vector every block reads, so the blocks' modulation grads are summed before
//! `time_projection`; the head reads `e` rather than `e0`, so `d e` has two
//! sources; and the embedded text context is one slab every block's
//! cross-attention reads, so the blocks' `dctx` are summed before
//! `text_embedding`'s backward.

use crate::devgrad::BlockDev;
use crate::grad::{affine, affine_bwd, dgelu, dsilu, gelu, layernorm, layernorm_bwd, linear, linear_bwd, silu, BlockGrads, Lin};
use crate::lora::{LoraAdapter, LoraGrads};
use crate::modelgrad::{loss, patchify, timestep_embedding, unpatchify, unpatchify_bwd, Batch, Cfg, ModelGrads, ModelWeights};

/// Everything the host computes BEFORE the block stack: the timestep path, the
/// text-embedding MLP and the patch embedding. Both device step paths share it
/// (only their block loop and what they read back differ), and the backward
/// tail reads the saved intermediates.
struct Front {
    e: Vec<f32>,
    e0: Vec<f32>,
    ctxe: Vec<f32>,
    x: Vec<f32>,
    h0pre: Vec<f32>,
    h0: Vec<f32>,
    eact: Vec<f32>,
    th: Vec<f32>,
    thg: Vec<f32>,
    flat: Vec<f32>,
    te: Vec<f32>,
}

fn front(cfg: &Cfg, w: &ModelWeights<f32>, b: &Batch<f32>) -> Front {
    let (dim, tl, td) = (cfg.dim, cfg.text_len, cfg.text_dim);
    let n = cfg.n_tokens();
    let (_, ph, pw) = cfg.patch;
    let te = timestep_embedding::<f32>(b.t, cfg.freq_dim);
    let h0pre = linear(&te, 1, cfg.freq_dim, &w.time0.w, &w.time0.b, dim);
    let h0: Vec<f32> = h0pre.iter().map(|&v| silu(v)).collect();
    let e = linear(&h0, 1, dim, &w.time2.w, &w.time2.b, dim);
    let eact: Vec<f32> = e.iter().map(|&v| silu(v)).collect();
    let e0 = linear(&eact, 1, dim, &w.time_proj.w, &w.time_proj.b, 6 * dim);

    let th = linear(&b.ctx, tl, td, &w.text0.w, &w.text0.b, dim);
    let thg: Vec<f32> = th.iter().map(|&v| gelu(v)).collect();
    let ctxe = linear(&thg, tl, dim, &w.text2.w, &w.text2.b, dim);

    let (lf, lh, lw) = cfg.latent;
    let flat = patchify(&b.latent, cfg.in_channels, lf, lh, lw, ph, pw);
    let x = linear(&flat, n, cfg.patch_dim(), &w.patch_embed.w, &w.patch_embed.b, dim);
    Front { e, e0, ctxe, x, h0pre, h0, eact, th, thg, flat, te }
}

/// The modulated head, the flow-matching loss and the head's backward - the
/// host work BETWEEN the two device sweeps. `dx` is what the reverse sweep
/// starts from.
struct Head {
    loss: f64,
    dx: Vec<f32>,
    de: Vec<f32>,
    head_mod: Vec<f32>,
    g_head: Lin<f32>,
}

fn head_pass(cfg: &Cfg, w: &ModelWeights<f32>, b: &Batch<f32>, x: &[f32], e: &[f32]) -> Head {
    let dim = cfg.dim;
    let (gf, gh, gw) = cfg.grid();
    let n = cfg.n_tokens();
    let (_, ph, pw) = cfg.patch;
    let shift_h: Vec<f32> = w.head_mod[..dim].iter().zip(e).map(|(&a, &c)| a + c).collect();
    let gamma_h: Vec<f32> = w.head_mod[dim..].iter().zip(e).map(|(&a, &c)| 1.0 + a + c).collect();
    let (xhat_h, inv_h) = layernorm(x, n, dim, cfg.eps);
    let nh = affine(&xhat_h, &gamma_h, &shift_h, n, dim);
    let rows = linear(&nh, n, dim, &w.head.w, &w.head.b, cfg.head_dim_out());
    let pred = unpatchify(&rows, cfg.out_channels, gf, gh, gw, ph, pw);
    let (l, dpred) = loss(&pred, &b.target);

    let drows = unpatchify_bwd(&dpred, cfg.out_channels, gf, gh, gw, ph, pw);
    let (dnh, g_head) = linear_bwd(&nh, n, dim, &w.head.w, cfg.head_dim_out(), &drows);
    let mut dgamma_h = vec![0f32; dim];
    let mut dshift_h = vec![0f32; dim];
    let dxhat_h = affine_bwd(&xhat_h, &gamma_h, n, dim, &dnh, &mut dgamma_h, &mut dshift_h);
    let mut head_mod = vec![0f32; 2 * dim];
    head_mod[..dim].copy_from_slice(&dshift_h);
    head_mod[dim..].copy_from_slice(&dgamma_h);
    // `e` enters BOTH head-modulation halves additively.
    let de: Vec<f32> = dshift_h.iter().zip(&dgamma_h).map(|(&a, &c)| a + c).collect();
    let dx = layernorm_bwd(&xhat_h, &inv_h, n, dim, &dxhat_h);
    Head { loss: l, dx, de, head_mod, g_head }
}

/// Persistent device trainer: one GPU engine sized for this config's block,
/// reused by every layer of the stack and across steps.
pub struct DeviceTrainer {
    eng: BlockDev,
    cfg: Cfg,
}

impl DeviceTrainer {
    /// Build on brain's default device.
    pub fn new(cfg: &Cfg) -> DeviceTrainer {
        DeviceTrainer::on_device(cfg, None)
    }

    /// Build on a named device (`"cpu"`, `"gpu"`, or `None` for the default).
    pub fn on_device(cfg: &Cfg, device: Option<&str>) -> DeviceTrainer {
        let d = cfg.dims();
        let mut eng = BlockDev::on_device(d, d.t, device);
        eng.reserve_slots(cfg.n_layers, resident_budget(&eng));
        DeviceTrainer { eng, cfg: *cfg }
    }

    /// Build over an already-open engine.
    pub fn with_engine(cfg: &Cfg, eng: BlockDev) -> DeviceTrainer {
        DeviceTrainer { eng, cfg: *cfg }
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    pub fn engine(&self) -> &BlockDev {
        &self.eng
    }

    /// `true` when the block stack runs on a real accelerator.
    pub fn is_accelerated(&self) -> bool {
        self.eng.is_accelerated()
    }

    /// Full forward + loss + backward for one batch. Returns `(loss, grads)`,
    /// identical in shape and meaning to [`crate::modelgrad::grads`].
    pub fn grads(&self, w: &ModelWeights<f32>, b: &Batch<f32>) -> (f64, ModelGrads<f32>) {
        let cfg = &self.cfg;
        let (dim, tl, td) = (cfg.dim, cfg.text_len, cfg.text_dim);
        let n = cfg.n_tokens();
        let d = cfg.dims();
        check_batch(cfg, b);
        // The two step paths bind the block's weights differently and are not
        // interchangeable on one engine: once the base is resident the graphs
        // read a FOLDED effective weight, which this entry point does not fill.
        assert!(!self.lora_resident(), "grads: this trainer is in LoRA mode - use lora_grads");

        // --- host front: timestep -> e -> e0, text -> ctxe, patches -> x ---
        let f = front(cfg, w, b);
        let mut x = f.x.clone();

        // --- device block stack (forward), saving each block's input ---
        // Each block's weights are uploaded into its own slot here; the reverse
        // sweep below reuses them where the engine had room for the whole
        // stack, and re-uploads where it did not.
        let cached = self.eng.slots() >= w.blocks.len();
        let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(w.blocks.len());
        for (l, bw) in w.blocks.iter().enumerate() {
            inputs.push(x.clone());
            self.eng.load_slot(l, bw, &f.e0);
            x = self.eng.forward_loaded(d, &x, &f.ctxe, &b.cos, &b.sin);
        }

        // --- host head, loss, head backward ---
        let h = head_pass(cfg, w, b, &x, &f.e);
        let mut de = h.de;
        let mut dx = h.dx;

        // --- device block stack (backward), accumulating the shared adjoints ---
        let mut de0 = vec![0f32; 6 * dim];
        let mut dctx = vec![0f32; tl * dim];
        let mut blocks: Vec<BlockGrads<f32>> = Vec::with_capacity(w.blocks.len());
        for (l, (bw, inp)) in w.blocks.iter().zip(&inputs).enumerate().rev() {
            if cached {
                self.eng.select_slot(l);
            } else {
                self.eng.load_slot(l, bw, &f.e0);
            }
            let g = self.eng.backward_loaded(d, inp, &f.ctxe, &b.cos, &b.sin, &dx);
            dx = g.dx.clone();
            for (a, c) in de0.iter_mut().zip(&g.modulation) {
                *a += *c;
            }
            for (a, c) in dctx.iter_mut().zip(&g.dctx) {
                *a += *c;
            }
            blocks.push(g);
        }
        blocks.reverse();

        // --- host tail: patch embed, text embed, timestep path ---
        let (_dflat, g_patch) = linear_bwd(&f.flat, n, cfg.patch_dim(), &w.patch_embed.w, dim, &dx);
        let (dthg, g_text2) = linear_bwd(&f.thg, tl, dim, &w.text2.w, dim, &dctx);
        let dth: Vec<f32> = dthg.iter().zip(&f.th).map(|(&g, &v)| g * dgelu(v)).collect();
        let (_dctx_in, g_text0) = linear_bwd(&b.ctx, tl, td, &w.text0.w, dim, &dth);

        let (deact, g_time_proj) = linear_bwd(&f.eact, 1, dim, &w.time_proj.w, 6 * dim, &de0);
        for (a, (&g, &v)) in de.iter_mut().zip(deact.iter().zip(&f.e)) {
            *a += g * dsilu(v);
        }
        let (dh0, g_time2) = linear_bwd(&f.h0, 1, dim, &w.time2.w, dim, &de);
        let dh0pre: Vec<f32> = dh0.iter().zip(&f.h0pre).map(|(&g, &v)| g * dsilu(v)).collect();
        let (_dte, g_time0) = linear_bwd(&f.te, 1, cfg.freq_dim, &w.time0.w, dim, &dh0pre);

        let g = ModelGrads {
            patch_embed: g_patch,
            text0: g_text0,
            text2: g_text2,
            time0: g_time0,
            time2: g_time2,
            time_proj: g_time_proj,
            blocks,
            head: h.g_head,
            head_mod: h.head_mod,
        };
        (h.loss, g)
    }

    /// Make every block's frozen base resident and switch the engine to the
    /// on-device LoRA path for rank `r`. `false` when the device has no room
    /// for the whole stack, in which case the caller keeps the host-apply path
    /// (residency is what makes the on-device fold worth doing: without it the
    /// base would have to be re-uploaded anyway).
    pub fn begin_lora(&mut self, base: &ModelWeights<f32>, r: usize) -> bool {
        let n = base.blocks.len();
        if self.eng.slots() < n {
            // The effective-weight set and the fold scratch come off the same
            // budget the resident base is measured against.
            let budget = lora_budget(&self.eng).saturating_sub(BlockDev::lora_bytes(self.cfg.dims()));
            self.eng.reserve_slots(n, budget);
        }
        if self.eng.slots() < n {
            return false;
        }
        self.eng.enable_lora(r);
        for (l, bw) in base.blocks.iter().enumerate() {
            self.eng.load_base_slot(l, bw);
        }
        true
    }

    /// `true` once [`DeviceTrainer::begin_lora`] has engaged.
    pub fn lora_resident(&self) -> bool {
        self.eng.lora_rank().is_some()
    }

    /// Full forward + loss + backward for one batch with `ad` applied to the
    /// resident frozen `base`, returning the loss and the ADAPTER gradients.
    ///
    /// Per block this uploads only the ten rank-sized `(A, B)` pairs and the
    /// six modulation vectors; `W_eff` is assembled on-device against the
    /// resident base, and the ten `dW` are projected back to `(dA, dB)` there
    /// too, so the full weight matrices never cross the bus in either
    /// direction.
    ///
    /// [`DeviceTrainer::begin_lora`] must have engaged first.
    pub fn lora_grads(&self, base: &ModelWeights<f32>, ad: &LoraAdapter, b: &Batch<f32>) -> (f64, LoraGrads) {
        let cfg = &self.cfg;
        let d = cfg.dims();
        check_batch(cfg, b);
        assert!(self.lora_resident(), "lora_grads: call begin_lora first");
        assert_eq!(ad.n_blocks(), base.blocks.len(), "lora_grads: adapter/base block count");

        let f = front(cfg, base, b);
        let mut x = f.x.clone();
        let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(base.blocks.len());
        for (l, bw) in base.blocks.iter().enumerate() {
            inputs.push(x.clone());
            self.eng.select_slot(l);
            self.eng.load_mods(bw, &f.e0);
            self.eng.upload_lora(&ad.block_ab(l), ad.scale(), false);
            x = self.eng.forward_loaded(d, &x, &f.ctxe, &b.cos, &b.sin);
        }

        let h = head_pass(cfg, base, b, &x, &f.e);
        let mut dx = h.dx;

        // The reverse sweep re-folds against the same resident base: the
        // modulation is already in the slot from the forward, so only the
        // adapter operands are re-uploaded.
        let mut blocks: Vec<crate::devgrad::AdapterGrads> = Vec::with_capacity(base.blocks.len());
        for (l, inp) in inputs.iter().enumerate().rev() {
            self.eng.select_slot(l);
            self.eng.upload_lora(&ad.block_ab(l), ad.scale(), true);
            let (ndx, pairs) = self.eng.backward_lora_loaded(d, inp, &f.ctxe, &b.cos, &b.sin, &dx);
            dx = ndx;
            blocks.push(pairs);
        }
        blocks.reverse();
        (h.loss, LoraGrads { blocks })
    }
}

fn check_batch(cfg: &Cfg, b: &Batch<f32>) {
    assert_eq!(b.latent.len(), cfg.latent_len(), "latent size");
    assert_eq!(b.ctx.len(), cfg.text_len * cfg.text_dim, "ctx size (pad to text_len before calling)");
    assert_eq!(b.cos.len(), cfg.n_tokens() * cfg.head_dim() / 2, "rope table size");
}

/// Bytes of resident weight storage the block engine may claim: half the
/// smallest card's VRAM, which leaves room for the activations, the gradient
/// set and whatever else shares the device. A backend with no card in the
/// registry (the CPU JIT) gets one slot, because there a "device upload" is a
/// host memcpy and residency would only cost RAM.
fn resident_budget(eng: &BlockDev) -> u64 {
    if !eng.is_accelerated() {
        return 0;
    }
    let vram = gpu_core::devices::gpus().iter().map(|d| d.identity.vram_bytes).filter(|&v| v > 0).min().unwrap_or(0);
    vram / 2
}

/// Bytes the on-device LoRA path may claim for the FROZEN base, which it needs
/// resident for the whole stack or not at all. On a real accelerator that is
/// the same half-VRAM budget above; on the host JIT, where a "device buffer" is
/// plain RAM, a modest cap - enough for the small configs the parity tests run
/// on the CPU backend, not enough for a full-size model to quietly double its
/// host footprint.
fn lora_budget(eng: &BlockDev) -> u64 {
    if eng.is_accelerated() {
        resident_budget(eng)
    } else {
        512 << 20
    }
}

/// The DiT trainer a run drives: the device engine where a real accelerator is
/// available, the host f32 reference otherwise.
///
/// Both arms compute the same `(loss, ModelGrads<f32>)` from the same
/// gradchecked math - `tests/device_train.rs` pins them equal - so a caller
/// picks one and needs no other branch.
pub enum Trainer {
    Device(Box<DeviceTrainer>),
    Host(Cfg),
}

impl Trainer {
    /// Open the trainer for `device`: `Some("cpu")` forces the host path,
    /// `Some("gpu")` the device engine, and `None` takes brain's default
    /// backend and uses the device engine only if that backend is a real
    /// accelerator - so a machine without one keeps the host path.
    pub fn open(cfg: &Cfg, device: Option<&str>) -> Trainer {
        if device == Some("cpu") {
            return Trainer::Host(*cfg);
        }
        let t = DeviceTrainer::on_device(cfg, device);
        if t.is_accelerated() {
            Trainer::Device(Box::new(t))
        } else {
            Trainer::Host(*cfg)
        }
    }

    pub fn is_device(&self) -> bool {
        matches!(self, Trainer::Device(_))
    }

    /// Short human-readable name of the path a step will take.
    pub fn label(&self) -> String {
        match self {
            Trainer::Device(t) => format!("device ({:?})", t.engine().gpu().caps().class),
            Trainer::Host(_) => "host f32 (CPU)".to_string(),
        }
    }

    pub fn grads(&self, w: &ModelWeights<f32>, b: &Batch<f32>) -> (f64, ModelGrads<f32>) {
        match self {
            Trainer::Device(t) => t.grads(w, b),
            Trainer::Host(cfg) => crate::modelgrad::grads(cfg, w, b),
        }
    }

    /// Offer the on-device LoRA path to a device trainer with room for the
    /// whole stack's frozen base. `false` means [`Trainer::lora_step`] keeps
    /// the host-apply route, which computes the same thing.
    pub fn begin_lora(&mut self, base: &ModelWeights<f32>, rank: usize) -> bool {
        match self {
            Trainer::Device(t) => t.begin_lora(base, rank),
            Trainer::Host(_) => false,
        }
    }

    /// One LoRA training step against a FROZEN `base`: loss, adapter grads,
    /// Adam. Returns the loss.
    ///
    /// The two routes differ only in where `W_eff = base + scale·B·A` is built
    /// and where `dL/dW_eff` is projected onto `(dA, dB)` - on the device where
    /// [`Trainer::begin_lora`] engaged, on the host otherwise.
    pub fn lora_step(&self, base: &ModelWeights<f32>, ad: &mut LoraAdapter, b: &Batch<f32>, lr: f32) -> f64 {
        match self {
            Trainer::Device(t) if t.lora_resident() => {
                let (loss, g) = t.lora_grads(base, ad, b);
                ad.step_projected(&g, lr);
                loss
            }
            _ => {
                let (loss, g) = self.grads(&ad.apply(base), b);
                ad.step(&g, lr);
                loss
            }
        }
    }
}
