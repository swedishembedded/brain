// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The VQGAN **training** graph: SSA forward + hand-written reverse, with the
//! vector-quantiser straight-through estimator.
//!
//! Composition again, not a second model. The encoder and generator are the
//! same [`vae::blocks::Builder`] blocks the inference graph records, put in
//! train mode ([`vae::blocks::Builder::set_train`]) so every stage keeps its own
//! buffer, and their reverse is [`vae::blocks::grad::Trace::backward`] — shared
//! with `AutoencoderKL`, added in this workstream. This module owns exactly one
//! thing the shared builder cannot: the quantiser seam.
//!
//! # The quantiser: forward
//!
//! `vq_argmin` is a piecewise-constant assignment. Its derivative is zero almost
//! everywhere and undefined on the cell boundaries, so VQ-VAE replaces it with
//! the **straight-through estimator**: the graph emits
//!
//! ```text
//! z_q_st = z + sub0,    sub0 = (q0 - z0)   a LATCHED CONSTANT
//! ```
//!
//! which is numerically `q0` (to fp32 rounding) but routes the decoder's
//! gradient straight onto `z`. On the device that is one `add2` against a
//! buffer refreshed by [`VqganTrainer::latch_assignment`] — no new kernel — and
//! `sub0` is simply never given a gradient buffer, which is what "detached"
//! means here. The inference graph in [`crate::model`] emits the raw gather
//! instead; the two agree in the forward and differ only in the backward.
//!
//! # The quantiser: backward — where the stop-gradients go
//!
//! Write `z` for the live encoder rows `[M, D]`, `k = argmin_k ||z - cb_k||^2`,
//! `q = cb[k]` for the live gather, `n = M*D`, and `z0`/`q0` for the values
//! latched at the current point. The three terms are:
//!
//! | term | formula | flows to | NOT to |
//! |---|---|---|---|
//! | straight-through | `dL_rec/dz  <-  dL_rec/dz_q` (identity) | encoder | codebook |
//! | codebook | `L_cb  = beta*\|\|sg[z] - q\|\|^2 / n`, `dL/dq = beta*2(q - z0)/n` | **codebook only** | encoder |
//! | commitment | `L_com = \|\|z - sg[q]\|\|^2 / n`, `dL/dz = 2(z - q0)/n` | **encoder only** | codebook |
//!
//! **Which term carries `beta` is the reference's, not the paper's.**
//! `vqgan_arch.py:55` is
//!
//! ```text
//! loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
//! ```
//!
//! `z_q.detach()` makes the FIRST term the one that reaches the encoder (the
//! commitment term) and it is UNWEIGHTED; `z.detach()` makes the SECOND the one
//! that reaches the codebook, and it carries `beta`. That is the opposite of the
//! VQ-VAE paper's convention AND of the reference's own comment on line 29
//! (`# commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2`) — the
//! comment and the code disagree, and `basicsr`'s executed code is what trained
//! `codeformer.pth` / `vqgan_code1024.pth`, so the code wins. Getting this
//! backwards changes the codebook/encoder pull ratio by `1/beta^2` = 16x and is
//! invisible to a finite-difference check, which only gates the backward against
//! whatever forward is emitted.
//!
//! `sg[.]` is the stop-gradient, and on the device it is literally which buffer
//! the `mse_grad` binds: the codebook term binds the **latched** `z0` (so `d_z`
//! gets nothing from it) and the commitment term binds the **latched** `q0` (so
//! the codebook gets nothing from it). The two terms have the same value at the
//! latch point and OPPOSITE stop-gradient placement — that is the whole
//! subtlety. Swapping them, or letting either flow both ways, still trains to a
//! plausible loss curve, so it is gated by finite differences
//! (`gradcheck::check_vqgan`) and not by inspection.
//!
//! **The straight-through gradient is a biased estimator, and this matters for
//! how it is gated.** Against the *literal* objective `L_rec(G(q))` the encoder's
//! true derivative is zero — `argmin` is flat and `q` does not move with `z`.
//! Finite differences of that objective therefore cannot validate the STE; they
//! measure the quantity the STE deliberately replaces. What IS checkable, and
//! what this module implements, is the **straight-through surrogate**: with
//! `sub0`, `z0` and `q0` held fixed the whole graph is smooth, its exact
//! gradient is the update a training step applies, and the two coincide at the
//! latch point. `crates/codec`'s `StraightThroughVq::surrogate_loss` is the same
//! construction on the host, for the same reason.
//!
//! `beta` is [`VqganConfig::beta`] — 0.25 in the CodeFormer preset. The codebook
//! is trained by **gradient**, not by the EMA alternative
//! (`wm_core::vq::ema_update`), so `quantize.embedding.weight` is a trainable
//! tensor here.
//!
//! # Reconstruction loss
//!
//! **L1**: `mean |out - target|`, the `masked_l1` / `masked_l1_grad` pair with an
//! all-ones mask. The real VQGAN recipe adds a **perceptual (LPIPS/VGG) term**
//! and an **adversarial patch-discriminator term** with an adaptive weight; both
//! need models this crate does not have (a VGG16 and a `NLayerDiscriminator`)
//! and are **out of scope** — they attach at the same `d_out` seed, so adding
//! them later is a second contribution into that buffer, not a restructuring.
//!
//! # The latch
//!
//! [`VqganTrainer::latch_assignment`] runs the encoder and `vq_argmin` at the
//! current parameters and pins four things: the code indices `k`, `q0 = cb[k]`,
//! `z0 = z`, and `sub0 = q0 - z0`. Pinning `k` removes the graph's only
//! non-smoothness (the same trick `check_moe` uses when it sets
//! `top_k == n_experts`); pinning the other three is the stop-gradient. A real
//! training loop re-latches once per step, before the forward whose gradient it
//! takes; the gradcheck latches once and holds it across the whole sweep.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use vae::blocks::grad::{BwdIds, Grads, Reverse, Trace};
use vae::blocks::{BlockNames, Builder, Tensors};

use crate::config::VqganConfig;
use crate::model::{record_assign, record_lookup, run_blocks};

/// Where [`vae::blocks::BWD_KERNELS`] sits in [`TRAIN_KERNELS`] — right after
/// the inference set ([`crate::KERNELS`]).
const BWD_BASE: usize = vae::blocks::NEXT_SLOT + 3;
const TAIL: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

const K_EMB_BWD: usize = TAIL;
const K_MASKED_L1: usize = TAIL + 1;
const K_MASKED_L1_GRAD: usize = TAIL + 2;
const K_MSE_VALUE: usize = TAIL + 3;
const K_MSE_GRAD: usize = TAIL + 4;
/// The quantiser seam is recorded outside any `Builder`, so it needs `axpy` and
/// `add2` directly — but it must REUSE the shared set's pipelines rather than
/// register a second copy under the same kernel name: the CPU backend's
/// Cranelift JIT rejects a duplicate definition outright
/// (`DuplicateDefinition("axpy")`), so a second registration is a hard failure
/// on `BRAIN_DEVICE=cpu` and silently fine on the GPU.
const K_AXPY: usize = BWD_BASE + 10;
const K_ADD2: usize = vae::blocks::ADD2_SLOT;

/// This model's TRAINING kernel set: the inference set ([`crate::KERNELS`] —
/// shared blocks, the two VQ assignment kernels, `embed`), then the shared
/// block backward set, then the quantiser-seam and loss kernels.
pub const TRAIN_KERNELS: [(&str, &str); TAIL + 5] = train_kernel_set();

/// [`TRAIN_KERNELS`] as a `'static` slice — what `gpu_core::testgpu::dev` and
/// `Gpu::new_like` want.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &TRAIN_KERNELS;

const fn train_kernel_set() -> [(&'static str, &'static str); TAIL + 5] {
    let mut k = [("", ""); TAIL + 5];
    let mut i = 0;
    while i < crate::KERNELS.len() {
        k[i] = crate::KERNELS[i];
        i += 1;
    }
    let mut j = 0;
    while j < vae::blocks::BWD_KERNELS.len() {
        k[BWD_BASE + j] = vae::blocks::BWD_KERNELS[j];
        j += 1;
    }
    k[K_EMB_BWD] = ("emb_bwd", kernels::EMB_BWD);
    k[K_MASKED_L1] = ("masked_l1", kernels::MASKED_L1);
    k[K_MASKED_L1_GRAD] = ("masked_l1_grad", kernels::MASKED_L1_GRAD);
    k[K_MSE_VALUE] = ("mse_value", kernels::MSE_VALUE);
    k[K_MSE_GRAD] = ("mse_grad", kernels::MSE_GRAD);
    k
}

/// A trainable VQGAN for a fixed input size: one forward step list, one reverse
/// step list, one gradient buffer per tensor.
pub struct VqganTrainer {
    gpu: Gpu,
    cfg: VqganConfig,
    hw: (u32, u32),
    lhw: (u32, u32),

    // ---- graphs -----------------------------------------------------------
    /// encoder blocks + `nchw_to_rows` (fills `z_flat`).
    enc_steps: Vec<Step>,
    /// `vq_argmin` on `z_flat` (fills `packed`).
    assign_steps: Vec<Step>,
    /// Latch the three stop-gradient constants: `q0 = cb[idx]`, `z0 = z`,
    /// `sub0 = q0 - z0`.
    latch_steps: Vec<Step>,
    latch_clears: Vec<DeviceBuffer>,
    /// The whole objective: encoder, gather, straight-through, generator, and
    /// the three per-element loss buffers.
    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
    bwd_clears: Vec<DeviceBuffer>,

    // ---- buffers ----------------------------------------------------------
    img_in: DeviceBuffer,
    target: DeviceBuffer,
    packed: DeviceBuffer,
    idx_in: DeviceBuffer,
    zq_st: DeviceBuffer,
    q0: DeviceBuffer,
    out: DeviceBuffer,
    l1_val: DeviceBuffer,
    cb_val: DeviceBuffer,
    com_val: DeviceBuffer,

    // ---- parameters -------------------------------------------------------
    enc: Trace,
    gen: Trace,
    enc_g: Grads,
    gen_g: Grads,
    codebook: DeviceBuffer,
    codebook_g: DeviceBuffer,
    names: Vec<(String, u64)>,
}

/// The codebook tensor's checkpoint name (it is a parameter like any other).
const CODEBOOK: &str = "quantize.embedding.weight";

impl VqganTrainer {
    /// Build the training graphs for a `[in_channels, h, w]` input on `gpu`
    /// (which MUST have been created with [`TRAIN_KERNELS`]).
    pub fn new(cfg: VqganConfig, tensors: &Tensors, h: u32, w: u32, gpu: Gpu) -> VqganTrainer {
        let scale = cfg.downscale();
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "vqgan: input {h}x{w} is not a multiple of the {scale}x downscale"
        );
        let (lh, lw) = (h / scale, w / scale);
        let (t, emb, ncode) = (lh * lw, cfg.emb_dim, cfg.codebook_size);
        let te = (t * emb) as u64;
        let n_out = (cfg.out_channels * h * w) as u64;
        let ids = BwdIds::at(BWD_BASE);
        let empty = Tensors::new();

        let codebook = {
            let (_, data) =
                tensors.get(CODEBOOK).unwrap_or_else(|| panic!("vqgan: missing {CODEBOOK}"));
            gpu.storage_init(CODEBOOK, data)
        };
        let img_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let target = gpu.storage(n_out);
        let mask = gpu.storage_init("recon.mask", &vec![1.0f32; n_out as usize]);
        let idx_in = gpu.storage(t as u64);

        // ---- encoder (train mode: SSA, direct lowerings, taped) ------------
        let mut be = Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        be.set_train(true);
        let z = run_blocks(&mut be, "encoder", &cfg.encoder_blocks(), 0, h, w, &img_in).0;
        let z_flat = be.nchw_to_rows(emb, t, &z);
        let enc = be.trace();
        let (enc_steps, _) = be.finish();

        // ---- the frozen assignment, through the ONE `vq_argmin` site -------
        let mut ba = Builder::new(&gpu, &empty, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        let packed = record_assign(&mut ba, &codebook, t, ncode, emb, &z_flat);
        let (assign_steps, _) = ba.finish();

        // ---- the codebook gather, through the ONE `embed` site -------------
        // TWO gathers, deliberately: `q_rows` is LIVE (it moves when the
        // codebook moves — the codebook term differentiates it), while `q0` is
        // the DETACHED copy latched at the linearisation point. `sg[q]` in the
        // commitment term is exactly `q0`.
        let mut bl = Builder::new(&gpu, &empty, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        let q_rows = record_lookup(&mut bl, &codebook, t, emb, &idx_in);
        let (lookup_steps, _) = bl.finish();
        let mut bl0 = Builder::new(&gpu, &empty, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        let q0 = record_lookup(&mut bl0, &codebook, t, emb, &idx_in);
        let (q0_steps, _) = bl0.finish();

        // ---- straight-through: z_q_st = z + sub0, sub0 = (q0 - z0) ---------
        // `sub0`, `z0` and `q0` are LATCHED (see `latch_assignment`). Freezing
        // them is what `.detach()` means as a finite-difference-checkable
        // object: at the latch point `z == z0` and `q == q0`, so the forward
        // VALUE is the quantised one (`z_q_st == q`, to fp32 rounding), while
        // the derivative w.r.t. `z` is the identity — the straight-through
        // estimator. `crates/codec`'s `StraightThroughVq::surrogate_loss` is
        // the same construction on the host (it feeds the decoder the raw `z`,
        // i.e. `sub0 == 0`; keeping `sub0` makes the forward value exact too).
        let z0 = gpu.storage(te);
        let sub0 = gpu.storage(te);
        let zq_st = gpu.storage(te);
        let mut latch_steps = q0_steps;
        latch_steps.push(gpu.step(K_AXPY, &[&z0, &z_flat], &[te as u32, f(1.0)], te as u32));
        latch_steps.push(gpu.step(K_AXPY, &[&sub0, &q0], &[te as u32, f(1.0)], te as u32));
        latch_steps.push(gpu.step(K_AXPY, &[&sub0, &z0], &[te as u32, f(-1.0)], te as u32));
        let ste_steps =
            vec![gpu.step(K_ADD2, &[&z_flat, &sub0, &zq_st], &[te as u32], te as u32)];

        // ---- generator -----------------------------------------------------
        let mut bg = Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        bg.set_train(true);
        let z_q = bg.rows_to_nchw(emb, t, &zq_st);
        let (out, (oh, ow)) =
            run_blocks(&mut bg, "generator", &cfg.generator_blocks(), 0, lh, lw, &z_q);
        assert_eq!((oh, ow), (h, w), "vqgan: generator output {oh}x{ow} != input {h}x{w}");
        let gen = bg.trace();
        let (gen_steps, _) = bg.finish();

        // ---- per-element loss values (the host sums them, as every brain
        //      loss kernel expects — see mse_value.wgsl) --------------------
        let l1_val = gpu.storage(n_out);
        let cb_val = gpu.storage(te);
        let com_val = gpu.storage(te);
        let loss_steps = vec![
            gpu.step(K_MASKED_L1, &[&out, &target, &mask, &l1_val], &[n_out as u32], n_out as u32),
            // L_cb  = ||sg[z] - q||^2 / n : z frozen (`z0`), q live. Weighted by
            // beta on the host, where the reduction happens — this is the
            // `self.beta * torch.mean((z_q - z.detach())**2)` half of
            // `vqgan_arch.py:55` (see the module docs: `beta` sits on the
            // CODEBOOK term in the reference's code, not the commitment one).
            gpu.step(K_MSE_VALUE, &[&q_rows, &z0, &cb_val], &[te as u32], te as u32),
            // L_com = ||z - sg[q]||^2 / n : q frozen (`q0`), z live. UNWEIGHTED
            // — `torch.mean((z_q.detach()-z)**2)` carries no beta.
            gpu.step(K_MSE_VALUE, &[&z_flat, &q0, &com_val], &[te as u32], te as u32),
        ];

        let mut fwd_steps = enc_steps.clone();
        fwd_steps.extend(lookup_steps);
        fwd_steps.extend(ste_steps);
        fwd_steps.extend(gen_steps);
        fwd_steps.extend(loss_steps);

        // ---- reverse -------------------------------------------------------
        let enc_g = enc.alloc_grads(&gpu);
        let gen_g = gen.alloc_grads(&gpu);
        let codebook_g = gpu.storage((ncode * emb) as u64);

        let d_out = gpu.storage(n_out);
        let d_z = gpu.storage(te);
        let dq_cb = gpu.storage(te);
        // `mse_grad` has no scale factor and `emb_bwd` has none either, so the
        // codebook term's `beta` is applied by one `axpy` into a cleared buffer
        // (assign would alias `dq_cb` against itself, which `axpy` forbids).
        let dq_cb_b = gpu.storage(te);
        let dz_com = gpu.storage(te);

        let gen_rev: Reverse = gen.backward(&gpu, ids, &gen_g, &out, &d_out);
        let d_zq_st = gen_rev
            .d(&zq_st)
            .expect("vqgan: the generator reverse did not reach its own input")
            .clone();

        let mut bwd_steps = vec![gpu.step(
            K_MASKED_L1_GRAD,
            &[&out, &target, &mask, &d_out],
            &[n_out as u32, f(1.0 / n_out as f32)],
            n_out as u32,
        )];
        bwd_steps.extend(gen_rev.steps);
        // Straight-through: the decoder gradient crosses the quantiser
        // UNCHANGED. `z_q_st = z + sub0` with `sub0` constant, so this identity
        // IS the exact adjoint of the emitted forward — the estimator's bias
        // lives entirely in the choice to freeze `sub0`, which is the
        // stop-gradient itself.
        bwd_steps.push(gpu.step(K_AXPY, &[&d_z, &d_zq_st], &[te as u32, f(1.0)], te as u32));
        // Codebook term dL_cb/dq = beta * 2(q - z0)/n, scattered into the
        // assigned codebook rows ONLY. `sg[z]` is why `z0` is bound and why
        // `d_z` gets nothing from this term; `beta` is here (not on the
        // commitment term) because `vqgan_arch.py:55` puts it on the
        // `z.detach()` half — see the module docs.
        bwd_steps.push(gpu.step(K_MSE_GRAD, &[&q_rows, &z0, &dq_cb], &[te as u32], te as u32));
        bwd_steps.push(gpu.step(K_AXPY, &[&dq_cb_b, &dq_cb], &[te as u32, f(cfg.beta)], te as u32));
        bwd_steps.push(gpu.step(
            K_EMB_BWD,
            &[&idx_in, &dq_cb_b, &codebook_g],
            &[t, emb, ncode],
            ncode * emb,
        ));
        // Commitment term dL_com/dz = 2(z - q0)/n, into the ENCODER only, and
        // UNWEIGHTED. `sg[q]` is why `q0` is bound and why the codebook gets
        // nothing here.
        bwd_steps.push(gpu.step(K_MSE_GRAD, &[&z_flat, &q0, &dz_com], &[te as u32], te as u32));
        bwd_steps.push(gpu.step(K_AXPY, &[&d_z, &dz_com], &[te as u32, f(1.0)], te as u32));
        let enc_rev: Reverse = enc.backward(&gpu, ids, &enc_g, &z_flat, &d_z);
        bwd_steps.extend(enc_rev.steps);

        let mut bwd_clears = gen_rev.clears;
        bwd_clears.push(d_z);
        bwd_clears.push(dq_cb_b);
        bwd_clears.extend(enc_rev.clears);

        let mut names: Vec<(String, u64)> = enc.params().to_vec();
        names.extend(gen.params().iter().cloned());
        names.push((CODEBOOK.to_string(), (ncode * emb) as u64));

        VqganTrainer {
            gpu,
            cfg,
            hw: (h, w),
            lhw: (lh, lw),
            enc_steps,
            assign_steps,
            latch_steps,
            latch_clears: vec![z0, sub0],
            fwd_steps,
            bwd_steps,
            bwd_clears,
            img_in,
            target,
            packed,
            idx_in,
            zq_st,
            q0,
            out,
            l1_val,
            cb_val,
            com_val,
            enc,
            gen,
            enc_g,
            gen_g,
            codebook,
            codebook_g,
            names,
        }
    }

    /// The forward dispatches of one training step, in submit order.
    ///
    /// Exposed for the PROFILER (`vqgan_bench`), not for driving: a caller that
    /// submits these itself skips the latch and the clears and would get a
    /// silently wrong gradient. `docs/kernel-checklist.md` §E wants a per-
    /// kernel-kind table before anyone optimises, and until this existed there
    /// was no way to get one for a BACKWARD anywhere in the tree.
    pub fn fwd_steps(&self) -> &[Step] {
        &self.fwd_steps
    }

    /// The backward dispatches of one training step, in submit order.
    pub fn bwd_steps(&self) -> &[Step] {
        &self.bwd_steps
    }

    /// The activation-gradient buffers the backward accumulates into, which
    /// MUST be zeroed before it runs.
    pub fn bwd_clears(&self) -> &[DeviceBuffer] {
        &self.bwd_clears
    }

    pub fn config(&self) -> &VqganConfig {
        &self.cfg
    }

    /// The latent grid `(lh, lw)`.
    pub fn latent_size(&self) -> (u32, u32) {
        self.lhw
    }

    /// Install the fixed training pair: `image` `[in_channels·H·W]` and the
    /// reconstruction `target` `[out_channels·H·W]`, both row-major NCHW at
    /// batch 1. Re-latches the code assignment for the new input.
    pub fn set_batch(&self, image: &[f32], target: &[f32]) {
        let n_in = (self.cfg.in_channels * self.hw.0 * self.hw.1) as usize;
        let n_out = (self.cfg.out_channels * self.hw.0 * self.hw.1) as usize;
        assert_eq!(image.len(), n_in, "vqgan: image has {} values, expected {n_in}", image.len());
        assert_eq!(target.len(), n_out, "vqgan: target has {} values, expected {n_out}", target.len());
        write_f32(&self.gpu, &self.img_in, image);
        write_f32(&self.gpu, &self.target, target);
        self.latch_assignment();
    }

    /// Run the encoder and `vq_argmin` at the current parameters and PIN the
    /// four stop-gradient constants the training objective is defined against:
    /// the code indices `k`, the gathered rows `q0 = cb[k]`, the encoder output
    /// `z0`, and the straight-through offset `sub0 = q0 - z0`.
    ///
    /// This is one training step's linearisation point. Pinning `k` removes the
    /// only non-smoothness in the graph (`argmin` is piecewise constant — the
    /// same reason `check_moe` sets `top_k == n_experts`); pinning `q0`/`z0`/
    /// `sub0` is what the reference's three `.detach()` calls do. A real
    /// training loop calls this once per step, before the forward whose
    /// gradient it takes; the gradcheck calls it once and holds it.
    pub fn latch_assignment(&self) {
        self.gpu.submit(&[], &self.enc_steps);
        self.gpu.submit(&[], &self.assign_steps);
        let t = (self.lhw.0 * self.lhw.1) as usize;
        let packed = self.gpu.read(&self.packed, 2 * t);
        let idx = wm_core::vq::indices(&packed);
        self.gpu.write(&self.idx_in, &idx);
        let clears: Vec<&DeviceBuffer> = self.latch_clears.iter().collect();
        self.gpu.submit(&clears, &self.latch_steps);
    }

    /// The pinned code indices. `Gpu::read` hands back the raw words as f32, so
    /// `to_bits` recovers the `u32` the `embed`/`emb_bwd` pair reads.
    pub fn codes(&self) -> Vec<u32> {
        let t = (self.lhw.0 * self.lhw.1) as usize;
        self.gpu.read(&self.idx_in, t).iter().map(|v| f32::to_bits(*v)).collect()
    }

    /// The straight-through latent `z + sub0` the generator actually consumed
    /// in the last [`Self::loss`], and the latched `q0` beside it — equal to
    /// fp32 rounding whenever the encoder has not moved since the latch. The
    /// forward-value half of the STE contract (the backward half is the FD
    /// gate).
    pub fn ste_vs_quantized(&self) -> (Vec<f32>, Vec<f32>) {
        let te = (self.lhw.0 * self.lhw.1 * self.cfg.emb_dim) as usize;
        (self.gpu.read(&self.zq_st, te), self.gpu.read(&self.q0, te))
    }

    /// One forward pass over the installed batch; returns the scalar objective
    ///
    /// ```text
    /// L = mean |G(z + sub0) - target|  +  beta*||sg[z] - q||^2/n  +  ||z - sg[q]||^2/n
    /// ```
    ///
    /// The two VQ terms are `vqgan_arch.py:55` verbatim (`beta` on the
    /// `z.detach()` / codebook half — see the module docs).
    ///
    /// At the latch point this equals the real VQ-VAE objective (`z + sub0` IS
    /// the quantised latent, `sg[z]` IS `z`, `sg[q]` IS `q`) — and unlike the
    /// real objective it is **smooth and exactly differentiable**, which is
    /// what makes a central difference a valid check of the gradient a training
    /// step applies.
    ///
    /// The reduction is in f64: the finite difference divides by `2*eps`, and
    /// an fp32 sum over 192 output pixels would put its own round-off there.
    pub fn loss(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd_steps);
        let n_out = (self.cfg.out_channels * self.hw.0 * self.hw.1) as usize;
        let te = (self.lhw.0 * self.lhw.1 * self.cfg.emb_dim) as usize;
        let sum = |b: &DeviceBuffer, n: usize| -> f64 {
            self.gpu.read(b, n).iter().map(|&v| v as f64).sum()
        };
        let rec = sum(&self.l1_val, n_out) / n_out as f64;
        let cb = sum(&self.cb_val, te);
        let com = sum(&self.com_val, te);
        (rec + self.cfg.beta as f64 * cb + com) as f32
    }

    /// The reconstruction `[out_channels·H·W]` from the last [`Self::loss`].
    pub fn output(&self) -> Vec<f32> {
        let n_out = (self.cfg.out_channels * self.hw.0 * self.hw.1) as usize;
        self.gpu.read(&self.out, n_out)
    }

    /// Zero every parameter gradient. This is the ONE place gradient buffers
    /// are cleared: `conv2d_dw`, `bias_grad`, `gn_dgamma`, `gn_dbeta` and
    /// `emb_bwd` all accumulate, so clearing them anywhere inside
    /// [`Self::backward`]'s submit would drop earlier contributions.
    pub fn zero_grads(&self) {
        let mut all = self.enc_g.all();
        all.extend(self.gen_g.all());
        all.push(&self.codebook_g);
        self.gpu.submit(&all, &[]);
    }

    /// One reverse pass. Requires a preceding [`Self::loss`] (the forward
    /// buffers ARE the activation cache).
    pub fn backward(&self) {
        let clears: Vec<&DeviceBuffer> = self.bwd_clears.iter().collect();
        self.gpu.submit(&clears, &self.bwd_steps);
    }

    /// Every trainable tensor, `(name, length)`, in graph order. GroupNorm
    /// appears once per module as the fused `{prefix}.gb[2C]`; an attention's
    /// q/k/v appear once as the fused `{prefix}.qkv.{w,b}`.
    pub fn param_names(&self) -> Vec<(String, u64)> {
        self.names.clone()
    }

    fn weight_buf(&self, name: &str) -> &DeviceBuffer {
        if name == CODEBOOK {
            return &self.codebook;
        }
        if name.starts_with("encoder.") {
            self.enc.weight(name)
        } else {
            self.gen.weight(name)
        }
    }

    fn grad_buf(&self, name: &str) -> &DeviceBuffer {
        if name == CODEBOOK {
            return &self.codebook_g;
        }
        if name.starts_with("encoder.") {
            self.enc_g.g(name)
        } else {
            self.gen_g.g(name)
        }
    }

    fn numel(&self, name: &str) -> usize {
        self.names
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("vqgan: no parameter {name}"))
            .1 as usize
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.gpu.read(self.weight_buf(name), self.numel(name))
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.numel(name), "vqgan: bad length for {name}");
        write_f32(&self.gpu, self.weight_buf(name), data);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.gpu.read(self.grad_buf(name), self.numel(name))
    }

    /// The device the graphs were built on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

fn write_f32(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    gpu.write(buf, &bits);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The training kernel set must keep the inference set at slots
    /// `0..KERNELS.len()` — the shared `Builder` and `record_assign` /
    /// `record_lookup` address those by position, and a set that drifts by one
    /// entry is silently wrong, not a crash.
    #[test]
    fn train_kernel_set_extends_the_inference_set() {
        assert_eq!(TRAIN_KERNELS[..crate::KERNELS.len()], crate::KERNELS[..]);
        assert_eq!(
            TRAIN_KERNELS[BWD_BASE..BWD_BASE + vae::blocks::BWD_KERNELS.len()],
            vae::blocks::BWD_KERNELS[..]
        );
        assert_eq!(TRAIN_KERNELS[K_EMB_BWD].0, "emb_bwd");
        // The seam reuses the shared pipelines; nothing is registered twice.
        assert_eq!(TRAIN_KERNELS[K_AXPY].0, "axpy");
        assert_eq!(TRAIN_KERNELS[K_ADD2].0, "add2");
        assert!(TRAIN_KERNELS.iter().all(|(n, _)| !n.is_empty()), "unfilled kernel slot");
        let mut names: Vec<&str> = TRAIN_KERNELS.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate kernel name (the CPU JIT rejects those)");
        assert_eq!(K_AXPY, vae::blocks::grad::BwdIds::at(BWD_BASE).axpy());
    }

    fn tiny() -> VqganConfig {
        VqganConfig {
            in_channels: 3,
            out_channels: 3,
            nf: 4,
            ch_mult: vec![1, 2],
            res_blocks: 1,
            attn_resolutions: vec![4],
            img_size: 8,
            codebook_size: 6,
            emb_dim: 4,
            beta: 0.25,
            norm_groups: 2,
            norm_eps: 1e-6,
        }
    }

    /// Deterministic fixture values — a fixed irrational-stride sequence, not an
    /// RNG (this test only needs reproducible non-degenerate weights).
    fn fill(n: usize, k: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| scale * (((i + 7 * k) as f32) * 0.754_877_7).sin()).collect()
    }

    fn fixture(cfg: &VqganConfig) -> Tensors {
        let mut t = Tensors::new();
        for (k, (name, shape)) in cfg.tensor_manifest().into_iter().enumerate() {
            let n: usize = shape.iter().product();
            let data = match shape.len() {
                1 if name.ends_with(".weight") => {
                    fill(n, k, 0.1).iter().map(|v| 1.0 + v).collect()
                }
                1 => fill(n, k, 0.1),
                2 => fill(n, k, 0.6),
                _ => fill(n, k, 1.0 / ((n / shape[0]) as f32).sqrt()),
            };
            t.insert(name, (shape, data));
        }
        t
    }

    /// The straight-through forward must be the QUANTISED forward: `z + sub0`
    /// with `sub0 = q0 - z0` equals `q0` whenever the encoder has not moved
    /// since the latch. This is the forward half of the STE contract — the
    /// backward half is `gradcheck::check_vqgan`.
    #[test]
    fn straight_through_forward_equals_the_raw_gather() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = tiny();
        let tensors = fixture(&cfg);
        let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
        let m = VqganTrainer::new(cfg.clone(), &tensors, cfg.img_size, cfg.img_size, gpu);
        let n_in = (cfg.in_channels * cfg.img_size * cfg.img_size) as usize;
        let n_out = (cfg.out_channels * cfg.img_size * cfg.img_size) as usize;
        m.set_batch(&fill(n_in, 1, 1.0), &fill(n_out, 2, 1.0));
        let l = m.loss();
        assert!(l.is_finite(), "loss is {l}");
        let (ste, q0) = m.ste_vs_quantized();
        let worst = ste
            .iter()
            .zip(&q0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "straight-through latent differs from the gather by {worst:e}");
        // And the codes are in range.
        assert!(m.codes().iter().all(|&i| i < cfg.codebook_size));
    }
}
