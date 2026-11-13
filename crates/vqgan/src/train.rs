// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The VQGAN **training** graph: SSA forward + explicit reverse, with the
//! vector-quantiser straight-through estimator.
//!
//! Composition again, not a second model. The encoder and generator are the
//! same [`vae::blocks::Builder`] blocks the inference graph records, put in
//! train mode ([`vae::blocks::Builder::set_train`]) so every stage keeps its own
//! buffer, and their reverse is [`vae::blocks::grad::Trace::backward`] — shared
//! with `AutoencoderKL`. This module owns what the shared builder cannot: the
//! quantiser seam, and the training step that adds the rest of the objective
//! ([`crate::objective`]: perceptual, adversarial with its discriminator, and
//! entropy terms) into the same output seed.
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
//! backwards changes the codebook/encoder pull ratio by `1/beta^2`, i.e. sixteenfold, and is
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
//! latch point. `crates/mimi`'s `ConvReconAE::surrogate_loss` is the same
//! construction on the host, for the same reason.
//!
//! `beta` is [`VqganConfig::beta`] - 0.25 in the CodeFormer preset - and which
//! term it weights is [`VqganConfig::beta_on`]: the codebook term above for
//! `basicsr` ([`BetaTerm::Codebook`]), the commitment term for LlamaGen
//! ([`BetaTerm::Commitment`], `commit_loss = beta * mean((z_q.detach() - z)**2)`,
//! with an unweighted `vq_loss`). The other term carries weight 1. The codebook
//! is trained by **gradient**, not by the EMA alternative
//! (`wm_core::vq::ema_update`), so `quantize.embedding.weight` is a trainable
//! tensor here. LlamaGen's entropy term is the third codebook term, below.
//!
//! # The L2-normalised codebook
//!
//! With [`VqganConfig::codebook_l2_norm`], LlamaGen normalises BOTH sides
//! before anything else sees them: `z = F.normalize(z)` replaces the query
//! (so the straight-through estimator and both loss terms are on the
//! normalised rows), and `F.normalize(embedding.weight)` replaces the codebook.
//! Both normalisations are on the graph, so their gradients flow: the query's
//! through `l2norm_scale_dx` into the encoder, the codebook's through the same
//! kernel into the raw `quantize.embedding.weight`, which stays the trainable
//! tensor. Everything written above about `z`, `q` and the latch applies to
//! the normalised rows.
//!
//! # The full objective
//!
//! [`Objective`] chooses the terms and their weights (presets: `codeformer`, `llamagen_vq16`; their values and where
//! they come from are in [`crate::objective`]). The generator minimises
//!
//! ```text
//! L_G = L_rec + perceptual_weight * LPIPS(x_hat, x) + L_cb + L_com + entropy_loss_ratio * L_ent
//!       + disc_weight * lambda * (-mean D(x_hat))              from step disc_start on
//! ```
//!
//! and the discriminator the hinge loss on `D(x)` and `D(x_hat)` ([`crate::patchgan`]), each with its own AdamW.
//! [`VqganTrainer::loss`] and [`VqganTrainer::backward`] are everything but the adversarial term (the part that needs
//! no discriminator and is a function of the parameters alone: what a finite-difference check can differentiate);
//! [`VqganTrainer::train_step`] runs the whole recipe.
//!
//! **Reconstruction.** L1 (`masked_l1`, an all-ones mask: `basicsr`, CodeFormer) or L2 (`mse_value`/`mse_grad`:
//! LlamaGen's default), [`ReconstructionLoss`]. Its gradient is written into `d_out`, the generator output's gradient
//! buffer, and every other image-space term is one more contribution into that same buffer.
//!
//! **Perceptual.** `perceptual_weight * LPIPS(x_hat, x)` ([`lpips::Lpips::distance_grad`]: the objective's trunk, VGG16 in both presets, and heads
//! frozen, the gradient taken to the reconstruction only). Images are in `[-1, 1]`, the range both nets are
//! trained on. The trainer owns the LPIPS network on its own handle of the device and refuses a nonzero weight without
//! the weights ([`VqganTrainer::lpips_weights_from_store`], or any `lpips::import::Tensors`), naming what is
//! missing.
//!
//! **Adversarial.** With the generator's output `x_hat` the discriminator's fake input, `-mean D(x_hat)` is
//! differentiated to the image and added to `d_out` scaled by `disc_weight * lambda`. `lambda` is taming's
//! `calculate_adaptive_weight`, `||grad_W L_nll|| / (||grad_W L_adv|| + 1e-4)` clamped to
//! `[0, adaptive_weight_max]` and detached, with `W` the generator's last convolution weight and `L_nll` the
//! reconstruction plus perceptual terms. Both norms come from one convolution's backward each
//! ([`vae::blocks::grad::Trace::last_conv`]) before the one full backward runs on the combined seed: the backward is
//! linear in its seed, so `lambda` is known before the full pass needs it.
//!
//! **Entropy.** On the device: `entropy_logp` and `entropy_cols` join the forward, `entropy_dz` and `entropy_dcb` lead
//! the reverse and assign their gradients (already scaled by `entropy_loss_ratio`), which are added into the same `d_z`
//! and codebook gradients as the commitment and codebook terms. Only the two scalar reductions (the mean of the
//! per-token entropies, the sum of the per-code terms) are read back, like every loss value here. [`crate::entropy`]
//! is the f64 oracle the kernels are tested against.
//!
//! # Reconstruction loss
//!
//! L1 is `mean |out - target|` and L2 `mean (out - target)^2`, both over every output element.
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

use std::cell::{Cell, RefCell};

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use lpips::Lpips;
use optim::Optim;
use paramstore::ParamStore;
use vae::blocks::grad::{BwdIds, Grads, Reverse, Trace};
use vae::blocks::{Activations, BlockNames, Builder, Tensors};

use crate::config::{BetaTerm, VqganConfig};
use crate::model::{l2norm_step, record_assign, record_lookup, run_blocks, L2_EPS};
use crate::objective::{Losses, Objective, ReconstructionLoss, StepOptions};
use crate::patchgan::{self, PatchGan, PatchGanConfig};

/// Where [`vae::blocks::BWD_KERNELS`] sits in [`TRAIN_KERNELS`] — right after
/// the inference set ([`crate::KERNELS`]).
const BWD_BASE: usize = crate::KERNELS.len();
const TAIL: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

const K_EMB_BWD: usize = TAIL;
const K_MASKED_L1: usize = TAIL + 1;
const K_MASKED_L1_GRAD: usize = TAIL + 2;
const K_MSE_VALUE: usize = TAIL + 3;
const K_MSE_GRAD: usize = TAIL + 4;
const K_L2NORM_DX: usize = TAIL + 5;
const K_ENTROPY_LOGP: usize = TAIL + 11;
const K_ENTROPY_COLS: usize = TAIL + 12;
const K_ENTROPY_DZ: usize = TAIL + 13;
const K_ENTROPY_DCB: usize = TAIL + 14;
/// The quantiser seam is recorded outside any `Builder`, so it needs `axpy` and
/// `add2` directly — but it must REUSE the shared set's pipelines rather than
/// register a second copy under the same kernel name: the CPU backend's
/// Cranelift JIT rejects a duplicate definition outright
/// (`DuplicateDefinition("axpy")`), so a second registration is a hard failure
/// on `BRAIN_DEVICE=cpu` and silently fine on the GPU.
/// NOT a literal offset. `BWD_BASE + 10` was one, and when `gn_dsum` was
/// removed from the shared set every slot after it shifted by one — so this
/// pointed at `attn_bwd_dscores_bidir` and failed as a bind-group arity
/// mismatch (3 bindings vs 5) — the exact signature of a hardcoded-offset
/// bug. `BwdIds::axpy()` exists so the offset lives in ONE place.
const K_AXPY: usize = BwdIds::at(BWD_BASE).axpy();
const K_ADD2: usize = vae::blocks::ADD2_SLOT;

/// This model's TRAINING kernel set: the inference set ([`crate::KERNELS`] —
/// shared blocks, the two VQ assignment kernels, `embed`, `l2norm_scale`), then
/// the shared block backward set, then the quantiser-seam and loss kernels and
/// the optimiser's.
pub const TRAIN_KERNELS: [(&str, &str); TAIL + 15] = vae::blocks::train_kernel_set::<{ TAIL + 15 }>(
    &crate::KERNELS,
    &[
        ("emb_bwd", kernels::EMB_BWD),
        ("masked_l1", kernels::MASKED_L1),
        ("masked_l1_grad", kernels::MASKED_L1_GRAD),
        ("mse_value", kernels::MSE_VALUE),
        ("mse_grad", kernels::MSE_GRAD),
        ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
        // The generator's AdamW (`optim::Optim` resolves the cooperative grad-norm pair by name beside them).
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        // The codebook entropy term ([`crate::entropy`] is its f64 oracle), forward and both gradients.
        ("entropy_logp", kernels::ENTROPY_LOGP),
        ("entropy_cols", kernels::ENTROPY_COLS),
        ("entropy_dz", kernels::ENTROPY_DZ),
        ("entropy_dcb", kernels::ENTROPY_DCB),
    ],
);

/// [`TRAIN_KERNELS`] as a `'static` slice — what `gpu_core::testgpu::dev` and
/// `Gpu::new_like` want.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &TRAIN_KERNELS;


/// The weights `(codebook term, commitment term)` of the two VQ losses:
/// `beta` on the one [`VqganConfig::beta_on`] names, 1 on the other.
fn term_weights(cfg: &VqganConfig) -> (f32, f32) {
    match cfg.beta_on {
        BetaTerm::Codebook => (cfg.beta, 1.0),
        BetaTerm::Commitment => (1.0, cfg.beta),
    }
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
    /// Per-element reconstruction value (L1 or L2, [`Objective::reconstruction`]), summed on the host.
    rec_val: DeviceBuffer,
    cb_val: DeviceBuffer,
    com_val: DeviceBuffer,
    /// What the quantiser reads: the encoder rows (normalised under `codebook_l2_norm`) and the codebook.
    query: DeviceBuffer,
    cb_q: DeviceBuffer,

    // ---- parameters -------------------------------------------------------
    enc: Trace,
    gen: Trace,
    enc_g: Grads,
    gen_g: Grads,
    codebook: DeviceBuffer,
    codebook_g: DeviceBuffer,
    names: Vec<(String, u64)>,

    // ---- the objective ----------------------------------------------------
    obj: Objective,
    /// `d_out`'s writers: the reconstruction gradient (assigned), then the terms added into it.
    d_out: DeviceBuffer,
    seed_steps: Vec<Step>,
    rest_steps: Vec<Step>,
    /// The generator's AdamW over the encoder, generator and codebook buffers the graphs already own.
    gen_ps: ParamStore,
    gen_opt: Optim,
    gen_updates: Cell<u32>,
    /// The generator's last convolution weight (the adaptive weight's reference layer), its one-layer reverse from
    /// the nll seed (`d_out`) and from the adversarial seed (`d_adv`), and their gradient buffers.
    last_weight: String,
    tail: Option<TailReverse>,
    perceptual: Option<Perceptual>,
    entropy: Option<EntropyBuffers>,
    adversarial: Option<Adversarial>,
    /// The terms of the last [`Self::loss`], as they entered it.
    last: RefCell<Terms>,
}

/// The adaptive weight's two one-layer reverses (only for [`Objective::adaptive_weight`]).
struct TailReverse {
    nll_grads: Grads,
    adv_grads: Grads,
    nll_steps: Vec<Step>,
    adv_steps: Vec<Step>,
}

/// The LPIPS network on its own handle of the device, the staged pair, and the gradient wrt the reconstruction.
struct Perceptual {
    lpips: RefCell<Lpips>,
    /// A handle on the LPIPS device, to stage into and read from it while the metric is borrowed.
    gpu: Gpu,
    pair: DeviceBuffer,
    /// `d LPIPS / d x_hat` on the trainer's device, scaled by `perceptual_weight` when it is added into `d_out`.
    d_image: DeviceBuffer,
}

/// The entropy term's device buffers: the forward's per-token `h` and per-code `term`, summed on the host into the
/// scalar (as every loss value here is), and the gradients the reverse adds into the quantiser's.
struct EntropyBuffers {
    /// `[tokens]`: each token's entropy.
    h: DeviceBuffer,
    /// `[codes]`: each code's share of the average-assignment entropy.
    term: DeviceBuffer,
    d_rows: DeviceBuffer,
    d_codebook: DeviceBuffer,
}

/// The discriminator, and the hinge loss's bookkeeping.
struct Adversarial {
    d: PatchGan,
    updates: Cell<u32>,
    /// The generator-loss gradient wrt the output image, the discriminator's input gradient copied to the trainer's
    /// device.
    d_adv: DeviceBuffer,
}

/// The terms of the last forward, as they entered the loss.
#[derive(Clone, Copy, Default)]
struct Terms {
    reconstruction: f64,
    perceptual: f64,
    codebook: f64,
    commitment: f64,
    entropy: f64,
}

impl Terms {
    fn total(&self) -> f64 {
        self.reconstruction + self.perceptual + self.codebook + self.commitment + self.entropy
    }
}

/// The codebook tensor's checkpoint name (it is a parameter like any other).
const CODEBOOK: &str = "quantize.embedding.weight";

impl VqganTrainer {
    /// Build the training graphs for a `[in_channels, h, w]` input on `gpu`
    /// (which MUST have been created with [`TRAIN_KERNELS`]), the encoder's and
    /// generator's block internals kept or recomputed (`activations`), for the
    /// objective [`Objective::reconstruction_only`]: L1 and the VQ terms.
    pub fn new(cfg: VqganConfig, tensors: &Tensors, h: u32, w: u32, gpu: Gpu, activations: Activations) -> VqganTrainer {
        Self::with_objective(cfg, Objective::reconstruction_only(), None, tensors, h, w, gpu, activations)
            .expect("the reconstruction-only objective needs nothing that can be missing")
    }

    /// [`Self::new`] for any [`Objective`]. `lpips_weights` is required exactly when `perceptual_weight` is nonzero
    /// ([`Self::lpips_weights_from_store`] reads the released ones; tests pass random ones). The perceptual network
    /// and the discriminator run on handles of `gpu`'s device of their own, built here.
    ///
    /// Errors, naming what is wrong: an invalid objective, a nonzero perceptual weight without LPIPS weights, a
    /// perceptual or adversarial term on an image that is not 3-channel or smaller than the metric's minimum side.
    #[allow(clippy::too_many_arguments)]
    pub fn with_objective(
        cfg: VqganConfig,
        obj: Objective,
        lpips_weights: Option<&lpips::import::Tensors>,
        tensors: &Tensors,
        h: u32,
        w: u32,
        gpu: Gpu,
        activations: Activations,
    ) -> Result<VqganTrainer, String> {
        obj.validate()?;
        if obj.perceptual_weight > 0.0 {
            if lpips_weights.is_none() {
                return Err(format!(
                    "vqgan: perceptual_weight {} needs the LPIPS v0.1 {} weights (the trunk's torchvision checkpoint and LPIPS's heads for it), \
                     and none were given; `VqganTrainer::lpips_weights_from_store({:?})` reads them from the model store",
                    obj.perceptual_weight,
                    obj.perceptual_net.name(),
                    obj.perceptual_net
                ));
            }
            let min = obj.perceptual_net.min_side();
            if cfg.out_channels != 3 || h < min || w < min {
                return Err(format!(
                    "vqgan: the perceptual term ({} trunk) compares 3-channel images of at least {min}x{min}, this is {} channels at {h}x{w}",
                    obj.perceptual_net.name(),
                    cfg.out_channels
                ));
            }
        }
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

        let (w_cb, w_com) = term_weights(&cfg);
        let codebook = {
            let (_, data) =
                tensors.get(CODEBOOK).unwrap_or_else(|| panic!("vqgan: missing {CODEBOOK}"));
            gpu.storage_init(CODEBOOK, data)
        };
        // The all-ones gain `l2norm_scale` and its backward take; allocated
        // only when the codebook search is L2-normalised.
        let l2_ones = cfg.codebook_l2_norm.then(|| gpu.storage_init("quantize.l2norm.ones", &vec![1.0f32; emb as usize]));
        // `cb_q` is the codebook the quantiser READS: the parameter itself, or
        // its row-normalised image, recomputed from the live parameter by
        // `cb_norm` at the head of every graph that reads it.
        let (cb_q, cb_norm) = match &l2_ones {
            Some(ones) => {
                let cb_n = gpu.storage((ncode * emb) as u64);
                let step = l2norm_step(&gpu, ncode, emb, &codebook, ones, &cb_n);
                (cb_n, vec![step])
            }
            None => (codebook.clone(), Vec::new()),
        };
        let img_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let target = gpu.storage(n_out);
        let mask = gpu.storage_init("recon.mask", &vec![1.0f32; n_out as usize]);
        let idx_in = gpu.storage(t as u64);

        // ---- encoder (train mode: SSA, direct lowerings, taped) ------------
        let mut be = Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        be.set_train(true);
        be.set_activations(activations);
        let z = run_blocks(&mut be, "encoder", &cfg.encoder_blocks(), 0, h, w, &img_in).0;
        let z_flat = be.nchw_to_rows(emb, t, &z);
        let enc = be.trace();
        // `query` is what the quantiser sees: the encoder rows, or their L2
        // normalisation. The normalisation is recorded OUTSIDE the builder's
        // tape (its reverse is the explicit `l2norm_scale_dx` below), so the
        // encoder trace still ends at `z_flat`.
        let query = match &l2_ones {
            Some(ones) => {
                let zn = gpu.storage(te);
                be.push_step(l2norm_step(&gpu, t, emb, &z_flat, ones, &zn));
                zn
            }
            None => z_flat.clone(),
        };
        let (enc_steps, _) = be.finish();

        // ---- the frozen assignment, through the ONE `vq_argmin` site -------
        let mut ba = Builder::new(&gpu, &empty, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        let packed = record_assign(&mut ba, &cb_q, t, ncode, emb, &query);
        let (record_steps, _) = ba.finish();
        let mut assign_steps = cb_norm.clone();
        assign_steps.extend(record_steps);

        // ---- the codebook gather, through the ONE `embed` site -------------
        // TWO gathers, deliberately: `q_rows` is LIVE (it moves when the
        // codebook moves — the codebook term differentiates it), while `q0` is
        // the DETACHED copy latched at the linearisation point. `sg[q]` in the
        // commitment term is exactly `q0`.
        let mut bl = Builder::new(&gpu, &empty, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        let q_rows = record_lookup(&mut bl, &cb_q, t, emb, &idx_in);
        let (lookup_steps, _) = bl.finish();
        let mut bl0 = Builder::new(&gpu, &empty, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        let q0 = record_lookup(&mut bl0, &cb_q, t, emb, &idx_in);
        let (q0_steps, _) = bl0.finish();

        // ---- straight-through: z_q_st = z + sub0, sub0 = (q0 - z0) ---------
        // `sub0`, `z0` and `q0` are LATCHED (see `latch_assignment`). Freezing
        // them is what `.detach()` means as a finite-difference-checkable
        // object: at the latch point `z == z0` and `q == q0`, so the forward
        // VALUE is the quantised one (`z_q_st == q`, to fp32 rounding), while
        // the derivative w.r.t. `z` is the identity — the straight-through
        // estimator. `crates/mimi`'s `ConvReconAE::surrogate_loss` is
        // the same construction on the host, with its latch `q0 - z0` in the
        // role of `sub0`.
        let z0 = gpu.storage(te);
        let sub0 = gpu.storage(te);
        let zq_st = gpu.storage(te);
        let mut latch_steps = q0_steps;
        latch_steps.push(gpu.step(K_AXPY, &[&z0, &query], &[te as u32, f(1.0)], te as u32));
        latch_steps.push(gpu.step(K_AXPY, &[&sub0, &q0], &[te as u32, f(1.0)], te as u32));
        latch_steps.push(gpu.step(K_AXPY, &[&sub0, &z0], &[te as u32, f(-1.0)], te as u32));
        let ste_steps =
            vec![gpu.step(K_ADD2, &[&query, &sub0, &zq_st], &[te as u32], te as u32)];

        // ---- generator -----------------------------------------------------
        let mut bg = Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), false);
        bg.set_train(true);
        bg.set_activations(activations);
        let z_q = bg.rows_to_nchw(emb, t, &zq_st);
        let (out, (oh, ow)) =
            run_blocks(&mut bg, "generator", &cfg.generator_blocks(), 0, lh, lw, &z_q);
        assert_eq!((oh, ow), (h, w), "vqgan: generator output {oh}x{ow} != input {h}x{w}");
        let gen = bg.trace();
        let (gen_steps, _) = bg.finish();

        // ---- per-element loss values (the host sums them, as every brain
        //      loss kernel expects — see mse_value.wgsl) --------------------
        let rec_val = gpu.storage(n_out);
        let cb_val = gpu.storage(te);
        let com_val = gpu.storage(te);
        let rec_step = match obj.reconstruction {
            ReconstructionLoss::L1 => gpu.step(K_MASKED_L1, &[&out, &target, &mask, &rec_val], &[n_out as u32], n_out as u32),
            // `mse_value` already divides by `n`: its host sum IS the mean.
            ReconstructionLoss::L2 => gpu.step(K_MSE_VALUE, &[&out, &target, &rec_val], &[n_out as u32], n_out as u32),
        };
        let loss_steps = vec![
            rec_step,
            // L_cb  = ||sg[z] - q||^2 / n : z frozen (`z0`), q live. Weighted
            // on the host, where the reduction happens (`term_weights`).
            gpu.step(K_MSE_VALUE, &[&q_rows, &z0, &cb_val], &[te as u32], te as u32),
            // L_com = ||z - sg[q]||^2 / n : q frozen (`q0`), z live.
            gpu.step(K_MSE_VALUE, &[&query, &q0, &com_val], &[te as u32], te as u32),
        ];

        let mut fwd_steps = enc_steps.clone();
        fwd_steps.extend(cb_norm);
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

        // The reconstruction gradient ASSIGNS `d_out`; every other image-space
        // term is added into it afterwards (`submit_seed`).
        let seed_steps = vec![match obj.reconstruction {
            ReconstructionLoss::L1 => gpu.step(
                K_MASKED_L1_GRAD,
                &[&out, &target, &mask, &d_out],
                &[n_out as u32, f(1.0 / n_out as f32)],
                n_out as u32,
            ),
            ReconstructionLoss::L2 => gpu.step(K_MSE_GRAD, &[&out, &target, &d_out], &[n_out as u32], n_out as u32),
        }];
        let mut bwd_steps = gen_rev.steps;
        // Straight-through: the decoder gradient crosses the quantiser
        // UNCHANGED. `z_q_st = z + sub0` with `sub0` constant, so this identity
        // IS the exact adjoint of the emitted forward — the estimator's bias
        // lives entirely in the choice to freeze `sub0`, which is the
        // stop-gradient itself.
        bwd_steps.push(gpu.step(K_AXPY, &[&d_z, &d_zq_st], &[te as u32, f(1.0)], te as u32));
        // Codebook term dL_cb/dq = w_cb * 2(q - z0)/n, scattered into the
        // assigned codebook rows ONLY. `sg[z]` is why `z0` is bound and why
        // `d_z` gets nothing from this term.
        bwd_steps.push(gpu.step(K_MSE_GRAD, &[&q_rows, &z0, &dq_cb], &[te as u32], te as u32));
        bwd_steps.push(gpu.step(K_AXPY, &[&dq_cb_b, &dq_cb], &[te as u32, f(w_cb)], te as u32));
        // The entropy term differentiates BOTH the rows and the codebook, with nothing detached. Its forward joins the
        // forward steps; its two gradient kernels (assigning, already scaled by the ratio) lead the reverse.
        let entropy = (obj.entropy_loss_ratio > 0.0).then(|| {
            let (logp, c, meanc) = (gpu.storage((t * ncode) as u64), gpu.storage(ncode as u64), gpu.storage(t as u64));
            let (h_tok, term) = (gpu.storage(t as u64), gpu.storage(ncode as u64));
            let (d_rows, d_codebook) = (gpu.storage(te), gpu.storage((ncode * emb) as u64));
            let tau = f(obj.entropy_temperature);
            fwd_steps.push(gpu.step(K_ENTROPY_LOGP, &[&query, &cb_q, &logp, &h_tok], &[t, ncode, emb, tau], t));
            fwd_steps.push(gpu.step(K_ENTROPY_COLS, &[&logp, &c, &term], &[t, ncode], ncode));
            let grad = [f(obj.entropy_loss_ratio)];
            let p = [t, ncode, emb, tau, grad[0]];
            bwd_steps.push(gpu.step(K_ENTROPY_DZ, &[&query, &cb_q, &logp, &c, &h_tok, &meanc, &d_rows], &p, t * emb));
            bwd_steps.push(gpu.step(K_ENTROPY_DCB, &[&query, &cb_q, &logp, &c, &h_tok, &meanc, &d_codebook], &p, ncode * emb));
            EntropyBuffers { h: h_tok, term, d_rows, d_codebook }
        });
        // The scatter lands on the codebook the quantiser read. Normalised,
        // that is `cb_q`, whose gradient is taken back through the
        // normalisation onto the parameter; raw, it IS the parameter.
        let mut bwd_clears = Vec::new();
        match &l2_ones {
            Some(ones) => {
                let d_cb_q = gpu.storage((ncode * emb) as u64);
                let d_cb_raw = gpu.storage((ncode * emb) as u64);
                bwd_steps.push(gpu.step(K_EMB_BWD, &[&idx_in, &dq_cb_b, &d_cb_q], &[t, emb, ncode], ncode * emb));
                if let Some(e) = &entropy {
                    bwd_steps.push(gpu.step(K_AXPY, &[&d_cb_q, &e.d_codebook], &[ncode * emb, f(1.0)], ncode * emb));
                }
                bwd_steps.push(gpu.step(
                    K_L2NORM_DX,
                    &[&codebook, ones, &d_cb_q, &d_cb_raw],
                    &[ncode, emb, f(L2_EPS)],
                    ncode * emb,
                ));
                bwd_steps.push(gpu.step(K_AXPY, &[&codebook_g, &d_cb_raw], &[ncode * emb, f(1.0)], ncode * emb));
                bwd_clears.push(d_cb_q);
            }
            None => {
                bwd_steps.push(gpu.step(K_EMB_BWD, &[&idx_in, &dq_cb_b, &codebook_g], &[t, emb, ncode], ncode * emb));
                if let Some(e) = &entropy {
                    bwd_steps.push(gpu.step(K_AXPY, &[&codebook_g, &e.d_codebook], &[ncode * emb, f(1.0)], ncode * emb));
                }
            }
        }
        // Commitment term dL_com/dz = w_com * 2(z - q0)/n, into the ENCODER
        // only. `sg[q]` is why `q0` is bound and why the codebook gets nothing
        // here.
        bwd_steps.push(gpu.step(K_MSE_GRAD, &[&query, &q0, &dz_com], &[te as u32], te as u32));
        bwd_steps.push(gpu.step(K_AXPY, &[&d_z, &dz_com], &[te as u32, f(w_com)], te as u32));
        if let Some(e) = &entropy {
            bwd_steps.push(gpu.step(K_AXPY, &[&d_z, &e.d_rows], &[te as u32, f(1.0)], te as u32));
        }
        // `d_z` is the gradient at the quantiser's query; through the
        // normalisation (when there is one) it becomes the encoder's.
        let d_enc = match &l2_ones {
            Some(ones) => {
                let d = gpu.storage(te);
                bwd_steps.push(gpu.step(K_L2NORM_DX, &[&z_flat, ones, &d_z, &d], &[t, emb, f(L2_EPS)], te as u32));
                d
            }
            None => d_z.clone(),
        };
        let enc_rev: Reverse = enc.backward(&gpu, ids, &enc_g, &z_flat, &d_enc);
        bwd_steps.extend(enc_rev.steps);

        bwd_clears.push(d_z);
        bwd_clears.push(dq_cb_b);

        let mut names: Vec<(String, u64)> = enc.params().to_vec();
        names.extend(gen.params().iter().cloned());
        names.push((CODEBOOK.to_string(), (ncode * emb) as u64));

        // ---- the generator's AdamW, over the buffers the graphs own -------
        let adopted = names
            .iter()
            .map(|(name, len)| {
                let (w, g) = if name == CODEBOOK {
                    (codebook.clone(), codebook_g.clone())
                } else if name.starts_with("encoder.") {
                    (enc.weight(name).clone(), enc_g.g(name).clone())
                } else {
                    (gen.weight(name).clone(), gen_g.g(name).clone())
                };
                (name.clone(), *len as usize, w, g)
            })
            .collect();
        let gen_ps = ParamStore::adopt(&gpu, adopted);
        let kernel = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("vqgan: kernel `{name}` is not registered; build the device with TRAIN_KERNELS"));
        let gen_opt = Optim::new(kernel("adamw"), kernel("gradnorm_sq"), kernel("grad_scale"), kernel("clip_coef"), kernel("grad_scale_buf"));

        // ---- the perceptual network ---------------------------------------
        let perceptual = match (obj.perceptual_weight > 0.0, lpips_weights) {
            (true, Some(weights)) => {
                let lp = Lpips::with_trunk(gpu.new_like(lpips::GRAD_PIPELINES), obj.perceptual_net, weights)?;
                let (lpips_gpu, pair) = (lp.gpu().share(), lp.gpu().storage(2 * n_out));
                Some(Perceptual { lpips: RefCell::new(lp), gpu: lpips_gpu, pair, d_image: gpu.storage(n_out) })
            }
            _ => None,
        };

        // ---- the discriminator and, when the recipe adapts its weight, the reference layer -------------
        let adversarial = obj.adversarial().then(|| {
            let seed = obj.disc_seed.unwrap_or_else(data::rng::random_seed);
            eprintln!("vqgan: discriminator weight seed {seed}");
            let disc_cfg = PatchGanConfig { in_channels: cfg.out_channels, ndf: obj.disc_ndf, n_layers: obj.disc_layers };
            Adversarial { d: PatchGan::new(gpu.new_like(patchgan::PIPELINES), disc_cfg, h, w, seed), updates: Cell::new(0), d_adv: gpu.storage(n_out) }
        });
        let (last_weight, tail) = match adversarial.as_ref().filter(|_| obj.adaptive_weight) {
            Some(adv) => {
                let (tail_trace, last_weight) = gen.last_conv();
                let (nll_grads, adv_grads) = (tail_trace.alloc_grads(&gpu), tail_trace.alloc_grads(&gpu));
                let nll_steps = tail_trace.backward(&gpu, ids, &nll_grads, &out, &d_out).steps;
                let adv_steps = tail_trace.backward(&gpu, ids, &adv_grads, &out, &adv.d_adv).steps;
                (last_weight, Some(TailReverse { nll_grads, adv_grads, nll_steps, adv_steps }))
            }
            None => (String::new(), None),
        };

        let mut bwd_all = seed_steps.clone();
        bwd_all.extend(bwd_steps.iter().cloned());
        Ok(VqganTrainer {
            gpu,
            cfg,
            hw: (h, w),
            lhw: (lh, lw),
            enc_steps,
            assign_steps,
            latch_steps,
            latch_clears: vec![z0, sub0],
            fwd_steps,
            bwd_steps: bwd_all,
            bwd_clears,
            img_in,
            target,
            packed,
            idx_in,
            zq_st,
            q0,
            out,
            rec_val,
            cb_val,
            com_val,
            query,
            cb_q,
            enc,
            gen,
            enc_g,
            gen_g,
            codebook,
            codebook_g,
            names,
            obj,
            d_out,
            seed_steps,
            rest_steps: bwd_steps,
            gen_ps,
            gen_opt,
            gen_updates: Cell::new(0),
            last_weight,
            tail,
            perceptual,
            entropy,
            adversarial,
            last: RefCell::new(Terms::default()),
        })
    }

    /// The forward dispatches of one training step, in submit order.
    ///
    /// Exposed for the PROFILER (`vqgan_bench`), not for driving: a caller that
    /// submits these itself skips the latch and the clears and would get a
    /// silently wrong gradient. A per-kernel-kind table is wanted before
    /// anyone optimises, and until this existed there
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

    /// The LPIPS weights of `net` in the model store (`lpips::spec::resolve_for`), for [`Self::with_objective`]
    /// (pass [`Objective::perceptual_net`]); the error names the file that is missing.
    pub fn lpips_weights_from_store(net: lpips::Trunk) -> Result<lpips::import::Tensors, String> {
        let (trunk, heads) = lpips::spec::resolve_for(net)?;
        lpips::import::read(net, &trunk, &heads)
    }

    /// The objective this trainer was built for.
    pub fn objective(&self) -> &Objective {
        &self.obj
    }

    /// What the quantiser reads in the last [`Self::loss`]: the query rows `[tokens, emb_dim]` (the encoder's, or their
    /// L2 normalisation) and the codebook `[codebook_size, emb_dim]` (the parameter, or its normalised image). The
    /// operands of the codebook terms and of [`crate::entropy`].
    pub fn quantizer_inputs(&self) -> (Vec<f32>, Vec<f32>) {
        let (emb, ncode) = (self.cfg.emb_dim as usize, self.cfg.codebook_size as usize);
        let t = (self.lhw.0 * self.lhw.1) as usize;
        (self.gpu.read(&self.query, t * emb), self.gpu.read(&self.cb_q, ncode * emb))
    }

    /// One forward pass over the installed batch; returns the scalar
    ///
    /// ```text
    /// L = L_rec(G(z + sub0), target) + perceptual_weight * LPIPS(G(z + sub0), target)
    ///     + w_cb*||sg[z] - q||^2/n  +  w_com*||z - sg[q]||^2/n  +  entropy_loss_ratio * L_ent
    /// ```
    ///
    /// with `beta` on the term [`VqganConfig::beta_on`] names and 1 on the
    /// other (see the module docs). The adversarial term is not in it: it is a
    /// function of the discriminator too, and [`Self::train_step`] adds it.
    /// The terms as they entered are kept for [`Self::train_step`]'s report.
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
        let rec = match self.obj.reconstruction {
            ReconstructionLoss::L1 => sum(&self.rec_val, n_out) / n_out as f64,
            ReconstructionLoss::L2 => sum(&self.rec_val, n_out),
        };
        let cb = sum(&self.cb_val, te);
        let com = sum(&self.com_val, te);
        let (w_cb, w_com) = term_weights(&self.cfg);
        let terms = Terms {
            reconstruction: rec,
            perceptual: self.perceptual_term(),
            codebook: w_cb as f64 * cb,
            commitment: w_com as f64 * com,
            entropy: self.entropy_term(),
        };
        *self.last.borrow_mut() = terms;
        terms.total() as f32
    }

    /// `perceptual_weight * LPIPS(out, target)` of the last forward, and `d_image` set to its gradient wrt `out`
    /// (unweighted); zero and untouched without a perceptual term.
    fn perceptual_term(&self) -> f64 {
        let Some(p) = &self.perceptual else { return 0.0 };
        let bytes = 4 * (self.cfg.out_channels * self.hw.0 * self.hw.1) as u64;
        let mut lp = p.lpips.borrow_mut();
        self.gpu.copy_to(&self.out, 0, &p.gpu, &p.pair, 0, bytes);
        self.gpu.copy_to(&self.target, 0, &p.gpu, &p.pair, bytes, bytes);
        let (dist, grad) = lp.distance_grad(&p.pair, self.hw.1, self.hw.0).unwrap_or_else(|e| panic!("vqgan: {e}"));
        // The gradient of the pair; image 0 (the reconstruction) is its first half.
        p.gpu.copy_to(grad, 0, &self.gpu, &p.d_image, 0, bytes);
        self.obj.perceptual_weight as f64 * dist.total
    }

    /// `entropy_loss_ratio * L_ent` of the last forward: the mean of the per-token entropies minus the sum of the
    /// per-code average-entropy terms, both reduced on the host from the device's per-element values; zero without
    /// an entropy term. The gradients are the reverse's kernels, not computed here.
    fn entropy_term(&self) -> f64 {
        let Some(e) = &self.entropy else { return 0.0 };
        let (t, ncode) = ((self.lhw.0 * self.lhw.1) as usize, self.cfg.codebook_size as usize);
        let total = |b: &DeviceBuffer, n: usize| -> f64 { self.gpu.read(b, n).iter().map(|&v| v as f64).sum() };
        self.obj.entropy_loss_ratio as f64 * (total(&e.h, t) / t as f64 - total(&e.term, ncode))
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

    /// `d_out` = the gradient of the reconstruction and perceptual terms wrt the output image.
    fn submit_seed(&self) {
        self.gpu.submit(&[], &self.seed_steps);
        if let Some(p) = &self.perceptual {
            let n = (self.cfg.out_channels * self.hw.0 * self.hw.1) as u32;
            let add = self.gpu.step(K_AXPY, &[&self.d_out, &p.d_image], &[n, f(self.obj.perceptual_weight)], n);
            self.gpu.submit(&[], &[add]);
        }
    }

    /// The reverse from the seed in `d_out` through the generator, the quantiser and the encoder.
    fn submit_rest(&self) {
        let clears: Vec<&DeviceBuffer> = self.bwd_clears.iter().collect();
        self.gpu.submit(&clears, &self.rest_steps);
    }

    /// One reverse pass of [`Self::loss`]'s objective. Requires a preceding [`Self::loss`] (the forward
    /// buffers ARE the activation cache).
    pub fn backward(&self) {
        self.submit_seed();
        self.submit_rest();
    }

    /// One whole training step at global step `step`: re-latch the assignment, the generator's forward, the
    /// reverse of the full objective (the adversarial term from `disc_start` on, weighted by the adaptive `lambda`),
    /// one AdamW update of the encoder, generator and codebook, and then, when the discriminator is live, one of its
    /// own ([`Self::discriminator_step`]). Needs the batch installed ([`Self::set_batch`]).
    pub fn train_step(&self, step: u32, opts: &StepOptions) -> Losses {
        self.latch_assignment();
        self.zero_grads();
        self.loss();
        let terms = *self.last.borrow();
        let live = self.obj.adversarial_at(step);
        let (mut adversarial, mut adaptive) = (None, None);
        self.submit_seed();
        if live {
            let (g_loss, lambda) = self.adversarial_seed();
            adversarial = Some(self.obj.disc_weight as f64 * lambda.unwrap_or(1.0) as f64 * g_loss);
            adaptive = lambda;
        }
        self.submit_rest();
        self.adamw_step(opts);
        let discriminator = live.then(|| self.discriminator_step(opts));
        Losses {
            reconstruction: terms.reconstruction,
            perceptual: terms.perceptual,
            codebook: terms.codebook,
            commitment: terms.commitment,
            entropy: terms.entropy,
            adversarial,
            adaptive_weight: adaptive,
            discriminator,
        }
    }

    /// The adversarial term of one step, after [`Self::submit_seed`]: the discriminator's verdict on the
    /// reconstruction, `-mean D(x_hat)` differentiated to the image, and `d_out += disc_weight * lambda *
    /// d(-mean D)/dx_hat`, where `lambda` is taming's adaptive weight from the last layer's two gradient norms
    /// ([`Objective::adaptive_weight`]) or 1. Returns the raw generator loss and the `lambda` applied, if adaptive.
    fn adversarial_seed(&self) -> (f64, Option<f32>) {
        let adv = self.adversarial.as_ref().expect("live implies a discriminator");
        let n = (self.cfg.out_channels * self.hw.0 * self.hw.1) as u32;
        adv.d.stage(true, &self.gpu, &self.out);
        let (g_loss, d_logits) = patchgan::generator_loss(&adv.d.forward(true));
        adv.d.zero_grads();
        adv.d.backward(true, &d_logits, true);
        adv.d.gpu().copy_to(adv.d.input_grad(), 0, &self.gpu, &adv.d_adv, 0, 4 * n as u64);

        let lambda = self.tail.as_ref().map(|tail| {
            // The last layer's weight gradient of the nll seed (now in `d_out`) and of the adversarial seed.
            let (nll, adv_g) = (tail.nll_grads.all(), tail.adv_grads.all());
            self.gpu.submit(&nll, &tail.nll_steps);
            self.gpu.submit(&adv_g, &tail.adv_steps);
            let len = self.numel(&self.last_weight);
            let norm = |grads: &Grads| -> f64 { self.gpu.read(grads.g(&self.last_weight), len).iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() };
            (norm(&tail.nll_grads) / (norm(&tail.adv_grads) + 1e-4)).clamp(0.0, self.obj.adaptive_weight_max as f64) as f32
        });
        let add = self.gpu.step(K_AXPY, &[&self.d_out, &adv.d_adv], &[n, f(self.obj.disc_weight * lambda.unwrap_or(1.0))], n);
        self.gpu.submit(&[], &[add]);
        (g_loss, lambda)
    }

    /// One AdamW step of the generator's parameters from the gradients the last backward left.
    fn adamw_step(&self, opts: &StepOptions) {
        let t = self.gen_updates.get() + 1;
        self.gen_updates.set(t);
        self.gen_opt.step(&self.gpu, &self.gen_ps, t, opts.lr_generator, opts.weight_decay, opts.adam, opts.clip, 1.0);
    }

    /// One update of the discriminator alone on the hinge loss, `real` the installed target and `fake` the last
    /// forward's reconstruction (what the reference calls `recons.detach()`): returns the loss as optimised
    /// ([`Objective::scale_disc_loss`]). The generator does not
    /// move, so repeated calls on one batch are the discriminator training against a fixed pair. Needs a discriminator
    /// ([`Objective::disc_weight`] above zero) and a preceding [`Self::loss`].
    pub fn discriminator_step(&self, opts: &StepOptions) -> f64 {
        let adv = self.adversarial.as_ref().expect("vqgan: discriminator_step needs an objective with disc_weight > 0");
        adv.d.stage(false, &self.gpu, &self.target);
        adv.d.stage(true, &self.gpu, &self.out);
        let (real, fake) = (adv.d.forward(false), adv.d.forward(true));
        let hinge = patchgan::hinge_d_loss(&real, &fake);
        // LlamaGen optimises `disc_weight * hinge`: the logit gradients and the reported loss carry the factor.
        let scale = if self.obj.scale_disc_loss { self.obj.disc_weight } else { 1.0 };
        let scaled = |g: &[f32]| -> Vec<f32> { g.iter().map(|v| v * scale).collect() };
        adv.d.zero_grads();
        adv.d.backward(false, &scaled(&hinge.d_real), false);
        adv.d.backward(true, &scaled(&hinge.d_fake), false);
        let t = adv.updates.get() + 1;
        adv.updates.set(t);
        adv.d.adamw_step(t, opts.lr_discriminator, opts.weight_decay, opts.adam, None);
        scale as f64 * hinge.loss
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
            dec_res_blocks: 1,
            attn_resolutions: vec![4],
            img_size: 8,
            codebook_size: 6,
            emb_dim: 4,
            z_channels: None,
            head_act: false,
            codebook_l2_norm: false,
            beta: 0.25,
            beta_on: BetaTerm::Codebook,
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
        let m = VqganTrainer::new(cfg.clone(), &tensors, cfg.img_size, cfg.img_size, gpu, Activations::Keep);
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

    /// [`tiny`] with every LlamaGen knob on: decoder resnet asymmetry, head
    /// SiLU, a `quant_conv`/`post_quant_conv` bridge from a wider latent, the
    /// L2-normalised codebook search, and `beta` on the commitment term.
    fn tiny_llamagen() -> VqganConfig {
        VqganConfig {
            dec_res_blocks: 2,
            z_channels: Some(6),
            head_act: true,
            codebook_l2_norm: true,
            beta_on: BetaTerm::Commitment,
            ..tiny()
        }
    }

    /// The training graph's forward IS the inference graph's: same code
    /// assignment, same reconstruction (the straight-through latent equals
    /// the gathered codes at the latch). A finite-difference check cannot
    /// see a forward that is missing a feature - it only checks the backward
    /// against whatever forward was emitted - so this pins the forward.
    #[test]
    fn llamagen_trainer_forward_matches_the_inference_graph() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = tiny_llamagen();
        let tensors = fixture(&cfg);
        let s = cfg.img_size;
        let n_in = (cfg.in_channels * s * s) as usize;
        let n_out = (cfg.out_channels * s * s) as usize;
        let image = fill(n_in, 1, 1.0);

        let inf = crate::Vqgan::new(cfg.clone(), &tensors, s, s, gpu_core::testgpu::dev(&crate::KERNELS), false);
        let want = inf.reconstruct(&image);

        let m = VqganTrainer::new(cfg.clone(), &tensors, s, s, gpu_core::testgpu::dev(TRAIN_PIPELINES), Activations::Keep);
        m.set_batch(&image, &fill(n_out, 2, 1.0));
        assert_eq!(m.codes(), want.indices, "training and inference graphs assign different codes");
        m.loss();
        let worst = m.output().iter().zip(&want.image).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "training forward differs from the inference forward by {worst:e}");
    }

    /// `beta` weights the term [`VqganConfig::beta_on`] names. The codebook's
    /// ONLY gradient is the codebook term (the straight-through path routes
    /// the decoder's gradient to the encoder), so moving `beta` off that term
    /// scales the codebook gradient by exactly `1/beta`.
    #[test]
    fn beta_weights_the_term_its_config_names() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let s = tiny().img_size;
        let codebook_grad = |beta_on| {
            let cfg = VqganConfig { beta_on, ..tiny_llamagen() };
            let tensors = fixture(&cfg);
            let m = VqganTrainer::new(cfg.clone(), &tensors, s, s, gpu_core::testgpu::dev(TRAIN_PIPELINES), Activations::Keep);
            let n_in = (cfg.in_channels * s * s) as usize;
            let n_out = (cfg.out_channels * s * s) as usize;
            m.set_batch(&fill(n_in, 1, 1.0), &fill(n_out, 2, 1.0));
            m.zero_grads();
            m.loss();
            m.backward();
            m.read_grad(CODEBOOK)
        };
        let on_codebook = codebook_grad(BetaTerm::Codebook);
        let on_commitment = codebook_grad(BetaTerm::Commitment);
        assert!(on_commitment.iter().any(|&g| g.abs() > 1e-6), "the codebook got no gradient");
        for (a, b) in on_codebook.iter().zip(&on_commitment) {
            assert!((a - 0.25 * b).abs() <= 1e-6 + 1e-4 * b.abs(), "codebook grad {a} != beta * {b}");
        }
    }
}
