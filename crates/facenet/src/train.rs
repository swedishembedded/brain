// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ArcFace training graph: the IResNet embedding backbone plus an
//! additive-angular-margin softmax head, with a hand-written device backward.
//!
//! # Scope — read this before trusting a number out of here
//!
//! This trains the **embedding backbone only**. SCRFD detection and the 5-point
//! similarity alignment are *preprocessing*: the detector selects a crop and the
//! Umeyama warp resamples it, and neither is differentiated here. That is not an
//! omission to be filled in later — the reference ArcFace recipe trains on
//! pre-aligned 112×112 crops exactly this way, so the detector carries no
//! recognition gradient at all. [`ArcFaceTrainer`]'s input is the aligned blob.
//!
//! # Folded, like the release
//!
//! `glintr100.onnx` ships with its BatchNorms folded into the convolutions, so
//! the trainable tensors here are the **folded** ones — every conv is
//! `Norm::None` with a learned bias, and the conv weight/bias ARE the parameters
//! that move. Training unfolded (conv → BN → PReLU, the torch module) would need
//! per-conv `gamma/beta/running_*` that are not in the file and would give a
//! parameterisation the checkpoint cannot round-trip. The three BatchNorms that
//! survive as real nodes (each block's entry `bn1`, the final `bn2`, and
//! `features` after `fc`) train as real BatchNorms.
//!
//! # BatchNorm mode
//!
//! Those three run in **TRAIN mode** (batch statistics), which is what makes
//! `gamma`/`beta` carry a gradient at all: `vision::bn::BatchNorm::backward` is
//! the train-mode formula (`bn_dstats` → `bn_dx`) and reads the `mvg` packing
//! that only a train-mode forward writes. Eval-mode BN would be a frozen affine
//! with `mvg` never populated — its backward would read stale garbage rather
//! than fail. The running-stat EMA is left OFF ([`BatchNorm::set_update_running`]
//! defaults to false) because it mutates `run_mean`/`run_var` during the forward
//! and so breaks the forward determinism a finite-difference check needs;
//! `run_mean`/`run_var` are therefore registered `Role::Frozen` (they carry no
//! train-mode gradient) and a real training loop must turn the EMA back on.
//!
//! # SSA
//!
//! Nothing is recomputed and nothing is overwritten: every `vision` block owns
//! its own output buffer for the object's lifetime (`Ctx::act` does not pool),
//! so the forward's activations ARE the backward's cache. The per-block backward
//! temporaries live in [`BlockBwd`], one set per block, for the same reason.
//!
//! # The margin head
//!
//! ```text
//! bn2 -> flatten -> fc(+bias) -> features(BN) -> e
//!   e_hat = e / ||e||           (l2norm_scale, gain pinned to 1)
//!   W_hat = W / ||W||  rowwise  (l2norm_scale, gain pinned to 1)
//!   cos   = e_hat · W_hatᵀ      (matmul)
//!   logit = s·cos, except the target column = s·cos(θ+m)   (arcface_margin)
//!   L     = mean cross-entropy over the batch              (ce_value/ce_grad)
//! ```
//!
//! The margin head is **not in the released checkpoint** — insightface ships the
//! embedding only. `head.weight` is therefore
//! initialised, not imported, exactly as it would be when fine-tuning onto a new
//! identity set.

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use vision::{Act, BatchNorm, BnNames, Conv, ConvNames, Ctx, PReLU, Shape};

use crate::config::ArcFaceConfig;
use crate::model::{folded, ids, kernel, ResBlock};

/// `l2norm_scale`'s epsilon. Small enough to be invisible against `‖e‖² ≈ E`,
/// large enough that a hypothetical all-zero row is finite rather than NaN.
const L2_EPS: f32 = 1e-8;

/// The trainable head + optimisation shape around [`ArcFaceConfig`].
#[derive(Clone, Debug, PartialEq)]
pub struct ArcFaceTrainConfig {
    pub arc: ArcFaceConfig,
    /// Batch size. Must be > 1: the three BatchNorms take batch statistics.
    pub batch: u32,
    /// Identity count of the margin head (`head.weight` is `[classes, E]`).
    pub classes: u32,
    /// ArcFace's logit scale `s`.
    pub scale: f32,
    /// ArcFace's additive angular margin `m`, in RADIANS.
    pub margin: f32,
}

impl ArcFaceTrainConfig {
    /// The paper's / insightface's hyper-parameters on the real IResNet-100:
    /// `s = 64`, `m = 0.5` rad (28.6°).
    pub fn insightface(batch: u32, classes: u32) -> ArcFaceTrainConfig {
        ArcFaceTrainConfig {
            arc: ArcFaceConfig::iresnet100(),
            batch,
            classes,
            scale: 64.0,
            margin: 0.5,
        }
    }

    /// A deliberately tiny IResNet for the gradient check: 5 blocks at a 32×32
    /// input, embedding 8, 5 identities.
    ///
    /// `layers` is `[2, 1, 1, 1]`, **not** `[1, 1, 1, 1]`, and the extra block is
    /// the point: a stage's FIRST block always strides, so it always has a
    /// `downsample` conv on the shortcut, while every LATER block is stride 1 at
    /// constant width and takes the IDENTITY shortcut — a different arm of both
    /// [`ResBlock::forward`] and [`block_backward`] (`d_skip` is `None` and the
    /// join's second operand is `d_out` itself). IResNet-100 is `[3, 13, 30, 3]`,
    /// so **45 of its 49 blocks** take the identity arm. With one block per stage
    /// that arm is never executed and the finite-difference gate cannot see it.
    ///
    /// `scale` is 8, not the paper's 64: the margin kernels are LINEAR in `s`
    /// (it multiplies the logit and multiplies the derivative), so the code path
    /// is identical, while `s = 64` over 5 classes saturates the softmax so hard
    /// that the finite difference of the mean CE is dominated by round-off. The
    /// margin `m = 0.5` is the real one — that is the part with curvature.
    pub fn tiny() -> ArcFaceTrainConfig {
        ArcFaceTrainConfig {
            arc: ArcFaceConfig {
                // 32 keeps `flatten()` = C3·(32>>4)² = C3·4 (a real spatial
                // flatten, not a degenerate 1×1) and gives `bn2` 4 spatial
                // positions per sample to take statistics over.
                image_size: 32,
                layers: [2, 1, 1, 1],
                channels: [4, 6, 8, 10],
                stem_channels: 4,
                embedding: 8,
                pre: ArcFaceConfig::iresnet100().pre,
            },
            batch: 4,
            classes: 5,
            scale: 8.0,
            margin: 0.5,
        }
    }
}

/// Per-block backward temporaries. One set per block, never shared: the block
/// backwards run back-to-back and a shared scratch would be read after the next
/// block overwrote it.
struct BlockBwd {
    /// grad wrt the block's PReLU output = `conv2`'s input grad.
    d_prelu_out: DeviceBuffer,
    /// grad wrt `bn1`'s output = `conv1`'s input grad.
    d_bn1_out: DeviceBuffer,
    /// grad wrt the block input arriving through the `bn1 → conv1 → prelu →
    /// conv2` branch.
    d_main: DeviceBuffer,
    /// grad wrt the block input arriving through the shortcut CONV, when there
    /// is one. With an identity shortcut the skip grad IS `d_out`, so no buffer.
    d_skip: Option<DeviceBuffer>,
    /// the two summed — grad wrt the block input.
    d_x: DeviceBuffer,
    /// element count of the block INPUT (= `d_main`/`d_skip`/`d_x`).
    n_in: u32,
}

impl BlockBwd {
    fn new(ctx: &Ctx, blk: &ResBlock, in_shape: Shape) -> BlockBwd {
        BlockBwd {
            d_prelu_out: ctx.act(blk.conv1.out_shape.numel()),
            d_bn1_out: ctx.act(in_shape.numel()),
            d_main: ctx.act(in_shape.numel()),
            d_skip: blk.downsample.as_ref().map(|_| ctx.act(in_shape.numel())),
            d_x: ctx.act(in_shape.numel()),
            n_in: in_shape.numel(),
        }
    }
}

/// `y = conv2(prelu(conv1(bn1(x)))) + shortcut(x)` backward.
///
/// The residual join is an `add2`, whose adjoint hands the SAME `d_out` to both
/// arms; the two input grads are then summed with another `add2` into a fresh
/// buffer (never accumulated in place — SSA).
///
/// The shortcut reads the block's **input**, not `bn1`'s output (the
/// pre-activation "v3" residual, see [`ResBlock`]'s doc). Routing the shortcut
/// gradient into `bn1`'s output instead still runs and still trains; it is
/// wrong, and only the forward's `s{n}b0_branch` golden ever showed it.
fn block_backward<'a>(
    ctx: &Ctx,
    ps: &ParamStore,
    blk: &ResBlock,
    bw: &'a BlockBwd,
    x: &DeviceBuffer,
    d_out: &DeviceBuffer,
) -> &'a DeviceBuffer {
    blk.conv2.backward(ctx, ps, blk.prelu.out(), d_out, &bw.d_prelu_out);
    // PReLU's `x` argument is the PRE-activation (conv1's output), not its own.
    blk.prelu.backward(ctx, ps, blk.conv1.out(), &bw.d_prelu_out);
    blk.conv1.backward(ctx, ps, blk.bn1.out(), blk.prelu.d_in(), &bw.d_bn1_out);
    blk.bn1.backward(ctx, ps, x, &bw.d_bn1_out, &bw.d_main);

    let skip: &DeviceBuffer = match (&blk.downsample, &bw.d_skip) {
        (Some(d), Some(ds)) => {
            d.backward(ctx, ps, x, d_out, ds);
            ds
        }
        // Identity shortcut: the branch grad passes straight through.
        _ => d_out,
    };
    let n = bw.n_in;
    let s = ctx.step(ctx.ids.need(ctx.ids.add2, "add2"), &[&bw.d_main, skip, &bw.d_x], &[n], n);
    ctx.gpu.submit(&[], &[s]);
    &bw.d_x
}

/// The trainable ArcFace: IResNet backbone + additive-angular-margin head.
pub struct ArcFaceTrainer {
    gpu: Gpu,
    cfg: ArcFaceTrainConfig,
    ps: ParamStore,
    in_shape: Shape,

    input: DeviceBuffer,
    /// `[batch]` ground-truth identities, as u32 — `ce_value`/`ce_grad`/
    /// `arcface_margin` all bind the label buffer as `array<u32>`.
    labels: DeviceBuffer,
    /// `[E]` all-ones gain pinned into `l2norm_scale`. NOT a parameter: a
    /// learnable gain in front of a softmax that is already scaled by `s` is
    /// redundant, and `l2norm_scale_dg` is deliberately never dispatched.
    ones: DeviceBuffer,

    stem_conv: Conv,
    stem_prelu: PReLU,
    blocks: Vec<ResBlock>,
    bwd: Vec<BlockBwd>,
    bn2: BatchNorm,
    fc_out: DeviceBuffer,
    features: BatchNorm,

    emb_n: DeviceBuffer,   // [B, E] row-normalised embedding
    w_n: DeviceBuffer,     // [K, E] row-normalised class centres
    cosv: DeviceBuffer,    // [B, K] cosine table
    logits: DeviceBuffer,  // [B, K] scaled, margin-applied
    ce_rows: DeviceBuffer, // [B] per-sample CE

    d_logits: DeviceBuffer,
    d_cos: DeviceBuffer,
    d_emb_n: DeviceBuffer,
    d_w_n: DeviceBuffer,
    d_head_w: DeviceBuffer,
    d_emb: DeviceBuffer,
    d_fc_out: DeviceBuffer,
    d_bn2_out: DeviceBuffer,
    d_last: DeviceBuffer,
    d_input: DeviceBuffer,

    fwd_done: Cell<bool>,
}

impl ArcFaceTrainer {
    /// Build on the default device (`Gpu::new`, honouring `--device` /
    /// `BRAIN_DEVICE`) — the `Gpt::new` / `Autoencoder::new` shape. Tests and
    /// anything sharing a device use [`Self::new_on`].
    pub fn new(cfg: ArcFaceTrainConfig, init: &dyn Fn(&str, usize) -> Vec<f32>) -> ArcFaceTrainer {
        ArcFaceTrainer::new_on(Gpu::new(crate::model::PIPELINES), cfg, init)
    }

    /// Build the graph and its ParamStore on an existing device handle.
    /// `init(name, numel)` supplies each tensor's initial values — a closure
    /// rather than a map because the parameter LIST is only known once the graph
    /// is built (widths come from the block shapes), which is the same order
    /// `ArcFace::new` establishes it in.
    pub fn new_on(
        gpu: Gpu,
        cfg: ArcFaceTrainConfig,
        init: &dyn Fn(&str, usize) -> Vec<f32>,
    ) -> ArcFaceTrainer {
        assert!(cfg.batch > 1, "ArcFace trains with train-mode BatchNorm; batch must be > 1");
        assert!(cfg.classes > 1, "a softmax head needs at least 2 identities");
        let idz = ids();
        let ctx = Ctx::new(&gpu, idz);
        let sz = cfg.arc.image_size;
        let in_shape = Shape::new(cfg.batch, 3, sz, sz);

        let stem_conv = Conv::with_names(
            &ctx,
            "stem.conv",
            ConvNames::torch_flat("stem.conv"),
            in_shape,
            folded(cfg.arc.stem_channels, 3, 1, 1, Act::None),
            false,
        );
        let stem_prelu = PReLU::new(&ctx, "stem.prelu", stem_conv.out_shape);

        let mut shape = stem_conv.out_shape;
        let mut blocks: Vec<ResBlock> = Vec::new();
        let mut bwd: Vec<BlockBwd> = Vec::new();
        for s in 0..4usize {
            for b in 0..cfg.arc.layers[s] as usize {
                let stride = if b == 0 { 2 } else { 1 };
                let blk =
                    ResBlock::new(&ctx, &format!("layer{}.{}", s + 1, b), shape, cfg.arc.channels[s], stride);
                // TRAIN-mode BN: batch statistics, and the `mvg` packing the
                // backward reads. `ResBlock::new` builds it eval-mode for
                // inference; flipping it here is the one difference.
                blk.bn1.set_eval(false);
                bwd.push(BlockBwd::new(&ctx, &blk, shape));
                shape = blk.out_shape;
                blocks.push(blk);
            }
        }

        let bn2 = BatchNorm::new(&ctx, BnNames::torch("bn2"), shape, true);
        let e = cfg.arc.embedding;
        let fl = cfg.arc.flatten();
        let fc_out = ctx.act(cfg.batch * e);
        let features = BatchNorm::new(&ctx, BnNames::torch("features"), Shape::new(cfg.batch, e, 1, 1), true);

        let k = cfg.classes;
        let (b, be, bk, ke) = (cfg.batch, cfg.batch * e, cfg.batch * k, k * e);

        let mut params: Vec<(String, usize)> = stem_conv.param_list();
        params.extend(stem_prelu.param_list());
        for blk in &blocks {
            params.extend(blk.param_list());
        }
        params.extend(bn2.param_list());
        params.push(("fc.weight".into(), (e * fl) as usize));
        params.push(("fc.bias".into(), e as usize));
        params.extend(features.param_list());
        params.push(("head.weight".into(), ke as usize));

        // `run_mean` / `run_var` are Frozen: train-mode BN never reads them and
        // they carry no gradient, so allocating grad + Adam moments for them
        // would give the optimiser (and the gradient check) tensors that can
        // only ever be zero.
        let mut init_map: HashMap<String, Vec<f32>> = HashMap::with_capacity(params.len());
        let mut roles = Vec::with_capacity(params.len());
        for (name, numel) in &params {
            let v = init(name, *numel);
            assert_eq!(v.len(), *numel, "init for {name} returned {} values, want {numel}", v.len());
            init_map.insert(name.clone(), v);
            let frozen = name.ends_with(".running_mean") || name.ends_with(".running_var");
            roles.push((name.clone(), *numel, if frozen { Role::Frozen } else { Role::Trainable }));
        }
        let ps = ParamStore::new_with_roles(&gpu, roles, &init_map);

        let ones = gpu.storage_init("facenet.l2.ones", &vec![1.0f32; e as usize]);
        let last = shape.numel();
        ArcFaceTrainer {
            input: gpu.storage(in_shape.numel() as u64),
            labels: gpu.storage(b as u64),
            ones,
            stem_conv,
            stem_prelu,
            blocks,
            bwd,
            bn2,
            fc_out,
            features,
            emb_n: gpu.storage(be as u64),
            w_n: gpu.storage(ke as u64),
            cosv: gpu.storage(bk as u64),
            logits: gpu.storage(bk as u64),
            ce_rows: gpu.storage(b as u64),
            d_logits: gpu.storage(bk as u64),
            d_cos: gpu.storage(bk as u64),
            d_emb_n: gpu.storage(be as u64),
            d_w_n: gpu.storage(ke as u64),
            d_head_w: gpu.storage(ke as u64),
            d_emb: gpu.storage(be as u64),
            d_fc_out: gpu.storage(be as u64),
            d_bn2_out: gpu.storage(last as u64),
            d_last: gpu.storage(last as u64),
            d_input: gpu.storage(in_shape.numel() as u64),
            in_shape,
            cfg,
            ps,
            gpu,
            fwd_done: Cell::new(false),
        }
    }

    pub fn config(&self) -> &ArcFaceTrainConfig {
        &self.cfg
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
    pub fn params(&self) -> &ParamStore {
        &self.ps
    }
    /// The grad wrt the input blob, valid after [`Self::backward`]. Exposed
    /// because the aligned crop is produced by `grid_sample`, whose own adjoint
    /// (`grid_sample_dx`) exists — so an end-to-end alignment-aware variant can
    /// be built on top without changing anything here.
    pub fn d_input(&self) -> &DeviceBuffer {
        &self.d_input
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, ids())
    }

    /// Pin the batch: `images` is NCHW `[B, 3, S, S]` (already aligned and
    /// normalised), `labels` one identity index per sample.
    pub fn set_batch(&self, images: &[f32], labels: &[u32]) {
        assert_eq!(images.len(), self.in_shape.numel() as usize, "images must be [B,3,S,S]");
        assert_eq!(labels.len(), self.cfg.batch as usize, "one label per sample");
        assert!(
            labels.iter().all(|&l| l < self.cfg.classes),
            "a label is out of range for {} classes",
            self.cfg.classes
        );
        self.gpu.write(&self.input, bytemuck::cast_slice(images));
        self.gpu.write(&self.labels, labels);
        self.fwd_done.set(false);
    }

    /// The margin kernels' shared uniform stream: `[rows, classes, cos_m,
    /// sin_m, scale]`, the last three bit-cast f32.
    fn margin_params(&self) -> [u32; 5] {
        [
            self.cfg.batch,
            self.cfg.classes,
            f(self.cfg.margin.cos()),
            f(self.cfg.margin.sin()),
            f(self.cfg.scale),
        ]
    }

    /// Full forward, returning the mean ArcFace cross-entropy over the batch.
    ///
    /// Every stage writes its own buffer, so this doubles as the backward's
    /// activation cache — calling it twice in a row is safe and idempotent.
    pub fn loss(&self) -> f32 {
        let ctx = self.ctx();
        let (b, e, k) = (self.cfg.batch, self.cfg.arc.embedding, self.cfg.classes);
        let fl = self.cfg.arc.flatten();

        self.stem_conv.forward(&ctx, &self.ps, &self.input);
        self.stem_prelu.forward(&ctx, &self.ps, self.stem_conv.out());
        let mut x: &DeviceBuffer = self.stem_prelu.out();
        for blk in &self.blocks {
            x = blk.forward(&ctx, &self.ps, x);
        }
        self.bn2.forward(&ctx, &self.ps, x);

        // Flatten is a no-op: `bn2`'s NCHW buffer IS the [B, fl] row block, and
        // `matmul` is out = x·Wᵀ with x [m,k], W [n,k] — the Gemm(transB=1)
        // layout the ONNX graph uses.
        let s_mm = ctx.step(
            kernel("matmul"),
            &[self.bn2.out(), self.ps.w("fc.weight"), &self.fc_out],
            &[b, fl, e],
            b * e,
        );
        let s_b = ctx.step(kernel("bias_add"), &[&self.fc_out, self.ps.w("fc.bias")], &[b, e], b * e);
        ctx.gpu.submit(&[], &[s_mm, s_b]);
        self.features.forward(&ctx, &self.ps, &self.fc_out);

        let eps = f(L2_EPS);
        let s_ne = ctx.step(
            kernel("l2norm_scale"),
            &[self.features.out(), &self.ones, &self.emb_n],
            &[b, e, eps],
            b * e,
        );
        let s_nw = ctx.step(
            kernel("l2norm_scale"),
            &[self.ps.w("head.weight"), &self.ones, &self.w_n],
            &[k, e, eps],
            k * e,
        );
        let s_cos = ctx.step(kernel("matmul"), &[&self.emb_n, &self.w_n, &self.cosv], &[b, e, k], b * k);
        let s_marg = ctx.step(
            kernel("arcface_margin"),
            &[&self.cosv, &self.labels, &self.logits],
            &self.margin_params(),
            b * k,
        );
        let s_ce = ctx.step(kernel("ce_value"), &[&self.logits, &self.labels, &self.ce_rows], &[b, k], b);
        ctx.gpu.submit(&[], &[s_ne, s_nw, s_cos, s_marg, s_ce]);

        self.fwd_done.set(true);
        let rows = self.gpu.read(&self.ce_rows, b as usize);
        rows.iter().sum::<f32>() / b as f32
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Reverse pass. Requires [`Self::loss`] to have run for this batch (it
    /// re-runs it otherwise): every stage's backward reads the forward's cache,
    /// and BatchNorm's in particular reads the `mvg` packing the train-mode
    /// forward writes on the host.
    pub fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        let ctx = self.ctx();
        let (b, e, k) = (self.cfg.batch, self.cfg.arc.embedding, self.cfg.classes);
        let fl = self.cfg.arc.flatten();
        let eps = f(L2_EPS);

        // --- loss head -----------------------------------------------------
        // `ce_grad` writes d(mean CE)/d(logit) directly (it divides by n_rows).
        let s_ceg = ctx.step(kernel("ce_grad"), &[&self.logits, &self.labels, &self.d_logits], &[b, k], b * k);
        let s_mb = ctx.step(
            kernel("arcface_margin_bwd"),
            &[&self.cosv, &self.labels, &self.d_logits, &self.d_cos],
            &self.margin_params(),
            b * k,
        );
        // cos = emb_n · w_nᵀ, so matmul's (m, k, n) is (batch, E, classes).
        let s_dx = ctx.step(
            kernel("matmul_dx"),
            &[&self.d_cos, &self.w_n, &self.d_emb_n],
            &[b, e, k, 0],
            b * e,
        );
        // `matmul_dw` ACCUMULATES and `d_w_n` is an intermediate that persists
        // across steps, so it goes in this submit's clear list. (A ParamStore
        // grad never does — `zero_grads` owns that, once per step.)
        let s_dw = ctx.step(kernel("matmul_dw"), &[&self.d_cos, &self.emb_n, &self.d_w_n], &[b, e, k], k * e);
        ctx.gpu.submit(&[&self.d_w_n], &[s_ceg, s_mb, s_dx, s_dw]);

        // Through the two row-normalisations. `l2norm_scale_dx` ASSIGNS, so the
        // head-weight grad lands in scratch and is then ACCUMULATED into the
        // ParamStore with `axpy` — assigning into `ps.g` would silently drop
        // any earlier contribution under gradient accumulation.
        let s_gw = ctx.step(
            kernel("l2norm_scale_dx"),
            &[self.ps.w("head.weight"), &self.ones, &self.d_w_n, &self.d_head_w],
            &[k, e, eps],
            k * e,
        );
        let s_acc = ctx.step(kernel("axpy"), &[self.ps.g("head.weight"), &self.d_head_w], &[k * e, f(1.0)], k * e);
        let s_ge = ctx.step(
            kernel("l2norm_scale_dx"),
            &[self.features.out(), &self.ones, &self.d_emb_n, &self.d_emb],
            &[b, e, eps],
            b * e,
        );
        ctx.gpu.submit(&[], &[s_gw, s_acc, s_ge]);

        // --- fc + the two head norms ---------------------------------------
        self.features.backward(&ctx, &self.ps, &self.fc_out, &self.d_emb, &self.d_fc_out);
        let s_bg = ctx.step(kernel("bias_grad"), &[&self.d_fc_out, self.ps.g("fc.bias")], &[b, e], e);
        let s_fdw = ctx.step(
            kernel("matmul_dw"),
            &[&self.d_fc_out, self.bn2.out(), self.ps.g("fc.weight")],
            &[b, fl, e],
            e * fl,
        );
        let s_fdx = ctx.step(
            kernel("matmul_dx"),
            &[&self.d_fc_out, self.ps.w("fc.weight"), &self.d_bn2_out],
            &[b, fl, e, 0],
            b * fl,
        );
        ctx.gpu.submit(&[], &[s_bg, s_fdw, s_fdx]);

        // --- backbone ------------------------------------------------------
        let last_in: &DeviceBuffer = match self.blocks.len() {
            0 => self.stem_prelu.out(),
            n => self.blocks[n - 1].out_ref(),
        };
        self.bn2.backward(&ctx, &self.ps, last_in, &self.d_bn2_out, &self.d_last);

        let mut d_out: &DeviceBuffer = &self.d_last;
        for i in (0..self.blocks.len()).rev() {
            let x: &DeviceBuffer =
                if i == 0 { self.stem_prelu.out() } else { self.blocks[i - 1].out_ref() };
            d_out = block_backward(&ctx, &self.ps, &self.blocks[i], &self.bwd[i], x, d_out);
        }
        self.stem_prelu.backward(&ctx, &self.ps, self.stem_conv.out(), d_out);
        self.stem_conv.backward(&ctx, &self.ps, &self.input, self.stem_prelu.d_in(), &self.d_input);
    }

    // ---- parameter access (what the gradient checker drives) --------------

    /// Only the OPTIMISED tensors — `run_mean`/`run_var` are frozen and have no
    /// gradient buffer at all.
    pub fn param_names(&self) -> Vec<String> {
        self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
        // Both `Conv` and `BatchNorm` cache their per-channel packings across
        // frames (`sb_ready`), keyed on eval mode. Train-mode BN re-packs every
        // forward, so nothing to invalidate — but the forward cache is stale.
        self.fwd_done.set(false);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init(name: &str, n: usize) -> Vec<f32> {
        if name.ends_with(".running_var") || name.ends_with("bn1.weight") {
            vec![1.0; n]
        } else {
            (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect()
        }
    }

    /// The trainable store must expose every conv/PReLU/BN/head tensor and NO
    /// running statistic — a `run_var` with a grad buffer is a tensor AdamW
    /// would happily walk to nonsense.
    #[test]
    fn trainable_set_excludes_running_stats_and_covers_the_prelu_slopes() {
        let gpu = gpu_core::testgpu::dev(crate::model::PIPELINES);
        let t = ArcFaceTrainer::new_on(gpu, ArcFaceTrainConfig::tiny(), &init);
        let names = t.param_names();
        assert!(names.iter().all(|n| !n.ends_with(".running_mean") && !n.ends_with(".running_var")));
        assert!(names.iter().any(|n| n == "stem.prelu.weight"), "{names:?}");
        assert!(names.iter().any(|n| n == "layer1.0.prelu.weight"), "{names:?}");
        assert!(names.iter().any(|n| n == "head.weight"), "{names:?}");
        // `tiny()` is [2,1,1,1]: four stage-first blocks with a downsample conv
        // (bn1 gamma/beta + conv1 w/b + prelu + conv2 w/b + down w/b = 9) and one
        // identity-shortcut block (the same minus the downsample pair = 7)
        // + stem (w/b/prelu) + bn2 (2) + fc (2) + features (2) + head (1).
        assert_eq!(names.len(), 4 * 9 + 7 + 3 + 2 + 2 + 2 + 1);
        // The identity block must really be identity — a `downsample` tensor
        // under `layer1.1` would mean the stride/width test fired and the
        // gradient check is back to covering only the strided arm.
        assert!(!names.iter().any(|n| n.starts_with("layer1.1.downsample")), "{names:?}");
        assert!(names.iter().any(|n| n == "layer1.1.conv2.weight"), "{names:?}");
    }

    /// The trainer rebuilds the IResNet graph rather than reusing `ArcFace`'s
    /// (the two differ in batch size and BN mode), so its parameterisation is a
    /// SECOND derivation of the same layout and can silently drift from the one
    /// the checkpoint is imported against. Pin it to `tensor_manifest()` — the
    /// importer's independent list, name for name, count for count, in order.
    /// Without this, a drifted trainer graph still gradient-checks perfectly
    /// (finite differences only ever compare the forward to its own backward)
    /// while training a net the release cannot round-trip.
    #[test]
    fn the_trainer_graph_matches_the_importers_tensor_manifest() {
        let gpu = gpu_core::testgpu::dev(crate::model::PIPELINES);
        let cfg = ArcFaceTrainConfig::tiny();
        let t = ArcFaceTrainer::new_on(gpu, cfg.clone(), &init);
        let want: Vec<(String, usize)> =
            cfg.arc.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
        let have = &t.params().params;
        assert_eq!(
            have.last().map(|(n, _)| n.as_str()),
            Some("head.weight"),
            "the margin head must be the trailing extra tensor"
        );
        assert_eq!(&have[..have.len() - 1], &want[..], "trainer graph vs import manifest");
    }

    /// Every stage's parameter gradient must ACCUMULATE into the ParamStore, not
    /// assign: gradient accumulation over micro-batches is the only way a real
    /// ArcFace run reaches its batch size, and a stage that assigns would silently
    /// keep only the LAST micro-batch. A finite-difference check cannot see this —
    /// it only ever calls `backward` once per `zero_grads`. Running it twice on the
    /// same batch must therefore double every gradient exactly.
    #[test]
    fn every_parameter_gradient_accumulates_across_backward_calls() {
        let gpu = gpu_core::testgpu::dev(crate::model::PIPELINES);
        let cfg = ArcFaceTrainConfig::tiny();
        let t = ArcFaceTrainer::new_on(gpu, cfg.clone(), &init);
        let n = (cfg.batch * 3 * cfg.arc.image_size * cfg.arc.image_size) as usize;
        let img: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let labels: Vec<u32> = (0..cfg.batch).map(|i| i % cfg.classes).collect();
        t.set_batch(&img, &labels);

        t.zero_grads();
        let _ = t.loss();
        t.backward();
        let once: Vec<Vec<f32>> = t.param_names().iter().map(|n| t.read_grad(n)).collect();
        t.backward();
        let twice: Vec<Vec<f32>> = t.param_names().iter().map(|n| t.read_grad(n)).collect();

        let mut nonzero = 0usize;
        for ((name, g1), g2) in t.param_names().iter().zip(&once).zip(&twice) {
            for (i, (&a, &b)) in g1.iter().zip(g2).enumerate() {
                if a.abs() > 1e-6 {
                    nonzero += 1;
                }
                assert!(
                    (b - 2.0 * a).abs() <= 1e-4 * (2.0 * a).abs().max(1e-3),
                    "{name}[{i}]: one backward gave {a}, two gave {b} — want 2x (this stage \
                     ASSIGNS into ps.g instead of accumulating)"
                );
            }
        }
        assert!(nonzero > 100, "only {nonzero} nonzero gradient entries — the batch is degenerate");
    }

    /// The margin must actually bite: at `m = 0` the head is a plain scaled
    /// cosine softmax, and turning `m` up must raise the loss on the same batch
    /// (the target logit is pushed down by `cos(θ+m) < cos θ`).
    #[test]
    fn the_angular_margin_increases_the_loss() {
        let gpu = gpu_core::testgpu::dev(crate::model::PIPELINES);
        let mut cfg = ArcFaceTrainConfig::tiny();
        cfg.margin = 0.0;
        let n = (cfg.batch * 3 * cfg.arc.image_size * cfg.arc.image_size) as usize;
        let img: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let labels: Vec<u32> = (0..cfg.batch).map(|i| i % cfg.classes).collect();

        let t0 = ArcFaceTrainer::new_on(gpu.share(), cfg.clone(), &init);
        t0.set_batch(&img, &labels);
        let l0 = t0.loss();

        cfg.margin = 0.5;
        let t1 = ArcFaceTrainer::new_on(gpu, cfg, &init);
        t1.set_batch(&img, &labels);
        let l1 = t1.loss();
        assert!(l1 > l0, "margin 0.5 loss {l1} must exceed margin 0 loss {l0}");
    }
}
