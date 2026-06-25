// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The full tiny YOLOv8 detector model (P3): backbone -> PAN-FPN neck ->
//! 3-scale decoupled head, wired into the architecture-agnostic
//! [`model::Model`] seam. The compute backend is selected at runtime via
//! [`Gpu::new`] (honouring `--device` / `BRAIN_DEVICE`): native CPU-JIT or wgpu.
//!
//! ## Graph
//! ```text
//! Backbone (strides /2 each Conv-s2):
//!   x -> Conv(s2) -> Conv(s2) -> C2f -> Conv(s2) -> C2f(=P3,/8)
//!     -> Conv(s2) -> C2f(=P4,/16) -> Conv(s2) -> C2f -> SPPF(=P5,/32)
//! Neck (PAN-FPN):
//!   top-down: up(P5) ++ P4 -> C2f(=T4); up(T4) ++ P3 -> C2f(=N3)
//!   bottom-up: down(N3) ++ T4 -> C2f(=N4); down(N4) ++ P5 -> C2f(=N5)
//! Head: ScaleHead on (N3,N4,N5) -> raw cls/box logits over A anchors.
//! ```
//! Backbone features P3/P4 each feed two consumers (a backbone Conv-s2 and a
//! neck concat); their grads are accumulated out-of-place via `add2` (the
//! multi-consumer pattern from P2's SPPF). P5 likewise feeds the top-down
//! upsample and the bottom-up concat.
//!
//! ## Loss-mode seam (P4)
//! [`LossMode::Proxy`] is implemented fully — it is the architecture-correctness
//! gate. A fixed seeded pseudo-random vector `r` (length = total raw-logit
//! elements, one slice per head branch in that branch's NCHW layout) defines
//! `L = Σ_k r_k · rawlogit_k`. `backward()` seeds every head branch's raw-logit
//! grad buffer with its slice of `r` (constant dL/draw = r) and runs the entire
//! reverse Step chain — exercising every conv/bn/silu/concat/upsample/pool +
//! head conv backward.
//!
//! [`LossMode::Detection`] is a documented STUB for P4: `set_targets` /
//! [`Yolo::raw_logits`] / [`Yolo::seed_head_grads`] are the hooks P4 fills with
//! the assigner + BCE+CIoU+DFL loss. The architecture here does not change.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{DeviceBuffer, Gpu};
use optim::Optim;
use paramstore::ParamStore;

use crate::blocks::{Conv, C2f, SPPF};
use crate::head::Head;
use crate::net::{Ctx, Shape, ADAMW, CLIP_COEF, GRADNORM_SQ, GRAD_SCALE, GRAD_SCALE_BUF, PIPELINES, UPSAMPLE2, UPSAMPLE2_DX, CONCAT2, CONCAT_SPLIT, ADD2};
use crate::YoloConfig;

/// A ground-truth box for the (P4) detection loss. Normalised xywh + class id.
#[derive(Clone, Copy, Debug)]
pub struct GtBox {
    pub img: u32,
    pub cls: u32,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Which loss the model differentiates. See the module docs for the P4 seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossMode {
    /// `L = <r, raw_logits>` for a fixed seeded `r`. The architecture gate.
    Proxy,
    /// Real detection loss (assigner + BCE+CIoU+DFL) — STUB, wired in P4.
    Detection,
}

// ===========================================================================
// Neck plumbing: upsample / downsample-conv / concat + their backward.
// ===========================================================================

/// A 2x nearest-neighbour upsample stage (forward + backward), SSA.
struct Up {
    in_shape: Shape,
    out_shape: Shape,
    out: DeviceBuffer,
    d_in: DeviceBuffer,
}
impl Up {
    fn new(ctx: &Ctx, in_shape: Shape) -> Up {
        let out_shape = Shape::new(in_shape.n, in_shape.c, in_shape.h * 2, in_shape.w * 2);
        Up { in_shape, out_shape, out: ctx.act(out_shape.numel()), d_in: ctx.act(in_shape.numel()) }
    }
    fn params(&self) -> [u32; 4] {
        [self.in_shape.n, self.in_shape.c, self.in_shape.h, self.in_shape.w]
    }
    fn forward(&self, ctx: &Ctx, x: &DeviceBuffer) {
        let s = ctx.step(UPSAMPLE2, &[x, &self.out], &self.params(), self.out_shape.numel());
        ctx.gpu.submit(&[], &[s]);
    }
    /// `d_out` (grad wrt upsampled output) -> `self.d_in` (grad wrt input).
    fn backward(&self, ctx: &Ctx, d_out: &DeviceBuffer) {
        let s = ctx.step(UPSAMPLE2_DX, &[d_out, &self.d_in], &self.params(), self.in_shape.numel());
        ctx.gpu.submit(&[], &[s]);
    }
}

/// A channel-concat of two equal-spatial feature maps `[a | b]` (forward +
/// backward via concat_split). SSA.
struct Cat {
    n: u32,
    ca: u32,
    cb: u32,
    h: u32,
    w: u32,
    out: DeviceBuffer,
    d_a: DeviceBuffer,
    d_b: DeviceBuffer,
}
impl Cat {
    fn new(ctx: &Ctx, a: Shape, b: Shape) -> Cat {
        assert_eq!((a.n, a.h, a.w), (b.n, b.h, b.w), "concat spatial mismatch");
        let out = Shape::new(a.n, a.c + b.c, a.h, a.w);
        Cat {
            n: a.n,
            ca: a.c,
            cb: b.c,
            h: a.h,
            w: a.w,
            out: ctx.act(out.numel()),
            d_a: ctx.act(a.numel()),
            d_b: ctx.act(b.numel()),
        }
    }
    fn out_shape(&self) -> Shape {
        Shape::new(self.n, self.ca + self.cb, self.h, self.w)
    }
    fn forward(&self, ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer) {
        let threads = (self.ca + self.cb) * self.h * self.w * self.n;
        let s = ctx.step(CONCAT2, &[a, b, &self.out], &[self.n, self.ca, self.cb, self.h, self.w], threads);
        ctx.gpu.submit(&[], &[s]);
    }
    /// Split `d_out` into `self.d_a` (channels [0,ca)) and `self.d_b` ([ca,..)).
    fn backward(&self, ctx: &Ctx, d_out: &DeviceBuffer) {
        let ctot = self.ca + self.cb;
        let na = self.ca * self.h * self.w * self.n;
        let nb = self.cb * self.h * self.w * self.n;
        // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
        let sa = ctx.step(CONCAT_SPLIT, &[d_out, &self.d_a], &[self.n, ctot, self.ca, 0, self.h, self.w], na);
        let sb = ctx.step(CONCAT_SPLIT, &[d_out, &self.d_b], &[self.n, ctot, self.cb, self.ca, self.h, self.w], nb);
        ctx.gpu.submit(&[], &[sa, sb]);
    }
}

/// A small out-of-place grad accumulator `dst = a + b` (the multi-consumer
/// pattern). Owns the destination so it survives across the backward chain.
struct Acc {
    out: DeviceBuffer,
    n: u32,
}
impl Acc {
    fn new(ctx: &Ctx, shape: Shape) -> Acc {
        Acc { out: ctx.act(shape.numel()), n: shape.numel() }
    }
    fn add(&self, ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer) {
        let s = ctx.step(ADD2, &[a, b, &self.out], &[self.n], self.n);
        ctx.gpu.submit(&[], &[s]);
    }
}

// ===========================================================================
// The model.
// ===========================================================================

pub struct Yolo {
    pub gpu: Gpu,
    pub cfg: YoloConfig,
    pub ps: ParamStore,
    opt: Optim,
    b: u32,
    mode: Cell<LossMode>,

    // input image buffer [N,3,H,W]
    img: DeviceBuffer,

    // ---- backbone ----
    b_conv0: Conv,
    b_conv1: Conv,
    b_c2f0: C2f,
    b_conv2: Conv,
    b_c2f1: C2f, // -> P3
    b_conv3: Conv,
    b_c2f2: C2f, // -> P4
    b_conv4: Conv,
    b_c2f3: C2f,
    b_sppf: SPPF, // -> P5

    // ---- neck (PAN-FPN) ----
    up5: Up,       // upsample P5
    cat_p5p4: Cat, // [up5 | P4]
    n_t4: C2f,     // top-down P4 stage
    up4: Up,       // upsample T4
    cat_t4p3: Cat, // [up4 | P3]
    n_n3: C2f,     // neck-P3 (head input 0)
    dn3: Conv,     // downsample N3 (stride 2)
    cat_n3t4: Cat, // [dn3 | T4]
    n_n4: C2f,     // neck-P4 (head input 1)
    dn4: Conv,     // downsample N4 (stride 2)
    cat_n4p5: Cat, // [dn4 | P5]
    n_n5: C2f,     // neck-P5 (head input 2)

    // ---- head ----
    head: Head,

    // ---- backward grad buffers for neck/head feature edges ----
    // head input grads (one per scale), produced by ScaleHead::backward.
    d_n3: DeviceBuffer,
    d_n4: DeviceBuffer,
    d_n5: DeviceBuffer,
    // C2f/Up input-grad scratch buffers.
    d_t4_from_up4: DeviceBuffer, // grad into T4 via up4 path
    d_t4_total: Acc,             // T4 grad = up4-path + concat-path
    d_p3_from_n3: DeviceBuffer,  // grad into P3 via N3.cv1
    d_p3_total: Acc,             // P3 grad = backbone-conv path + neck-concat path
    d_p4_from_t4: DeviceBuffer,  // grad into P4 via cat_p5p4
    d_p4_total: Acc,             // P4 grad = backbone-conv path + neck-concat path
    d_p5_from_up5: DeviceBuffer, // grad into P5 via up5
    d_p5_total: Acc,             // P5 grad = sppf-out path (cat_n4p5) + up5 path
    // generic per-stage input-grad scratch (sized to that stage's input).
    d_catp5p4: DeviceBuffer,
    d_catt4p3: DeviceBuffer,
    d_catn3t4: DeviceBuffer,
    d_catn4p5: DeviceBuffer,
    d_n3_in: DeviceBuffer, // grad into N3 from dn3 (= grad wrt n_n3 output)
    d_n4_in: DeviceBuffer, // grad into N4 from dn4
    d_dn3_in: DeviceBuffer,
    d_dn4_in: DeviceBuffer,
    // backbone input-grad scratch (chain back to image; unused after b_conv0).
    d_back: RefCell<Vec<DeviceBuffer>>,

    // ---- proxy loss ----
    /// Per-head-branch fixed proxy vector `r` (NCHW layout, one Vec per branch
    /// in head order: scale0.cls, scale0.reg, scale1.cls, ...).
    r: Vec<Vec<f32>>,
    /// Device grad buffers seeded with `r` for each head branch (head order).
    d_logit: Vec<DeviceBuffer>,

    // ---- detection loss (P4) ----
    /// Ground-truth boxes for the current batch (set via `set_targets`).
    gts: RefCell<Vec<GtBox>>,
    /// Frozen assignment: once `Some`, the detection forward/backward reuse it
    /// instead of re-running the (non-differentiable) assigner. This is what the
    /// gradcheck relies on so finite-difference weight perturbations do not move
    /// the assignment and create discontinuities. `None` = recompute each call.
    frozen: RefCell<Option<crate::loss::Assignment>>,
    /// Scalar detection loss from the most recent `detection_eval` (debug aid).
    det_loss: Cell<f32>,
}

impl Yolo {
    /// Load a model from a `.weights` checkpoint, sized for batch `b`. The config
    /// (channels/depths/nc/input) is read from the checkpoint header; the `t`
    /// (sequence) seam is unused by detection so it is passed as 0. Mirrors
    /// [`gpt::Gpt::load`].
    pub fn load(path: &str, b: u32) -> Yolo {
        let c = checkpoint::load(path);
        let cfg = YoloConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Yolo::new(cfg, b, 0, &init)
    }

    pub fn new(cfg: YoloConfig, b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Yolo {
        // Honour an EXPLICIT backend choice (`brain ... --device cpu|gpu`) so
        // `--device gpu` actually runs the WGSL kernels on the wgpu/GPU backend,
        // while still defaulting to the native CPU-JIT when nothing was selected
        // (preserving the `cargo test` / tooling convention that yolo is CPU).
        let gpu = if gpu_core::backend_selected() {
            Gpu::new(PIPELINES)
        } else {
            Gpu::new_cpu(PIPELINES)
        };
        let ps = ParamStore::new(&gpu, ModelConfigParamList::param_list(&cfg), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let train = true;
        let ctx = Ctx::new(&gpu);
        let side = cfg.input;
        let img_shape = Shape::new(b, 3, side, side);
        let img = gpu.buffer("img", (img_shape.numel() as u64) * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);

        // Explicit per-stage layout (P12 canonical yolov8n; tiny shares the shape).
        let bch = cfg.backbone_ch; // out-ch of backbone stages 0..=9
        let bd = cfg.backbone_depth; // C2f depths for stages 2,4,6,8
        let nck = cfg.neck_ch; // out-ch of neck.0..=5
        let ndepth = cfg.neck_depth.max(1);

        // ---- backbone ----
        // stem: two stride-2 convs (/4).
        let b_conv0 = Conv::new(&ctx, "backbone.0", img_shape, bch[0], 3, 2, 1, train);
        let b_conv1 = Conv::new(&ctx, "backbone.1", b_conv0.out_shape, bch[1], 3, 2, 1, train);
        let b_c2f0 = C2f::new(&ctx, "backbone.2", b_conv1.out_shape, bch[2], bd[0], true, train);
        let b_conv2 = Conv::new(&ctx, "backbone.3", b_c2f0.out_shape, bch[3], 3, 2, 1, train); // /8
        let b_c2f1 = C2f::new(&ctx, "backbone.4", b_conv2.out_shape, bch[4], bd[1], true, train); // P3
        let b_conv3 = Conv::new(&ctx, "backbone.5", b_c2f1.out_shape, bch[5], 3, 2, 1, train); // /16
        let b_c2f2 = C2f::new(&ctx, "backbone.6", b_conv3.out_shape, bch[6], bd[2], true, train); // P4
        let b_conv4 = Conv::new(&ctx, "backbone.7", b_c2f2.out_shape, bch[7], 3, 2, 1, train); // /32
        let b_c2f3 = C2f::new(&ctx, "backbone.8", b_conv4.out_shape, bch[8], bd[3], true, train);
        let b_sppf = SPPF::new(&ctx, "backbone.9", b_c2f3.out_shape, bch[9], train); // P5

        let p3 = b_c2f1.out_shape; // [b,P3,/8,/8]
        let p4 = b_c2f2.out_shape; // [b,P4,/16,/16]
        let p5 = b_sppf.out_shape; // [b,P5,/32,/32]

        // ---- neck (PAN-FPN) ----
        // top-down
        let up5 = Up::new(&ctx, p5);
        let cat_p5p4 = Cat::new(&ctx, up5.out_shape, p4);
        let n_t4 = C2f::new(&ctx, "neck.0", cat_p5p4.out_shape(), nck[0], ndepth, false, train);
        let t4 = n_t4.out_shape;
        let up4 = Up::new(&ctx, t4);
        let cat_t4p3 = Cat::new(&ctx, up4.out_shape, p3);
        let n_n3 = C2f::new(&ctx, "neck.1", cat_t4p3.out_shape(), nck[1], ndepth, false, train);
        let n3 = n_n3.out_shape;
        // bottom-up
        let dn3 = Conv::new(&ctx, "neck.2", n3, nck[2], 3, 2, 1, train); // down to /16
        let cat_n3t4 = Cat::new(&ctx, dn3.out_shape, t4);
        let n_n4 = C2f::new(&ctx, "neck.3", cat_n3t4.out_shape(), nck[3], ndepth, false, train);
        let n4 = n_n4.out_shape;
        let dn4 = Conv::new(&ctx, "neck.4", n4, nck[4], 3, 2, 1, train); // down to /32
        let cat_n4p5 = Cat::new(&ctx, dn4.out_shape, p5);
        let n_n5 = C2f::new(&ctx, "neck.5", cat_n4p5.out_shape(), nck[5], ndepth, false, train);
        let n5 = n_n5.out_shape;

        // ---- head ----
        let cls_mid = cfg.cls_mid;
        let reg_mid = cfg.reg_mid;
        let head = Head::new(
            &ctx,
            "head",
            [n3, n4, n5],
            cfg.nc,
            cfg.reg_max,
            cls_mid,
            reg_mid,
            cfg.strides,
            train,
        );

        // ---- backward scratch ----
        let d_n3 = ctx.act(n3.numel());
        let d_n4 = ctx.act(n4.numel());
        let d_n5 = ctx.act(n5.numel());
        let d_t4_from_up4 = ctx.act(t4.numel());
        let d_t4_total = Acc::new(&ctx, t4);
        let d_p3_from_n3 = ctx.act(p3.numel());
        let d_p3_total = Acc::new(&ctx, p3);
        let d_p4_from_t4 = ctx.act(p4.numel());
        let d_p4_total = Acc::new(&ctx, p4);
        let d_p5_from_up5 = ctx.act(p5.numel());
        let d_p5_total = Acc::new(&ctx, p5);
        let d_catp5p4 = ctx.act(cat_p5p4.out_shape().numel());
        let d_catt4p3 = ctx.act(cat_t4p3.out_shape().numel());
        let d_catn3t4 = ctx.act(cat_n3t4.out_shape().numel());
        let d_catn4p5 = ctx.act(cat_n4p5.out_shape().numel());
        let d_n3_in = ctx.act(n3.numel());
        let d_n4_in = ctx.act(n4.numel());
        // dn3/dn4 backward write grad wrt their INPUT (n3 / n4 resp.).
        let d_dn3_in = ctx.act(n3.numel());
        let d_dn4_in = ctx.act(n4.numel());

        // backbone input-grad scratch (one per backbone stage input).
        let back_shapes = [
            img_shape,
            b_conv0.out_shape,
            b_conv1.out_shape,
            b_c2f0.out_shape,
            b_conv2.out_shape,
            b_c2f1.out_shape, // P3
            b_conv3.out_shape,
            b_c2f2.out_shape, // P4
            b_conv4.out_shape,
            b_c2f3.out_shape,
        ];
        let d_back: Vec<DeviceBuffer> = back_shapes.iter().map(|s| ctx.act(s.numel())).collect();

        // ---- proxy vectors ----
        let mut r: Vec<Vec<f32>> = Vec::new();
        let mut d_logit: Vec<DeviceBuffer> = Vec::new();
        let mut seed = 0xC0FFEEu64;
        for sc in &head.scales {
            for branch_shape in [sc.cls.out_shape, sc.reg.out_shape] {
                let n = branch_shape.numel();
                let rv = proxy_vec(&mut seed, n);
                r.push(rv);
                d_logit.push(ctx.act(n));
            }
        }

        Yolo {
            gpu,
            cfg,
            ps,
            opt,
            b,
            mode: Cell::new(LossMode::Proxy),
            img,
            b_conv0,
            b_conv1,
            b_c2f0,
            b_conv2,
            b_c2f1,
            b_conv3,
            b_c2f2,
            b_conv4,
            b_c2f3,
            b_sppf,
            up5,
            cat_p5p4,
            n_t4,
            up4,
            cat_t4p3,
            n_n3,
            dn3,
            cat_n3t4,
            n_n4,
            dn4,
            cat_n4p5,
            n_n5,
            head,
            d_n3,
            d_n4,
            d_n5,
            d_t4_from_up4,
            d_t4_total,
            d_p3_from_n3,
            d_p3_total,
            d_p4_from_t4,
            d_p4_total,
            d_p5_from_up5,
            d_p5_total,
            d_catp5p4,
            d_catt4p3,
            d_catn3t4,
            d_catn4p5,
            d_n3_in,
            d_n4_in,
            d_dn3_in,
            d_dn4_in,
            d_back: RefCell::new(d_back),
            r,
            d_logit,
            gts: RefCell::new(Vec::new()),
            frozen: RefCell::new(None),
            det_loss: Cell::new(0.0),
        }
    }

    pub fn set_mode(&self, mode: LossMode) {
        self.mode.set(mode);
    }

    /// Flip the WHOLE network's BatchNorm between eval-mode (running stats, used
    /// for inference / [`Yolo::detect`]) and train-mode (batch stats). This is an
    /// inference-only toggle: it changes only which BN kernel each `Conv` forward
    /// dispatches — no buffers, no graph edges, no parameters change — so it is
    /// safe to flip back and forth around an inference call. Training paths
    /// (the loss gradchecks) leave this untouched and stay in train-mode BN.
    pub fn set_eval(&self, eval: bool) {
        self.b_conv0.set_eval(eval);
        self.b_conv1.set_eval(eval);
        self.b_c2f0.set_eval(eval);
        self.b_conv2.set_eval(eval);
        self.b_c2f1.set_eval(eval);
        self.b_conv3.set_eval(eval);
        self.b_c2f2.set_eval(eval);
        self.b_conv4.set_eval(eval);
        self.b_c2f3.set_eval(eval);
        self.b_sppf.set_eval(eval);
        self.n_t4.set_eval(eval);
        self.n_n3.set_eval(eval);
        self.dn3.set_eval(eval);
        self.n_n4.set_eval(eval);
        self.dn4.set_eval(eval);
        self.n_n5.set_eval(eval);
        self.head.set_eval(eval);
    }

    /// Enable/disable the BN running-stat momentum EMA update across the WHOLE
    /// network (backbone + neck + head). Must be ON during real training so every
    /// `Conv`'s `bn.run_mean`/`bn.run_var` track the data — those running stats
    /// are what eval-mode BN (and hence [`Yolo::detect`]) reads. Left OFF for the
    /// gradchecks, whose finite-difference forward passes must stay deterministic.
    pub fn set_update_running(&self, on: bool) {
        self.b_conv0.set_update_running(on);
        self.b_conv1.set_update_running(on);
        self.b_c2f0.set_update_running(on);
        self.b_conv2.set_update_running(on);
        self.b_c2f1.set_update_running(on);
        self.b_conv3.set_update_running(on);
        self.b_c2f2.set_update_running(on);
        self.b_conv4.set_update_running(on);
        self.b_c2f3.set_update_running(on);
        self.b_sppf.set_update_running(on);
        self.n_t4.set_update_running(on);
        self.n_n3.set_update_running(on);
        self.dn3.set_update_running(on);
        self.n_n4.set_update_running(on);
        self.dn4.set_update_running(on);
        self.n_n5.set_update_running(on);
        self.head.set_update_running(on);
    }

    /// Upload the image batch `[N*3*H*W]` (f32 CHW).
    pub fn set_image(&self, inputs: &[f32]) {
        let want = (self.b * 3 * self.cfg.input * self.cfg.input) as usize;
        assert_eq!(inputs.len(), want, "image size mismatch: {} != {}", inputs.len(), want);
        self.gpu.write(&self.img, bytemuck::cast_slice(inputs));
    }

    /// P4: set the ground-truth boxes for the detection loss. The assigner reads
    /// these to build the per-anchor target tensors. Changing targets clears any
    /// frozen assignment.
    pub fn set_targets(&self, gts: &[GtBox]) {
        *self.gts.borrow_mut() = gts.to_vec();
        *self.frozen.borrow_mut() = None;
    }

    /// P4: compute the Task-Aligned assignment ONCE from the current logits and
    /// hold it, so subsequent `forward()`/`backward()` calls (e.g. the
    /// gradcheck's weight perturbations) reuse the same assignment instead of
    /// re-running the assigner. Without this the finite-difference perturbations
    /// would move the (piecewise-constant) assignment and the central difference
    /// would straddle a discontinuity. Requires `set_image`/`set_targets` first.
    pub fn freeze_assignment(&self) {
        self.forward_net();
        let (cls, boxl) = self.raw_logits();
        let anchors = self.head.anchor_geometry();
        let inp = self.loss_input(&cls, &boxl, &anchors);
        let asg = crate::loss::compute_assignment(&inp, &self.gts.borrow(), self.cfg.input as f32);
        *self.frozen.borrow_mut() = Some(asg);
    }

    /// Drop a frozen assignment (return to per-forward recomputation).
    pub fn unfreeze_assignment(&self) {
        *self.frozen.borrow_mut() = None;
    }

    /// Assemble the loss-module input view over the (already-read) flat logits.
    fn loss_input<'a>(
        &'a self,
        cls: &'a [f32],
        boxl: &'a [f32],
        anchors: &'a [crate::assign::Anchor],
    ) -> crate::loss::LossInput<'a> {
        crate::loss::LossInput {
            gpu: &self.gpu,
            n: self.b as usize,
            a: self.head.num_anchors() as usize,
            nc: self.cfg.nc as usize,
            reg_max: self.cfg.reg_max as usize,
            anchors,
            cls_logits: cls,
            box_logits: boxl,
            gains: crate::loss::Gains::default(),
        }
    }

    /// Run the full detection loss (forward + backward grads) against the current
    /// logits, using a frozen assignment if present else recomputing it.
    /// Caches the scalar loss and scatters the flat cls/box logit grads into the
    /// per-branch NCHW head grad buffers. Returns the scalar loss.
    fn detection_eval(&self) -> f32 {
        let (cls, boxl) = self.raw_logits();
        let anchors = self.head.anchor_geometry();
        let inp = self.loss_input(&cls, &boxl, &anchors);

        // The frozen assignment (or a fresh one) — clone out to drop the borrow.
        let asg = match self.frozen.borrow().as_ref() {
            Some(a) => a.clone(),
            None => crate::loss::compute_assignment(&inp, &self.gts.borrow(), self.cfg.input as f32),
        };
        let out = crate::loss::eval(&inp, &asg);
        self.det_loss.set(out.loss);
        self.scatter_head_grads(&out.d_cls, &out.d_box);
        out.loss
    }

    /// Scatter flat cls grads `[N,A,nc]` and box grads `[N,A,4*reg_max]` into the
    /// per-branch NCHW head grad buffers `self.d_logit` (head order:
    /// s0.cls,s0.reg,s1.cls,s1.reg,s2.cls,s2.reg). Inverse of
    /// `Head::gather_flat`: flat anchor `(n*A + base + p)` channel `ch` maps to
    /// the per-scale NCHW index `((n*C + ch)*hw + p)`.
    fn scatter_head_grads(&self, d_cls: &[f32], d_box: &[f32]) {
        let n = self.b as usize;
        let a = self.head.num_anchors() as usize;
        let nc = self.cfg.nc as usize;
        let four_rm = 4 * self.cfg.reg_max as usize;

        let mut anchor_base = 0usize;
        for (s, scale) in self.head.scales.iter().enumerate() {
            let sh = scale.cls.out_shape; // [n, nc, h, w]
            let hw = (sh.h * sh.w) as usize;
            // cls branch.
            let mut cls_nchw = vec![0.0f32; n * nc * hw];
            for nn in 0..n {
                for ch in 0..nc {
                    for p in 0..hw {
                        let src = (nn * a + anchor_base + p) * nc + ch;
                        cls_nchw[(nn * nc + ch) * hw + p] = d_cls[src];
                    }
                }
            }
            self.gpu.write(&self.d_logit[s * 2], bytemuck::cast_slice(&cls_nchw));
            // reg branch.
            let mut box_nchw = vec![0.0f32; n * four_rm * hw];
            for nn in 0..n {
                for ch in 0..four_rm {
                    for p in 0..hw {
                        let src = (nn * a + anchor_base + p) * four_rm + ch;
                        box_nchw[(nn * four_rm + ch) * hw + p] = d_box[src];
                    }
                }
            }
            self.gpu.write(&self.d_logit[s * 2 + 1], bytemuck::cast_slice(&box_nchw));
            anchor_base += hw;
        }
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu)
    }

    /// Run the whole backbone+neck+head forward. Returns nothing; outputs live
    /// in the head branch buffers (read by [`Self::loss`] / `raw_logits`).
    fn forward_net(&self) {
        let ctx = self.ctx();
        let ps = &self.ps;

        // ---- backbone ----
        self.b_conv0.forward(&ctx, ps, &self.img);
        self.b_conv1.forward(&ctx, ps, self.b_conv0.out());
        self.b_c2f0.forward(&ctx, ps, self.b_conv1.out());
        self.b_conv2.forward(&ctx, ps, self.b_c2f0.out());
        self.b_c2f1.forward(&ctx, ps, self.b_conv2.out()); // P3
        self.b_conv3.forward(&ctx, ps, self.b_c2f1.out());
        self.b_c2f2.forward(&ctx, ps, self.b_conv3.out()); // P4
        self.b_conv4.forward(&ctx, ps, self.b_c2f2.out());
        self.b_c2f3.forward(&ctx, ps, self.b_conv4.out());
        self.b_sppf.forward(&ctx, ps, self.b_c2f3.out()); // P5

        let p3 = self.b_c2f1.out();
        let p4 = self.b_c2f2.out();
        let p5 = self.b_sppf.out();

        // ---- neck top-down ----
        self.up5.forward(&ctx, p5);
        self.cat_p5p4.forward(&ctx, &self.up5.out, p4);
        self.n_t4.forward(&ctx, ps, &self.cat_p5p4.out);
        let t4 = self.n_t4.out();
        self.up4.forward(&ctx, t4);
        self.cat_t4p3.forward(&ctx, &self.up4.out, p3);
        self.n_n3.forward(&ctx, ps, &self.cat_t4p3.out);
        let n3 = self.n_n3.out();

        // ---- neck bottom-up ----
        self.dn3.forward(&ctx, ps, n3);
        self.cat_n3t4.forward(&ctx, self.dn3.out(), t4);
        self.n_n4.forward(&ctx, ps, &self.cat_n3t4.out);
        let n4 = self.n_n4.out();
        self.dn4.forward(&ctx, ps, n4);
        self.cat_n4p5.forward(&ctx, self.dn4.out(), p5);
        self.n_n5.forward(&ctx, ps, &self.cat_n4p5.out);
        let n5 = self.n_n5.out();

        // ---- head ----
        self.head.forward(&ctx, ps, &[n3, n4, n5]);
    }

    /// Proxy loss `L = Σ_branch <r_branch, branch_out>`.
    fn proxy_loss(&self) -> f32 {
        let mut acc = 0.0f32;
        let mut idx = 0;
        for sc in &self.head.scales {
            for branch in [&sc.cls, &sc.reg] {
                let n = self.r[idx].len();
                let out = self.gpu.read(branch.out(), n);
                acc += out.iter().zip(&self.r[idx]).map(|(o, rr)| o * rr).sum::<f32>();
                idx += 1;
            }
        }
        acc
    }

    pub fn forward(&self) -> f32 {
        self.forward_net();
        match self.mode.get() {
            LossMode::Proxy => self.proxy_loss(),
            LossMode::Detection => self.detection_eval(),
        }
    }

    /// Seed every head branch's raw-logit grad buffer (Detection: from the P4
    /// loss kernels; Proxy: with the fixed `r`). Returns the per-branch device
    /// grad buffers in head order so [`Self::backward_net`] can consume them.
    fn seed_head_grads(&self) {
        match self.mode.get() {
            LossMode::Proxy => {
                for (i, rv) in self.r.iter().enumerate() {
                    self.gpu.write(&self.d_logit[i], bytemuck::cast_slice(rv));
                }
            }
            LossMode::Detection => {
                // The detection loss writes dL/draw into self.d_logit[i] (cls
                // then reg per scale). `detection_eval` runs the full loss
                // forward+grad and scatters those grads into the head buffers, so
                // they are consistent with the CURRENT weights even when called
                // standalone (it reuses the frozen assignment when present).
                self.detection_eval();
            }
        }
    }

    pub fn backward(&self) {
        self.seed_head_grads();
        self.backward_net();
    }

    /// Reverse Step chain: head -> neck -> backbone, accumulating multi-consumer
    /// grads (P3/P4/P5/T4) out-of-place via `add2`.
    fn backward_net(&self) {
        let ctx = self.ctx();
        let ps = &self.ps;
        let n3 = self.n_n3.out();
        let n4 = self.n_n4.out();

        // ---- head backward (per scale): seeds d_n3/d_n4/d_n5 ----
        // d_logit order: [s0.cls, s0.reg, s1.cls, s1.reg, s2.cls, s2.reg].
        self.head.scales[0].backward(&ctx, ps, n3, &self.d_logit[0], &self.d_logit[1], &self.d_n3);
        self.head.scales[1].backward(&ctx, ps, n4, &self.d_logit[2], &self.d_logit[3], &self.d_n4);
        self.head.scales[2].backward(&ctx, ps, self.n_n5.out(), &self.d_logit[4], &self.d_logit[5], &self.d_n5);

        // ---- neck bottom-up backward ----
        // n_n5 = C2f(cat_n4p5);  d(cat_n4p5) -> split into d(dn4_out) and d(p5_via_cat)
        self.n_n5.backward(&ctx, ps, &self.cat_n4p5.out, &self.d_n5, &self.d_catn4p5);
        self.cat_n4p5.backward(&ctx, &self.d_catn4p5); // -> d_a=dn4_out grad, d_b=p5-path grad
        // dn4: Conv(n4) -> grad wrt n4 input
        self.dn4.backward(&ctx, ps, n4, &self.cat_n4p5.d_a, &self.d_dn4_in);
        // p5 path via cat_n4p5.d_b accumulated later.

        // n_n4 = C2f(cat_n3t4); grad wrt n4 output = d_dn4_in (from dn4) + d_n4 (head)
        self.d_n4_in_acc(&ctx);
        self.n_n4.backward(&ctx, ps, &self.cat_n3t4.out, &self.d_n4_in, &self.d_catn3t4);
        self.cat_n3t4.backward(&ctx, &self.d_catn3t4); // d_a=dn3_out grad, d_b=t4-path grad
        // dn3: Conv(n3) -> grad wrt n3 input
        self.dn3.backward(&ctx, ps, n3, &self.cat_n3t4.d_a, &self.d_dn3_in);

        // ---- neck top-down backward ----
        // n_n3 = C2f(cat_t4p3); grad wrt n3 output = d_dn3_in (from dn3) + d_n3 (head)
        self.d_n3_in_acc(&ctx);
        self.n_n3.backward(&ctx, ps, &self.cat_t4p3.out, &self.d_n3_in, &self.d_catt4p3);
        self.cat_t4p3.backward(&ctx, &self.d_catt4p3); // d_a=up4_out grad, d_b=p3-via-concat grad
        // up4 backward -> grad wrt T4 (via up4 path)
        self.up4.backward(&ctx, &self.cat_t4p3.d_a);
        // copy up4.d_in into d_t4_from_up4 (own it before T4 grad is combined).
        self.copy(&self.up4.d_in, &self.d_t4_from_up4, self.n_t4.out_shape.numel());

        // T4 grad = up4-path (d_t4_from_up4) + concat-path (cat_n3t4.d_b).
        self.d_t4_total.add(&ctx, &self.d_t4_from_up4, &self.cat_n3t4.d_b);
        // n_t4 = C2f(cat_p5p4); grad wrt T4 output = d_t4_total
        self.n_t4.backward(&ctx, ps, &self.cat_p5p4.out, &self.d_t4_total.out, &self.d_catp5p4);
        self.cat_p5p4.backward(&ctx, &self.d_catp5p4); // d_a=up5_out grad, d_b=p4-via-concat grad
        // up5 backward -> grad wrt P5 (via up5 path)
        self.up5.backward(&ctx, &self.cat_p5p4.d_a);
        self.copy(&self.up5.d_in, &self.d_p5_from_up5, self.b_sppf.out_shape.numel());

        // ---- accumulate backbone feature grads (multi-consumer) ----
        // P5 grad = sppf-out path (cat_n4p5.d_b) + up5 path (d_p5_from_up5).
        self.d_p5_total.add(&ctx, &self.cat_n4p5.d_b, &self.d_p5_from_up5);
        // P4 grad = backbone-conv path (b_conv4) + neck-concat path (cat_p5p4.d_b).
        self.copy(&self.cat_p5p4.d_b, &self.d_p4_from_t4, self.b_c2f2.out_shape.numel());
        // P3 grad = backbone-conv path (b_conv3) + neck-concat path (cat_t4p3.d_b).
        self.copy(&self.cat_t4p3.d_b, &self.d_p3_from_n3, self.b_c2f1.out_shape.numel());

        // ---- backbone backward (reverse), routing the P3/P4/P5 grads in ----
        let d = self.d_back.borrow_mut();
        // sppf: input grad -> d[9] (= grad wrt b_c2f3 output)
        self.b_sppf.backward(&ctx, ps, self.b_c2f3.out(), &self.d_p5_total.out, &d[9]);
        self.b_c2f3.backward(&ctx, ps, self.b_conv4.out(), &d[9], &d[8]);
        // d[7] receives grad wrt P4 from the backbone-conv4 path.
        self.b_conv4.backward(&ctx, ps, self.b_c2f2.out(), &d[8], &d[7]);
        // add the neck path (cat_p5p4 -> d_p4_from_t4) to get the total P4 grad.
        self.d_p4_total.add(&ctx, &d[7], &self.d_p4_from_t4);
        self.b_c2f2.backward(&ctx, ps, self.b_conv3.out(), &self.d_p4_total.out, &d[6]);
        self.b_conv3.backward(&ctx, ps, self.b_c2f1.out(), &d[6], &d[5]);
        // d[5] holds grad wrt P3 from backbone-conv3 path; add the neck path.
        self.d_p3_total.add(&ctx, &d[5], &self.d_p3_from_n3);
        self.b_c2f1.backward(&ctx, ps, self.b_conv2.out(), &self.d_p3_total.out, &d[4]);
        self.b_conv2.backward(&ctx, ps, self.b_c2f0.out(), &d[4], &d[3]);
        self.b_c2f0.backward(&ctx, ps, self.b_conv1.out(), &d[3], &d[2]);
        self.b_conv1.backward(&ctx, ps, self.b_conv0.out(), &d[2], &d[1]);
        self.b_conv0.backward(&ctx, ps, &self.img, &d[1], &d[0]);
    }

    /// d_n4_in = d_dn4_in (from dn4 backward) + d_n4 (from head).
    fn d_n4_in_acc(&self, ctx: &Ctx) {
        let n = self.n_n4.out_shape.numel();
        let s = ctx.step(ADD2, &[&self.d_dn4_in, &self.d_n4, &self.d_n4_in], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }
    /// d_n3_in = d_dn3_in (from dn3 backward) + d_n3 (from head).
    fn d_n3_in_acc(&self, ctx: &Ctx) {
        let n = self.n_n3.out_shape.numel();
        let s = ctx.step(ADD2, &[&self.d_dn3_in, &self.d_n3, &self.d_n3_in], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }

    /// Snapshot a transient grad buffer (`src`) into an owned buffer (`dst`)
    /// before its producer is reused later in the backward chain. There is no
    /// device-side copy/identity kernel, so on the CPU backend we round-trip
    /// through the host (these neck grad buffers are small).
    fn copy(&self, src: &DeviceBuffer, dst: &DeviceBuffer, numel: u32) {
        let v = self.gpu.read(src, numel as usize);
        self.gpu.write(dst, bytemuck::cast_slice(&v));
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
    }

    /// P6 inference accessors. The configured batch size.
    pub fn batch(&self) -> u32 {
        self.b
    }
    /// Total anchor count across the 3 scales.
    pub fn num_anchors(&self) -> u32 {
        self.head.num_anchors()
    }
    /// Per-anchor geometry (pixel center, feature anchor point, stride),
    /// scale-major — the DFL decode needs the anchor point + stride.
    pub fn anchor_geometry(&self) -> Vec<crate::assign::Anchor> {
        self.head.anchor_geometry()
    }
    /// Run the backbone+neck+head forward only (no loss). P6 inference path.
    pub fn forward_net_pub(&self) {
        self.forward_net();
    }
    /// Whether the network is currently in eval-mode BN (running stats). All
    /// Convs share the same mode after `set_eval`, so we read the stem's.
    pub fn is_eval(&self) -> bool {
        self.b_conv0.is_eval()
    }

    /// P4 SEAM: read the concatenated raw cls/box logits `[N,A,nc]` /
    /// `[N,A,4*reg_max]` (the loss module's input).
    pub fn raw_logits(&self) -> (Vec<f32>, Vec<f32>) {
        let ctx = self.ctx();
        (self.head.cls_logits_flat(&ctx), self.head.box_logits_flat(&ctx))
    }

    pub fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }
}

/// A reproducible pseudo-random vector in (-1, 1) for the proxy loss.
fn proxy_vec(seed: &mut u64, n: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (*seed >> 32) as u32;
        // Small amplitude keeps the proxy loss + its gradients numerically
        // modest, so the directional-derivative dot product (a sum over
        // hundreds of thousands of weight terms) and the central-difference
        // forward passes stay well-conditioned in fp32 on the CPU JIT.
        out.push(((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.1);
    }
    out
}

// Helper trait to call YoloConfig::param_list without ambiguity vs ModelConfig.
trait ModelConfigParamList {
    fn param_list(&self) -> Vec<(String, usize)>;
}
impl ModelConfigParamList for YoloConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        <YoloConfig as model::ModelConfig>::param_list(self)
    }
}

// ---- model::Model seam ----

impl model::Model for Yolo {
    type Config = YoloConfig;

    fn new(cfg: YoloConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Yolo::new(cfg, b, t, init)
    }

    fn init_weights(cfg: &YoloConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_model(cfg, seed)
    }

    fn config(&self) -> &YoloConfig {
        &self.cfg
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            // Detection targets arrive via `set_targets`, NOT the `targets`
            // slice (which the detection loss ignores). We upload only the image.
            model::Batch::Tensor { inputs, .. } => Yolo::set_image(self, inputs),
            _ => panic!("yolo::Yolo only supports Batch::Tensor"),
        }
    }

    fn forward(&self) -> f32 {
        Yolo::forward(self)
    }
    fn backward(&self) {
        Yolo::backward(self)
    }
    fn zero_grads(&self) {
        Yolo::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Yolo::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        Yolo::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Yolo::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Yolo::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Yolo::read_grad(self, name)
    }
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, path: &str) {
        Yolo::save(self, path)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}
