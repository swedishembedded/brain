// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 decoupled anchor-free detection head (P2).
//!
//! Per input feature map (`Cin x H x W`) the head has two branches:
//!   * cls: `Conv(K3,s1, Cin->nc')` -> `Conv(K3,s1, nc'->nc')` ->
//!     BIASED `conv2d(K1, nc'->nc)`  => `nc` class logits per cell.
//!   * reg: `Conv(K3,s1, Cin->reg')` -> `Conv(K3,s1)` ->
//!     BIASED `conv2d(K1, reg'->4*reg_max)` => box-distribution logits.
//!
//! The two `Conv`s in each branch are the full Conv (conv+BN+SiLU); the final
//! 1x1 is a bare convolution (no BN/activation) plus a per-output-channel learned
//! bias, matching Ultralytics' detection head (its last layer is a plain biased
//! `nn.Conv2d`).
//!
//! ## Head bias (P12): NCHW + the `[M,N]` bias kernels
//!
//! The logits are NCHW `[N,C,H,W]` and the bias is per-channel `C`. The shared
//! `bias_add`/`bias_grad` kernels are hardwired to `[M,N]` row-major where the
//! biased dim `N` is the TRAILING (contiguous) dim — i.e. they compute
//! `out[idx] += b[idx % N]` / `db[col] += sum_m dy[m*N + col]`. In NCHW the
//! channel is NOT the trailing dim, so we cannot pass the `[C]` bias directly.
//!
//! Instead we view the buffer as `[M=N, N=C*H*W]` and use a per-image-independent
//! BROADCAST bias `bcast[c*HW + p] = bias[c]` (length `C*H*W`):
//!   * forward: `bias_add(m=N, n=C*HW, bcast)` -> `out[idx] += bcast[idx % (C*HW)]`
//!     and `idx % (C*HW) = c*HW + p`, so each element gets `bias[c]`. Correct.
//!   * backward: `bias_grad(m=N, n=C*HW)` -> `dbcast[c*HW+p] = sum_n dy[...]`
//!     (sum over the N images); then host-reduce `dbias[c] = sum_p dbcast[c*HW+p]`
//!     (sum over spatial), giving the per-channel grad summed over N and H*W.
//! Both use the EXISTING kernels verbatim; no new kernel is added.
//!
//! [`Head`] wires the three scales; [`Head::forward`] runs them. The raw logit
//! maps are concatenated across scales into `[A, nc]` (cls) and
//! `[A, 4*reg_max]` (box) by [`Head::cls_logits_flat`] / [`Head::box_logits_flat`]
//! (`A = sum of H*W over the 3 scales`) — the network's raw output. DFL decode
//! (the `dfl_decode` kernel) and the anchor/stride tables are STUBBED here
//! ([`Head::anchors`]/[`Head::strides`] return the geometry only); the full
//! decode->box path is P6.
//!
//! Param-naming: per scale `s`, the cls branch is `head.{s}.cls.0` / `.1` (the
//! two Convs) + `head.{s}.cls.2.weight` (the final bias-free 1x1); the reg
//! branch is `head.{s}.reg.{0,1,2}` likewise.

use gpu_core::DeviceBuffer;
use paramstore::ParamStore;

use crate::blocks::Conv;
use crate::net::{Ctx, Shape, BIAS_ADD, BIAS_GRAD, CONV2D, CONV2D_DW, CONV2D_DX};

/// One scale's cls or reg branch: two `Conv`s then a BIASED 1x1 conv.
pub struct Branch {
    pub c0: Conv,
    pub c1: Conv,
    prefix: String,
    pub mid: u32,
    pub out_c: u32,
    pub out_shape: Shape,

    logits: DeviceBuffer, // final 1x1 output [n,out_c,h,w]
    d_c1: DeviceBuffer,   // grad wrt c1.out  [n,mid,h,w]
    d_c0: DeviceBuffer,   // grad wrt c0.out  [n,mid,h,w]

    // head-bias plumbing (see module docs). C*H*W elements.
    bcast: DeviceBuffer,  // broadcast bias bcast[c*HW+p] = bias[c]
    dbcast: DeviceBuffer, // bias_grad output before the host spatial-reduce
    /// Whether `bcast` holds the current (constant in eval) broadcast bias, so
    /// inference packs it once instead of a host readback+write every frame.
    bcast_ready: std::cell::Cell<bool>,
}

impl Branch {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, mid: u32, out_c: u32, train: bool) -> Branch {
        let c0 = Conv::new(ctx, &format!("{prefix}.0"), in_shape, mid, 3, 1, 1, train);
        let c1 = Conv::new(ctx, &format!("{prefix}.1"), c0.out_shape, mid, 3, 1, 1, train);
        let out_shape = c1.out_shape.conv_out(out_c, 1, 1, 0);
        let mid_n = c1.out_shape.numel();
        let chw = out_c * out_shape.h * out_shape.w; // C*H*W (per image)
        Branch {
            c0,
            c1,
            prefix: prefix.to_string(),
            mid,
            out_c,
            out_shape,
            logits: ctx.act(out_shape.numel()),
            d_c1: ctx.act(mid_n),
            d_c0: ctx.act(mid_n),
            bcast: ctx.act(chw),
            dbcast: ctx.act(chw),
            bcast_ready: std::cell::Cell::new(false),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.logits
    }

    /// Propagate the eval/train BN toggle to the two Convs (the final bias-free
    /// 1x1 has no BN, so nothing to flip there).
    pub fn set_eval(&self, eval: bool) {
        self.c0.set_eval(eval);
        self.c1.set_eval(eval);
        // Re-entering train mode means the bias can change again: recompute bcast.
        if !eval {
            self.bcast_ready.set(false);
        }
    }

    /// Propagate the BN running-stat update toggle to the two Convs (the final
    /// bias-free 1x1 has no BN, so nothing to toggle there).
    pub fn set_update_running(&self, on: bool) {
        self.c0.set_update_running(on);
        self.c1.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.c0.param_list();
        v.extend(self.c1.param_list());
        v.push((format!("{}.2.weight", self.prefix), (self.out_c * self.mid) as usize));
        v.push((format!("{}.2.bias", self.prefix), self.out_c as usize));
        v
    }

    fn conv1x1_params(&self) -> [u32; 10] {
        let cin = self.c1.out_shape;
        [cin.n, cin.c, cin.h, cin.w, self.out_c, 1, 1, 0, cin.h, cin.w]
    }

    /// Build the broadcast bias `bcast[c*HW+p] = bias[c]` on the host from the
    /// `[C]` bias param (see module docs). Returns `(N, C*HW)` for the kernel.
    fn pack_bcast(&self, ctx: &Ctx, ps: &ParamStore) -> (u32, u32) {
        let c = self.out_c as usize;
        let hw = (self.out_shape.h * self.out_shape.w) as usize;
        let bias = ctx.gpu.read(ps.w(&format!("{}.2.bias", self.prefix)), c);
        let mut bcast = vec![0.0f32; c * hw];
        for ch in 0..c {
            let b = bias[ch];
            let base = ch * hw;
            for p in 0..hw {
                bcast[base + p] = b;
            }
        }
        ctx.gpu.write(&self.bcast, bytemuck::cast_slice(&bcast));
        (self.out_shape.n, (self.out_c * self.out_shape.h * self.out_shape.w))
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.c0.forward(ctx, ps, x_in);
        self.c1.forward(ctx, ps, self.c0.out());
        let w = ps.w(&format!("{}.2.weight", self.prefix));
        let s = ctx.step(CONV2D, &[self.c1.out(), w, &self.logits], &self.conv1x1_params(), self.out_shape.numel());
        ctx.gpu.submit(&[], &[s]);
        // Add the per-channel bias via the [M,N] bias_add on the [N, C*HW] view.
        // In eval the bias is constant, so pack the broadcast buffer once (skip the
        // per-frame host readback+write); in train it can change, so always repack.
        if !(self.c0.is_eval() && self.bcast_ready.get()) {
            self.pack_bcast(ctx, ps);
            if self.c0.is_eval() {
                self.bcast_ready.set(true);
            }
        }
        let m = self.out_shape.n;
        let n = self.out_c * self.out_shape.h * self.out_shape.w;
        let sb = ctx.step(BIAS_ADD, &[&self.logits, &self.bcast], &[m, n], m * n);
        ctx.gpu.submit(&[], &[sb]);
    }

    /// Backward. `d_out` = grad wrt this branch's raw-logit output; `d_in`
    /// receives grad wrt `x_in`.
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let wname = format!("{}.2.weight", self.prefix);
        let bname = format!("{}.2.bias", self.prefix);
        let dw_n = self.out_c * self.mid; // K=1
        // Bias grad: sum d_out over N (via bias_grad on the [N, C*HW] view) into
        // dbcast, then host-reduce over the HW spatial positions per channel and
        // accumulate into the [C] bias grad buffer. d(conv_out) = d_out (the bias
        // add is the identity wrt the conv output, so it does not change d_c1/d_x).
        let hw = self.out_shape.h * self.out_shape.w;
        let n = self.out_c * hw; // C*HW
        let m = self.out_shape.n; // N images
        // Clear dbcast first: bias_grad ACCUMULATES (`dbcast[col] += ...`), and
        // dbcast is reused across backward passes, so it must start at zero.
        let s_bias = ctx.step(BIAS_GRAD, &[d_out, &self.dbcast], &[m, n], n);
        ctx.gpu.submit(&[&self.dbcast], &[s_bias]);
        let dbcast = ctx.gpu.read(&self.dbcast, n as usize);
        let mut dbias = vec![0.0f32; self.out_c as usize];
        for ch in 0..self.out_c as usize {
            let base = ch * hw as usize;
            let mut acc = 0.0f32;
            for p in 0..hw as usize {
                acc += dbcast[base + p];
            }
            dbias[ch] = acc;
        }
        // Accumulate into the (pre-zeroed) grad buffer (single consumer/backward).
        let cur = ctx.gpu.read(ps.g(&bname), self.out_c as usize);
        let merged: Vec<f32> = cur.iter().zip(&dbias).map(|(a, b)| a + b).collect();
        ctx.gpu.write(ps.g(&bname), bytemuck::cast_slice(&merged));

        let s_dw = ctx.step(CONV2D_DW, &[d_out, self.c1.out(), ps.g(&wname)], &self.conv1x1_params(), dw_n);
        let s_dx = ctx.step(CONV2D_DX, &[d_out, ps.w(&wname), &self.d_c1], &self.conv1x1_params(), self.c1.out_shape.numel());
        ctx.gpu.submit(&[], &[s_dw, s_dx]);
        self.c1.backward(ctx, ps, self.c0.out(), &self.d_c1, &self.d_c0);
        self.c0.backward(ctx, ps, x_in, &self.d_c0, d_in);
    }
}

/// One pyramid scale's decoupled head: a cls branch + a reg branch sharing the
/// scale's input feature map.
pub struct ScaleHead {
    pub cls: Branch,
    pub reg: Branch,
    pub in_shape: Shape,
    d_in_cls: DeviceBuffer, // grad wrt input from the cls branch
}

impl ScaleHead {
    pub fn new(
        ctx: &Ctx,
        prefix: &str,
        in_shape: Shape,
        nc: u32,
        reg_max: u32,
        cls_mid: u32,
        reg_mid: u32,
        train: bool,
    ) -> ScaleHead {
        let cls = Branch::new(ctx, &format!("{prefix}.cls"), in_shape, cls_mid, nc, train);
        let reg = Branch::new(ctx, &format!("{prefix}.reg"), in_shape, reg_mid, 4 * reg_max, train);
        ScaleHead { cls, reg, in_shape, d_in_cls: ctx.act(in_shape.numel()) }
    }

    /// Propagate the eval/train BN toggle to both branches.
    pub fn set_eval(&self, eval: bool) {
        self.cls.set_eval(eval);
        self.reg.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to both branches.
    pub fn set_update_running(&self, on: bool) {
        self.cls.set_update_running(on);
        self.reg.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cls.param_list();
        v.extend(self.reg.param_list());
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.cls.forward(ctx, ps, x_in);
        self.reg.forward(ctx, ps, x_in);
    }

    /// Backward for both branches. The shared input grad is the sum of the two
    /// branch contributions: reg writes `d_in`, cls writes `d_in_cls`, then
    /// add2 merges them into `d_in`.
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_cls: &DeviceBuffer,
        d_reg: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        self.reg.backward(ctx, ps, x_in, d_reg, d_in);
        self.cls.backward(ctx, ps, x_in, d_cls, &self.d_in_cls);
        let n = self.in_shape.numel();
        let s = ctx.step(crate::net::ADD2, &[d_in, &self.d_in_cls, d_in], &[n], n);
        ctx.gpu.submit(&[], &[s]);
    }
}

/// The full decoupled head over the 3 pyramid scales. Owns a [`ScaleHead`] per
/// input feature map and exposes the raw per-scale logit maps flattened +
/// concatenated as the network's raw output.
pub struct Head {
    pub scales: Vec<ScaleHead>,
    pub nc: u32,
    pub reg_max: u32,
    pub strides: [u32; 3],
    in_shapes: Vec<Shape>,
}

impl Head {
    /// Build a head for the 3 `in_shapes` (P3/P4/P5 feature maps). `cls_mid` /
    /// `reg_mid` are the small intermediate widths for the tiny config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Ctx,
        prefix: &str,
        in_shapes: [Shape; 3],
        nc: u32,
        reg_max: u32,
        cls_mid: u32,
        reg_mid: u32,
        strides: [u32; 3],
        train: bool,
    ) -> Head {
        let scales = (0..3)
            .map(|s| {
                ScaleHead::new(
                    ctx,
                    &format!("{prefix}.{s}"),
                    in_shapes[s],
                    nc,
                    reg_max,
                    cls_mid,
                    reg_mid,
                    train,
                )
            })
            .collect();
        Head { scales, nc, reg_max, strides, in_shapes: in_shapes.to_vec() }
    }

    /// Propagate the eval/train BN toggle to every scale head.
    pub fn set_eval(&self, eval: bool) {
        for s in &self.scales {
            s.set_eval(eval);
        }
    }

    /// Propagate the BN running-stat update toggle to every scale head.
    pub fn set_update_running(&self, on: bool) {
        for s in &self.scales {
            s.set_update_running(on);
        }
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        self.scales.iter().flat_map(|s| s.param_list()).collect()
    }

    /// Run every scale forward on its feature map `xs[s]`.
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, xs: &[&DeviceBuffer; 3]) {
        for (s, scale) in self.scales.iter().enumerate() {
            scale.forward(ctx, ps, xs[s]);
        }
    }

    /// Total anchor count `A = sum_s H_s * W_s` across the 3 scales.
    pub fn num_anchors(&self) -> u32 {
        self.in_shapes.iter().map(|s| s.h * s.w).sum()
    }

    /// Class logits flattened + concatenated across scales into a host `[N, A,
    /// nc]` row-major tensor (anchors ordered scale-major, then row-major over
    /// H,W). This is the network's raw cls output; the loss (P3) consumes it.
    pub fn cls_logits_flat(&self, ctx: &Ctx) -> Vec<f32> {
        self.gather_flat(ctx, |sc| sc.cls.out(), self.nc, |sc| sc.cls.out_shape)
    }

    /// Box-distribution logits flattened + concatenated across scales into a host
    /// `[N, A, 4*reg_max]` tensor (DFL bins kept interleaved per side). Raw box
    /// output; DFL decode (P6) turns these into boxes.
    pub fn box_logits_flat(&self, ctx: &Ctx) -> Vec<f32> {
        self.gather_flat(ctx, |sc| sc.reg.out(), 4 * self.reg_max, |sc| sc.reg.out_shape)
    }

    /// Read each scale's `[N,C,H,W]` logit map and repack to `[N, (sum H*W), C]`
    /// with anchors scale-major then row-major. `c` is the per-cell channel
    /// count (nc or 4*reg_max).
    fn gather_flat(
        &self,
        ctx: &Ctx,
        out: impl Fn(&ScaleHead) -> &DeviceBuffer,
        c: u32,
        shape: impl Fn(&ScaleHead) -> Shape,
    ) -> Vec<f32> {
        let n = self.in_shapes[0].n as usize;
        let a = self.num_anchors() as usize;
        let cc = c as usize;
        let mut flat = vec![0.0f32; n * a * cc];
        let mut anchor_base = 0usize;
        for scale in &self.scales {
            let sh = shape(scale);
            let hw = (sh.h * sh.w) as usize;
            let data = ctx.gpu.read(out(scale), (sh.numel()) as usize);
            // data is NCHW: index ((nn*C + ch)*H + h)*W + w = nn*C*hw + ch*hw + p
            for nn in 0..n {
                for ch in 0..cc {
                    for p in 0..hw {
                        let src = (nn * cc + ch) * hw + p;
                        let dst = (nn * a + anchor_base + p) * cc + ch;
                        flat[dst] = data[src];
                    }
                }
            }
            anchor_base += hw;
        }
        flat
    }

    /// Anchor-point centers `(ax, ay)` per cell, in FEATURE units (`ax =
    /// w + 0.5`, `ay = h + 0.5`), concatenated scale-major. The DFL decode
    /// (P4 loss) scales these by the per-anchor stride to reach pixel boxes.
    pub fn anchors(&self) -> Vec<(f32, f32)> {
        let mut v = Vec::with_capacity(self.num_anchors() as usize);
        for sh in &self.in_shapes {
            for h in 0..sh.h {
                for w in 0..sh.w {
                    v.push((w as f32 + 0.5, h as f32 + 0.5));
                }
            }
        }
        v
    }

    /// Full per-anchor geometry the P4 assigner needs: pixel-space center
    /// `(cx,cy) = (ax*stride, ay*stride)`, the feature-unit anchor point
    /// `(ax,ay)`, and the anchor's stride. Scale-major, one entry per anchor.
    pub fn anchor_geometry(&self) -> Vec<crate::assign::Anchor> {
        let mut v = Vec::with_capacity(self.num_anchors() as usize);
        for (s, sh) in self.in_shapes.iter().enumerate() {
            let stride = self.strides[s] as f32;
            for h in 0..sh.h {
                for w in 0..sh.w {
                    let (ax, ay) = (w as f32 + 0.5, h as f32 + 0.5);
                    v.push(crate::assign::Anchor {
                        cx: ax * stride,
                        cy: ay * stride,
                        ax,
                        ay,
                        stride,
                    });
                }
            }
        }
        v
    }

    /// STUB (P6): per-anchor stride, scale-major (each anchor inherits its
    /// scale's stride). Full decode->box uses these to scale DFL expectations.
    pub fn anchor_strides(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.num_anchors() as usize);
        for (s, sh) in self.in_shapes.iter().enumerate() {
            let stride = self.strides[s];
            for _ in 0..(sh.h * sh.w) {
                v.push(stride);
            }
        }
        v
    }
}
