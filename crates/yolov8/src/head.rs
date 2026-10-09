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
//! `nn.Conv2d`). It is the shared [`Conv`] unit too, specified as exactly that
//! (`Norm::None`, `Act::None`, biased), so its forward, input/weight gradients
//! and bias gradient are the ones every other biased conv in brain runs.
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
//! two Convs) + `head.{s}.cls.2.{weight,bias}` (the final biased 1x1); the reg
//! branch is `head.{s}.reg.{0,1,2}` likewise.

use gpu_core::DeviceBuffer;
use paramstore::ParamStore;

use crate::blocks::{Act, Conv, ConvNames, ConvSpec, Norm};
use crate::net::{Ctx, Shape};

/// Repack per-scale NCHW logit maps into a flat `[N, A, C]` host tensor with
/// anchors ordered scale-major then row-major over `(H,W)` — the layout the loss
/// and the inference decode consume. `scales[s] = (data_nchw, h, w)`, each
/// `data_nchw` laid out `[N,C,H,W]` with the SAME `c` across scales;
/// `a = Σ_s h·w`.
///
/// This is the one definition shared by the engine path ([`Head::gather_flat`])
/// and the NPU path (which feeds OpenVINO's per-scale outputs straight in), so
/// both produce byte-identical anchor ordering.
///
/// A transpose between the two layouts, blocked by [`TRANSPOSE_TILE`] anchors:
/// the `[tile, c]` side is a small contiguous block that stays in cache while
/// each of the `c` channel rows contributes one contiguous run of `tile`
/// floats. Walking either layout element by element instead strides the other
/// by `c` (or by `H*W`) floats, and with power-of-two map sizes the `c`
/// channel rows are exactly a cache-way apart, so every access misses.
pub fn repack_heads_to_flat(scales: &[(&[f32], u32, u32)], n: usize, c: usize, a: usize) -> Vec<f32> {
    let mut flat = vec![0.0f32; n * a * c];
    let mut anchor_base = 0usize;
    for &(data, h, w) in scales {
        let hw = (h * w) as usize;
        for nn in 0..n {
            let src = &data[nn * c * hw..(nn + 1) * c * hw];
            let dst = &mut flat[(nn * a + anchor_base) * c..(nn * a + anchor_base + hw) * c];
            for p0 in (0..hw).step_by(TRANSPOSE_TILE) {
                let t = TRANSPOSE_TILE.min(hw - p0);
                for ch in 0..c {
                    let row = &src[ch * hw + p0..ch * hw + p0 + t];
                    for (i, &v) in row.iter().enumerate() {
                        dst[(p0 + i) * c + ch] = v;
                    }
                }
            }
        }
        anchor_base += hw;
    }
    flat
}

/// The inverse of [`repack_heads_to_flat`] for one scale: the `[N, hw, c]`
/// slice of a flat `[N, A, c]` tensor starting at anchor `anchor_base`, back to
/// that scale's NCHW `[N, c, H, W]` map, each channel `ch` multiplied by
/// `gate(ch)` (a gate of exactly 0 writes 0, whatever the flat value).
/// Blocked like the repack, for the same reason.
pub fn unpack_flat_to_head(flat: &[f32], n: usize, c: usize, a: usize, anchor_base: usize, hw: usize, gate: impl Fn(usize) -> f32) -> Vec<f32> {
    let gates: Vec<f32> = (0..c).map(gate).collect();
    let mut nchw = vec![0.0f32; n * c * hw];
    for nn in 0..n {
        let src = &flat[(nn * a + anchor_base) * c..(nn * a + anchor_base + hw) * c];
        let dst = &mut nchw[nn * c * hw..(nn + 1) * c * hw];
        for p0 in (0..hw).step_by(TRANSPOSE_TILE) {
            let t = TRANSPOSE_TILE.min(hw - p0);
            for (ch, &g) in gates.iter().enumerate() {
                let row = &mut dst[ch * hw + p0..ch * hw + p0 + t];
                for (i, v) in row.iter_mut().enumerate() {
                    *v = if g == 0.0 { 0.0 } else { src[(p0 + i) * c + ch] * g };
                }
            }
        }
    }
    nchw
}

/// Anchors per block of the layout transposes: one 64-byte cache line of a
/// channel row.
const TRANSPOSE_TILE: usize = 16;

/// One scale's cls or reg branch: two `Conv`s then a BIASED 1x1 conv.
pub struct Branch {
    pub c0: Conv,
    pub c1: Conv,
    /// The final biased 1x1 (`P.2.{weight,bias}`): a raw conv, no BN, no act.
    pub c2: Conv,
    pub mid: u32,
    pub out_c: u32,
    pub out_shape: Shape,

    d_c1: DeviceBuffer, // grad wrt c1.out  [n,mid,h,w]
    d_c0: DeviceBuffer, // grad wrt c0.out  [n,mid,h,w]
}

impl Branch {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, mid: u32, out_c: u32, train: bool) -> Branch {
        let c0 = Conv::new(ctx, &format!("{prefix}.0"), in_shape, mid, 3, 1, 1, train);
        let c1 = Conv::new(ctx, &format!("{prefix}.1"), c0.out_shape, mid, 3, 1, 1, train);
        // Ultralytics names the plain `nn.Conv2d` by its position: `P.2.weight`
        // and `P.2.bias`. It has no BatchNorm, so the BN names are never read.
        let c2_prefix = format!("{prefix}.2");
        let names = ConvNames { weight: format!("{c2_prefix}.weight"), bias: format!("{c2_prefix}.bias"), ..ConvNames::torch_flat(&c2_prefix) };
        let spec = ConvSpec { norm: Norm::None, act: Act::None, ..ConvSpec::silu(out_c, 1, 1, 0) }.with_bias();
        // Not a tap site: the quantized export keeps these logits' projection
        // in full precision (no Q/DQ pair in front of it).
        let c2 = Conv::with_names(ctx, &c2_prefix, names, c1.out_shape, spec, train).without_tap();
        let out_shape = c2.out_shape;
        let mid_n = c1.out_shape.numel();
        Branch { c0, c1, c2, mid, out_c, out_shape, d_c1: ctx.act(mid_n), d_c0: ctx.act(mid_n) }
    }

    pub fn out(&self) -> &DeviceBuffer {
        self.c2.out()
    }

    /// Propagate the eval/train BN toggle to the two Convs (the final bias-free
    /// 1x1 has no BN, so nothing to flip there).
    pub fn set_eval(&self, eval: bool) {
        self.c0.set_eval(eval);
        self.c1.set_eval(eval);
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
        v.extend(self.c2.param_list());
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.c0.forward(ctx, ps, x_in);
        self.c1.forward(ctx, ps, self.c0.out());
        self.c2.forward(ctx, ps, self.c1.out());
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
        self.c2.backward(ctx, ps, self.c1.out(), d_out, &self.d_c1);
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
        let s = ctx.step(crate::net::ADD_INPLACE, &[d_in, &self.d_in_cls], &[n], n);
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
    /// count (nc or 4*reg_max). Delegates the host repack to
    /// [`repack_heads_to_flat`] so the engine path and the NPU decode path share
    /// one definition.
    fn gather_flat(
        &self,
        ctx: &Ctx,
        out: impl Fn(&ScaleHead) -> &DeviceBuffer,
        c: u32,
        shape: impl Fn(&ScaleHead) -> Shape,
    ) -> Vec<f32> {
        let n = self.in_shapes[0].n as usize;
        let a = self.num_anchors() as usize;
        let datas: Vec<(Vec<f32>, u32, u32)> = self
            .scales
            .iter()
            .map(|scale| {
                let sh = shape(scale);
                (ctx.gpu.read(out(scale), sh.numel() as usize), sh.h, sh.w)
            })
            .collect();
        let refs: Vec<(&[f32], u32, u32)> = datas.iter().map(|(d, h, w)| (d.as_slice(), *h, *w)).collect();
        repack_heads_to_flat(&refs, n, c as usize, a)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack is the plain `NCHW -> [N, A, C]` index map, anchors
    /// scale-major then row-major, and the per-scale unpack is its exact
    /// inverse (a gate of 1 is a bit-exact copy; a gate of 0 writes 0).
    #[test]
    fn the_flat_repack_and_its_unpack_are_inverse_index_maps() {
        let (n, c) = (3usize, 5usize);
        let scales_hw = [(4u32, 3u32), (2, 2), (1, 1)];
        let a: usize = scales_hw.iter().map(|&(h, w)| (h * w) as usize).sum();
        let maps: Vec<Vec<f32>> = scales_hw
            .iter()
            .enumerate()
            .map(|(s, &(h, w))| (0..n * c * (h * w) as usize).map(|i| (s * 1000 + i) as f32 - 0.5).collect())
            .collect();
        let refs: Vec<(&[f32], u32, u32)> = maps.iter().zip(&scales_hw).map(|(m, &(h, w))| (m.as_slice(), h, w)).collect();
        let flat = repack_heads_to_flat(&refs, n, c, a);

        let mut base = 0usize;
        for (m, &(h, w)) in maps.iter().zip(&scales_hw) {
            let hw = (h * w) as usize;
            for nn in 0..n {
                for ch in 0..c {
                    for p in 0..hw {
                        assert_eq!(flat[(nn * a + base + p) * c + ch], m[(nn * c + ch) * hw + p]);
                    }
                }
            }
            assert_eq!(&unpack_flat_to_head(&flat, n, c, a, base, hw, |_| 1.0), m);
            let gated = unpack_flat_to_head(&flat, n, c, a, base, hw, |ch| if ch == 2 { 0.0 } else { 2.0 });
            for nn in 0..n {
                for p in 0..hw {
                    assert_eq!(gated[(nn * c + 2) * hw + p], 0.0);
                    assert_eq!(gated[(nn * c + 1) * hw + p], 2.0 * m[(nn * c + 1) * hw + p]);
                }
            }
            base += hw;
        }
    }
}
