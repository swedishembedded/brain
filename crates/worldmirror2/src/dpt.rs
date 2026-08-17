// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DPT dense heads (Depth-Anything-V2 style, `dense_head.py` parity).
//!
//! One builder, four instances (depth/pts/norm 3–4 ch; the GS head adds the
//! RGB `input_merger` and the `gs_renderer` gaussian-parameter convs).
//! Processes ONE frame per invocation (the reference chunks frames for the
//! same reason: the 256×296² and 128×518² intermediates are the memory peak).
//!
//! Direct kernel dispatch (conv2d / conv2d_dx-as-ConvTranspose / bilinear /
//! leaky_relu(0) etc.) rather than `vision::Conv` — every op maps 1:1 onto a
//! reference module for exact parity; fused eval paths can adopt later. No
//! buffer is ever bound read+write in one dispatch (wgpu aliasing rules).

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use paramstore::ParamStore;

use crate::config::MirrorConfig;

/// Kernel ids the DPT builder dispatches (indices into the model pipeline).
#[derive(Clone, Copy)]
pub struct DptKernels {
    pub layernorm: usize,
    pub nlc_nchw: usize,
    pub conv2d: usize,
    pub conv2d_dx: usize,
    pub add_chan_inplace: usize,
    pub resize_bilinear: usize,
    pub leaky_relu: usize,
    pub relu_inplace: usize,
    pub add2: usize,
    pub add_inplace: usize,
    pub axpy: usize,
}

/// Sinusoidal positional embedding `[C, ph, pw]` (grid.py parity: uv grid
/// normalized by the plane diagonal, f64 omega bands, `ratio` pre-applied).
pub fn pos_embed_chw(c: usize, ph: usize, pw: usize, ratio: f32) -> Vec<f32> {
    let aspect = pw as f64 / ph as f64; // == image W/H
    let diag = (aspect * aspect + 1.0).sqrt();
    let (span_x, span_y) = (aspect / diag, 1.0 / diag);
    let lin = |n: usize, span: f64, i: usize| {
        let lo = -span * (n as f64 - 1.0) / n as f64;
        let hi = span * (n as f64 - 1.0) / n as f64;
        if n == 1 { lo } else { lo + (hi - lo) * i as f64 / (n as f64 - 1.0) }
    };
    let quarter = c / 4;
    let mut omega = vec![0.0f64; quarter];
    for (k, o) in omega.iter_mut().enumerate() {
        *o = 1.0 / 100f64.powf(k as f64 / quarter as f64);
    }
    let mut out = vec![0.0f32; c * ph * pw];
    for y in 0..ph {
        for x in 0..pw {
            let u = lin(pw, span_x, x);
            let v = lin(ph, span_y, y);
            let px = y * pw + x;
            for k in 0..quarter {
                let (su, cu) = (u * omega[k]).sin_cos();
                let (sv, cv) = (v * omega[k]).sin_cos();
                // channel layout: [sin(u)·D/4, cos(u), sin(v), cos(v)]
                out[k * ph * pw + px] = (su * ratio as f64) as f32;
                out[(quarter + k) * ph * pw + px] = (cu * ratio as f64) as f32;
                out[(2 * quarter + k) * ph * pw + px] = (sv * ratio as f64) as f32;
                out[(3 * quarter + k) * ph * pw + px] = (cv * ratio as f64) as f32;
            }
        }
    }
    out
}

/// Weight accessor for one head prefix (`depth_head`, `gs_head`, …).
pub struct HeadWeights<'a> {
    pub ps: &'a ParamStore,
    pub prefix: &'a str,
}

impl<'a> HeadWeights<'a> {
    fn get(&self, name: &str) -> &'a DeviceBuffer {
        self.ps.w(&format!("{}.{name}", self.prefix))
    }
}

/// GS-head extra branch: RGB input-merger + gaussian-parameter convs.
pub struct GsBranch<'a> {
    /// Raw [0,1] CHW frame `[3, H, W]` on device.
    pub rgb: &'a DeviceBuffer,
    pub im_w: &'a DeviceBuffer,
    pub im_b: &'a DeviceBuffer,
    pub g0_w: &'a DeviceBuffer,
    pub g2_w: &'a DeviceBuffer,
    pub g2_b: &'a DeviceBuffer,
    /// Output `[12, H, W]`: quat(4) scale(3) opacity(1) sh_dc(3) weight(1).
    pub out: &'a DeviceBuffer,
}

/// Scratch buffers for one frame at grid (ph, pw); shared by all four heads.
pub struct DptScratch {
    pub tok_n: DeviceBuffer,   // [P, 2C]
    pub feat: DeviceBuffer,    // [2C, ph, pw]
    pub proj: DeviceBuffer,    // up to [1024, ph, pw]
    pub s0: DeviceBuffer,      // [256, 4ph, 4pw]
    pub s1: DeviceBuffer,      // [512, 2ph, 2pw]
    pub s3: DeviceBuffer,      // [1024, ⌈ph/2⌉, ⌈pw/2⌉]
    pub rn: [DeviceBuffer; 4], // [256, spatial_i]
    pub a: DeviceBuffer,       // fusion ping [256, 8ph, 8pw]
    pub b: DeviceBuffer,       // fusion pong
    pub t: DeviceBuffer,       // fusion tmp
    pub u: DeviceBuffer,       // fusion tmp 2 (RCU needs 4 distinct buffers)
    pub full_a: DeviceBuffer,  // [128, H, W]
    pub full_b: DeviceBuffer,  // [128, H, W]
    pub head32: DeviceBuffer,  // [f2/8 (= 32), H, W]
    pub gs256: DeviceBuffer,   // [256, H, W]
    pub pos: [DeviceBuffer; 4],
    pub pos_full: DeviceBuffer, // [128, H, W]
}

impl DptScratch {
    pub fn new(gpu: &Gpu, cfg: &MirrorConfig, ph: usize, pw: usize) -> DptScratch {
        let p = ph * pw;
        let (h, w) = (ph * cfg.patch, pw * cfg.patch);
        let f2 = cfg.dpt_feat; // 256
        let mk = |n: usize| gpu.storage(n as u64);
        let spat = [16 * p, 4 * p, p, ph.div_ceil(2) * pw.div_ceil(2)];
        let pos = [
            gpu.storage_init("dpt.pos0", &pos_embed_chw(cfg.dpt_proj[0], ph, pw, 0.1)),
            gpu.storage_init("dpt.pos1", &pos_embed_chw(cfg.dpt_proj[1], ph, pw, 0.1)),
            gpu.storage_init("dpt.pos2", &pos_embed_chw(cfg.dpt_proj[2], ph, pw, 0.1)),
            gpu.storage_init("dpt.pos3", &pos_embed_chw(cfg.dpt_proj[3], ph, pw, 0.1)),
        ];
        DptScratch {
            tok_n: mk(p * 2 * cfg.dim),
            feat: mk(2 * cfg.dim * p),
            proj: mk(cfg.dpt_proj[3] * p),
            s0: mk(cfg.dpt_proj[0] * 16 * p),
            s1: mk(cfg.dpt_proj[1] * 4 * p),
            s3: mk(cfg.dpt_proj[3] * spat[3]),
            rn: [mk(f2 * spat[0]), mk(f2 * spat[1]), mk(f2 * spat[2]), mk(f2 * spat[3])],
            a: mk(f2 * 64 * p),
            b: mk(f2 * 64 * p),
            t: mk(f2 * 64 * p),
            u: mk(f2 * 64 * p),
            full_a: mk((f2 / 2) * h * w),
            full_b: mk((f2 / 2) * h * w),
            head32: mk((f2 / 8) * h * w),
            gs256: mk(f2 * h * w),
            pos,
            pos_full: gpu.storage_init("dpt.pos_full", &pos_embed_chw(f2 / 2, h, w, 0.1)),
        }
    }
}

pub struct DptCtx<'a> {
    pub gpu: &'a Gpu,
    pub k: DptKernels,
    pub cfg: &'a MirrorConfig,
    pub scr: &'a DptScratch,
    pub eps: f32, // token-norm eps (torch default 1e-5)
}

impl<'a> DptCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &self,
        x: &DeviceBuffer,
        wt: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        out: &DeviceBuffer,
        (cin, hin, win): (usize, usize, usize),
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        steps: &mut Vec<Step>,
    ) -> (usize, usize, usize) {
        let ho = (hin + 2 * pad - k) / stride + 1;
        let wo = (win + 2 * pad - k) / stride + 1;
        let total = (cout * ho * wo) as u32;
        steps.push(self.gpu.step(
            self.k.conv2d,
            &[x, wt, out],
            &[1, cin as u32, hin as u32, win as u32, cout as u32, k as u32, stride as u32, pad as u32, ho as u32, wo as u32],
            total,
        ));
        if let Some(bv) = bias {
            steps.push(self.gpu.step(
                self.k.add_chan_inplace,
                &[out, bv],
                &[total, cout as u32, (ho * wo) as u32],
                total,
            ));
        }
        (cout, ho, wo)
    }

    /// ConvTranspose2d k×k stride k via `conv2d_dx` in gather form: the
    /// checkpoint weight `[in, out, k, k]` binds verbatim (dx-Cout = ConvT-in,
    /// dx-Cin = ConvT-out).
    #[allow(clippy::too_many_arguments)]
    pub fn deconv(
        &self,
        x: &DeviceBuffer,
        wt: &DeviceBuffer,
        bias: &DeviceBuffer,
        out: &DeviceBuffer,
        (cin, hin, win): (usize, usize, usize),
        cout: usize,
        k: usize,
        steps: &mut Vec<Step>,
    ) -> (usize, usize, usize) {
        let (ho, wo) = (hin * k, win * k);
        let total = (cout * ho * wo) as u32;
        steps.push(self.gpu.step(
            self.k.conv2d_dx,
            &[x, wt, out],
            &[1, cout as u32, ho as u32, wo as u32, cin as u32, k as u32, k as u32, 0, hin as u32, win as u32],
            total,
        ));
        steps.push(self.gpu.step(
            self.k.add_chan_inplace,
            &[out, bias],
            &[total, cout as u32, (ho * wo) as u32],
            total,
        ));
        (cout, ho, wo)
    }

    pub fn relu(&self, x: &DeviceBuffer, out: &DeviceBuffer, n: usize, steps: &mut Vec<Step>) {
        steps.push(self.gpu.step(self.k.leaky_relu, &[x, out], &[n as u32, f(0.0)], n as u32));
    }

    pub fn relu_inplace(&self, buf: &DeviceBuffer, n: usize, steps: &mut Vec<Step>) {
        steps.push(self.gpu.step(self.k.relu_inplace, &[buf], &[n as u32], n as u32));
    }

    pub fn bilinear(
        &self,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        (c, hin, win): (usize, usize, usize),
        (ho, wo): (usize, usize),
        steps: &mut Vec<Step>,
    ) {
        steps.push(self.gpu.step(
            self.k.resize_bilinear,
            &[x, out],
            &[1, c as u32, hin as u32, win as u32, ho as u32, wo as u32, 1],
            (c * ho * wo) as u32,
        ));
    }

    /// ResidualConvUnit — reference parity note: the reference uses
    /// `nn.ReLU(inplace=True)`, which MUTATES the block input, so the skip
    /// connection adds **relu(x)**, not x:
    ///   out = conv2(relu(conv1(relu(x)))) + relu(x)
    /// x, t1, t2, out must be 4 distinct buffers; result lands in `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn rcu(
        &self,
        hw: &HeadWeights,
        name: &str,
        x: &DeviceBuffer,
        t1: &DeviceBuffer,
        t2: &DeviceBuffer,
        out: &DeviceBuffer,
        dims: (usize, usize, usize),
        steps: &mut Vec<Step>,
    ) {
        let n = dims.0 * dims.1 * dims.2;
        self.relu(x, out, n, steps); // out = relu(x) — kept for the skip add
        self.conv(out, hw.get(&format!("{name}.conv1.weight")), Some(hw.get(&format!("{name}.conv1.bias"))), t1, dims, dims.0, 3, 1, 1, steps);
        self.relu_inplace(t1, n, steps);
        self.conv(t1, hw.get(&format!("{name}.conv2.weight")), Some(hw.get(&format!("{name}.conv2.bias"))), t2, dims, dims.0, 3, 1, 1, steps);
        steps.push(self.gpu.step(self.k.add_inplace, &[out, t2], &[n as u32], n as u32));
    }

    /// Record one head for one frame: `tap_bufs` = the 4 trunk taps
    /// `[s*td, 2C]`; result `[out_ch, H, W]` lands in `out` (pre-activation —
    /// exp/sigmoid/normalize happen on the host during assembly). The GS
    /// branch additionally writes its 12-channel parameter map.
    #[allow(clippy::too_many_arguments)]
    pub fn head_frame(
        &self,
        hw: &HeadWeights,
        tap_bufs: &[DeviceBuffer],
        frame: usize,
        td: usize,
        (ph, pw): (usize, usize),
        out_ch: usize,
        out: &DeviceBuffer,
        gs: Option<&GsBranch>,
        steps: &mut Vec<Step>,
    ) {
        let cfg = self.cfg;
        let scr = self.scr;
        let c2 = 2 * cfg.dim;
        let p = ph * pw;
        let (h, w) = (ph * cfg.patch, pw * cfg.patch);
        let f2 = cfg.dpt_feat;
        let patch_row0 = (frame * td + crate::model::PATCH_START) as u64;

        let dims = [
            (f2, 4 * ph, 4 * pw),
            (f2, 2 * ph, 2 * pw),
            (f2, ph, pw),
            (f2, ph.div_ceil(2), pw.div_ceil(2)),
        ];

        // `i` is the pyramid-scale index, not a cursor over `tap_bufs`: it also
        // selects `cfg.dpt_proj[i]`, `scr.pos[i]`, `scr.rn[i]` and the
        // `resize_layers.{i}` weights, and the scale count is fixed at 4 while
        // `tap_bufs` is caller-supplied, so its length is not provably 4.
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            steps.push(self.gpu.step_sliced(
                self.k.layernorm,
                &[&tap_bufs[i], hw.get("norm.weight"), hw.get("norm.bias"), &scr.tok_n],
                &[(patch_row0 * c2 as u64, (p * c2) as u64), (0, 0), (0, 0), (0, 0)],
                &[c2 as u32, p as u32, f(self.eps)],
                p as u32,
            ));
            steps.push(self.gpu.step(
                self.k.nlc_nchw,
                &[&scr.tok_n, &scr.feat],
                &[(c2 * p) as u32, c2 as u32, p as u32],
                (c2 * p) as u32,
            ));
            let oc = cfg.dpt_proj[i];
            self.conv(&scr.feat, hw.get(&format!("projects.{i}.weight")), Some(hw.get(&format!("projects.{i}.bias"))), &scr.proj, (c2, ph, pw), oc, 1, 1, 0, steps);
            steps.push(self.gpu.step(
                self.k.axpy,
                &[&scr.proj, &scr.pos[i]],
                &[(oc * p) as u32, f(1.0)],
                (oc * p) as u32,
            ));
            let (rin, rdims): (&DeviceBuffer, (usize, usize, usize)) = match i {
                0 => {
                    let d = self.deconv(&scr.proj, hw.get("resize_layers.0.weight"), hw.get("resize_layers.0.bias"), &scr.s0, (oc, ph, pw), oc, 4, steps);
                    (&scr.s0, d)
                }
                1 => {
                    let d = self.deconv(&scr.proj, hw.get("resize_layers.1.weight"), hw.get("resize_layers.1.bias"), &scr.s1, (oc, ph, pw), oc, 2, steps);
                    (&scr.s1, d)
                }
                2 => (&scr.proj, (oc, ph, pw)),
                _ => {
                    let d = self.conv(&scr.proj, hw.get("resize_layers.3.weight"), Some(hw.get("resize_layers.3.bias")), &scr.s3, (oc, ph, pw), oc, 3, 2, 1, steps);
                    (&scr.s3, d)
                }
            };
            self.conv(rin, hw.get(&format!("scratch.layer{}_rn.weight", i + 1)), None, &scr.rn[i], rdims, f2, 3, 1, 1, steps);
        }

        // ---- fusion 4 → 1 with the a/b/t ping-pong ----
        // refinenet4 (no residual unit): RCU2(rn4) → up(size of scale2) → out_conv
        self.rcu(hw, "scratch.refinenet4.resConfUnit2", &scr.rn[3], &scr.a, &scr.t, &scr.b, dims[3], steps);
        self.bilinear(&scr.b, &scr.a, dims[3], (dims[2].1, dims[2].2), steps);
        self.conv(&scr.a, hw.get("scratch.refinenet4.out_conv.weight"), Some(hw.get("scratch.refinenet4.out_conv.bias")), &scr.b, dims[2], f2, 1, 1, 0, steps);
        // refinenet3, 2, 1 — `b` holds the running fused map at dims[rn_i]
        for (r, rn_i) in [(3usize, 2usize), (2, 1), (1, 0)] {
            let pre = format!("scratch.refinenet{r}");
            let n = dims[rn_i].0 * dims[rn_i].1 * dims[rn_i].2;
            // rcu1(residual input) -> u; running (b) + u -> a; rcu2(a) -> b
            self.rcu(hw, &format!("{pre}.resConfUnit1"), &scr.rn[rn_i], &scr.a, &scr.t, &scr.u, dims[rn_i], steps);
            steps.push(self.gpu.step(self.k.add2, &[&scr.b, &scr.u, &scr.a], &[n as u32], n as u32));
            self.rcu(hw, &format!("{pre}.resConfUnit2"), &scr.a, &scr.t, &scr.u, &scr.b, dims[rn_i], steps);
            let target = if rn_i == 0 { (8 * ph, 8 * pw) } else { (dims[rn_i - 1].1, dims[rn_i - 1].2) };
            self.bilinear(&scr.b, &scr.a, dims[rn_i], target, steps);
            self.conv(&scr.a, hw.get(&format!("{pre}.out_conv.weight")), Some(hw.get(&format!("{pre}.out_conv.bias"))), &scr.b, (f2, target.0, target.1), f2, 1, 1, 0, steps);
        }

        // ---- output_conv1 → full-res bilinear (align_corners) → +pos ----
        self.conv(&scr.b, hw.get("scratch.output_conv1.weight"), Some(hw.get("scratch.output_conv1.bias")), &scr.a, (f2, 8 * ph, 8 * pw), f2 / 2, 3, 1, 1, steps);
        self.bilinear(&scr.a, &scr.full_a, (f2 / 2, 8 * ph, 8 * pw), (h, w), steps);
        steps.push(self.gpu.step(
            self.k.axpy,
            &[&scr.full_a, &scr.pos_full],
            &[((f2 / 2) * h * w) as u32, f(1.0)],
            ((f2 / 2) * h * w) as u32,
        ));

        // ---- output_conv2: conv3 → relu → conv1 ----
        self.conv(&scr.full_a, hw.get("scratch.output_conv2.0.weight"), Some(hw.get("scratch.output_conv2.0.bias")), &scr.head32, (f2 / 2, h, w), f2 / 8, 3, 1, 1, steps);
        self.relu_inplace(&scr.head32, (f2 / 8) * h * w, steps);
        self.conv(&scr.head32, hw.get("scratch.output_conv2.2.weight"), Some(hw.get("scratch.output_conv2.2.bias")), out, (f2 / 8, h, w), out_ch, 1, 1, 0, steps);

        // ---- GS branch: fused += relu(conv7(rgb)); gaussian-param convs ----
        if let Some(g) = gs {
            let n_full = (f2 / 2) * h * w;
            self.conv(g.rgb, g.im_w, Some(g.im_b), &scr.full_b, (3, h, w), f2 / 2, 7, 1, 3, steps);
            self.relu_inplace(&scr.full_b, n_full, steps);
            steps.push(self.gpu.step(
                self.k.axpy,
                &[&scr.full_a, &scr.full_b],
                &[n_full as u32, f(1.0)],
                n_full as u32,
            ));
            self.conv(&scr.full_a, g.g0_w, None, &scr.gs256, (f2 / 2, h, w), f2, 3, 1, 1, steps);
            self.relu_inplace(&scr.gs256, f2 * h * w, steps);
            self.conv(&scr.gs256, g.g2_w, Some(g.g2_b), g.out, (f2, h, w), 12, 1, 1, 0, steps);
        }
    }
}
