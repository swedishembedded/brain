// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND fine-tuning: the UNet as a TRAINING graph (forward + backward,
//! SSA-recorded once) behind the `model::Model` seam, which buys the blanket
//! finite-difference gradcheck and the AdamW/checkpoint machinery.
//!
//! Scope (v1): every CONVOLUTION's weight+bias is trainable (~3.7M params —
//! conv_in/conv_out, all resblock conv1/conv2/proj, down/upsample convs,
//! attention qkv/out projections). The conditioning path (Fourier, action
//! embedding, cond MLP, AdaGroupNorm linears) and the three tiny affine-GN
//! sites stay FROZEN: their gamma/beta arrive as constant per-batch inputs,
//! and gradients flow THROUGH GroupNorm via gn_dx. Loss is the reference's
//! F-space MSE: target_F = (clean - c_skip * noisy) / c_out
//! (denoiser.py::forward), sigma ~ exp(N(loc, scale^2)) clipped.
//!
//! Batch is 1 (one transition per step) in v1; determinism contract for the
//! gradcheck: noise is PRE-APPLIED by the batch builder, so forward() is a
//! pure function of (weights, batch).

use crate::cond::{conditioners, AdaGnSite, CondNet};
use crate::config::DiamondConfig;
use crate::model::Tensors;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use paramstore::ParamStore;
use std::cell::RefCell;
use std::collections::HashMap;

// Training kernel table (indices are local to this table).
const K_CONV: usize = 0; // conv_bias_reg fwd
const K_CONV_DX: usize = 1;
const K_CONV_DW: usize = 2;
const K_BIAS_GRAD: usize = 3;
const K_GN_PART: usize = 4;
const K_GN_STATS2: usize = 5;
const K_GN_APPLY: usize = 6;
const K_SCALE_CHAN: usize = 7;
const K_GN_DSUM: usize = 8;
const K_GN_DX: usize = 9;
const K_SILU: usize = 10;
const K_SILU_BWD: usize = 11;
const K_ADD2: usize = 12;
const K_CONCAT2: usize = 13;
const K_CONCAT_SPLIT: usize = 14;
const K_UPSAMPLE2: usize = 15;
const K_UPSAMPLE2_DX: usize = 16;
const K_NCHW_NLC: usize = 17;
const K_NLC_NCHW: usize = 18;
const K_ATTN_SCORES: usize = 19;
const K_ATTN_SOFTMAX: usize = 20;
const K_ATTN_APPLY: usize = 21;
const K_ATTN_BWD_DSCORES: usize = 22;
const K_ATTN_BWD_DV: usize = 23;
const K_ATTN_BWD_DQ: usize = 24;
const K_ATTN_BWD_DK: usize = 25;
const K_MSE_VALUE: usize = 26;
const K_MSE_GRAD: usize = 27;
const K_ADAMW: usize = 28;
const K_GRADNORM_SQ: usize = 29;
const K_GRAD_SCALE: usize = 30;
const K_CLIP_COEF: usize = 31;
const K_GRAD_SCALE_BUF: usize = 32;

const KERNELS: [(&str, &str); 33] = [
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("gn_part", kernels::GN_PART),
    ("gn_stats2", kernels::GN_STATS2),
    ("gn_apply", kernels::GN_APPLY),
    ("scale_chan", kernels::SCALE_CHAN),
    ("gn_dsum", kernels::GN_DSUM),
    ("gn_dx", kernels::GN_DX),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("add2", kernels::ADD2),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("upsample2", kernels::UPSAMPLE2),
    ("upsample2_dx", kernels::UPSAMPLE2_DX),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("mse_value", kernels::MSE_VALUE),
    ("mse_grad", kernels::MSE_GRAD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

const GN_EPS: f32 = 1e-5;
const GN_P: u32 = 64;
const ATTN_HEAD_DIM: u32 = 8;

fn num_groups(c: u32) -> u32 {
    (c / 32).max(1)
}

fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    gpu.write(buf, &bits);
}

/// A value in the training graph: activation buffer + its gradient buffer.
#[derive(Clone)]
struct TVal {
    buf: DeviceBuffer,
    grad: DeviceBuffer,
    len: u32,
}

struct TB<'a> {
    gpu: &'a Gpu,
    ps: &'a ParamStore,
    t: &'a Tensors,
    cc: usize,
    fwd: Vec<Step>,
    /// Backward step groups in FORWARD order; flattened reversed at the end.
    bwd: Vec<Vec<Step>>,
    adagn: Vec<(AdaGnSite, DeviceBuffer)>,
}

impl<'a> TB<'a> {
    fn val(&self, len: u32) -> TVal {
        TVal { buf: self.gpu.storage(len as u64), grad: self.gpu.storage(len as u64), len }
    }


    fn host(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        self.t.get(name).unwrap_or_else(|| panic!("diamond train: missing tensor {name}")).clone()
    }

    /// Trainable conv weight/bias from the ParamStore (with their grads).
    fn pw(&self, name: &str) -> (&'a DeviceBuffer, &'a DeviceBuffer) {
        let w = self.ps.weight.get(name).unwrap_or_else(|| panic!("ps missing {name}"));
        let g = self.ps.grad.get(name).unwrap_or_else(|| panic!("ps missing grad {name}"));
        (w, g)
    }

    /// Fork a value into two consumer slots; their grads are summed back.
    fn fork(&mut self, v: &TVal) -> (TVal, TVal) {
        let a = TVal { buf: v.buf.clone(), grad: self.gpu.storage(v.len as u64), len: v.len };
        let b = TVal { buf: v.buf.clone(), grad: self.gpu.storage(v.len as u64), len: v.len };
        self.bwd.push(vec![self.gpu.step(
            K_ADD2,
            &[&a.grad, &b.grad, &v.grad],
            &[v.len],
            v.len,
        )]);
        (a, b)
    }

    /// Trainable conv (+bias). Records fwd (register-tiled) and bwd
    /// (dx via conv2d_dx, dw via conv2d_dw, dbias via permute+bias_grad).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        h: u32,
        w: u32,
        x: &TVal,
    ) -> (TVal, u32, u32) {
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        let (wgt, dwgt) = self.pw(&format!("{prefix}.weight"));
        let (bias, dbias) = self.pw(&format!("{prefix}.bias"));
        let y = self.val(cout * ho * wo);
        let dims = [1, cin, h, w, cout, k, stride, pad, ho, wo];
        let threads = cout.div_ceil(8) * (ho * wo).div_ceil(4);
        self.fwd.push(self.gpu.step(K_CONV, &[&x.buf, wgt, bias, &y.buf], &dims, threads));
        // Backward: dy is y.grad.
        let dy_nlc = self.gpu.storage((cout * ho * wo) as u64);
        self.bwd.push(vec![
            self.gpu.step(K_CONV_DX, &[&y.grad, wgt, &x.grad], &dims, cin * h * w),
            self.gpu.step(K_CONV_DW, &[&y.grad, &x.buf, dwgt], &dims, cout * cin * k * k),
            // dbias[c] = sum over positions: permute [C,HW] -> [HW,C], col-sum.
            self.gpu.step(K_NCHW_NLC, &[&y.grad, &dy_nlc], &[cout * ho * wo, cout, ho * wo], cout * ho * wo),
            self.gpu.step(K_BIAS_GRAD, &[&dy_nlc, dbias], &[ho * wo, cout], cout),
        ]);
        (y, ho, wo)
    }

    /// GroupNorm with a FROZEN dynamic gamma/beta buffer (AdaGN, conditioned)
    /// or a static affine one. Backward: gn_dx only (gb frozen).
    fn gn(&mut self, c: u32, h: u32, w: u32, x: &TVal, gb: &DeviceBuffer) -> TVal {
        let g = num_groups(c);
        let stats = self.gpu.storage(2 * g as u64);
        let part = self.gpu.storage(2 * g as u64 * GN_P as u64);
        let y = self.val(c * h * w);
        let n = c * h * w;
        self.fwd.push(self.gpu.step(K_GN_PART, &[&x.buf, &part], &[1, c, h, w, g, GN_P], g * GN_P));
        self.fwd.push(self.gpu.step(K_GN_STATS2, &[&part, &stats], &[1, c, h, w, g, GN_P, f(GN_EPS)], g));
        self.fwd.push(self.gpu.step(K_GN_APPLY, &[&x.buf, &stats, gb, &y.buf], &[1, c, h, w, g], n));
        // Backward (wm_core::gn recipe, dgamma/dbeta skipped — frozen):
        let dyg = self.gpu.storage(n as u64);
        let sums = self.gpu.storage(4 * g as u64);
        let dims5 = [1, c, h, w, g];
        self.bwd.push(vec![
            self.gpu.step(K_SCALE_CHAN, &[&y.grad, gb, &dyg], &[n, c, h * w], n),
            self.gpu.step(K_GN_DSUM, &[&x.buf, &dyg, &stats, &sums], &dims5, g),
            self.gpu.step(K_GN_DX, &[&x.buf, &dyg, &sums, &x.grad], &dims5, n),
        ]);
        y
    }

    /// AdaGN site: frozen host linear producing gb per batch.
    fn adagn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &TVal) -> TVal {
        let (wsh, wdat) = self.host(&format!("{prefix}.linear.weight"));
        let (_, bdat) = self.host(&format!("{prefix}.linear.bias"));
        assert_eq!(wsh, vec![2 * c as usize, self.cc]);
        let gb = self.gpu.storage(2 * c as u64);
        self.adagn.push((AdaGnSite { w: wdat, b: bdat, c: c as usize }, gb.clone()));
        self.gn(c, h, w, x, &gb)
    }

    /// Frozen affine GN (attn.norm / norm_out).
    fn affine_gn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &TVal) -> TVal {
        let (_, gamma) = self.host(&format!("{prefix}.weight"));
        let (_, beta) = self.host(&format!("{prefix}.bias"));
        let mut gbv = gamma;
        gbv.extend_from_slice(&beta);
        let gb = self.gpu.storage_init(prefix, &gbv);
        self.gn(c, h, w, x, &gb)
    }

    fn silu(&mut self, x: &TVal) -> TVal {
        let y = self.val(x.len);
        self.fwd.push(self.gpu.step(K_SILU, &[&x.buf, &y.buf], &[x.len], x.len));
        self.bwd.push(vec![self.gpu.step(
            K_SILU_BWD,
            &[&x.buf, &y.grad, &x.grad],
            &[x.len],
            x.len,
        )]);
        y
    }

    /// Residual add: dy flows to BOTH inputs' grad slots (copy via add2 with
    /// a shared zero buffer would waste a pass; instead consumers get dy by
    /// add2(dy, zero)? — no: both inputs simply RECEIVE y.grad, so we route
    /// by making their grads distinct buffers and copying. To stay
    /// overwrite-safe we give each input its own grad and add2-copy.
    fn add(&mut self, a: &TVal, b: &TVal, zero: &DeviceBuffer) -> TVal {
        let y = self.val(a.len);
        self.fwd.push(self.gpu.step(K_ADD2, &[&a.buf, &b.buf, &y.buf], &[a.len], a.len));
        self.bwd.push(vec![
            self.gpu.step(K_ADD2, &[&y.grad, zero, &a.grad], &[a.len], a.len),
            self.gpu.step(K_ADD2, &[&y.grad, zero, &b.grad], &[a.len], a.len),
        ]);
        y
    }

    fn concat(&mut self, ca: u32, cb: u32, h: u32, w: u32, a: &TVal, b: &TVal) -> TVal {
        let y = self.val((ca + cb) * h * w);
        self.fwd.push(self.gpu.step(
            K_CONCAT2,
            &[&a.buf, &b.buf, &y.buf],
            &[1, ca, cb, h, w],
            (ca + cb) * h * w,
        ));
        // concat_split extracts a channel slice of dy into each input's grad.
        self.bwd.push(vec![
            self.gpu.step(
                K_CONCAT_SPLIT,
                &[&y.grad, &a.grad],
                &[1, ca + cb, ca, 0, h, w],
                ca * h * w,
            ),
            self.gpu.step(
                K_CONCAT_SPLIT,
                &[&y.grad, &b.grad],
                &[1, ca + cb, cb, ca, h, w],
                cb * h * w,
            ),
        ]);
        y
    }

    fn upsample(&mut self, c: u32, h: u32, w: u32, x: &TVal) -> TVal {
        let y = self.val(c * 4 * h * w);
        self.fwd.push(self.gpu.step(K_UPSAMPLE2, &[&x.buf, &y.buf], &[1, c, h, w], c * 4 * h * w));
        self.bwd.push(vec![self.gpu.step(
            K_UPSAMPLE2_DX,
            &[&y.grad, &x.grad],
            &[1, c, h, w],
            c * h * w,
        )]);
        y
    }

    /// Mid-block self-attention with full backward. Trainable qkv/out convs.
    fn attn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &TVal, zero: &DeviceBuffer) -> TVal {
        let t = h * w;
        let heads = (c / ATTN_HEAD_DIM).max(1);
        let normed = self.affine_gn(&format!("{prefix}.norm.norm"), c, h, w, x);
        let (normed_a, normed_b) = self.fork(&normed);
        let (qkv_chw, _, _) =
            self.conv(&format!("{prefix}.qkv_proj"), c, 3 * c, 1, 1, 0, h, w, &normed_a);
        // NCHW -> [T, 3C] rows.
        let qkv = self.val(3 * c * t);
        self.fwd.push(self.gpu.step(
            K_NCHW_NLC,
            &[&qkv_chw.buf, &qkv.buf],
            &[3 * c * t, 3 * c, t],
            3 * c * t,
        ));
        self.bwd.push(vec![self.gpu.step(
            K_NLC_NCHW,
            &[&qkv.grad, &qkv_chw.grad],
            &[3 * c * t, 3 * c, t],
            3 * c * t,
        )]);
        let scores = self.gpu.storage((heads * t * t) as u64);
        let probs = self.gpu.storage((heads * t * t) as u64);
        let attn_out = self.val(t * c);
        let sp = [1, heads, t, ATTN_HEAD_DIM, 3 * c, 0, c]; // scores: q_off 0, k_off c
        let ap = [1, heads, t, ATTN_HEAD_DIM, 3 * c, 2 * c, c]; // apply/dv: v_off 2c
        self.fwd.push(self.gpu.step(K_ATTN_SCORES, &[&qkv.buf, &scores], &sp, heads * t * t));
        self.fwd.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, heads, t], heads * t));
        self.fwd.push(self.gpu.step(
            K_ATTN_APPLY,
            &[&probs, &qkv.buf, &attn_out.buf],
            &ap,
            heads * t * ATTN_HEAD_DIM,
        ));
        // Backward: d_scores (softmax jacobian folded), then dv/dq/dk into
        // the disjoint regions of qkv.grad (every element written).
        let d_scores = self.gpu.storage((heads * t * t) as u64);
        self.bwd.push(vec![
            self.gpu.step(
                K_ATTN_BWD_DSCORES,
                &[&attn_out.grad, &qkv.buf, &probs, &d_scores],
                &ap,
                heads * t,
            ),
            self.gpu.step(
                K_ATTN_BWD_DV,
                &[&probs, &attn_out.grad, &qkv.grad],
                &ap,
                heads * t * ATTN_HEAD_DIM,
            ),
            self.gpu.step(
                K_ATTN_BWD_DQ,
                &[&d_scores, &qkv.buf, &qkv.grad],
                &sp,
                heads * t * ATTN_HEAD_DIM,
            ),
            self.gpu.step(
                K_ATTN_BWD_DK,
                &[&d_scores, &qkv.buf, &qkv.grad],
                &sp,
                heads * t * ATTN_HEAD_DIM,
            ),
        ]);
        let attn_chw = self.val(c * t);
        self.fwd.push(self.gpu.step(
            K_NLC_NCHW,
            &[&attn_out.buf, &attn_chw.buf],
            &[c * t, c, t],
            c * t,
        ));
        self.bwd.push(vec![self.gpu.step(
            K_NCHW_NLC,
            &[&attn_chw.grad, &attn_out.grad],
            &[c * t, c, t],
            c * t,
        )]);
        let (proj, _, _) =
            self.conv(&format!("{prefix}.out_proj"), c, c, 1, 1, 0, h, w, &attn_chw);
        // Residual on the NORMED input (reference quirk).
        self.add(&normed_b, &proj, zero)
    }

    #[allow(clippy::too_many_arguments)]
    fn resblock(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        attn: bool,
        h: u32,
        w: u32,
        x: &TVal,
        zero: &DeviceBuffer,
    ) -> TVal {
        let (xa, xb) = self.fork(x);
        let r = if cin != cout {
            let (p, _, _) = self.conv(&format!("{prefix}.proj"), cin, cout, 1, 1, 0, h, w, &xa);
            p
        } else {
            xa
        };
        let n1 = self.adagn(&format!("{prefix}.norm1"), cin, h, w, &xb);
        let s1 = self.silu(&n1);
        let (c1, _, _) = self.conv(&format!("{prefix}.conv1"), cin, cout, 3, 1, 1, h, w, &s1);
        let n2 = self.adagn(&format!("{prefix}.norm2"), cout, h, w, &c1);
        let s2 = self.silu(&n2);
        let (c2, _, _) = self.conv(&format!("{prefix}.conv2"), cout, cout, 3, 1, 1, h, w, &s2);
        let y = self.add(&c2, &r, zero);
        let out = if attn {
            self.attn(&format!("{prefix}.attn"), cout, h, w, &y, zero)
        } else {
            y
        };
        out
    }
}

/// The trainable-parameter manifest: every conv weight/bias.
pub fn trainable_list(cfg: &DiamondConfig) -> Vec<(String, usize)> {
    cfg.param_list()
        .into_iter()
        .filter(|(n, _)| {
            (n.contains("conv") || n.contains("proj") || n.contains("samples"))
                && !n.contains("cond_proj")
                && !n.contains("norm")
        })
        .map(|(n, s)| (n, s.iter().product()))
        .collect()
}

pub struct DiamondTrainer {
    pub gpu: Gpu,
    pub cfg: DiamondConfig,
    pub ps: ParamStore,
    opt: optim::Optim,
    cond: CondNet,
    adagn: Vec<(AdaGnSite, DeviceBuffer)>,
    fwd: Vec<Step>,
    bwd: Vec<Step>,
    x_in: DeviceBuffer,
    x_in_grad: DeviceBuffer,
    obs_in: DeviceBuffer,
    y_out: DeviceBuffer,
    y_grad: DeviceBuffer,
    tgt: DeviceBuffer,
    loss_parts: DeviceBuffer,
    pub n_px: u32,
    /// Batch state (sigma + actions) for forward/backward determinism.
    batch: RefCell<Option<(f32, Vec<u32>)>>,
}

impl DiamondTrainer {
    /// Build from full host tensors (imported weights). Conv params become
    /// trainable in the ParamStore; everything else stays frozen host-side.
    pub fn from_tensors(cfg: DiamondConfig, tensors: &Tensors, device: Option<&str>) -> DiamondTrainer {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let list = trainable_list(&cfg);
        let init: HashMap<String, Vec<f32>> =
            list.iter().map(|(n, _)| (n.clone(), tensors[n].1.clone())).collect();
        let ps = ParamStore::new(&gpu, list, &init);
        let opt = optim::Optim::new(K_ADAMW, K_GRADNORM_SQ, K_GRAD_SCALE, K_CLIP_COEF, K_GRAD_SCALE_BUF);

        let cc = cfg.cond_channels as usize;
        let get = |n: &str| tensors[n].1.clone();
        let cond = CondNet {
            cond_channels: cc,
            num_steps_conditioning: cfg.num_steps_conditioning as usize,
            fourier_w: get("noise_emb.weight"),
            act_emb: get("act_emb.0.weight"),
            num_actions: cfg.num_actions as usize,
            mlp0_w: get("cond_proj.0.weight"),
            mlp0_b: get("cond_proj.0.bias"),
            mlp2_w: get("cond_proj.2.weight"),
            mlp2_b: get("cond_proj.2.bias"),
        };

        let ic = cfg.img_channels;
        let (h0, w0) = (cfg.h, cfg.w);
        let nsc = cfg.num_steps_conditioning;
        let n_lv = cfg.levels();
        let n_px = ic * h0 * w0;

        let mut b = TB { gpu: &gpu, ps: &ps, t: tensors, cc, fwd: vec![], bwd: vec![], adagn: vec![] };

        let obs_in = gpu.storage((nsc * ic * h0 * w0) as u64);
        let obs_grad = gpu.storage((nsc * ic * h0 * w0) as u64); // discarded
        let x_in = gpu.storage(n_px as u64);
        let x_in_grad = gpu.storage(n_px as u64); // discarded (input grad)
        // Shared zeros for the residual-copy backward (`add2(y.grad, zero, .)`).
        // MUST cover the LARGEST activation any add() operates on — the widest
        // resblock/attention output is `max_channels * h0 * w0`, NOT the input
        // `(nsc+1)*ic`. Undersizing here reads OOB zeros → garbage gradients
        // for any channel width above the input's (the cpg>8 explosion bug).
        let max_c = *cfg.channels.iter().max().unwrap();
        let zero = gpu.storage((max_c * h0 * w0) as u64);

        let obs_v = TVal { buf: obs_in.clone(), grad: obs_grad, len: nsc * ic * h0 * w0 };
        let x_v = TVal { buf: x_in.clone(), grad: x_in_grad.clone(), len: n_px };

        let cat = b.concat(nsc * ic, ic, h0, w0, &obs_v, &x_v);
        let c0 = cfg.channels[0];
        let (mut x, _, _) = b.conv("conv_in", (nsc + 1) * ic, c0, 3, 1, 1, h0, w0, &cat);

        let mut hw = (h0, w0);
        let mut d_skips: Vec<Vec<(TVal, u32)>> = vec![];
        for i in 0..n_lv {
            let c1 = cfg.channels[i.saturating_sub(1)];
            let c2 = cfg.channels[i];
            if i > 0 {
                let (y, nh, nw) =
                    b.conv(&format!("unet.downsamples.{i}.conv"), c1, c1, 3, 2, 1, hw.0, hw.1, &x);
                x = y;
                hw = (nh, nw);
            }
            let mut level: Vec<(TVal, u32)> = vec![];
            // Each level entry (x_down + per-resblock outputs) is consumed by
            // both the forward chain and one up-path concat: fork each.
            let (x_chain, x_skip) = b.fork(&x);
            level.push((x_skip, c1));
            x = x_chain;
            let n = cfg.depths[i];
            for r in 0..n {
                let cin = if r == 0 { c1 } else { c2 };
                let y = b.resblock(
                    &format!("unet.d_blocks.{i}.resblocks.{r}"),
                    cin,
                    c2,
                    cfg.attn_depths[i],
                    hw.0,
                    hw.1,
                    &x,
                    &zero,
                );
                let (y_chain, y_skip) = b.fork(&y);
                level.push((y_skip, c2));
                x = y_chain;
            }
            d_skips.push(level);
        }

        let cl = *cfg.channels.last().unwrap();
        for r in 0..2 {
            x = b.resblock(
                &format!("unet.mid_blocks.resblocks.{r}"),
                cl,
                cl,
                true,
                hw.0,
                hw.1,
                &x,
                &zero,
            );
        }

        for j in 0..n_lv {
            let i = n_lv - 1 - j;
            let c1 = cfg.channels[i.saturating_sub(1)];
            let c2 = cfg.channels[i];
            if j > 0 {
                let cx = c2;
                let up = b.upsample(cx, hw.0, hw.1, &x);
                hw = (hw.0 * 2, hw.1 * 2);
                let (y, _, _) =
                    b.conv(&format!("unet.upsamples.{j}.conv"), cx, cx, 3, 1, 1, hw.0, hw.1, &up);
                x = y;
            }
            let skips = &d_skips[i];
            let n = cfg.depths[i] as usize;
            for r in 0..=n {
                let (skip, skip_c) = &skips[n - r];
                let xc = if r == 0 { c2 } else { cfg.channels[i] };
                let cat = b.concat(xc, *skip_c, hw.0, hw.1, &x, skip);
                let (cin, cout) = if r < n { (2 * c2, c2) } else { (c1 + c2, c1) };
                debug_assert_eq!(cin, xc + skip_c);
                x = b.resblock(
                    &format!("unet.u_blocks.{j}.resblocks.{r}"),
                    cin,
                    cout,
                    cfg.attn_depths[i],
                    hw.0,
                    hw.1,
                    &cat,
                    &zero,
                );
            }
        }

        let hn = b.affine_gn("norm_out.norm", c0, hw.0, hw.1, &x);
        let hs = b.silu(&hn);
        let (y_v, _, _) = b.conv("conv_out", c0, ic, 3, 1, 1, hw.0, hw.1, &hs);
        assert_eq!(hw, (h0, w0));

        // Loss plumbing: mse_value partials + mse_grad seed into y_v.grad.
        let tgt = gpu.storage(n_px as u64);
        let loss_parts = gpu.storage(n_px as u64);
        let mut fwd = std::mem::take(&mut b.fwd);
        fwd.push(gpu.step(K_MSE_VALUE, &[&y_v.buf, &tgt, &loss_parts], &[n_px], n_px));
        let mut bwd: Vec<Step> =
            vec![gpu.step(K_MSE_GRAD, &[&y_v.buf, &tgt, &y_v.grad], &[n_px], n_px)];
        // End the builder's borrows of `gpu`/`ps` before moving them.
        let TB { bwd: bwd_groups, adagn, gpu: _, ps: _, t: _, cc: _, fwd: _ } = b;
        for group in bwd_groups.into_iter().rev() {
            bwd.extend(group);
        }

        DiamondTrainer {
            gpu,
            cfg,
            ps,
            opt,
            cond,
            adagn,
            fwd,
            bwd,
            x_in,
            x_in_grad,
            obs_in,
            y_out: y_v.buf,
            y_grad: y_v.grad,
            tgt,
            loss_parts,
            n_px,
            batch: RefCell::new(None),
        }
    }

    /// Prepare one transition: obs (nsc frames, [-1,1]), clean next frame
    /// ([-1,1]), pre-sampled noise, sigma, actions. Deterministic given its
    /// arguments (noise applied here, not in forward) — FD-safe.
    pub fn set_transition(
        &self,
        obs: &[f32],
        clean: &[f32],
        noise: &[f32],
        sigma: f32,
        actions: &[u32],
    ) {
        let cs = conditioners(sigma, self.cfg.sigma_data, self.cfg.sigma_offset_noise);
        let sd = self.cfg.sigma_data;
        // apply_noise uses the RAW sigma + offset folded via conditioners'
        // s' — the reference adds offset noise; v1 folds it into one draw:
        // noisy = clean + s' * noise (s'^2 = sigma^2 + so^2).
        let s_eff = (sigma * sigma + self.cfg.sigma_offset_noise * self.cfg.sigma_offset_noise).sqrt();
        let noisy: Vec<f32> = clean.iter().zip(noise).map(|(c, n)| c + s_eff * n).collect();
        let x_scaled: Vec<f32> = noisy.iter().map(|v| v * cs.c_in).collect();
        let obs_rescaled: Vec<f32> = obs.iter().map(|v| v / sd).collect();
        let target_f: Vec<f32> = clean
            .iter()
            .zip(&noisy)
            .map(|(c, x)| (c - cs.c_skip * x) / cs.c_out)
            .collect();
        wf(&self.gpu, &self.x_in, &x_scaled);
        wf(&self.gpu, &self.obs_in, &obs_rescaled);
        wf(&self.gpu, &self.tgt, &target_f);
        let cond = self.cond.cond(cs.c_noise, actions);
        for (site, gb) in &self.adagn {
            wf(&self.gpu, gb, &site.gb(&cond));
        }
        *self.batch.borrow_mut() = Some((sigma, actions.to_vec()));
    }


    pub fn forward_loss(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd);
        self.gpu.read(&self.loss_parts, self.n_px as usize).iter().sum()
    }


    pub fn backward(&self) {
        self.gpu.submit(&[], &self.bwd);
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, 1.0);
    }


    /// The inner-model output F of the last forward.

    pub fn read_output(&self) -> Vec<f32> {
        self.gpu.read(&self.y_out, self.n_px as usize)
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.ps.weight[name], &bits);
    }

    /// Inference-graph forward on the SAME trainable weights (for the
    /// anti-drift test): read params back into a Tensors map.
    pub fn export_tensors(&self, base: &Tensors) -> Tensors {
        let mut out = base.clone();
        for name in self.ps.weight.keys() {
            let (shape, _) = base[name].clone();
            let data = self.ps.read_weight(&self.gpu, name);
            out.insert(name.clone(), (shape, data));
        }
        out
    }
}

/// Reference training sigma distribution: exp(N(loc, scale^2)) clipped
/// (trainer.yaml: loc -0.4, scale 1.2, clip [2e-3, 20]).
pub struct SigmaDist {
    pub loc: f32,
    pub scale: f32,
    pub min: f32,
    pub max: f32,
}

impl Default for SigmaDist {
    fn default() -> Self {
        SigmaDist { loc: -0.4, scale: 1.2, min: 2e-3, max: 20.0 }
    }
}

impl SigmaDist {
    pub fn sample(&self, rng: &mut crate::play::NormalRng) -> f32 {
        (rng.normal() * self.scale + self.loc).exp().clamp(self.min, self.max)
    }
}

/// One transition of training data in [-1,1]: context frames (oldest first,
/// nsc * frame_len), the clean next frame, and the nsc context actions.
pub struct Transition {
    pub obs: Vec<f32>,
    pub clean: Vec<f32>,
    pub actions: Vec<u32>,
}

/// Fine-tune loop over a transition sampler (the CLI glues an episode
/// dataset in; tests use synthetic closures). Returns (first, last) loss.
#[allow(clippy::too_many_arguments)]
pub fn finetune(
    tr: &DiamondTrainer,
    mut sample: impl FnMut(&mut crate::play::NormalRng) -> Transition,
    steps: u32,
    lr: f32,
    weight_decay: f32,
    clip: Option<f32>,
    seed: u64,
    mut on_log: impl FnMut(u32, f32),
) -> (f32, f32) {
    let mut rng = crate::play::NormalRng::new(seed);
    let dist = SigmaDist::default();
    let n_px = tr.n_px as usize;
    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for t in 1..=steps {
        let tx = sample(&mut rng);
        let sigma = dist.sample(&mut rng);
        let noise: Vec<f32> = (0..n_px).map(|_| rng.normal()).collect();
        tr.set_transition(&tx.obs, &tx.clean, &noise, sigma, &tx.actions);
        tr.zero_grads();
        let loss = tr.forward_loss();
        tr.backward();
        tr.adamw_step(t, lr, weight_decay, clip);
        if first.is_nan() {
            first = loss;
        }
        last = loss;
        if t % 25 == 0 || t == 1 || t == steps {
            on_log(t, loss);
        }
        if !loss.is_finite() {
            // Divergence: stop immediately — continuing would only burn time
            // and the caller must NOT save these weights.
            on_log(t, loss);
            break;
        }
    }
    (first, last)
}

impl DiamondTrainer {
    /// Save the fine-tuned model as a standard DIAMOND `.weights` (trained
    /// convs from the ParamStore merged over the frozen base tensors), so
    /// `brain wm play --model diamond` loads it unchanged.
    pub fn save(&self, base: &Tensors, path: &str) -> Result<(), String> {
        let merged = self.export_tensors(base);
        let mut out: Vec<(String, Vec<u64>, Vec<f32>)> = vec![];
        for (name, shape) in self.cfg.param_list() {
            let (s, d) = merged
                .get(&name)
                .ok_or_else(|| format!("save: missing {name}"))?;
            assert_eq!(s, &shape);
            out.push((name, shape.iter().map(|&x| x as u64).collect(), d.clone()));
        }
        let config: serde_json::Value = serde_json::from_str(&self.cfg.to_json())
            .map_err(|e| format!("config json: {e}"))?;
        checkpoint::save(path, config, &out);
        Ok(())
    }
}
