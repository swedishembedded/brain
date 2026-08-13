// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth -> ONNX, for the Intel NPU (the `blend`/`where_conv` upsampler variant).
//!
//! Walks the exact graph `zipdepth::ZipDepth::build` constructs, in the same op order,
//! and emits it as fp32 ONNX. The NPU (blend) variant is chosen deliberately: its
//! upsampler is Conv/BN/Relu/Sigmoid/Resize/Mul/Add only — no `unfold`, no
//! softmax-over-9, no `pixel_shuffle` — so every op maps to a standard, NPU-friendly
//! ONNX node. `zipdepth::fuse_qarep` collapses each RepVGG block to one biased 3x3
//! before export, and BN is folded into every conv, matching the fused inference
//! form the checkpoint's 6.1M headline refers to.
//!
//! Parity is checked in `tests/depth_onnx.rs` against `zipdepth::ZipDepth`'s own CPU
//! forward, on OpenVINO-CPU and on the real NPU.

use zipdepth::config::{pick_groups, GlobalMode, ZipConfig};
use zipdepth::fuse::{fuse_qarep, Branch};
use onnx::{GraphBuilder, Node};
use vision::{fold_bn, BN_EPS};

use crate::topology::WeightSource;

/// A named feature tensor plus its NCHW shape (N is always 1 for export).
#[derive(Clone)]
struct Feat {
    name: String,
    c: u32,
    h: u32,
    w: u32,
}

struct Exporter<'a> {
    g: &'a mut GraphBuilder,
    w: &'a dyn WeightSource,
    /// Monotonic id for anonymous intermediate tensors.
    uid: u32,
}

impl<'a> Exporter<'a> {
    fn t(&mut self, tag: &str) -> String {
        self.uid += 1;
        format!("{tag}_{}", self.uid)
    }

    /// Emit a Conv from an already-folded weight `[cout, cin/groups, k, k]` + bias.
    #[allow(clippy::too_many_arguments)]
    fn conv_raw(
        &mut self,
        prefix: &str,
        x: &Feat,
        wp: &[f32],
        bias: &[f32],
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        groups: u32,
        dilation: u32,
    ) -> Feat {
        let cin_g = x.c / groups;
        let wname = format!("{prefix}.w");
        self.g.init_f32(&wname, &[cout as i64, cin_g as i64, k as i64, k as i64], wp.to_vec());
        let bname = format!("{prefix}.b");
        self.g.init_f32(&bname, &[cout as i64], bias.to_vec());
        let out = format!("{prefix}.conv");
        // Dilated pad: effective kernel is k + (k-1)(d-1); reference pad already
        // accounts for it, so `pad` is passed through.
        self.g.add(
            Node::new("Conv", &[&x.name, &wname, &bname], &[&out])
                .name(&format!("{prefix}/Conv"))
                .attr_ints("kernel_shape", &[k as i64, k as i64])
                .attr_ints("strides", &[stride as i64, stride as i64])
                .attr_ints("pads", &[pad as i64, pad as i64, pad as i64, pad as i64])
                .attr_ints("dilations", &[dilation as i64, dilation as i64])
                .attr_int("group", groups as i64),
        );
        let ho = (x.h + 2 * pad - (dilation * (k - 1) + 1)) / stride + 1;
        let wo = (x.w + 2 * pad - (dilation * (k - 1) + 1)) / stride + 1;
        Feat { name: out, c: cout, h: ho, w: wo }
    }

    fn relu(&mut self, x: &Feat) -> Feat {
        let out = self.t("relu");
        self.g.add(Node::new("Relu", &[&x.name], &[&out]));
        Feat { name: out, ..x.clone() }
    }

    /// `ConvBN` (torch names): conv(bias-free) + folded BN [+ Relu].
    #[allow(clippy::too_many_arguments)]
    fn conv_bn(&mut self, prefix: &str, x: &Feat, cout: u32, k: u32, stride: u32, pad: u32, groups: u32, relu: bool) -> Feat {
        let w = self.w.get(&format!("{prefix}.conv.weight"));
        let (wp, bias) = self.fold_torch_bn(&w, prefix, cout);
        let f = self.conv_raw(prefix, x, &wp, &bias, cout, k, stride, pad, groups, 1);
        if relu {
            self.relu(&f)
        } else {
            f
        }
    }

    /// Fold `<prefix>.bn.*` (torch names) into a conv weight.
    fn fold_torch_bn(&self, w: &[f32], prefix: &str, cout: u32) -> (Vec<f32>, Vec<f32>) {
        let gamma = self.w.get(&format!("{prefix}.bn.weight"));
        let beta = self.w.get(&format!("{prefix}.bn.bias"));
        let rm = self.w.get(&format!("{prefix}.bn.running_mean"));
        let rv = self.w.get(&format!("{prefix}.bn.running_var"));
        fold_bn(w, &gamma, &beta, &rm, &rv, cout as usize)
    }

    /// Apply a folded standalone BN (over `x`, no preceding conv here) as a
    /// per-channel affine: `y = x*scale + shift`, `scale=gamma/sqrt(var+eps)`,
    /// `shift=beta - mean*scale`. Emitted as Mul + Add with `[1,C,1,1]` constants.
    fn apply_bn(&mut self, prefix: &str, x: &Feat) -> Feat {
        let gamma = self.w.get(&format!("{prefix}.weight"));
        let beta = self.w.get(&format!("{prefix}.bias"));
        let rm = self.w.get(&format!("{prefix}.running_mean"));
        let rv = self.w.get(&format!("{prefix}.running_var"));
        let c = x.c as usize;
        let scale: Vec<f32> = (0..c).map(|i| gamma[i] / (rv[i] + BN_EPS).sqrt()).collect();
        let shift: Vec<f32> = (0..c).map(|i| beta[i] - rm[i] * scale[i]).collect();
        let sn = format!("{prefix}.bn_scale");
        self.g.init_f32(&sn, &[1, c as i64, 1, 1], scale);
        let bn = format!("{prefix}.bn_shift");
        self.g.init_f32(&bn, &[1, c as i64, 1, 1], shift);
        let m = self.t("bnmul");
        self.g.add(Node::new("Mul", &[&x.name, &sn], &[&m]));
        let out = self.t("bnadd");
        self.g.add(Node::new("Add", &[&m, &bn], &[&out]));
        Feat { name: out, ..x.clone() }
    }

    fn add(&mut self, a: &Feat, b: &Feat) -> Feat {
        let out = self.t("add");
        self.g.add(Node::new("Add", &[&a.name, &b.name], &[&out]));
        Feat { name: out, ..a.clone() }
    }
    fn mul(&mut self, a: &Feat, b: &Feat) -> Feat {
        let out = self.t("mul");
        self.g.add(Node::new("Mul", &[&a.name, &b.name], &[&out]));
        Feat { name: out, ..a.clone() }
    }
    fn sigmoid(&mut self, x: &Feat) -> Feat {
        let out = self.t("sig");
        self.g.add(Node::new("Sigmoid", &[&x.name], &[&out]));
        Feat { name: out, ..x.clone() }
    }

    /// Bilinear or nearest `Resize` to a target H×W, `align_corners=False`
    /// (`half_pixel`, matching brain's resize kernels).
    fn resize(&mut self, x: &Feat, to_h: u32, to_w: u32, bilinear: bool) -> Feat {
        // Use the `sizes` input (4th) with an empty scales input.
        let sizes = self.t("size");
        self.g.init_i64(&sizes, &[4], vec![1, x.c as i64, to_h as i64, to_w as i64]);
        let out = self.t("resize");
        let mode = if bilinear { "linear" } else { "nearest" };
        let mut node = Node::new("Resize", &[&x.name, "", "", &sizes], &[&out])
            .attr_str("mode", mode)
            .attr_str("coordinate_transformation_mode", if bilinear { "half_pixel" } else { "asymmetric" });
        if !bilinear {
            node = node.attr_str("nearest_mode", "floor");
        }
        self.g.add(node);
        Feat { name: out, c: x.c, h: to_h, w: to_w }
    }

    /// A scalar-broadcast multiply, `y = x * k`.
    fn scale(&mut self, x: &Feat, k: f32) -> Feat {
        let kn = self.t("k");
        self.g.init_f32(&kn, &[1], vec![k]);
        let out = self.t("scaled");
        self.g.add(Node::new("Mul", &[&x.name, &kn], &[&out]));
        Feat { name: out, ..x.clone() }
    }
}

impl<'a> Exporter<'a> {
    /// QARepBlock -> one fused biased 3x3 -> Relu.
    fn qarep(&mut self, prefix: &str, x: &Feat, cout: u32, stride: u32) -> Feat {
        let cin = x.c as usize;
        let has_id = x.c == cout && stride == 1;
        let w3 = self.w.get(&format!("{prefix}.branch_3x3.0.weight"));
        let g3 = self.w.get(&format!("{prefix}.branch_3x3.1.weight"));
        let be3 = self.w.get(&format!("{prefix}.branch_3x3.1.bias"));
        let m3 = self.w.get(&format!("{prefix}.branch_3x3.1.running_mean"));
        let v3 = self.w.get(&format!("{prefix}.branch_3x3.1.running_var"));
        let w1 = self.w.get(&format!("{prefix}.branch_1x1.0.weight"));
        let g1 = self.w.get(&format!("{prefix}.branch_1x1.1.weight"));
        let be1 = self.w.get(&format!("{prefix}.branch_1x1.1.bias"));
        let m1 = self.w.get(&format!("{prefix}.branch_1x1.1.running_mean"));
        let v1 = self.w.get(&format!("{prefix}.branch_1x1.1.running_var"));
        let b3 = Branch { weight: &w3, gamma: &g3, beta: &be3, run_mean: &m3, run_var: &v3 };
        let b1 = Branch { weight: &w1, gamma: &g1, beta: &be1, run_mean: &m1, run_var: &v1 };
        let (kernel, bias) = fuse_qarep(&b3, &b1, cin, cout as usize, 1, has_id);
        let f = self.conv_raw(prefix, x, &kernel, &bias, cout, 3, stride, 1, 1, 1);
        self.relu(&f)
    }

    /// MinimalMultiScale: x + BN(dw_d1(x) + dw_d2(x)). Two depthwise 3x3 branches
    /// (dilation 1 / pad 1 and dilation 2 / pad 2), one shared BN, residual, no act.
    fn mms(&mut self, prefix: &str, x: &Feat) -> Feat {
        let c = x.c;
        let w1 = self.w.get(&format!("{prefix}.branch1.weight"));
        let b1 = self.conv_raw(&format!("{prefix}.branch1"), x, &w1, &vec![0.0; c as usize], c, 3, 1, 1, c, 1);
        let w2 = self.w.get(&format!("{prefix}.branch2.weight"));
        let b2 = self.conv_raw(&format!("{prefix}.branch2"), x, &w2, &vec![0.0; c as usize], c, 3, 1, 2, c, 2);
        let sum = self.add(&b1, &b2);
        let bn = self.apply_bn(&format!("{prefix}.bn"), &sum);
        self.add(x, &bn)
    }

    /// StripPoolingAttention: x * sigmoid(BN(dw1x1(mean_W(x) + mean_H(x)))).
    fn strip(&mut self, prefix: &str, x: &Feat) -> Feat {
        // mean over W (axis 3) -> [B,C,H,1]; mean over H (axis 2) -> [B,C,1,W].
        let hstrip = self.t("hstrip");
        self.g.add(Node::new("ReduceMean", &[&x.name], &[&hstrip]).attr_ints("axes", &[3]).attr_int("keepdims", 1));
        let wstrip = self.t("wstrip");
        self.g.add(Node::new("ReduceMean", &[&x.name], &[&wstrip]).attr_ints("axes", &[2]).attr_int("keepdims", 1));
        // Broadcast add -> [B,C,H,W].
        let summ = self.t("stripsum");
        self.g.add(Node::new("Add", &[&hstrip, &wstrip], &[&summ]));
        let s = Feat { name: summ, c: x.c, h: x.h, w: x.w };
        // Depthwise 1x1 gate + folded BN + Sigmoid. gate_conv.0 bias-free, .1 = BN.
        let gw = self.w.get(&format!("{prefix}.gate_conv.0.weight"));
        let (gwp, gb) = self.fold_torch_bn2(&gw, &format!("{prefix}.gate_conv.1"), x.c);
        let gate = self.conv_raw(&format!("{prefix}.gate_conv"), &s, &gwp, &gb, x.c, 1, 1, 0, x.c, 1);
        let gate = self.sigmoid(&gate);
        self.mul(x, &gate)
    }

    /// Fold a BN whose params live at `<prefix>.{weight,bias,running_mean,
    /// running_var}` into a conv weight (torch ConvBN with the BN at an arbitrary
    /// sub-name, e.g. `gate_conv.1`).
    fn fold_torch_bn2(&self, w: &[f32], bn_prefix: &str, cout: u32) -> (Vec<f32>, Vec<f32>) {
        let gamma = self.w.get(&format!("{bn_prefix}.weight"));
        let beta = self.w.get(&format!("{bn_prefix}.bias"));
        let rm = self.w.get(&format!("{bn_prefix}.running_mean"));
        let rv = self.w.get(&format!("{bn_prefix}.running_var"));
        fold_bn(w, &gamma, &beta, &rm, &rv, cout as usize)
    }

    /// ChannelAttention (SE): x * sigmoid(fc2(relu(fc1(GAP(x))))). Bias-free 1x1s.
    fn se(&mut self, prefix: &str, x: &Feat) -> Feat {
        let c = x.c;
        let hidden = (c / 8).max(4);
        let pooled = self.t("gap");
        self.g.add(Node::new("GlobalAveragePool", &[&x.name], &[&pooled]));
        let pf = Feat { name: pooled, c, h: 1, w: 1 };
        let w0 = self.w.get(&format!("{prefix}.fc.0.weight"));
        let h0 = self.conv_raw(&format!("{prefix}.fc0"), &pf, &w0, &vec![0.0; hidden as usize], hidden, 1, 1, 0, 1, 1);
        let h0 = self.relu(&h0);
        let w2 = self.w.get(&format!("{prefix}.fc.2.weight"));
        let g = self.conv_raw(&format!("{prefix}.fc2"), &h0, &w2, &vec![0.0; c as usize], c, 1, 1, 0, 1, 1);
        let g = self.sigmoid(&g);
        self.mul(x, &g)
    }

    /// GlobalContextBlock: x + transform(bmm(x, softmax(score(x)))).
    fn gcb(&mut self, prefix: &str, x: &Feat) -> Feat {
        let (c, hw) = (x.c, x.h * x.w);
        let hidden = (c / 4).max(8);
        // score: biased 1x1 conv -> [B,1,H,W]; reshape to [B,1,HW]; softmax over HW.
        let sw = self.w.get(&format!("{prefix}.context_weight.weight"));
        let sb = self.w.get(&format!("{prefix}.context_weight.bias"));
        let score = self.conv_raw(&format!("{prefix}.score"), x, &sw, &sb, 1, 1, 1, 0, 1, 1);
        let sc_shape = self.t("scsh");
        self.g.init_i64(&sc_shape, &[3], vec![1, 1, hw as i64]);
        let sc_flat = self.t("scflat");
        self.g.add(Node::new("Reshape", &[&score.name, &sc_shape], &[&sc_flat]));
        let sm = self.t("softmax");
        self.g.add(Node::new("Softmax", &[&sc_flat], &[&sm]).attr_int("axis", 2));
        // x -> [B,C,HW]; context = x_flat @ sm^T -> [B,C,1] -> [B,C,1,1].
        let x_shape = self.t("xsh");
        self.g.init_i64(&x_shape, &[3], vec![1, c as i64, hw as i64]);
        let x_flat = self.t("xflat");
        self.g.add(Node::new("Reshape", &[&x.name, &x_shape], &[&x_flat]));
        // transpose sm [B,1,HW] -> [B,HW,1].
        let sm_t = self.t("smt");
        self.g.add(Node::new("Transpose", &[&sm], &[&sm_t]).attr_ints("perm", &[0, 2, 1]));
        let ctx_flat = self.t("ctxf");
        self.g.add(Node::new("MatMul", &[&x_flat, &sm_t], &[&ctx_flat])); // [B,C,1]
        let ctx_shape = self.t("ctxsh");
        self.g.init_i64(&ctx_shape, &[4], vec![1, c as i64, 1, 1]);
        let ctx4 = self.t("ctx4");
        self.g.add(Node::new("Reshape", &[&ctx_flat, &ctx_shape], &[&ctx4]));
        let ctxf = Feat { name: ctx4, c, h: 1, w: 1 };
        // transform: Conv(1x1,C->hidden) biased + BN + Relu, Conv(1x1,hidden->C) biased.
        let t0w = self.w.get(&format!("{prefix}.transform.0.weight"));
        let t0b = self.w.get(&format!("{prefix}.transform.0.bias"));
        let t0 = self.conv_raw(&format!("{prefix}.t0"), &ctxf, &t0w, &t0b, hidden, 1, 1, 0, 1, 1);
        let t0 = self.apply_bn(&format!("{prefix}.transform.1"), &t0);
        let t0 = self.relu(&t0);
        let t3w = self.w.get(&format!("{prefix}.transform.3.weight"));
        let t3b = self.w.get(&format!("{prefix}.transform.3.bias"));
        let t3 = self.conv_raw(&format!("{prefix}.t3"), &t0, &t3w, &t3b, c, 1, 1, 0, 1, 1);
        // residual: x + context (context broadcasts [B,C,1,1] over H,W).
        self.add(x, &t3)
    }

    /// LightweightSPPF: cv1 -> 3x maxpool(k5,s1,pad2) -> concat -> cv2, ReLU acts.
    fn sppf(&mut self, prefix: &str, x: &Feat, cout: u32) -> Feat {
        let hidden = x.c / 4;
        let cv1 = self.conv_bn(&format!("{prefix}.cv1"), x, hidden, 1, 1, 0, 1, true);
        let pool = |e: &mut Self, inp: &Feat| -> Feat {
            let out = e.t("mp");
            e.g.add(
                Node::new("MaxPool", &[&inp.name], &[&out])
                    .attr_ints("kernel_shape", &[5, 5])
                    .attr_ints("strides", &[1, 1])
                    .attr_ints("pads", &[2, 2, 2, 2]),
            );
            Feat { name: out, ..inp.clone() }
        };
        let m1 = pool(self, &cv1);
        let m2 = pool(self, &m1);
        let m3 = pool(self, &m2);
        let cat = self.t("sppcat");
        self.g.add(
            Node::new("Concat", &[&cv1.name, &m1.name, &m2.name, &m3.name], &[&cat]).attr_int("axis", 1),
        );
        let catf = Feat { name: cat, c: hidden * 4, h: x.h, w: x.w };
        self.conv_bn(&format!("{prefix}.cv2"), &catf, cout, 1, 1, 0, 1, true)
    }

    /// MinimalCrossScale: returns (x_high + 0.3*up(l2h(x_low)), x_low + 0.3*pool(h2l(x_high))).
    fn cross_scale(&mut self, prefix: &str, high: &Feat, low: &Feat) -> (Feat, Feat) {
        let g_h = pick_groups(low.c, high.c, 4);
        let g_l = pick_groups(high.c, low.c, 4);
        let l2hw = self.w.get(&format!("{prefix}.low_to_high.weight"));
        let l2h = self.conv_raw(&format!("{prefix}.l2h"), low, &l2hw, &vec![0.0; high.c as usize], high.c, 1, 1, 0, g_h, 1);
        let up = self.resize(&l2h, high.h, high.w, false); // nearest
        let up = self.scale(&up, 0.3);
        let out_high = self.add(high, &up);

        let h2lw = self.w.get(&format!("{prefix}.high_to_low.weight"));
        let h2l = self.conv_raw(&format!("{prefix}.h2l"), high, &h2lw, &vec![0.0; low.c as usize], low.c, 1, 1, 0, g_l, 1);
        // adaptive_avg_pool2d(high -> low size): exact box pool with k=stride=high/low.
        let kh = high.h / low.h;
        let kw = high.w / low.w;
        let down = self.t("csdown");
        self.g.add(
            Node::new("AveragePool", &[&h2l.name], &[&down])
                .attr_ints("kernel_shape", &[kh as i64, kw as i64])
                .attr_ints("strides", &[kh as i64, kw as i64]),
        );
        let down = Feat { name: down, c: low.c, h: low.h, w: low.w };
        let down = self.scale(&down, 0.3);
        let out_low = self.add(low, &down);
        (out_high, out_low)
    }

    /// UltraLightFusion: relu(BN(proj_high(x_high) + proj_low(up_bilinear(x_low)))).
    fn fusion(&mut self, prefix: &str, high: &Feat, low: &Feat, out_ch: u32) -> Feat {
        let g_high = pick_groups(high.c, out_ch, 4);
        let g_low = pick_groups(low.c, out_ch, 4);
        let up = self.resize(low, high.h, high.w, true); // bilinear
        let phw = self.w.get(&format!("{prefix}.proj_high.weight"));
        let ph = self.conv_raw(&format!("{prefix}.ph"), high, &phw, &vec![0.0; out_ch as usize], out_ch, 1, 1, 0, g_high, 1);
        let plw = self.w.get(&format!("{prefix}.proj_low.weight"));
        let pl = self.conv_raw(&format!("{prefix}.pl"), &up, &plw, &vec![0.0; out_ch as usize], out_ch, 1, 1, 0, g_low, 1);
        let sum = self.add(&ph, &pl);
        let bn = self.apply_bn(&format!("{prefix}.bn"), &sum);
        self.relu(&bn)
    }

    /// FastConvexUpsample, BLEND variant: relu(a*nn(d) + (1-a)*bi(d)),
    /// a = sigmoid(up_bilinear(where_conv(feat))).
    fn convex_up_blend(&mut self, prefix: &str, feat: &Feat, depth: &Feat, scale: u32) -> Feat {
        let (oh, ow) = (depth.h * scale, depth.w * scale);
        let wh = (feat.c / 2).max(8);
        // where_conv: 1x1+BN+Relu, dw5x5+BN+Relu, 1x1 (all bias-free).
        let w0 = self.w.get(&format!("{prefix}.where_conv.0.weight"));
        let (w0p, w0b) = self.fold_torch_bn2(&w0, &format!("{prefix}.where_conv.1"), wh);
        let a = self.conv_raw(&format!("{prefix}.wc0"), feat, &w0p, &w0b, wh, 1, 1, 0, 1, 1);
        let a = self.relu(&a);
        let w3 = self.w.get(&format!("{prefix}.where_conv.3.weight"));
        let (w3p, w3b) = self.fold_torch_bn2(&w3, &format!("{prefix}.where_conv.4"), wh);
        let a = self.conv_raw(&format!("{prefix}.wc3"), &a, &w3p, &w3b, wh, 5, 1, 2, wh, 1);
        let a = self.relu(&a);
        let w6 = self.w.get(&format!("{prefix}.where_conv.6.weight"));
        let a = self.conv_raw(&format!("{prefix}.wc6"), &a, &w6, &[0.0; 1], 1, 1, 1, 0, 1, 1);
        // alpha upsampled (bilinear) BEFORE sigmoid.
        let a_up = self.resize(&a, oh, ow, true);
        let alpha = self.sigmoid(&a_up);
        // depth upsamples.
        let nn = self.resize(depth, oh, ow, false);
        let bi = self.resize(depth, oh, ow, true);
        // out = a*nn + (1-a)*bi = bi + a*(nn-bi).
        let diff = {
            let out = self.t("diff");
            self.g.add(Node::new("Sub", &[&nn.name, &bi.name], &[&out]));
            Feat { name: out, c: 1, h: oh, w: ow }
        };
        let blended = self.mul(&alpha, &diff);
        let out = self.add(&bi, &blended);
        self.relu(&out)
    }
}

/// Build ZipDepth's fp32 ONNX graph (NPU/blend variant) into `g`. `w` supplies the
/// imported weights; `cfg.upsample_unfold` must be false (this exporter only emits
/// the blend upsampler — the unfold path's unfold/pixel-shuffle are not NPU ops).
pub fn build_depth_graph(cfg: &ZipConfig, w: &dyn WeightSource, g: &mut GraphBuilder) {
    let sz = cfg.input;
    build_depth_graph_hw(cfg, w, sz, sz, g);
}

/// As [`build_depth_graph`], for an arbitrary `h × w` input (both multiples of 32).
/// The reference feeds an aspect-preserving rectangular input, not a padded square,
/// so the NPU path exports at the resized frame size to match — see
/// `zipdepth::predict::target_size`.
pub fn build_depth_graph_hw(cfg: &ZipConfig, w: &dyn WeightSource, in_h: u32, in_w: u32, g: &mut GraphBuilder) {
    assert!(!cfg.upsample_unfold, "the NPU exporter emits the blend (where_conv) upsampler; set upsample_unfold=false");
    assert_ne!(cfg.global_mode, GlobalMode::Full, "GlobalMode::Full (EGA) is not exported");
    assert_eq!(in_h % 32, 0, "input height must be a multiple of 32");
    assert_eq!(in_w % 32, 0, "input width must be a multiple of 32");
    let d = cfg.dims;
    let half = cfg.half_ch();
    let use_global = cfg.global_mode != GlobalMode::None;

    g.input_f32("input", &[1, 3, in_h as i64, in_w as i64]);
    let mut e = Exporter { g, w, uid: 0 };

    // normalize: (x - mean) / std, both [1,3,1,1] buffers in the state_dict.
    let mean = e.w.get("mean");
    let std = e.w.get("std");
    e.g.init_f32("norm.mean", &[1, 3, 1, 1], mean);
    let inv_std: Vec<f32> = std.iter().map(|s| 1.0 / s).collect();
    e.g.init_f32("norm.invstd", &[1, 3, 1, 1], inv_std);
    e.g.add(Node::new("Sub", &["input", "norm.mean"], &["norm.sub"]));
    e.g.add(Node::new("Mul", &["norm.sub", "norm.invstd"], &["norm.x"]));
    let x = Feat { name: "norm.x".into(), c: 3, h: in_h, w: in_w };

    // ---- encoder ----
    let x = e.conv_bn("encoder.stem_half", &x, half, 3, 2, 1, 1, true);
    let s_half = x.clone();
    let x = e.conv_bn("encoder.stem_quarter", &x, d[0], 3, 2, 1, 1, true);

    let mut x = x;
    for i in 0..cfg.depths[0] {
        x = e.qarep(&format!("encoder.stage1.{i}"), &x, d[0], 1);
    }
    let c1 = x.clone();

    let mut x = e.qarep("encoder.down2", &x, d[1], 2);
    for i in 0..cfg.depths[1] {
        x = e.qarep(&format!("encoder.stage2.{i}"), &x, d[1], 1);
    }
    let mut idx = cfg.depths[1];
    x = e.mms(&format!("encoder.stage2.{idx}"), &x);
    idx += 1;
    if use_global {
        x = e.strip(&format!("encoder.stage2.{idx}"), &x);
    }
    let c2 = x.clone();

    let mut x = e.qarep("encoder.down3", &x, d[2], 2);
    for i in 0..cfg.depths[2] {
        x = e.qarep(&format!("encoder.stage3.{i}"), &x, d[2], 1);
    }
    let mut idx = cfg.depths[2];
    x = e.se(&format!("encoder.stage3.{idx}"), &x);
    idx += 1;
    if use_global {
        x = e.gcb(&format!("encoder.stage3.{idx}"), &x);
    }
    let c3 = x.clone();

    let mut x = e.qarep("encoder.down4", &x, d[3], 2);
    for i in 0..cfg.depths[3] {
        x = e.qarep(&format!("encoder.stage4.{i}"), &x, d[3], 1);
    }
    let x = e.sppf("encoder.spp", &x, d[3]);
    let (c3, c4) = e.cross_scale("encoder.cross_scale", &c3, &x);

    // ---- decoder ----
    let (ch4, ch3, ch2, ch1) = cfg.dec_chans();
    let f4 = e.conv_bn("decoder.proj4", &c4, ch4, 1, 1, 0, 1, true);
    let f3 = e.fusion("decoder.fuse3", &c3, &f4, ch3);
    let f2 = e.fusion("decoder.fuse2", &c2, &f3, ch2);
    let f1 = e.fusion("decoder.fuse1", &c1, &f2, ch1);
    let f_half = e.fusion("decoder.fuse_half", &s_half, &f1, cfg.half_dec_ch);
    // head_half: biased 3x3, no act.
    let hw = e.w.get("decoder.head_half.weight");
    let hb = e.w.get("decoder.head_half.bias");
    let depth_half = e.conv_raw("decoder.head_half", &f_half, &hw, &hb, 1, 3, 1, 1, 1, 1);
    let out = e.convex_up_blend("decoder.convex_up", &f_half, &depth_half, 2);

    e.g.add(Node::new("Identity", &[&out.name], &["output"]));
    e.g.output_f32("output", &[1, 1, out.h as i64, out.w as i64]);
}

/// Local weight source over an imported map.
impl WeightSource for std::collections::HashMap<String, Vec<f32>> {
    fn get(&self, name: &str) -> Vec<f32> {
        self.get(name).cloned().unwrap_or_else(|| panic!("weights missing tensor `{name}`"))
    }
}
