// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 → ONNX topology builder.
//!
//! Walks the exact graph that `yolo::model::Yolo::new` constructs (same op order
//! as `YoloConfig::full_param_list`), emitting ONNX nodes + initializers. BN is
//! folded into each conv ([`crate::fold`]); SiLU becomes `Sigmoid`+`Mul`; the
//! graph outputs the 6 raw head tensors (per-scale cls/reg, NCHW) — DFL decode +
//! NMS stay on the host.
//!
//! The SAME walker emits both the fp32 graph (`quant = None`) and the INT8-QDQ
//! graph (`quant = Some`), the only difference being `QuantizeLinear`/
//! `DequantizeLinear` pairs inserted around each quantized conv.

use onnx::{GraphBuilder, Node};

use crate::fold::{fold_bn, quantize_weight_per_channel};
use crate::quant::Quant;

/// Reads model tensors by name (the conv weights + BN affine/stats + head bias).
pub trait WeightSource {
    /// Tensor data for `name`. Panics with a clear message if absent.
    fn get(&self, name: &str) -> Vec<f32>;
}

impl WeightSource for checkpoint::Container {
    fn get(&self, name: &str) -> Vec<f32> {
        self.find(name, "")
            .cloned()
            .unwrap_or_else(|| panic!("checkpoint is missing tensor `{name}`"))
    }
}

/// Streaming source: decodes exactly one tensor from the mmap per call (no
/// whole-checkpoint host copy) — see `checkpoint::weightio`.
impl WeightSource for checkpoint::weightio::WeightReader {
    fn get(&self, name: &str) -> Vec<f32> {
        self.tensor(name).unwrap_or_else(|| panic!("checkpoint is missing tensor `{name}`"))
    }
}

/// A feature-map edge in the graph being built: its ONNX tensor name + NCHW dims.
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
    quant: Option<&'a Quant>,
}

impl<'a> Exporter<'a> {
    /// conv (BN folded) → SiLU. `prefix` matches the brain `Conv` prefix (and the
    /// calibration tap name).
    fn conv(&mut self, prefix: &str, x: &Feat, cout: u32, k: u32, stride: u32) -> Feat {
        let pad = if k == 3 { 1 } else { 0 };
        let cin = x.c;
        let w = self.w.get(&format!("{prefix}.conv.weight"));
        let gamma = self.w.get(&format!("{prefix}.bn.gamma"));
        let beta = self.w.get(&format!("{prefix}.bn.beta"));
        let rm = self.w.get(&format!("{prefix}.bn.run_mean"));
        let rv = self.w.get(&format!("{prefix}.bn.run_var"));
        let (wp, bias) = fold_bn(&w, &gamma, &beta, &rm, &rv, cout as usize);

        let conv_out = self.conv_node(prefix, &x.name, &wp, &bias, cin, cout, k, stride, pad);

        // SiLU = x * sigmoid(x). OpenVINO fuses Sigmoid+Mul into Swish.
        let sig = format!("{prefix}.sig");
        self.g.add(Node::new("Sigmoid", &[&conv_out], &[&sig]).name(&format!("{prefix}/Sigmoid")));
        let act = format!("{prefix}.act");
        self.g.add(Node::new("Mul", &[&conv_out, &sig], &[&act]).name(&format!("{prefix}/SiLU")));

        let ho = (x.h + 2 * pad - k) / stride + 1;
        let wo = (x.w + 2 * pad - k) / stride + 1;
        Feat { name: act, c: cout, h: ho, w: wo }
    }

    /// Emit the Conv op (fp32, or INT8-QDQ when calibrated), returning its output
    /// tensor name. Weight/bias are initializers; BN is already folded in.
    #[allow(clippy::too_many_arguments)]
    fn conv_node(
        &mut self,
        prefix: &str,
        x_name: &str,
        wp: &[f32],
        bias: &[f32],
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
    ) -> String {
        let out = format!("{prefix}.conv");
        let dims = [cout as i64, cin as i64, k as i64, k as i64];

        let (w_input, x_input) = match self.quant {
            None => {
                let wname = format!("{prefix}.weight");
                self.g.init_f32(&wname, &dims, wp.to_vec());
                (wname, x_name.to_string())
            }
            Some(q) => {
                // Per-channel INT8 weights + DequantizeLinear(axis=0).
                let per = (cin * k * k) as usize;
                let (wq, wscales) = quantize_weight_per_channel(wp, cout as usize, per);
                let wq_name = format!("{prefix}.weight_i8");
                self.g.init_i8(&wq_name, &dims, wq);
                let wsc_name = format!("{prefix}.weight_scale");
                self.g.init_f32(&wsc_name, &[cout as i64], wscales);
                let wzp_name = format!("{prefix}.weight_zp");
                self.g.init_i8(&wzp_name, &[cout as i64], vec![0i8; cout as usize]);
                let wdq = format!("{prefix}.weight_dq");
                self.g.add(
                    Node::new("DequantizeLinear", &[&wq_name, &wsc_name, &wzp_name], &[&wdq])
                        .name(&format!("{prefix}/WeightDQ"))
                        .attr_int("axis", 0),
                );
                // Per-tensor activation Q/DQ.
                let a_scale = q.act_scale(prefix).unwrap_or_else(|| {
                    panic!("no calibrated activation scale for conv `{prefix}`")
                });
                let xq = self.qdq_activation(prefix, x_name, a_scale);
                (wdq, xq)
            }
        };

        let bname = format!("{prefix}.bias");
        self.g.init_f32(&bname, &[cout as i64], bias.to_vec());
        self.g.add(
            Node::new("Conv", &[&x_input, &w_input, &bname], &[&out])
                .name(&format!("{prefix}/Conv"))
                .attr_ints("kernel_shape", &[k as i64, k as i64])
                .attr_ints("strides", &[stride as i64, stride as i64])
                .attr_ints("pads", &[pad as i64, pad as i64, pad as i64, pad as i64])
                .attr_ints("dilations", &[1, 1])
                .attr_int("group", 1),
        );
        out
    }

    /// Per-tensor symmetric `QuantizeLinear`→`DequantizeLinear` on an activation.
    fn qdq_activation(&mut self, prefix: &str, x_name: &str, scale: f32) -> String {
        let sc = format!("{prefix}.act_scale");
        self.g.init_f32(&sc, &[], vec![scale]);
        let zp = format!("{prefix}.act_zp");
        self.g.init_i8(&zp, &[], vec![0i8]);
        let q = format!("{prefix}.act_q");
        self.g.add(Node::new("QuantizeLinear", &[x_name, &sc, &zp], &[&q]).name(&format!("{prefix}/ActQ")));
        let dq = format!("{prefix}.act_dq");
        self.g.add(Node::new("DequantizeLinear", &[&q, &sc, &zp], &[&dq]).name(&format!("{prefix}/ActDQ")));
        dq
    }

    /// Bottleneck: two K3 convs, optional residual `Add` (matches
    /// `yolo::blocks::Bottleneck`: shortcut only when `shortcut && cin == cout`).
    fn bottleneck(&mut self, prefix: &str, x: &Feat, cout: u32, shortcut: bool) -> Feat {
        let a = self.conv(&format!("{prefix}.cv1"), x, cout, 3, 1);
        let b = self.conv(&format!("{prefix}.cv2"), &a, cout, 3, 1);
        if shortcut && x.c == cout {
            let out = format!("{prefix}.add");
            self.g.add(Node::new("Add", &[&x.name, &b.name], &[&out]).name(&format!("{prefix}/Add")));
            Feat { name: out, c: cout, h: b.h, w: b.w }
        } else {
            b
        }
    }

    /// C2f: cv1 (1×1 → 2c) → Split → bottleneck chain → Concat[y0,y1,b…] → cv2.
    fn c2f(&mut self, prefix: &str, x: &Feat, cout: u32, n: u32, shortcut: bool) -> Feat {
        let c = cout / 2;
        let y = self.conv(&format!("{prefix}.cv1"), x, 2 * c, 1, 1);
        // Split into y0 (channels [0,c)) and y1 ([c,2c)) along axis 1. Opset-13
        // Split takes the split sizes as a second (int64) input.
        let split_sz = format!("{prefix}.split_sz");
        self.g.init_i64(&split_sz, &[2], vec![c as i64, c as i64]);
        let y0 = format!("{prefix}.y0");
        let y1 = format!("{prefix}.y1");
        self.g.add(
            Node::new("Split", &[&y.name, &split_sz], &[&y0, &y1])
                .name(&format!("{prefix}/Split"))
                .attr_int("axis", 1),
        );

        let mut chunks: Vec<String> = vec![y0, y1.clone()];
        let mut prev = Feat { name: y1, c, h: y.h, w: y.w };
        for i in 0..n {
            let b = self.bottleneck(&format!("{prefix}.m.{i}"), &prev, c, shortcut);
            chunks.push(b.name.clone());
            prev = b;
        }
        let cat = format!("{prefix}.cat");
        let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        self.g.add(Node::new("Concat", &refs, &[&cat]).name(&format!("{prefix}/Concat")).attr_int("axis", 1));
        let cat_c = (2 + n) * c;
        let catf = Feat { name: cat, c: cat_c, h: y.h, w: y.w };
        self.conv(&format!("{prefix}.cv2"), &catf, cout, 1, 1)
    }

    /// SPPF: cv1 (1×1 → c) → 3 chained 5×5 maxpools → Concat[x,m1,m2,m3] → cv2.
    fn sppf(&mut self, prefix: &str, x: &Feat, cout: u32) -> Feat {
        let c = cout / 2;
        let cv1 = self.conv(&format!("{prefix}.cv1"), x, c, 1, 1);
        let m1 = self.maxpool(prefix, "m1", &cv1);
        let m2 = self.maxpool(prefix, "m2", &m1);
        let m3 = self.maxpool(prefix, "m3", &m2);
        let cat = format!("{prefix}.cat");
        self.g.add(
            Node::new("Concat", &[&cv1.name, &m1.name, &m2.name, &m3.name], &[&cat])
                .name(&format!("{prefix}/Concat"))
                .attr_int("axis", 1),
        );
        let catf = Feat { name: cat, c: 4 * c, h: cv1.h, w: cv1.w };
        self.conv(&format!("{prefix}.cv2"), &catf, cout, 1, 1)
    }

    fn maxpool(&mut self, prefix: &str, tag: &str, x: &Feat) -> Feat {
        let out = format!("{prefix}.{tag}");
        self.g.add(
            Node::new("MaxPool", &[&x.name], &[&out])
                .name(&format!("{prefix}/{tag}"))
                .attr_ints("kernel_shape", &[5, 5])
                .attr_ints("strides", &[1, 1])
                .attr_ints("pads", &[2, 2, 2, 2])
                .attr_int("ceil_mode", 0),
        );
        Feat { name: out, c: x.c, h: x.h, w: x.w }
    }

    /// 2× nearest upsample = ONNX `Resize` (scales input `[1,1,2,2]`, asymmetric/
    /// floor) — matches brain's `upsample2` integer doubling.
    fn upsample(&mut self, prefix: &str, x: &Feat) -> Feat {
        let scales = format!("{prefix}.scales");
        self.g.init_f32(&scales, &[4], vec![1.0, 1.0, 2.0, 2.0]);
        let out = format!("{prefix}.up");
        // Resize inputs: (X, roi, scales). roi is skipped via an empty input "".
        self.g.add(
            Node::new("Resize", &[&x.name, "", &scales], &[&out])
                .name(&format!("{prefix}/Resize"))
                .attr_str("mode", "nearest")
                .attr_str("coordinate_transformation_mode", "asymmetric")
                .attr_str("nearest_mode", "floor"),
        );
        Feat { name: out, c: x.c, h: x.h * 2, w: x.w * 2 }
    }

    fn concat2(&mut self, prefix: &str, a: &Feat, b: &Feat) -> Feat {
        let out = format!("{prefix}.cat");
        self.g.add(
            Node::new("Concat", &[&a.name, &b.name], &[&out]).name(&format!("{prefix}/Concat")).attr_int("axis", 1),
        );
        Feat { name: out, c: a.c + b.c, h: a.h, w: a.w }
    }

    /// A head branch: two K3 convs then a BIASED 1×1 (kept fp32 — the head logits
    /// stay full precision; the NPU runs them as FP16/FP32 alongside the INT8
    /// backbone/neck). `out_name` becomes the graph output tensor name.
    fn head_branch(&mut self, prefix: &str, x: &Feat, mid: u32, out_c: u32, out_name: &str) -> Feat {
        let a = self.conv(&format!("{prefix}.0"), x, mid, 3, 1);
        let b = self.conv(&format!("{prefix}.1"), &a, mid, 3, 1);
        // Final biased 1×1 (Ultralytics' plain nn.Conv2d), fp32.
        let w = self.w.get(&format!("{prefix}.2.weight")); // [out_c * mid]
        let bias = self.w.get(&format!("{prefix}.2.bias"));
        let wname = format!("{prefix}.2.weight");
        self.g.init_f32(&wname, &[out_c as i64, b.c as i64, 1, 1], w);
        let bname = format!("{prefix}.2.bias");
        self.g.init_f32(&bname, &[out_c as i64], bias);
        self.g.add(
            Node::new("Conv", &[&b.name, &wname, &bname], &[out_name])
                .name(&format!("{prefix}/Head1x1"))
                .attr_ints("kernel_shape", &[1, 1])
                .attr_ints("strides", &[1, 1])
                .attr_ints("pads", &[0, 0, 0, 0])
                .attr_ints("dilations", &[1, 1])
                .attr_int("group", 1),
        );
        Feat { name: out_name.to_string(), c: out_c, h: b.h, w: b.w }
    }
}

/// Build the full YOLOv8 ONNX graph into `g`. `quant = None` ⇒ fp32; `Some` ⇒
/// INT8 Q/DQ. The graph has one static input `images:[1,3,input,input]` and 6
/// outputs `head.{0,1,2}.{cls,reg}` (NCHW).
pub fn build_graph(cfg: &yolo::YoloConfig, w: &dyn WeightSource, quant: Option<&Quant>, g: &mut GraphBuilder) {
    let s = cfg.input as i64;
    g.input_f32("images", &[1, 3, s, s]);

    // (name, channels, h, w) of the 6 outputs, registered after the borrow ends.
    let mut outs: Vec<(String, i64, i64, i64)> = Vec::new();
    {
        let mut ex = Exporter { g: &mut *g, w, quant };
        let img = Feat { name: "images".to_string(), c: 3, h: cfg.input, w: cfg.input };
        let bch = cfg.backbone_ch;
        let bd = cfg.backbone_depth;

        // ---- backbone ----
        let c0 = ex.conv("backbone.0", &img, bch[0], 3, 2);
        let c1 = ex.conv("backbone.1", &c0, bch[1], 3, 2);
        let c2 = ex.c2f("backbone.2", &c1, bch[2], bd[0], true);
        let c3 = ex.conv("backbone.3", &c2, bch[3], 3, 2);
        let p3 = ex.c2f("backbone.4", &c3, bch[4], bd[1], true); // P3
        let c5 = ex.conv("backbone.5", &p3, bch[5], 3, 2);
        let p4 = ex.c2f("backbone.6", &c5, bch[6], bd[2], true); // P4
        let c7 = ex.conv("backbone.7", &p4, bch[7], 3, 2);
        let c8 = ex.c2f("backbone.8", &c7, bch[8], bd[3], true);
        let p5 = ex.sppf("backbone.9", &c8, bch[9]); // P5

        // ---- neck (PAN-FPN) ----
        let nck = cfg.neck_ch;
        let nd = cfg.neck_depth.max(1);
        let up5 = ex.upsample("neck.up5", &p5);
        let cat0 = ex.concat2("neck.cat0", &up5, &p4);
        let t4 = ex.c2f("neck.0", &cat0, nck[0], nd, false);
        let up4 = ex.upsample("neck.up4", &t4);
        let cat1 = ex.concat2("neck.cat1", &up4, &p3);
        let n3 = ex.c2f("neck.1", &cat1, nck[1], nd, false);
        let dn3 = ex.conv("neck.2", &n3, nck[2], 3, 2);
        let cat2 = ex.concat2("neck.cat2", &dn3, &t4);
        let n4 = ex.c2f("neck.3", &cat2, nck[3], nd, false);
        let dn4 = ex.conv("neck.4", &n4, nck[4], 3, 2);
        let cat3 = ex.concat2("neck.cat3", &dn4, &p5);
        let n5 = ex.c2f("neck.5", &cat3, nck[5], nd, false);

        // ---- head (3 scales on N3/N4/N5) ----
        let scales = [n3, n4, n5];
        let four_rm = 4 * cfg.reg_max;
        for (si, feat) in scales.iter().enumerate() {
            let cls_out = format!("head.{si}.cls");
            let reg_out = format!("head.{si}.reg");
            let cls = ex.head_branch(&format!("head.{si}.cls"), feat, cfg.cls_mid, cfg.nc, &cls_out);
            let reg = ex.head_branch(&format!("head.{si}.reg"), feat, cfg.reg_mid, four_rm, &reg_out);
            outs.push((cls.name, cfg.nc as i64, cls.h as i64, cls.w as i64));
            outs.push((reg.name, four_rm as i64, reg.h as i64, reg.w as i64));
        }
    }

    for (name, c, h, w) in &outs {
        g.output_f32(name, &[1, *c, *h, *w]);
    }
}
