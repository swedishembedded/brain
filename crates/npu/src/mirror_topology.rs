// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 → ONNX for the Intel NPU (whole-graph OpenVINO path).
//!
//! Stage 6a: the per-frame DINOv2 ViT-L/14-reg encoder as a fixed
//! `[1,3,518,518] → [1,1369,1024]` graph — the biggest per-frame win and the
//! template for the trunk export (6b: same block emitter + RoPE tables as
//! initializers; 6c: DPT heads). LayerNorm and erf-GELU are decomposed
//! (no native ONNX ops in the builder); attention is plain
//! MatMul/Softmax/MatMul at fp32.
//!
//! Weight source = the mirror init map (checkpoint-verbatim names under
//! `visual_geometry_transformer.patch_embed.`); Linear weights are
//! transposed host-side to `[in,out]` for MatMul.

use std::collections::HashMap;

use onnx::builder::GraphBuilder;
use onnx::graph::Node;

pub type W = HashMap<String, Vec<f32>>;

const C: i64 = 1024;
const HEADS: i64 = 16;
const HD: i64 = 64;
const T: i64 = 1374; // cls + 4 reg + 1369 patches
const P: i64 = 1369;
const REG: i64 = 4;
const EPS: f32 = 1e-6;
const PE: &str = "visual_geometry_transformer.patch_embed";

struct Topo<'a> {
    g: &'a mut GraphBuilder,
    w: &'a W,
    n: usize,
}

impl<'a> Topo<'a> {
    fn tmp(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("mir_{tag}_{}", self.n)
    }
    fn node(&mut self, op: &str, ins: &[&str], out: &str) {
        self.g.add(Node::new(op, ins, &[out]));
    }
    fn unary(&mut self, op: &str, x: &str, tag: &str) -> String {
        let o = self.tmp(tag);
        self.node(op, &[x], &o);
        o
    }
    fn binary(&mut self, op: &str, a: &str, b: &str, tag: &str) -> String {
        let o = self.tmp(tag);
        self.node(op, &[a, b], &o);
        o
    }
    fn host(&self, name: &str) -> &Vec<f32> {
        self.w.get(name).unwrap_or_else(|| panic!("missing weight {name}"))
    }
    /// Linear weight `[out,in]` → initializer `[in,out]` for MatMul.
    fn linear_t(&mut self, name: &str, out_d: i64, in_d: i64) -> String {
        let src = self.host(name).clone();
        let mut t = vec![0.0f32; src.len()];
        for o in 0..out_d as usize {
            for i in 0..in_d as usize {
                t[i * out_d as usize + o] = src[o * in_d as usize + i];
            }
        }
        let iname = format!("{name}.T");
        self.g.init_f32(&iname, &[in_d, out_d], t);
        iname
    }
    fn init_vec(&mut self, name: &str, dims: &[i64]) -> String {
        let v = self.host(name).clone();
        self.g.init_f32(name, dims, v);
        name.to_string()
    }
    fn reshape_to(&mut self, x: &str, shape: &[i64]) -> String {
        let sname = self.tmp("shape");
        self.g.init_i64(&sname, &[shape.len() as i64], shape.to_vec());
        self.binary("Reshape", x, &sname, "rs")
    }
    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        let o = self.tmp("tr");
        self.g.add(Node::new("Transpose", &[x], &[&o]).attr_ints("perm", perm));
        o
    }
    /// Decomposed LayerNorm over the last axis with affine weights.
    fn layernorm(&mut self, x: &str, wname: &str, bname: &str) -> String {
        self.layernorm_d(x, wname, bname, C, "mir_eps")
    }
    /// LayerNorm with explicit affine width and eps initializer (the trunk
    /// uses eps 1e-5 and 64-wide per-head QK norms).
    fn layernorm_d(&mut self, x: &str, wname: &str, bname: &str, dim: i64, eps: &str) -> String {
        let mean = {
            let o = self.tmp("ln_mean");
            self.g.add(Node::new("ReduceMean", &[x], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
            o
        };
        let xc = self.binary("Sub", x, &mean, "ln_xc");
        let sq = self.binary("Mul", &xc, &xc, "ln_sq");
        let var = {
            let o = self.tmp("ln_var");
            self.g.add(Node::new("ReduceMean", &[&sq], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
            o
        };
        let veps = self.binary("Add", &var, eps, "ln_veps");
        let std = self.unary("Sqrt", &veps, "ln_std");
        let xn = self.binary("Div", &xc, &std, "ln_xn");
        let g1 = self.init_vec(wname, &[dim]);
        let sc = self.binary("Mul", &xn, &g1, "ln_g");
        let b1 = self.init_vec(bname, &[dim]);
        self.binary("Add", &sc, &b1, "ln_out")
    }
    /// Constant-range Slice along `axis`.
    fn slice(&mut self, x: &str, axis: i64, start: i64, end: i64) -> String {
        let st = self.tmp("sl_s");
        self.g.init_i64(&st, &[1], vec![start]);
        let en = self.tmp("sl_e");
        self.g.init_i64(&en, &[1], vec![end]);
        let ax = self.tmp("sl_a");
        self.g.init_i64(&ax, &[1], vec![axis]);
        let o = self.tmp("sl");
        self.g.add(Node::new("Slice", &[x, &st, &en, &ax], &[&o]));
        o
    }
    /// Normalized 2D RoPE on `[b,H,t,64]` via precomputed `[1,1,t,32]`
    /// cos/sin initializers: pairs (d, d+32) share angle d (rotate-half).
    fn rope(&mut self, x: &str, cos: &str, sin: &str) -> String {
        let x1 = self.slice(x, 3, 0, HD / 2);
        let x2 = self.slice(x, 3, HD / 2, HD);
        let a = self.binary("Mul", &x1, cos, "rope_a");
        let b = self.binary("Mul", &x2, sin, "rope_b");
        let r1 = self.binary("Sub", &a, &b, "rope_r1");
        let c1 = self.binary("Mul", &x2, cos, "rope_c");
        let d1 = self.binary("Mul", &x1, sin, "rope_d");
        let r2 = self.binary("Add", &c1, &d1, "rope_r2");
        let o = self.tmp("rope");
        self.g.add(Node::new("Concat", &[&r1, &r2], &[&o]).attr_int("axis", 3));
        o
    }
    /// One trunk block (QK-LN + 2D RoPE + LayerScale) on `[b,t,C]`. `cos`/
    /// `sin` are `[1,1,t,32]` tables matching this block's attention span.
    fn trunk_block(&mut self, x: &str, p: &str, b: i64, t: i64, cos: &str, sin: &str) -> String {
        let ln1 = self.layernorm_d(x, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"), C, "mir_eps5");
        let qkv = self.linear(&ln1, &format!("{p}.attn.qkv"), 3 * C, C);
        let q5 = self.reshape_to(&qkv, &[b, t, 3, HEADS, HD]);
        let names: Vec<String> = (0..3).map(|i| self.tmp(&format!("tqkv{i}"))).collect();
        {
            let outs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            self.g.add(
                Node::new("Split", &[&q5], &outs).attr_int("axis", 2).attr_ints("split", &[1, 1, 1]),
            );
        }
        let mut qh = Vec::new();
        for nm in &names {
            let sq = self.reshape_to(nm, &[b, t, HEADS, HD]);
            qh.push(self.transpose(&sq, &[0, 2, 1, 3])); // [b,H,t,D]
        }
        let qn = self.layernorm_d(&qh[0], &format!("{p}.attn.q_norm.weight"), &format!("{p}.attn.q_norm.bias"), HD, "mir_eps5");
        let kn = self.layernorm_d(&qh[1], &format!("{p}.attn.k_norm.weight"), &format!("{p}.attn.k_norm.bias"), HD, "mir_eps5");
        let qr = self.rope(&qn, cos, sin);
        let kr = self.rope(&kn, cos, sin);
        let kt = self.transpose(&kr, &[0, 1, 3, 2]);
        let s0 = self.binary("MatMul", &qr, &kt, "tscores");
        let s1 = self.binary("Mul", &s0, "mir_attn_scale", "tscores_s");
        let probs = {
            let o = self.tmp("tprobs");
            self.g.add(Node::new("Softmax", &[&s1], &[&o]).attr_int("axis", 3));
            o
        };
        let ctx = self.binary("MatMul", &probs, &qh[2], "tctx");
        let cxt = self.transpose(&ctx, &[0, 2, 1, 3]);
        let cflat = self.reshape_to(&cxt, &[b, t, C]);
        let proj = self.linear(&cflat, &format!("{p}.attn.proj"), C, C);
        let ls1 = self.init_vec(&format!("{p}.ls1.gamma"), &[C]);
        let a = self.binary("Mul", &proj, &ls1, "tls1");
        let x1 = self.binary("Add", x, &a, "tres1");
        let ln2 = self.layernorm_d(&x1, &format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"), C, "mir_eps5");
        let h = self.linear(&ln2, &format!("{p}.mlp.fc1"), 4 * C, C);
        let hg = self.gelu(&h);
        let m = self.linear(&hg, &format!("{p}.mlp.fc2"), C, 4 * C);
        let ls2 = self.init_vec(&format!("{p}.ls2.gamma"), &[C]);
        let bb = self.binary("Mul", &m, &ls2, "tls2");
        self.binary("Add", &x1, &bb, "tres2")
    }
    /// erf-GELU: 0.5 * x * (1 + Erf(x / sqrt(2)))
    fn gelu(&mut self, x: &str) -> String {
        let xr = self.binary("Mul", x, "mir_inv_sqrt2", "gelu_xr");
        let e = self.unary("Erf", &xr, "gelu_erf");
        let e1 = self.binary("Add", &e, "mir_one", "gelu_e1");
        let xh = self.binary("Mul", x, "mir_half", "gelu_xh");
        self.binary("Mul", &xh, &e1, "gelu_out")
    }
    /// Linear `prefix.{weight,bias}` on `[1,T,in]` tokens.
    fn linear(&mut self, x: &str, prefix: &str, out_d: i64, in_d: i64) -> String {
        let wt = self.linear_t(&format!("{prefix}.weight"), out_d, in_d);
        let mm = self.binary("MatMul", x, &wt, "lin");
        let b = self.init_vec(&format!("{prefix}.bias"), &[out_d]);
        self.binary("Add", &mm, &b, "lin_b")
    }
    /// One DINOv2 block (pre-LN, plain MHA, LayerScale) on `[1,T,C]`.
    fn block(&mut self, x: &str, p: &str) -> String {
        let ln1 = self.layernorm(x, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"));
        let qkv = self.linear(&ln1, &format!("{p}.attn.qkv"), 3 * C, C);
        // [1,T,3C] -> [1,T,3,H,D] -> Split along axis 2
        let q5 = self.reshape_to(&qkv, &[1, T, 3, HEADS, HD]);
        let names: Vec<String> = (0..3).map(|i| self.tmp(&format!("qkv{i}"))).collect();
        {
            let outs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            self.g.add(
                Node::new("Split", &[&q5], &outs).attr_int("axis", 2).attr_ints("split", &[1, 1, 1]),
            );
        }
        let mut qh = Vec::new();
        for nm in &names {
            let sq = self.reshape_to(nm, &[1, T, HEADS, HD]);
            qh.push(self.transpose(&sq, &[0, 2, 1, 3])); // [1,H,T,D]
        }
        let kt = self.transpose(&qh[1], &[0, 1, 3, 2]); // [1,H,D,T]
        let s0 = self.binary("MatMul", &qh[0], &kt, "scores");
        let s1 = self.binary("Mul", &s0, "mir_attn_scale", "scores_s");
        let probs = {
            let o = self.tmp("probs");
            self.g.add(Node::new("Softmax", &[&s1], &[&o]).attr_int("axis", 3));
            o
        };
        let ctx = self.binary("MatMul", &probs, &qh[2], "ctx"); // [1,H,T,D]
        let cxt = self.transpose(&ctx, &[0, 2, 1, 3]);
        let cflat = self.reshape_to(&cxt, &[1, T, C]);
        let proj = self.linear(&cflat, &format!("{p}.attn.proj"), C, C);
        let ls1 = self.init_vec(&format!("{p}.ls1.gamma"), &[C]);
        let a = self.binary("Mul", &proj, &ls1, "ls1");
        let x1 = self.binary("Add", x, &a, "res1");
        let ln2 = self.layernorm(&x1, &format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"));
        let h = self.linear(&ln2, &format!("{p}.mlp.fc1"), 4 * C, C);
        let hg = self.gelu(&h);
        let m = self.linear(&hg, &format!("{p}.mlp.fc2"), C, 4 * C);
        let ls2 = self.init_vec(&format!("{p}.ls2.gamma"), &[C]);
        let b = self.binary("Mul", &m, &ls2, "ls2");
        self.binary("Add", &x1, &b, "res2")
    }
}

/// Build the DINOv2 encoder graph: input `frame` `[1,3,518,518]` (ImageNet-
/// normalized CHW), output `patch_tokens` `[1,1369,1024]`. `blocks` = 24 for
/// the real model (smaller for structural tests).
pub fn build_dinov2_graph(w: &W, g: &mut GraphBuilder, blocks: usize) {
    g.input_f32("frame", &[1, 3, 518, 518]);
    g.init_f32("mir_eps", &[1], vec![EPS]);
    g.init_f32("mir_one", &[1], vec![1.0]);
    g.init_f32("mir_half", &[1], vec![0.5]);
    g.init_f32("mir_inv_sqrt2", &[1], vec![std::f32::consts::FRAC_1_SQRT_2]);
    g.init_f32("mir_attn_scale", &[1], vec![1.0 / (HD as f32).sqrt()]);

    let mut tp = Topo { g, w, n: 0 };
    // patch conv 14x14 s14
    let pw = tp.host(&format!("{PE}.patch_embed.proj.weight")).clone();
    tp.g.init_f32(&format!("{PE}.patch_embed.proj.weight"), &[C, 3, 14, 14], pw);
    let pb = tp.host(&format!("{PE}.patch_embed.proj.bias")).clone();
    tp.g.init_f32(&format!("{PE}.patch_embed.proj.bias"), &[C], pb);
    let conv = tp.tmp("patch_conv");
    tp.g.add(
        Node::new(
            "Conv",
            &["frame", &format!("{PE}.patch_embed.proj.weight"), &format!("{PE}.patch_embed.proj.bias")],
            &[&conv],
        )
        .attr_ints("kernel_shape", &[14, 14])
        .attr_ints("strides", &[14, 14])
        .attr_ints("pads", &[0, 0, 0, 0]),
    );
    let flat = tp.reshape_to(&conv, &[1, C, P]);
    let patches = tp.transpose(&flat, &[0, 2, 1]); // [1,P,C]
    // + patch positional embedding (pos_embed rows 1..)
    let pos = tp.host(&format!("{PE}.pos_embed")).clone();
    tp.g.init_f32("mir_pos_patch", &[1, P, C], pos[C as usize..].to_vec());
    let patches = tp.binary("Add", &patches, "mir_pos_patch", "patch_pos");
    // head rows: [cls+pos0, reg x4] (registers get no pos)
    let cls = tp.host(&format!("{PE}.cls_token")).clone();
    let regs = tp.host(&format!("{PE}.register_tokens")).clone();
    let mut head = Vec::with_capacity(((1 + REG) * C) as usize);
    for d in 0..C as usize {
        head.push(cls[d] + pos[d]);
    }
    head.extend_from_slice(&regs);
    tp.g.init_f32("mir_head_rows", &[1, 1 + REG, C], head);
    let toks = {
        let o = tp.tmp("tokens");
        tp.g.add(Node::new("Concat", &["mir_head_rows", &patches], &[&o]).attr_int("axis", 1));
        o
    };

    let mut x = toks;
    for b in 0..blocks {
        x = tp.block(&x, &format!("{PE}.blocks.{b}"));
    }
    let xn = tp.layernorm(&x, &format!("{PE}.norm.weight"), &format!("{PE}.norm.bias"));
    // drop cls+registers: Slice rows [5, 1374)
    let starts = tp.tmp("sl_starts");
    tp.g.init_i64(&starts, &[1], vec![1 + REG]);
    let ends = tp.tmp("sl_ends");
    tp.g.init_i64(&ends, &[1], vec![T]);
    let axes = tp.tmp("sl_axes");
    tp.g.init_i64(&axes, &[1], vec![1]);
    tp.g.add(Node::new("Slice", &[&xn, &starts, &ends, &axes], &["patch_tokens"]));
    tp.g.output_f32("patch_tokens", &[1, P, C]);
}

/// Build the trunk graph (stage 6b): input `patch_tokens` `[s, hp*wp, 1024]`
/// (per-frame DINOv2 outputs from the 6a graph), outputs `tap{i}`
/// `[s, td, 2048]` — the concat(frame, global) taps the DPT/camera heads
/// consume, `td = 7 + hp*wp`. Token assembly (cam/reg specials, zero
/// pose/ray) and the normalized 2D RoPE tables are baked in as initializers;
/// frame attention batches over `s`, global attention runs on the flattened
/// `[1, s*td, C]` sequence (fixed-shape graph, one per (s, grid)).
pub fn build_trunk_graph(
    w: &W,
    g: &mut GraphBuilder,
    s: usize,
    hp: usize,
    wp: usize,
    levels: usize,
    tap_levels: &[usize],
) {
    const VGT: &str = "visual_geometry_transformer";
    let patches = (hp * wp) as i64;
    let td = mirror::model::PATCH_START as i64 + patches;
    let sb = s as i64;
    g.input_f32("patch_tokens", &[sb, patches, C]);
    g.init_f32("mir_eps5", &[1], vec![1e-5]);
    g.init_f32("mir_one", &[1], vec![1.0]);
    g.init_f32("mir_half", &[1], vec![0.5]);
    g.init_f32("mir_inv_sqrt2", &[1], vec![std::f32::consts::FRAC_1_SQRT_2]);
    g.init_f32("mir_attn_scale", &[1], vec![1.0 / (HD as f32).sqrt()]);

    let mut tp = Topo { g, w, n: 0 };
    // special head rows [s, 7, C]: cam+reg variant 0 for frame 0, variant 1
    // for later frames; pose/ray rows zero (no-prior path).
    let ps = mirror::model::PATCH_START;
    let cam_t = tp.host(&format!("{VGT}.cam_token")).clone();
    let reg_t = tp.host(&format!("{VGT}.reg_token")).clone();
    let c = C as usize;
    let head = |variant: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(ps * c);
        v.extend_from_slice(&cam_t[variant * c..(variant + 1) * c]);
        v.extend_from_slice(&reg_t[variant * 4 * c..(variant + 1) * 4 * c]);
        v.resize(ps * c, 0.0);
        v
    };
    let mut heads = head(0);
    for _ in 1..s {
        heads.extend_from_slice(&head(1));
    }
    tp.g.init_f32("mir_trunk_heads", &[sb, ps as i64, C], heads);
    let x0 = {
        let o = tp.tmp("trunk_tokens");
        tp.g.add(Node::new("Concat", &["mir_trunk_heads", "patch_tokens"], &[&o]).attr_int("axis", 1));
        o
    };

    // RoPE tables: frame span [1,1,td,32]; global span tiles them s times.
    let periods = tp.host(&format!("{VGT}.frame_blocks.0.attn.rope.periods")).clone();
    let (cos, sin) = mirror::rope2d::rope_tables(&periods, hp, wp, ps);
    let mut gcos = Vec::with_capacity(s * cos.len());
    let mut gsin = Vec::with_capacity(s * sin.len());
    for _ in 0..s {
        gcos.extend_from_slice(&cos);
        gsin.extend_from_slice(&sin);
    }
    tp.g.init_f32("mir_rope_cos_f", &[1, 1, td, HD / 2], cos);
    tp.g.init_f32("mir_rope_sin_f", &[1, 1, td, HD / 2], sin);
    tp.g.init_f32("mir_rope_cos_g", &[1, 1, sb * td, HD / 2], gcos);
    tp.g.init_f32("mir_rope_sin_g", &[1, 1, sb * td, HD / 2], gsin);

    let mut x = x0;
    let mut tap_i = 0usize;
    for l in 0..levels {
        x = tp.trunk_block(&x, &format!("{VGT}.frame_blocks.{l}"), sb, td, "mir_rope_cos_f", "mir_rope_sin_f");
        let frame_out = x.clone();
        let flat = tp.reshape_to(&x, &[1, sb * td, C]);
        let gout = tp.trunk_block(&flat, &format!("{VGT}.global_blocks.{l}"), 1, sb * td, "mir_rope_cos_g", "mir_rope_sin_g");
        x = tp.reshape_to(&gout, &[sb, td, C]);
        if tap_levels.contains(&l) {
            let name = format!("tap{tap_i}");
            tp.g.add(Node::new("Concat", &[&frame_out, &x], &[&name]).attr_int("axis", 2));
            tp.g.output_f32(&name, &[sb, td, 2 * C]);
            tap_i += 1;
        }
    }
}

/// Build one DPT head graph (stage 6c): inputs `tap0..tap3` `[1, p, 2048]`
/// (one frame's PATCH rows of the trunk taps), output `head_out`
/// `[1, out_ch, H, W]` (pre-activation, exactly the brain/T5 buffers). With
/// `gs` an extra `rgb` `[1,3,H,W]` input and `gs_params` `[1,12,H,W]` output
/// replicate the gaussian-parameter branch (input_merger + gs_renderer).
/// `prefix` = "depth_head" | "pts_head" | "norm_head" | "gs_head".
#[allow(clippy::too_many_arguments)]
pub fn build_dpt_head_graph(
    w: &W,
    g: &mut GraphBuilder,
    cfg: &mirror::config::MirrorConfig,
    prefix: &str,
    out_ch: i64,
    hp: usize,
    wp: usize,
    gs: bool,
) {
    let (ph, pw) = (hp as i64, wp as i64);
    let p = ph * pw;
    let c2 = 2 * cfg.dim as i64;
    let f2 = cfg.dpt_feat as i64;
    let (h, wdt) = (ph * cfg.patch as i64, pw * cfg.patch as i64);
    for i in 0..4 {
        g.input_f32(&format!("tap{i}"), &[1, p, c2]);
    }
    if gs {
        g.input_f32("rgb", &[1, 3, h, wdt]);
    }
    g.init_f32("mir_eps5", &[1], vec![1e-5]);

    let mut tp = Topo { g, w, n: 0 };
    let nm = |s: &str| format!("{prefix}.{s}");
    // conv with the bias as a direct Conv input
    fn conv(
        tp: &mut Topo,
        x: &str,
        wname: &str,
        bias: Option<&str>,
        shape: &[i64],
        k: i64,
        stride: i64,
        pad: i64,
    ) -> String {
        let wv = tp.host(wname).clone();
        tp.g.init_f32(wname, shape, wv);
        let mut ins = vec![x, wname];
        if let Some(b) = bias {
            let bv = tp.host(b).clone();
            tp.g.init_f32(b, &[shape[0]], bv);
            ins.push(b);
        }
        let o = tp.tmp("conv");
        tp.g.add(
            Node::new("Conv", &ins, &[&o])
                .attr_ints("kernel_shape", &[k, k])
                .attr_ints("strides", &[stride, stride])
                .attr_ints("pads", &[pad, pad, pad, pad]),
        );
        o
    }
    fn deconv(tp: &mut Topo, x: &str, wname: &str, bname: &str, cin: i64, cout: i64, k: i64) -> String {
        let wv = tp.host(wname).clone();
        tp.g.init_f32(wname, &[cin, cout, k, k], wv);
        let bv = tp.host(bname).clone();
        tp.g.init_f32(bname, &[cout], bv);
        let o = tp.tmp("deconv");
        tp.g.add(
            Node::new("ConvTranspose", &[x, wname, bname], &[&o])
                .attr_ints("kernel_shape", &[k, k])
                .attr_ints("strides", &[k, k]),
        );
        o
    }
    fn resize_ac(tp: &mut Topo, x: &str, c: i64, ho: i64, wo: i64) -> String {
        let sz = tp.tmp("rs_sz");
        tp.g.init_i64(&sz, &[4], vec![1, c, ho, wo]);
        let o = tp.tmp("resize");
        tp.g.add(
            Node::new("Resize", &[x, "", "", &sz], &[&o])
                .attr_str("mode", "linear")
                .attr_str("coordinate_transformation_mode", "align_corners"),
        );
        o
    }
    // reference ResidualConvUnit with the inplace-ReLU quirk: skip = relu(x)
    fn rcu(tp: &mut Topo, x: &str, c1w: &str, c1b: &str, c2w: &str, c2b: &str, f2: i64) -> String {
        let r = tp.unary("Relu", x, "rcu_r");
        let t1 = conv(tp, &r, c1w, Some(c1b), &[f2, f2, 3, 3], 3, 1, 1);
        let t1r = tp.unary("Relu", &t1, "rcu_t1");
        let t2 = conv(tp, &t1r, c2w, Some(c2b), &[f2, f2, 3, 3], 3, 1, 1);
        tp.binary("Add", &r, &t2, "rcu_out")
    }

    let dims: [(i64, i64); 4] = [(4 * ph, 4 * pw), (2 * ph, 2 * pw), (ph, pw), ((ph + 1) / 2, (pw + 1) / 2)];
    let mut rn: Vec<String> = Vec::new();
    for i in 0..4usize {
        let ln = tp.layernorm_d(&format!("tap{i}"), &nm("norm.weight"), &nm("norm.bias"), c2, "mir_eps5");
        let tr = tp.transpose(&ln, &[0, 2, 1]); // [1, 2C, p]
        let feat = tp.reshape_to(&tr, &[1, c2, ph, pw]);
        let oc = cfg.dpt_proj[i] as i64;
        let proj = conv(&mut tp, &feat, &nm(&format!("projects.{i}.weight")), Some(&nm(&format!("projects.{i}.bias"))), &[oc, c2, 1, 1], 1, 1, 0);
        let pos_name = format!("mir_{prefix}_pos{i}");
        tp.g.init_f32(&pos_name, &[1, oc, ph, pw], mirror::dpt::pos_embed_chw(oc as usize, hp, wp, 0.1));
        let posed = tp.binary("Add", &proj, &pos_name, "pos");
        let resized = match i {
            0 => deconv(&mut tp, &posed, &nm("resize_layers.0.weight"), &nm("resize_layers.0.bias"), oc, oc, 4),
            1 => deconv(&mut tp, &posed, &nm("resize_layers.1.weight"), &nm("resize_layers.1.bias"), oc, oc, 2),
            2 => posed,
            _ => conv(&mut tp, &posed, &nm("resize_layers.3.weight"), Some(&nm("resize_layers.3.bias")), &[oc, oc, 3, 3], 3, 2, 1),
        };
        rn.push(conv(&mut tp, &resized, &nm(&format!("scratch.layer{}_rn.weight", i + 1)), None, &[f2, oc, 3, 3], 3, 1, 1));
    }

    // fusion 4 -> 1 (refinenet4 has no residual unit 1)
    let r4 = format!("{prefix}.scratch.refinenet4");
    let mut fused = rcu(&mut tp, &rn[3], &format!("{r4}.resConfUnit2.conv1.weight"), &format!("{r4}.resConfUnit2.conv1.bias"), &format!("{r4}.resConfUnit2.conv2.weight"), &format!("{r4}.resConfUnit2.conv2.bias"), f2);
    fused = resize_ac(&mut tp, &fused, f2, dims[2].0, dims[2].1);
    fused = conv(&mut tp, &fused, &format!("{r4}.out_conv.weight"), Some(&format!("{r4}.out_conv.bias")), &[f2, f2, 1, 1], 1, 1, 0);
    for (r, rn_i) in [(3usize, 2usize), (2, 1), (1, 0)] {
        let pre = format!("{prefix}.scratch.refinenet{r}");
        let u = rcu(&mut tp, &rn[rn_i], &format!("{pre}.resConfUnit1.conv1.weight"), &format!("{pre}.resConfUnit1.conv1.bias"), &format!("{pre}.resConfUnit1.conv2.weight"), &format!("{pre}.resConfUnit1.conv2.bias"), f2);
        let sum = tp.binary("Add", &fused, &u, "fuse_add");
        fused = rcu(&mut tp, &sum, &format!("{pre}.resConfUnit2.conv1.weight"), &format!("{pre}.resConfUnit2.conv1.bias"), &format!("{pre}.resConfUnit2.conv2.weight"), &format!("{pre}.resConfUnit2.conv2.bias"), f2);
        let target = if rn_i == 0 { (8 * ph, 8 * pw) } else { dims[rn_i - 1] };
        fused = resize_ac(&mut tp, &fused, f2, target.0, target.1);
        fused = conv(&mut tp, &fused, &format!("{pre}.out_conv.weight"), Some(&format!("{pre}.out_conv.bias")), &[f2, f2, 1, 1], 1, 1, 0);
    }

    // output_conv1 -> full-res align-corners resize -> +pos_full
    let oc1 = conv(&mut tp, &fused, &nm("scratch.output_conv1.weight"), Some(&nm("scratch.output_conv1.bias")), &[f2 / 2, f2, 3, 3], 3, 1, 1);
    let full = resize_ac(&mut tp, &oc1, f2 / 2, h, wdt);
    tp.g.init_f32(&format!("mir_{prefix}_pos_full"), &[1, f2 / 2, h, wdt], mirror::dpt::pos_embed_chw((f2 / 2) as usize, (h) as usize, wdt as usize, 0.1));
    let full = tp.binary("Add", &full, &format!("mir_{prefix}_pos_full"), "pos_full");

    // output_conv2: conv3 -> relu -> conv1 -> head_out
    let h32 = conv(&mut tp, &full, &nm("scratch.output_conv2.0.weight"), Some(&nm("scratch.output_conv2.0.bias")), &[f2 / 8, f2 / 2, 3, 3], 3, 1, 1);
    let h32r = tp.unary("Relu", &h32, "oc2_r");
    let wv = tp.host(&nm("scratch.output_conv2.2.weight")).clone();
    tp.g.init_f32(&nm("scratch.output_conv2.2.weight"), &[out_ch, f2 / 8, 1, 1], wv);
    let bv = tp.host(&nm("scratch.output_conv2.2.bias")).clone();
    tp.g.init_f32(&nm("scratch.output_conv2.2.bias"), &[out_ch], bv);
    tp.g.add(
        Node::new("Conv", &[&h32r, &nm("scratch.output_conv2.2.weight"), &nm("scratch.output_conv2.2.bias")], &["head_out"])
            .attr_ints("kernel_shape", &[1, 1])
            .attr_ints("strides", &[1, 1])
            .attr_ints("pads", &[0, 0, 0, 0]),
    );
    tp.g.output_f32("head_out", &[1, out_ch, h, wdt]);

    if gs {
        let fb = conv(&mut tp, "rgb", &nm("input_merger.0.weight"), Some(&nm("input_merger.0.bias")), &[f2 / 2, 3, 7, 7], 7, 1, 3);
        let fbr = tp.unary("Relu", &fb, "gs_r");
        let merged = tp.binary("Add", &full, &fbr, "gs_merge");
        let g0 = conv(&mut tp, &merged, "gs_renderer.gs_head.0.weight", None, &[f2, f2 / 2, 3, 3], 3, 1, 1);
        let g0r = tp.unary("Relu", &g0, "gs0_r");
        let wv = tp.host("gs_renderer.gs_head.2.weight").clone();
        tp.g.init_f32("gs_renderer.gs_head.2.weight", &[12, f2, 1, 1], wv);
        let bv = tp.host("gs_renderer.gs_head.2.bias").clone();
        tp.g.init_f32("gs_renderer.gs_head.2.bias", &[12], bv);
        tp.g.add(
            Node::new("Conv", &[&g0r, "gs_renderer.gs_head.2.weight", "gs_renderer.gs_head.2.bias"], &["gs_params"])
                .attr_ints("kernel_shape", &[1, 1])
                .attr_ints("strides", &[1, 1])
                .attr_ints("pads", &[0, 0, 0, 0]),
        );
        tp.g.output_f32("gs_params", &[1, 12, h, wdt]);
    }
}
