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
        let veps = self.binary("Add", &var, "mir_eps", "ln_veps");
        let std = self.unary("Sqrt", &veps, "ln_std");
        let xn = self.binary("Div", &xc, &std, "ln_xn");
        let g1 = self.init_vec(wname, &[C]);
        let sc = self.binary("Mul", &xn, &g1, "ln_g");
        let b1 = self.init_vec(bname, &[C]);
        self.binary("Add", &sc, &b1, "ln_out")
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
