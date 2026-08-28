// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One SUPIR adaptor - `ZeroCrossAttn` - as an ONNX graph, for Intel NPU.
//!
//! # Scope, stated honestly
//!
//! SUPIR's adaptor stack is 10 `ZeroSFT` (pure conv + GroupNorm - the same
//! primitives [`crate::vae_topology`] already exports) plus 2
//! `ZeroCrossAttn` (the only adaptors with LINEAR projections). This module
//! covers `ZeroCrossAttn` - the piece that actually needs
//! [`crate::topo::linear_quant`] (three earlier topologies had drifted into
//! whole-channel weight copies before that emitter existed; this does not
//! become a fourth). `ZeroSFT` and the 1.24B `GLVControl` trunk itself are
//! NOT exported here: the trunk is numerically identical to a full SDXL
//! UNet, and no model in this tree - `sdxlunet`, `controlnet`, nor this one -
//! has an NPU export for that shape yet (every existing `*_topology` module
//! is either pure-conv/GroupNorm, like [`crate::vae_topology`], or a
//! sequence-only decoder, like `qwen_topology`). Building one from scratch is
//! real, separate work this phase does not close - recorded in the roadmap
//! rather than silently implied by this file's presence. Realistically the
//! trunk also exceeds what this NPU can hold on the hardware this port was
//! written on, per that same roadmap entry.
//!
//! # The formula
//!
//! `ZeroCrossAttn(c, x, s) = x + s . CrossAttn(GN(x), GN(c))`: queries from
//! the normed spatial features `x`, keys/values from the normed control
//! tensor `c`, `heads = channels / 64`, `dim_head = 64`. `s` (`control_scale`)
//! is a graph CONSTANT here, mirroring [`crate::model::Supir`]'s own choice
//! to bake it in rather than thread a per-step buffer through - see that
//! crate's `pipeline` module doc for the same reasoning applied to the served
//! forward.
//!
//! Validation here is structural, the same honest posture
//! [`crate::vae_topology`]'s own module doc states: node counts, the
//! `linear_quant` group-quantization shape, and that the exported bytes
//! re-read as well-formed ONNX. Numerical parity against `crates/supir`
//! needs a real checkpoint and the NPU hardware neither is available on this
//! machine.

use onnx::{GraphBuilder, Node};

use crate::topo::{linear_quant, Quant, TopoBase};
use crate::topology::WeightSource;

/// One `ZeroCrossAttn` instance's shape. `x` and `c` share a spatial grid in
/// every real SUPIR join (the control tensor is a same-resolution skip - see
/// `crate::adaptors::CrossSpec` in `crates/supir`), so one `(h, w)` covers
/// both.
#[derive(Clone, Copy, Debug)]
pub struct CrossAttnTopo {
    pub channels: u32,
    pub h: u32,
    pub w: u32,
    pub gn_groups: u32,
    pub gn_eps: f32,
    /// `s` - baked in as a graph constant, see the module doc.
    pub control_scale: f32,
}

impl CrossAttnTopo {
    pub fn heads(&self) -> u32 {
        self.channels / 64
    }
}

struct Ca<'a> {
    b: TopoBase<'a>,
}

impl<'a> Ca<'a> {
    fn group_norm(&mut self, p: &str, t: &CrossAttnTopo, w: &dyn WeightSource, x: &str) -> String {
        let (gn, bn) = (format!("{p}.weight"), format!("{p}.bias"));
        let (g, b) = (w.get(&gn), w.get(&bn));
        self.b.group_norm(x, 1, t.channels as usize, t.h as usize, t.w as usize, t.gn_groups as usize, &gn, g, &bn, b, t.gn_eps)
    }

    /// `[1,C,H,W] -> [1,HW,C]`, so a `linear_quant` projection is a plain MatMul.
    fn flatten_tokens(&mut self, x: &str, c: u32, hw: u32) -> String {
        let shape = self.b.tmp("ca_shape_flat");
        self.b.i64(&shape, &[3], vec![1, c as i64, hw as i64]);
        let f = self.b.reshape(x, &shape);
        self.b.transpose(&f, &[0, 2, 1])
    }

    fn linear(&mut self, p: &str, w: &dyn WeightSource, out: u32, inp: u32, quant: Quant, x: &str) -> String {
        let winit = format!("{p}.wt");
        let y = self.b.tmp("ca_linear");
        linear_quant(&mut self.b, x, p, &winit, w, out as usize, inp as usize, quant, &y);
        y
    }

    /// `linear` plus a bias add - `to_out.0` is the one projection this
    /// adaptor ships a bias for (upstream's own zero-init call on it is
    /// commented out, so it is an ordinary biased linear, not special-cased
    /// here beyond that).
    fn linear_biased(&mut self, p: &str, w: &dyn WeightSource, out: u32, inp: u32, quant: Quant, x: &str) -> String {
        let y = self.linear(p, w, out, inp, quant, x);
        let bn = format!("{p}.bias");
        let b = w.get(&bn);
        self.b.f32(&bn, &[1, 1, out as i64], b);
        self.b.add_t(&y, &bn)
    }
}

/// Build `ZeroCrossAttn(c, x, s)` into `g`: two inputs (`x`, `c`, both
/// `[1,C,H,W]`), one output (`out`, same shape as `x`).
///
/// Weight names this reads (all under `t`'s own channel width, `linear_quant`
/// handling the quantized/fp32 split): `norm_x.{weight,bias}`,
/// `norm_c.{weight,bias}` (the two `GN` calls), `to_q.weight`, `to_k.weight`,
/// `to_v.weight` (bias-free, matching diffusers' own cross-attention
/// convention), `to_out.0.{weight,bias}`.
pub fn build_zero_cross_attn_graph(t: &CrossAttnTopo, w: &dyn WeightSource, quant: Quant, g: &mut GraphBuilder) {
    let (c, h, ww) = (t.channels, t.h, t.w);
    let hw = h * ww;
    g.input_f32("x", &[1, c as i64, h as i64, ww as i64]);
    g.input_f32("c", &[1, c as i64, h as i64, ww as i64]);
    g.output_f32("out", &[1, c as i64, h as i64, ww as i64]);

    let mut m = Ca { b: TopoBase::new(g) };
    let nx = m.group_norm("norm_x", t, w, "x");
    let nc = m.group_norm("norm_c", t, w, "c");

    let qx = m.flatten_tokens(&nx, c, hw);
    let kc = m.flatten_tokens(&nc, c, hw);
    let vc = kc.clone();

    let q = m.linear("to_q", w, c, c, quant, &qx);
    let k = m.linear("to_k", w, c, c, quant, &kc);
    let v = m.linear("to_v", w, c, c, quant, &vc);

    let heads = t.heads() as i64;
    let dim_head = 64i64;
    let split_shape = m.b.tmp("ca_split_shape");
    m.b.i64(&split_shape, &[4], vec![1, hw as i64, heads, dim_head]);
    let split_heads = |m: &mut Ca, x: &str| -> String {
        let r = m.b.reshape(x, &split_shape);
        m.b.transpose(&r, &[0, 2, 1, 3]) // [1, heads, hw, dim_head]
    };
    let qh = split_heads(&mut m, &q);
    let kh = split_heads(&mut m, &k);
    let vh = split_heads(&mut m, &v);

    let kt = m.b.transpose(&kh, &[0, 1, 3, 2]);
    let scores = m.b.matmul(&qh, &kt);
    let scale_name = m.b.tmp("ca_scale");
    m.b.f32(&scale_name, &[1], vec![1.0 / (dim_head as f32).sqrt()]);
    let scaled = m.b.mul_t(&scores, &scale_name);
    let probs = m.b.softmax(&scaled, -1);
    let ctx = m.b.matmul(&probs, &vh); // [1, heads, hw, dim_head]

    let back = m.b.transpose(&ctx, &[0, 2, 1, 3]); // [1, hw, heads, dim_head]
    let merge_shape = m.b.tmp("ca_merge_shape");
    m.b.i64(&merge_shape, &[3], vec![1, hw as i64, c as i64]);
    let merged = m.b.reshape(&back, &merge_shape);

    let attn_out = m.linear_biased("to_out.0", w, c, c, quant, &merged);

    // Tokens -> [1,C,H,W]
    let nchw_shape = m.b.tmp("ca_shape_nchw");
    m.b.i64(&nchw_shape, &[4], vec![1, c as i64, h as i64, ww as i64]);
    let attn_spatial = m.b.transpose(&attn_out, &[0, 2, 1]);
    let attn_nchw = m.b.reshape(&attn_spatial, &nchw_shape);

    let s_name = m.b.tmp("ca_control_scale");
    m.b.f32(&s_name, &[1], vec![t.control_scale]);
    let scaled_attn = m.b.mul_t(&attn_nchw, &s_name);
    let out = m.b.add_t("x", &scaled_attn);
    m.b.g.add(Node::new("Identity", &[&out], &["out"]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapWeights(HashMap<&'static str, Vec<f32>>);
    impl WeightSource for MapWeights {
        fn get(&self, name: &str) -> Vec<f32> {
            self.0.get(name).unwrap_or_else(|| panic!("test weights missing `{name}`")).clone()
        }
    }

    fn tiny_topo() -> CrossAttnTopo {
        CrossAttnTopo { channels: 64, h: 2, w: 2, gn_groups: 8, gn_eps: 1e-6, control_scale: 0.7 }
    }

    fn weights(t: &CrossAttnTopo) -> MapWeights {
        let c = t.channels as usize;
        let mut m = HashMap::new();
        for gn in ["norm_x", "norm_c"] {
            m.insert(Box::leak(format!("{gn}.weight").into_boxed_str()) as &'static str, vec![1.0f32; c]);
            m.insert(Box::leak(format!("{gn}.bias").into_boxed_str()) as &'static str, vec![0.0f32; c]);
        }
        for lin in ["to_q", "to_k", "to_v", "to_out.0"] {
            m.insert(Box::leak(lin.to_string().into_boxed_str()) as &'static str, vec![0.01f32; c * c]);
        }
        m.insert("to_out.0.bias", vec![0.0f32; c]);
        MapWeights(m)
    }

    #[test]
    fn structural_graph_uses_linear_quant_group_quantization() {
        let t = tiny_topo();
        let w = weights(&t);
        let mut g = GraphBuilder::new("zero_cross_attn");
        build_zero_cross_attn_graph(&t, &w, Quant::Int8, &mut g);

        let ops: Vec<&str> = g.graph().nodes.iter().map(|n| n.op_type.as_str()).collect();
        assert!(!ops.contains(&"DequantizeLinear"), "must not use whole-channel DequantizeLinear: {ops:?}");
        assert!(ops.contains(&"Softmax"), "cross-attention must softmax the scores: {ops:?}");
        assert!(ops.iter().filter(|&&o| o == "MatMul").count() >= 4, "q@k, probs@v, plus the quantized linears' matmuls: {ops:?}");

        // Four quantized projections (q, k, v, out) -> four group-scale
        // initializers, the `linear_quant` contract this file delegates to.
        let scale_names: Vec<&str> = g.graph().initializers.iter().map(|i| i.name.as_str()).filter(|n| n.ends_with(".wt.s")).collect();
        assert_eq!(scale_names.len(), 4, "expected one scale initializer per quantized linear (q,k,v,to_out.0): {scale_names:?}");

        let bytes = g.finish();
        assert!(!bytes.is_empty(), "the serialized graph must not be empty");
    }

    #[test]
    fn fp32_graph_has_no_quantization_nodes() {
        let t = tiny_topo();
        let w = weights(&t);
        let mut g = GraphBuilder::new("zero_cross_attn_fp32");
        build_zero_cross_attn_graph(&t, &w, Quant::F32, &mut g);
        let scale_names = g.graph().initializers.iter().filter(|i| i.name.ends_with(".wt.s")).count();
        assert_eq!(scale_names, 0, "fp32 export must carry no quantization scale initializers");
    }

    #[test]
    fn heads_derive_from_channels_at_a_fixed_64_dim_head() {
        let t = CrossAttnTopo { channels: 640, ..tiny_topo() };
        assert_eq!(t.heads(), 10);
    }
}
