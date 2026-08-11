// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build a GLM-5.2 (`glm_moe_dsa`) decoder as an ONNX graph (fixed sequence
//! length `T`) for whole-graph compilation on OpenVINO (NPU/GPU/CPU). Cache-free
//! prefill: `input_ids:[1,T]` (i64) -> `logits:[1,T,vocab]` (f32).
//!
//! Dense-expert MoE: every expert FFN is evaluated and gate-weighted (the same
//! dense formulation brain's `scale_add` uses), with the top-k gate built in-graph
//! via `TopK` + `ScatterElements`. MLA runs dense (the DSA indexer is a no-op at
//! `index_topk >= T`). Standard ONNX ops only — brain linear weights `[out,in]`
//! are transposed once to ONNX `[in,out]`. RoPE uses brain's **interleaved**
//! convention (base 10000, matching `rope_train`).

use onnx::builder::GraphBuilder;
use onnx::graph::Node;
use glm::config::GlmConfig;

use crate::qwen_topology::Quant;
use crate::topology::WeightSource;

/// Assemble the GLM decoder graph into `g` (fp32). `w` = checkpoint tensors (role "").
pub fn build_glm_graph(cfg: &GlmConfig, w: &dyn WeightSource, t: usize, g: &mut GraphBuilder) {
    build_glm_graph_quant(cfg, w, t, g, Quant::F32);
}

/// As [`build_glm_graph`] but with a weight quantization mode (`Int8` stores the
/// linear weights per-output-channel and dequantises in-graph — ~4x smaller).
pub fn build_glm_graph_quant(cfg: &GlmConfig, w: &dyn WeightSource, t: usize, g: &mut GraphBuilder, quant: Quant) {
    let d = cfg.d_model as usize;
    let vocab = cfg.vocab as usize;
    let ti = t as i64;
    let mut tp = Topo { b: crate::topo::TopoBase::new(g), quant };

    tp.g.input_i64("input_ids", &[1, ti]);
    tp.g.output_f32("logits", &[1, ti, vocab as i64]);

    tp.f32("tok.weight", &[vocab as i64, d as i64], w.get("tok.weight"));
    let x = tp.gather("tok.weight", "input_ids", 0, "emb"); // [1,T,d]

    let xf = build_stack(&mut tp, cfg, w, t, &x);
    let head = if cfg.tie_embeddings { "tok.weight" } else { "lm_head.weight" };
    tp.linear_named(&xf, head, "lm_head.wt", w, vocab, d, "logits");
}

fn build_stack(tp: &mut Topo, cfg: &GlmConfig, w: &dyn WeightSource, t: usize, x_in: &str) -> String {
    let d = cfg.d_model as usize;
    let nh = cfg.n_heads as usize;
    let nope = cfg.qk_nope_head_dim as usize;
    let rope = cfg.qk_rope_head_dim as usize;
    let vhd = cfg.v_head_dim as usize;
    let qkhd = nope + rope;
    let ql = cfg.q_lora_rank as usize;
    let kvl = cfg.kv_lora_rank as usize;
    let ndim = nh * nope; // all-heads nope width
    let qrdim = nh * rope;
    let vdim = nh * vhd;
    let e = cfg.n_routed_experts as usize;
    let moe_ff = cfg.moe_intermediate_size as usize;
    let dense_ff = cfg.intermediate_size as usize;
    let shared_ff = cfg.shared_ff() as usize;
    let ti = t as i64;

    // ---- shared constants ----
    tp.f32("c_eps", &[1], vec![cfg.rms_eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (qkhd as f32).sqrt()]);
    tp.f32("c_rscale", &[1], vec![cfg.routed_scaling_factor]);
    // interleaved-RoPE cos/sin half tables [1,T,1,rope/2] (base 10000, matches rope_train)
    let half = rope / 2;
    let (mut cos, mut sin) = (vec![0f32; t * half], vec![0f32; t * half]);
    for p in 0..t {
        for j in 0..half {
            let ang = p as f32 * 10000f32.powf(-(2.0 * j as f32) / rope as f32);
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, ti, 1, half as i64], cos);
    tp.f32("rope_sin", &[1, ti, 1, half as i64], sin);
    // causal mask [1,1,T,T]
    let mut mask = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask[i * t + j] = -1.0e9;
            }
        }
    }
    tp.f32("causal_mask", &[1, 1, ti, ti], mask);
    // interleave even/odd slice bounds (last axis, step 2)
    tp.i64("ev_lo", &[1], vec![0]);
    tp.i64("od_lo", &[1], vec![1]);
    tp.i64("ei_hi", &[1], vec![rope as i64]);
    tp.i64("ei_ax", &[1], vec![3]);
    tp.i64("ei_st", &[1], vec![2]);
    let _ = qkhd;
    // reshape shapes
    tp.i64("sh_qk_nope", &[4], vec![1, ti, nh as i64, nope as i64]);
    tp.i64("sh_qr", &[4], vec![1, ti, nh as i64, rope as i64]);
    tp.i64("sh_kr", &[4], vec![1, ti, 1, rope as i64]);
    tp.i64("sh_v", &[4], vec![1, ti, nh as i64, vhd as i64]);
    tp.i64("sh_ctx", &[3], vec![1, ti, vdim as i64]);
    // interleave merge shapes for both head counts used (nh for q, 1 for k)
    tp.i64(&format!("sh_ilv1_{nh}"), &[5], vec![1, ti, nh as i64, half as i64, 1]);
    tp.i64(&format!("sh_ilv2_{nh}"), &[4], vec![1, ti, nh as i64, rope as i64]);
    tp.i64("sh_ilv1_1", &[5], vec![1, ti, 1, half as i64, 1]);
    tp.i64("sh_ilv2_1", &[4], vec![1, ti, 1, rope as i64]);
    // zeros for the MoE gate scatter [1,T,E]
    tp.f32("moe_zeros", &[1, ti, e as i64], vec![0f32; t * e]);

    let mut x = x_in.to_string();
    for l in 0..cfg.n_layers as usize {
        let p = |s: &str| format!("blocks.{l}.{s}");
        // ---- MLA attention (dense) ----
        let h1 = tp.rmsnorm(&x, &p("input_ln.weight"), w, d);
        // Q: low-rank down -> norm -> nope / rope up
        let qc = tp.linear(&h1, &p("attn.q_a.weight"), w, ql, d);
        let qc = tp.rmsnorm(&qc, &p("attn.q_a_norm.weight"), w, ql);
        let q_pass = tp.linear(&qc, &p("attn.q_b_nope.weight"), w, ndim, ql); // [1,T,ndim]
        let q_rot = tp.linear(&qc, &p("attn.q_b_rope.weight"), w, qrdim, ql); // [1,T,qrdim]
        // KV: compressed latent (+ shared rope key) -> norm -> nope / v up
        let kv = tp.linear(&h1, &p("attn.kv_a_c.weight"), w, kvl, d);
        let kv = tp.rmsnorm(&kv, &p("attn.kv_a_norm.weight"), w, kvl);
        let k_pass = tp.linear(&kv, &p("attn.kv_b_nope.weight"), w, ndim, kvl);
        let v = tp.linear(&kv, &p("attn.kv_b_v.weight"), w, vdim, kvl);
        let k_rot = tp.linear(&h1, &p("attn.kv_a_rope.weight"), w, rope, d); // shared single-head [1,T,rope]

        // reshape to heads and rope the rope slices (interleaved)
        let q_rot4 = tp.reshape(&q_rot, "sh_qr"); // [1,T,nh,rope]
        let k_rot4 = tp.reshape(&k_rot, "sh_kr"); // [1,T,1,rope]
        let q_rot4 = tp.rope_interleave(&q_rot4, nh);
        let k_rot4 = tp.rope_interleave(&k_rot4, 1);
        let q_pass4 = tp.reshape(&q_pass, "sh_qk_nope"); // [1,T,nh,nope]
        let k_pass4 = tp.reshape(&k_pass, "sh_qk_nope");
        let v4 = tp.reshape(&v, "sh_v"); // [1,T,nh,vhd]
        // assemble per-head q,k = [nope | rope]; k_rot broadcasts over heads
        let k_rot_b = tp.expand(&k_rot4, "sh_qr"); // broadcast [1,T,1,rope] -> [1,T,nh,rope]
        let q_full = tp.concat2(&q_pass4, &q_rot4, 3); // [1,T,nh,qkhd]
        let k_full = tp.concat2(&k_pass4, &k_rot_b, 3);
        // to [1,heads,T,*]
        let q = tp.transpose(&q_full, &[0, 2, 1, 3]);
        let k = tp.transpose(&k_full, &[0, 2, 1, 3]);
        let vt = tp.transpose(&v4, &[0, 2, 1, 3]);
        let kt = tp.transpose(&k, &[0, 1, 3, 2]);
        let scores = tp.matmul(&q, &kt);
        let scores = tp.mul(&scores, "c_scale");
        let scores = tp.add(&scores, "causal_mask");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &vt); // [1,nh,T,vhd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
        let ctx = tp.reshape(&ctx, "sh_ctx"); // [1,T,vdim]
        let attn = tp.linear(&ctx, &p("attn.o.weight"), w, d, vdim);
        x = tp.add_t(&x, &attn);

        // ---- MLP: dense SwiGLU or MoE ----
        let h2 = tp.rmsnorm(&x, &p("post_ln.weight"), w, d);
        let mlp = if cfg.is_dense_layer(l as u32) {
            tp.swiglu(&h2, &p("mlp.gate.weight"), &p("mlp.up.weight"), &p("mlp.down.weight"), w, dense_ff, d)
        } else {
            tp.moe(&h2, l, cfg, w, e, moe_ff, shared_ff, d)
        };
        x = tp.add_t(&x, &mlp);
    }
    tp.rmsnorm(&x, "norm.weight", w, d)
}

/// ONNX graph assembly helper: unique temp names + node/initializer emission.
struct Topo<'a> {
    b: crate::topo::TopoBase<'a>,
    quant: Quant,
}

// DSL + shared math emitters live on `TopoBase` (crate::topo); Deref keeps this
// file's call sites unchanged. Only model-specific emission stays here.
impl<'a> std::ops::Deref for Topo<'a> {
    type Target = crate::topo::TopoBase<'a>;
    fn deref(&self) -> &Self::Target {
        &self.b
    }
}
impl<'a> std::ops::DerefMut for Topo<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.b
    }
}

impl<'a> Topo<'a> {
    fn expand(&mut self, x: &str, shape: &str) -> String {
        let o = self.tmp("exp");
        self.node("Expand", &[x, shape], &o);
        o
    }

    /// Interleaved RoPE on `[1,T,heads,rope]`: rotate each even/odd pair
    /// (x0,x1) -> (x0*cos - x1*sin, x1*cos + x0*sin), then re-interleave via
    /// reshape([..,half,1])+concat+reshape([..,rope]). Reuses the shared
    /// `rope_cos`/`rope_sin` half tables (broadcast over heads). `heads` selects
    /// the pre-declared merge shapes.
    fn rope_interleave(&mut self, x: &str, heads: usize) -> String {
        let sh1 = format!("sh_ilv1_{heads}");
        let sh2 = format!("sh_ilv2_{heads}");
        let ev = {
            let o = self.tmp("ev");
            self.g.add(Node::new("Slice", &[x, "ev_lo", "ei_hi", "ei_ax", "ei_st"], &[&o]));
            o
        };
        let od = {
            let o = self.tmp("od");
            self.g.add(Node::new("Slice", &[x, "od_lo", "ei_hi", "ei_ax", "ei_st"], &[&o]));
            o
        };
        let ec = self.mul(&ev, "rope_cos");
        let os = self.mul(&od, "rope_sin");
        let oe = {
            let o = self.tmp("oe");
            self.node("Sub", &[&ec, &os], &o); // even' = ev*cos - od*sin
            o
        };
        let oc = self.mul(&od, "rope_cos");
        let es = self.mul(&ev, "rope_sin");
        let oo = self.add_t(&oc, &es); // odd' = od*cos + ev*sin
        let oe_m = self.reshape(&oe, &sh1); // [..,half,1]
        let oo_m = self.reshape(&oo, &sh1);
        let merged = self.concat2(&oe_m, &oo_m, 4); // [..,half,2]
        self.reshape(&merged, &sh2) // [..,rope] interleaved
    }

    fn rmsnorm(&mut self, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
        let gain = format!("{name}.g");
        self.b.rmsnorm(x, &gain, w.get(name), dim, "c_eps")
    }

    fn linear(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        self.linear_named(x, name, &format!("{name}.wt"), w, out, inp, &o);
        o
    }
    fn linear_named(&mut self, x: &str, name: &str, winit: &str, w: &dyn WeightSource, out: usize, inp: usize, y: &str) {
        crate::topo::linear_quant(&mut self.b, x, name, winit, w, out, inp, self.quant, y);
    }

    /// SwiGLU MLP: down(silu(gate(x)) * up(x)).
    fn swiglu(&mut self, x: &str, gate: &str, up: &str, down: &str, w: &dyn WeightSource, ff: usize, d: usize) -> String {
        let g = self.linear(x, gate, w, ff, d);
        let u = self.linear(x, up, w, ff, d);
        let sig = self.unary("Sigmoid", &g);
        let silu = self.mul_t(&g, &sig);
        let hmul = self.mul_t(&silu, &u);
        self.linear(&hmul, down, w, d, ff)
    }

    /// Dense-expert MoE: sigmoid noaux_tc router (top-k gate via TopK +
    /// ScatterElements) + every expert FFN gate-weighted + shared expert.
    #[allow(clippy::too_many_arguments)]
    fn moe(&mut self, x: &str, l: usize, cfg: &GlmConfig, w: &dyn WeightSource, e: usize, moe_ff: usize, shared_ff: usize, d: usize) -> String {
        let p = |s: &str| format!("blocks.{l}.moe.{s}");
        let topk = cfg.num_experts_per_tok as i64;
        // router: s = sigmoid(x·Wr^T) ; choice = s + bias
        let logits = self.linear(x, &p("router.weight"), w, e, d); // [1,T,E]
        let s = self.unary("Sigmoid", &logits);
        self.f32(&format!("{}.b", p("router.bias")), &[1, 1, e as i64], w.get(&p("router.bias")));
        let choice = self.add(&s, &format!("{}.b", p("router.bias")));
        // TopK over experts -> indices [1,T,k]
        let (tv, tidx) = {
            let vals = self.tmp("topk_v");
            let idx = self.tmp("topk_i");
            self.i64("topk_k", &[1], vec![topk]);
            self.g.add(Node::new("TopK", &[&choice, "topk_k"], &[&vals, &idx]).attr_int("axis", 2).attr_int("largest", 1).attr_int("sorted", 0));
            (vals, idx)
        };
        let _ = tv;
        // gathered raw sigmoid scores at the selected experts, renormalize, scale
        let gs = {
            let o = self.tmp("ge");
            self.g.add(Node::new("GatherElements", &[&s, &tidx], &[&o]).attr_int("axis", 2));
            o
        };
        let denom = {
            let o = self.tmp("rsum");
            self.g.add(Node::new("ReduceSum", &[&gs], &[&o]).attr_ints("axes", &[2]).attr_int("keepdims", 1));
            o
        };
        let wnorm = {
            let o = self.tmp("wn");
            self.node("Div", &[&gs, &denom], &o);
            o
        };
        let wscaled = self.mul(&wnorm, "c_rscale"); // [1,T,k]
        // scatter the weights back to a dense [1,T,E] gate (zeros elsewhere)
        let gate = {
            let o = self.tmp("gate");
            self.g.add(Node::new("ScatterElements", &["moe_zeros", &tidx, &wscaled], &[&o]).attr_int("axis", 2));
            o
        };
        // dense expert eval, gate-weighted accumulate
        let mut acc: Option<String> = None;
        self.i64("ge_ax", &[1], vec![2]);
        for ei in 0..e {
            let ep = |s: &str| format!("blocks.{l}.moe.experts.{ei}.{s}");
            let out = self.swiglu(x, &ep("gate.weight"), &ep("up.weight"), &ep("down.weight"), w, moe_ff, d);
            // gate column e: Slice gate[:,:,ei:ei+1] -> [1,T,1], broadcast-mul
            self.i64(&format!("gsl_lo{ei}"), &[1], vec![ei as i64]);
            self.i64(&format!("gsl_hi{ei}"), &[1], vec![(ei + 1) as i64]);
            let gcol = {
                let o = self.tmp("gcol");
                self.g.add(Node::new("Slice", &[&gate, &format!("gsl_lo{ei}"), &format!("gsl_hi{ei}"), "ge_ax"], &[&o]));
                o
            };
            let scaled = self.mul_t(&out, &gcol);
            acc = Some(match acc {
                None => scaled,
                Some(a) => self.add_t(&a, &scaled),
            });
        }
        // shared expert (always on)
        let sh = self.swiglu(x, &p("shared.gate.weight"), &p("shared.up.weight"), &p("shared.down.weight"), w, shared_ff, d);
        self.add_t(&acc.unwrap(), &sh)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Widening `w` from a hardcoded `HashMap` alias to `&dyn WeightSource` must
    /// not change the emitted graph: build once from an eager in-memory HashMap
    /// (the old call shape), once from a streaming `WeightReader` opened on the
    /// same checkpoint written to disk, and diff the raw ONNX bytes.
    #[test]
    fn streaming_weight_source_matches_eager_hashmap() {
        let cfg = GlmConfig::tiny();
        let block = cfg.block_size;
        let init = glm::init_weights(&cfg, 11);
        let model = glm::Glm::new(cfg.clone(), 1, block, &init);
        let dir = std::env::temp_dir().join(format!("brain_glm_topo_parity_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        let path = path.to_str().unwrap();
        model.save(path);

        let w_eager: HashMap<String, Vec<f32>> = checkpoint::load(path).by_role("");
        let reader = checkpoint::weightio::WeightReader::open(path).unwrap();

        let t = 4usize;
        let mut g1 = GraphBuilder::new("glm_decoder");
        build_glm_graph(&cfg, &w_eager, t, &mut g1);
        let mut g2 = GraphBuilder::new("glm_decoder");
        build_glm_graph(&cfg, &reader, t, &mut g2);

        assert_eq!(g1.finish(), g2.finish(), "HashMap vs WeightReader graphs must be byte-identical");
        std::fs::remove_dir_all(&dir).ok();
    }
}
