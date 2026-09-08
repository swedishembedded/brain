// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build `deepseek2::DeepseekV2` (the DeepSeek-OCR / DeepSeek-OCR-2 decoder -
//! plain MHA, sparse top-k MoE with an unweighted fused shared expert) as a
//! fixed-sequence-length ONNX graph, for a best-effort OpenVINO/NPU compile
//! attempt. Pure Rust - no NPU needed to produce the file.
//!
//! **Input is `inputs_embeds:[1,T,d]`, not `input_ids`.** This decoder is
//! never run text-only in production: a real request already has the
//! composite's own row-gather splice its vision rows into the embedding
//! stream before the decoder ever runs (`model::vlm::splice_fwd`). Exporting
//! a token-embedding Gather here would ask the graph to re-derive a step it
//! never actually performs standalone, and would silently drop the spliced
//! rows in anything that used it. The caller (host code) is responsible for
//! assembling `inputs_embeds` exactly as the real composite does.
//!
//! **MoE routing** follows the same sparse gather-and-weight scheme
//! `qwen35moe_topology.rs::Topo::moe_layer` (itself built on
//! `glm_topology.rs::Topo::moe`) already established for this repo's other
//! MoE architectures: every expert's weight is stacked once into a
//! `Gather`-indexable `[E,in,out]` initializer, `TopK` picks the winning
//! `top_k` experts per token from the (here: plain softmax, no
//! renormalisation) router probabilities, and a single broadcasting `MatMul`
//! runs only the selected experts. Two real differences from that precedent,
//! both read from `cfg` rather than assumed, so a future caller with a
//! differently-configured `DeepseekV2Config` cannot silently export the
//! wrong router:
//! - `cfg.norm_topk_prob`/`cfg.routed_scaling` gate the renormalising
//!   `Div`/scaling `Mul` this file emits - the real DeepSeek-OCR-2 checkpoint
//!   carries `norm_topk_prob=false, routed_scaling=1.0` (`DeepseekV2Config`'s
//!   own default, matching llama.cpp's compiled-in behaviour for this arch
//!   when the GGUF carries neither key), so neither op is emitted for it,
//!   but a differently configured decoder gets the op it actually needs.
//! - the shared expert here is **unweighted** (`model::moe::
//!   shared_expert_fwd`'s `None` gate arm: one dense SwiGLU pass, added
//!   directly to the routed sum) rather than qwen35moe's sigmoid-gated one -
//!   simpler, one fewer linear and no `Sigmoid`.
//!
//! Standard ONNX ops only (Gather/MatMul/Mul/Add/Softmax/TopK/Sigmoid-free
//! here/ReduceMean/Sqrt/Div/Reshape/Transpose/Slice/Neg/Concat), targeting
//! `onnx::DEFAULT_OPSET` (13).

use deepseek2::config::DeepseekV2Config;
use onnx::builder::GraphBuilder;
use onnx::graph::Node;

use crate::topo::TopoBase;
use crate::topology::WeightSource;

/// Assemble the decoder graph into `g`. `w` is the real (or tiny) checkpoint's
/// flat tensors (`blocks.{l}.*`/`norm.weight`/`lm_head.weight` naming -
/// `deepseek2::model`'s own doc), `t` the fixed sequence length.
pub fn build_deepseek2_graph(cfg: &DeepseekV2Config, w: &dyn WeightSource, t: usize, g: &mut GraphBuilder) {
    let mut tp = TopoBase::new(g);
    let d = cfg.d_model() as usize;
    let nh = cfg.n_heads() as usize;
    let hd = cfg.head_dim() as usize;
    let half = hd / 2;
    let vocab = cfg.vocab() as usize;
    let ti = t as i64;
    let theta = cfg.rope_theta();
    let eps = cfg.rms_eps();

    tp.g.input_f32("inputs_embeds", &[1, ti, d as i64]);
    tp.g.output_f32("logits", &[1, ti, vocab as i64]);

    // ---- shared constants ----
    tp.f32("c_eps", &[1], vec![eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    let (mut cos, mut sin) = (vec![0f32; t * hd], vec![0f32; t * hd]);
    for p in 0..t {
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = p as f32 * theta.powf(-2.0 * m / hd as f32);
            cos[p * hd + j] = ang.cos();
            sin[p * hd + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, ti, 1, hd as i64], cos);
    tp.f32("rope_sin", &[1, ti, 1, hd as i64], sin);
    let mut mask = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask[i * t + j] = -1.0e9;
        }
    }
    tp.f32("causal_mask", &[1, 1, ti, ti], mask);
    tp.i64("rh_ax", &[1], vec![3]);
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    tp.i64("sh_heads", &[4], vec![1, ti, nh as i64, hd as i64]);
    tp.i64("sh_ctx", &[3], vec![1, ti, (nh * hd) as i64]);

    let mut x = "inputs_embeds".to_string();
    for l in 0..cfg.n_layers() as usize {
        let p = |leaf: &str| format!("blocks.{l}.{leaf}");

        // ---- plain MHA: n_kv_heads == n_heads, so no head-repeat needed ----
        let xn1 = rmsnorm(&mut tp, &x, &p("ln1.weight"), w, d);
        let q = linear(&mut tp, &xn1, &p("self_attn.q_proj.weight"), w, d, d);
        let k = linear(&mut tp, &xn1, &p("self_attn.k_proj.weight"), w, d, d);
        let v = linear(&mut tp, &xn1, &p("self_attn.v_proj.weight"), w, d, d);
        let q = tp.reshape(&q, "sh_heads");
        let k = tp.reshape(&k, "sh_heads");
        let v = tp.reshape(&v, "sh_heads");
        let q = rope_half_split(&mut tp, &q);
        let k = rope_half_split(&mut tp, &k);
        let q = tp.transpose(&q, &[0, 2, 1, 3]); // [1,nh,T,hd]
        let k = tp.transpose(&k, &[0, 2, 1, 3]);
        let v = tp.transpose(&v, &[0, 2, 1, 3]);
        let kt = tp.transpose(&k, &[0, 1, 3, 2]); // [1,nh,hd,T]
        let scores = tp.matmul(&q, &kt);
        let scores = tp.mul(&scores, "c_scale");
        let scores = tp.add_t(&scores, "causal_mask");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &v); // [1,nh,T,hd]
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]); // [1,T,nh,hd]
        let ctx = tp.reshape(&ctx, "sh_ctx");
        let attn_out = linear(&mut tp, &ctx, &p("self_attn.o_proj.weight"), w, d, d);
        x = tp.add_t(&x, &attn_out);

        // ---- MLP: dense (leading blocks) or sparse MoE ----
        let xn2 = rmsnorm(&mut tp, &x, &p("ln2.weight"), w, d);
        let mlp_out = if cfg.is_moe_layer(l as u32) {
            moe_layer(&mut tp, l, &xn2, w, cfg, t)
        } else {
            let ff = cfg.ffn_hidden() as usize;
            let gate = linear(&mut tp, &xn2, &p("mlp.gate.weight"), w, ff, d);
            let up = linear(&mut tp, &xn2, &p("mlp.up.weight"), w, ff, d);
            let h = swiglu(&mut tp, &gate, &up);
            linear(&mut tp, &h, &p("mlp.down.weight"), w, d, ff)
        };
        x = tp.add_t(&x, &mlp_out);
    }

    let xf = rmsnorm(&mut tp, &x, "norm.weight", w, d);
    linear_to(&mut tp, &xf, cfg.head_weight(), w, vocab, d, "logits");
}

fn rmsnorm(tp: &mut TopoBase, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
    let gain = format!("{name}.g");
    tp.rmsnorm(x, &gain, w.get(name), dim, "c_eps")
}

fn linear(tp: &mut TopoBase, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
    let y = tp.tmp("lin");
    linear_to(tp, x, name, w, out, inp, &y);
    y
}

/// `y = x . Wᵀ` (brain stores `[out,in]`; ONNX `MatMul` wants `[in,out]`, so
/// the weight is transposed once into a fresh initializer per distinct name).
fn linear_to(tp: &mut TopoBase, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize, y: &str) {
    let winit = format!("{name}.wt");
    if !tp.has(&winit) {
        let raw = w.get(name);
        let mut t = vec![0f32; raw.len()];
        for r in 0..out {
            for c in 0..inp {
                t[c * out + r] = raw[r * inp + c];
            }
        }
        tp.f32(&winit, &[inp as i64, out as i64], t);
    }
    tp.node("MatMul", &[x, &winit], y);
}

/// RoPE, half-split (NeoX) convention, over `[1,T,heads,hd]`:
/// `x*cos + rotate_half(x)*sin`, `rotate_half = concat(-x[..,half:], x[..,:half])`.
fn rope_half_split(tp: &mut TopoBase, x: &str) -> String {
    let x_hi = tp.slice(x, "rh_lo1", "rh_hi1", "rh_ax");
    let x_lo = tp.slice(x, "rh_lo0", "rh_hi0", "rh_ax");
    let neg_hi = tp.unary("Neg", &x_hi);
    let rot = tp.concat2(&neg_hi, &x_lo, 3);
    let a = tp.mul(x, "rope_cos");
    let b = tp.mul(&rot, "rope_sin");
    tp.add_t(&a, &b)
}

fn swiglu(tp: &mut TopoBase, gate: &str, up: &str) -> String {
    let act = tp.silu(gate);
    tp.mul_t(&act, up)
}

/// Sparse top-k MoE + an unweighted, always-active fused shared expert.
/// Returns the combined `[1,T,d]` MLP output (routed sum + shared output).
fn moe_layer(tp: &mut TopoBase, l: usize, xn2: &str, w: &dyn WeightSource, cfg: &DeepseekV2Config, t: usize) -> String {
    let d = cfg.d_model() as usize;
    let e = cfg.n_experts() as usize;
    let ff = cfg.moe_ff() as usize;
    let sff = cfg.shared_ff() as usize;
    let k = cfg.top_k() as usize;
    let ti = t as i64;
    let p = |leaf: &str| format!("blocks.{l}.{leaf}");

    let router_w = p("mlp.router.weight");
    let logits = linear(tp, xn2, &router_w, w, e, d);
    let probs = tp.softmax(&logits, -1);

    let k_name = format!("moe_k_{k}");
    tp.i64(&k_name, &[1], vec![k as i64]);
    let vals = tp.tmp("moe_tkv");
    let idx = tp.tmp("moe_tki");
    tp.g.add(Node::new("TopK", &[&probs, &k_name], &[&vals, &idx]).attr_int("axis", -1).attr_int("largest", 1).attr_int("sorted", 1));

    // The real checkpoint's router policy (M0/M6): no renormalisation, unit
    // scale. Both knobs are read from `cfg`, not hardcoded, so a decoder
    // built with a different policy exports the router it actually runs.
    let mut gate_w = vals;
    if cfg.norm_topk_prob {
        let denom = reduce_sum(tp, &gate_w, -1, true);
        let g = tp.tmp("moe_gwn");
        tp.node("Div", &[&gate_w, &denom], &g);
        gate_w = g;
    }
    if cfg.routed_scaling != 1.0 {
        tp.f32("moe_rs", &[1], vec![cfg.routed_scaling]);
        gate_w = tp.mul(&gate_w, "moe_rs");
    }

    let gate_stack = expert_stack(tp, &format!("moe_gs_{l}"), w, e, ff, d, |ei| format!("blocks.{l}.mlp.experts.{ei}.gate.weight"));
    let up_stack = expert_stack(tp, &format!("moe_us_{l}"), w, e, ff, d, |ei| format!("blocks.{l}.mlp.experts.{ei}.up.weight"));
    let down_stack = expert_stack(tp, &format!("moe_ds_{l}"), w, e, d, ff, |ei| format!("blocks.{l}.mlp.experts.{ei}.down.weight"));

    let gk = tp.gather(&gate_stack, &idx, 0, "moe_gk"); // [1,T,k,d,ff]
    let gu = tp.gather(&up_stack, &idx, 0, "moe_gu");
    let gd = tp.gather(&down_stack, &idx, 0, "moe_gd"); // [1,T,k,ff,d]

    let sh_x5 = format!("moe_sh_x5_{l}");
    tp.i64(&sh_x5, &[5], vec![1, ti, 1, 1, d as i64]);
    let x5 = tp.reshape(xn2, &sh_x5);
    let gate_pre = tp.matmul(&x5, &gk); // [1,T,k,1,ff]
    let up = tp.matmul(&x5, &gu);
    let h = swiglu(tp, &gate_pre, &up);
    let expert_out5 = tp.matmul(&h, &gd); // [1,T,k,1,d]

    let sh_out4 = format!("moe_sh_out4_{l}");
    tp.i64(&sh_out4, &[4], vec![1, ti, k as i64, d as i64]);
    let expert_out4 = tp.reshape(&expert_out5, &sh_out4);
    let sh_gw4 = format!("moe_sh_gw4_{l}");
    tp.i64(&sh_gw4, &[4], vec![1, ti, k as i64, 1]);
    let gate_w4 = tp.reshape(&gate_w, &sh_gw4);
    let weighted = tp.mul_t(&expert_out4, &gate_w4);
    let routed_sum = reduce_sum(tp, &weighted, 2, false);

    // Unweighted fused shared expert (`model::moe::shared_expert_fwd`'s
    // `None` gate arm): one dense SwiGLU pass, added directly.
    let sh_gate = linear(tp, xn2, &p("mlp.shared.gate.weight"), w, sff, d);
    let sh_up = linear(tp, xn2, &p("mlp.shared.up.weight"), w, sff, d);
    let sh_h = swiglu(tp, &sh_gate, &sh_up);
    let sh_out = linear(tp, &sh_h, &p("mlp.shared.down.weight"), w, d, sff);

    tp.add_t(&routed_sum, &sh_out)
}

/// `ReduceSum(x, axis)`, opset-13-correct: `axes` is an INPUT tensor, not an
/// attribute (that form arrives only at opset 18, past this builder's
/// target). Each caller of this shape keeps its own copy for the same reason
/// `qwen35moe_topology.rs`/`fincast_topology.rs` do.
fn reduce_sum(tp: &mut TopoBase, x: &str, axis: i64, keepdims: bool) -> String {
    let ax = format!("c_i64_{axis}");
    if !tp.has(&ax) {
        tp.i64(&ax, &[1], vec![axis]);
    }
    let o = tp.tmp("rsum");
    tp.g.add(Node::new("ReduceSum", &[x, &ax], &[&o]).attr_int("keepdims", keepdims as i64));
    o
}

/// Stack every expert's `[out,in]` weight into one `[E,in,out]`
/// `Gather`-indexable initializer, transposed once like every other linear
/// weight this file exports. Registered once per `name` (layer-scoped).
fn expert_stack(tp: &mut TopoBase, name: &str, w: &dyn WeightSource, e: usize, out: usize, inp: usize, namer: impl Fn(usize) -> String) -> String {
    if !tp.has(name) {
        let mut data = Vec::with_capacity(e * out * inp);
        for ei in 0..e {
            let raw = w.get(&namer(ei));
            for c in 0..inp {
                for r in 0..out {
                    data.push(raw[r * inp + c]);
                }
            }
        }
        tp.f32(name, &[e as i64, inp as i64, out as i64], data);
    }
    name.to_string()
}
