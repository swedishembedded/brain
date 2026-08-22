// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build CosyVoice's speech-token LM backbone (the `qwen3::Qwen`-hosted
//! Qwen2.5-0.5B decoder both CosyVoice 2 and CosyVoice 3 share, per
//! `cosyvoice::config::CosyVoiceLmConfig`) as an **input-embedding-driven**
//! ONNX graph: `inputs_embeds:[1,T,d] -> hidden:[1,T,d]` (full causal
//! attention over `T`, no KV cache - the prefill shape). Mirrors
//! `crate::qwen_topology::build_talker_hidden_graph`'s exact contract and
//! purpose (feed an externally-assembled embedding stream in, read the
//! post-final-norm hidden state back, leave the model's own bolted-on
//! vocabulary/head off the graph) for the identical reason CosyVoice needs
//! it: `sos ++ text_emb ++ task_id ++ prompt_speech_emb` is assembled from
//! THREE disjoint embedding tables (the Qwen backbone's own `tok.weight` for
//! text, plus CosyVoice's own `llm_embedding`/`speech_embedding`), which does
//! not fit a plain `Gather(tok.weight, input_ids)` front end - so, exactly as
//! `qwen3tts`'s Talker already does for its own bolted-on codec vocabulary,
//! that assembly and the head/sampling stay host-side and only the decoder
//! BODY goes on the graph.
//!
//! **Why this is a NEW file rather than a call into `crate::qwen_topology`,
//! a deliberate judgment call**: `qwen_topology::build_stack` hardcodes
//! Qwen3's attention shape - **QK-norm always applied**, q/k/v/o **always
//! bias-free** - because every existing caller (the Qwen3-TTS Talker/MTP, the
//! Qwen3 decoder itself) is genuinely Qwen3-shaped. CosyVoice's backbone is
//! Qwen2.5, confirmed from the real checkpoint and asserted in
//! `cosyvoice::config`'s own tests: `qk_norm = false` (no `q_norm`/`k_norm`
//! weights exist in `llm.pt` at all) and `attn_bias = true` (q/k/v carry a
//! bias; `o_proj` and the MLP do not, matching `llm_import::backbone_name`'s
//! own tensor-name mapping). Reusing `build_stack` unmodified would silently
//! apply a QK-norm this checkpoint has no weights for and drop the q/k/v bias
//! terms entirely - a WRONG graph that would still build and compile. Making
//! `build_stack` generic over `qk_norm`/`attn_bias` is the structurally
//! cleaner fix, but it is a shared function every existing Qwen3-family NPU
//! export already depends on (the Talker, the MTP, `qwen3`'s own decoder) -
//! retrofitting it safely needs re-verifying every one of those unchanged,
//! which is real, separate work from this milestone. This file duplicates the
//! attention/MLP block shape rather than risk that regression under this
//! milestone's own time budget; hoisting the two into one generic
//! implementation is a recorded, honest follow-up, not attempted here.
//!
//! Weight-only INT8/INT4 quantization (`crate::topo::Quant`) is supported via
//! the SAME shared emitter every other Qwen-family export uses
//! (`crate::topo::linear_quant`) - nothing about quantization is
//! architecture-specific, only the block wiring above it is.

use cosyvoice::config::CosyVoiceLmConfig;
use onnx::builder::GraphBuilder;
use onnx::graph::Node;

pub use crate::topo::Quant;
use crate::topo::{linear_quant, TopoBase};
use crate::topology::WeightSource;

// `cosyvoice::llm_import::LmWeights::backbone` is already keyed exactly by
// `qwen3::QwenConfig::param_list()`'s names (`tok.weight`, `norm.weight`,
// `blocks.{l}.attn.wq.weight`, ...) - the SAME convention every `WeightSource`
// reader in this crate uses - so it needs no name-remapping layer, only the
// blanket `impl WeightSource for HashMap<String, Vec<f32>>` `depth_topology`
// already registers (one implementation, not a second copy here).

struct Topo<'a> {
    b: TopoBase<'a>,
    quant: Quant,
}

impl<'a> std::ops::Deref for Topo<'a> {
    type Target = TopoBase<'a>;
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
    fn rmsnorm(&mut self, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
        let gain = format!("{name}.g");
        self.b.rmsnorm(x, &gain, w.get(name), dim, "c_eps")
    }

    /// RoPE (half-split) on `[1,T,heads,hd]`: `x*cos + rotate_half(x)*sin` -
    /// identical formula to `qwen_topology::Topo::rope`, duplicated per this
    /// module's own doc (no shared private-struct method to call instead).
    fn rope(&mut self, x: &str) -> String {
        let x2 = {
            let o = self.tmp("rh_x2");
            self.g.add(Node::new("Slice", &[x, "rh_lo1", "rh_hi1", "rh_ax"], &[&o]));
            o
        };
        let x1 = {
            let o = self.tmp("rh_x1");
            self.g.add(Node::new("Slice", &[x, "rh_lo0", "rh_hi0", "rh_ax"], &[&o]));
            o
        };
        let nx2 = self.unary("Neg", &x2);
        let rot = {
            let o = self.tmp("rh");
            self.g.add(Node::new("Concat", &[&nx2, &x1], &[&o]).attr_int("axis", 3));
            o
        };
        let a = self.mul(x, "rope_cos");
        let b = self.mul(&rot, "rope_sin");
        self.add_t(&a, &b)
    }

    fn expand_kv(&mut self, x: &str) -> String {
        let r5 = self.reshape(x, "sh_q5");
        let e = self.tmp("exp");
        self.node("Expand", &[&r5, "sh_exp"], &e);
        self.reshape(&e, "sh_nh")
    }

    /// `y = x·Wᵀ` (no bias), weight-quantizable via `crate::topo::linear_quant`.
    fn linear(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        let winit = format!("{name}.wt");
        let quant = self.quant;
        linear_quant(&mut self.b, x, name, &winit, w, out, inp, quant, &o);
        o
    }

    /// `y = x·Wᵀ + b` - the q/k/v projections, the one place this Qwen2.5
    /// backbone differs from the bias-free `linear` every other projection
    /// uses (`o_proj`/MLP carry no bias - see `llm_import::backbone_name`).
    fn linear_biased(&mut self, x: &str, wname: &str, bname: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let y = self.linear(x, wname, w, out, inp);
        if !self.has(bname) {
            let bv = w.get(bname);
            self.f32(bname, &[out as i64], bv);
        }
        self.add(&y, bname)
    }
}

/// Assemble the graph into `g`. `t` is the fixed prefix length (`1 + sos +
/// text_ids.len() + 1 + prompt_speech_tokens.len()`, matching
/// `CosyVoiceLm::prefill`'s own row count) - a fresh graph per distinct length
/// bucket, the same static-shape convention every NPU export in this crate
/// uses. `quant` selects weight precision; norms/RoPE/mask/embeddings stay
/// fp32 regardless (`crate::topo::linear_quant`'s own convention).
pub fn build_cosyvoice_lm_hidden_graph(cfg: &CosyVoiceLmConfig, w: &dyn WeightSource, t: usize, quant: Quant, g: &mut GraphBuilder) {
    let qc = &cfg.qwen;
    debug_assert!(qc.attn_bias && !qc.qk_norm, "cosyvoice_llm_topology assumes Qwen2-style attention (biased QKV, no QK-norm) - got attn_bias={} qk_norm={}", qc.attn_bias, qc.qk_norm);

    let d = qc.d_model as usize;
    let nh = qc.n_heads as usize;
    let nkv = qc.n_kv_heads as usize;
    let hd = qc.head_dim as usize;
    let half = hd / 2;
    let group = nh / nkv;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let ff = qc.d_ff as usize;
    let eps = qc.rms_eps;
    let ti = t as i64;
    let mut tp = Topo { b: TopoBase::new(g), quant };

    tp.g.input_f32("inputs_embeds", &[1, ti, d as i64]);
    tp.g.output_f32("hidden", &[1, ti, d as i64]);

    tp.f32("c_eps", &[1], vec![eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
    let (mut cos, mut sin) = (vec![0f32; t * hd], vec![0f32; t * hd]);
    for p in 0..t {
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = p as f32 * qc.rope_theta.powf(-2.0 * m / hd as f32);
            cos[p * hd + j] = ang.cos();
            sin[p * hd + j] = ang.sin();
        }
    }
    tp.f32("rope_cos", &[1, ti, 1, hd as i64], cos);
    tp.f32("rope_sin", &[1, ti, 1, hd as i64], sin);
    let mut mask = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask[i * t + j] = -1.0e9;
            }
        }
    }
    tp.f32("causal_mask", &[1, 1, ti, ti], mask);
    tp.i64("rh_ax", &[1], vec![3]);
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    tp.i64("sh_q", &[4], vec![1, ti, nh as i64, hd as i64]);
    tp.i64("sh_kv", &[4], vec![1, ti, nkv as i64, hd as i64]);
    tp.i64("sh_q5", &[5], vec![1, nkv as i64, 1, ti, hd as i64]);
    tp.i64("sh_exp", &[5], vec![1, nkv as i64, group as i64, ti, hd as i64]);
    tp.i64("sh_nh", &[4], vec![1, nh as i64, ti, hd as i64]);
    tp.i64("sh_ctx", &[3], vec![1, ti, hq as i64]);

    let mut x = "inputs_embeds".to_string();
    for l in 0..qc.n_layers as usize {
        let p = |s: &str| format!("blocks.{l}.{s}");
        let h1 = tp.rmsnorm(&x, &p("ln1.weight"), w, d);
        let q = tp.linear_biased(&h1, &p("attn.wq.weight"), &p("attn.wq.bias"), w, hq, d);
        let k = tp.linear_biased(&h1, &p("attn.wk.weight"), &p("attn.wk.bias"), w, hkv, d);
        let v = tp.linear_biased(&h1, &p("attn.wv.weight"), &p("attn.wv.bias"), w, hkv, d);
        let q = tp.reshape(&q, "sh_q");
        let k = tp.reshape(&k, "sh_kv");
        let v = tp.reshape(&v, "sh_kv");
        // No QK-norm - Qwen2, unlike Qwen3 (see this module's doc).
        let q = tp.rope(&q);
        let k = tp.rope(&k);
        let q = tp.transpose(&q, &[0, 2, 1, 3]);
        let k = tp.transpose(&k, &[0, 2, 1, 3]);
        let v = tp.transpose(&v, &[0, 2, 1, 3]);
        let k = tp.expand_kv(&k);
        let v = tp.expand_kv(&v);
        let kt = tp.transpose(&k, &[0, 1, 3, 2]);
        let scores = tp.matmul(&q, &kt);
        let scores = tp.mul(&scores, "c_scale");
        let scores = tp.add(&scores, "causal_mask");
        let probs = tp.softmax(&scores, -1);
        let ctx = tp.matmul(&probs, &v);
        let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
        let ctx = tp.reshape(&ctx, "sh_ctx");
        let attn = tp.linear(&ctx, &p("attn.wo.weight"), w, d, hq); // o_proj is bias-free.
        x = tp.add_t(&x, &attn);

        let h2 = tp.rmsnorm(&x, &p("ln2.weight"), w, d);
        let gate = tp.linear(&h2, &p("mlp.gate.weight"), w, ff, d);
        let up = tp.linear(&h2, &p("mlp.up.weight"), w, ff, d);
        let sig = tp.unary("Sigmoid", &gate);
        let silu = tp.mul_t(&gate, &sig);
        let hmul = tp.mul_t(&silu, &up);
        let down = tp.linear(&hmul, &p("mlp.down.weight"), w, d, ff); // MLP is bias-free.
        x = tp.add_t(&x, &down);
    }
    let xf = tp.rmsnorm(&x, "norm.weight", w, d);
    tp.node("Identity", &[&xf], "hidden");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_cfg() -> CosyVoiceLmConfig {
        let mut cfg = CosyVoiceLmConfig::cosyvoice2();
        cfg.qwen = qwen3::QwenConfig::qwen2(29, 2, 16, 4, 2, 32, true);
        cfg
    }

    fn tiny_backbone(cfg: &CosyVoiceLmConfig) -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        for (name, numel) in cfg.qwen.param_list() {
            m.insert(name, (0..numel).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect());
        }
        m
    }

    #[test]
    fn builds_a_structurally_correct_graph() {
        let cfg = tiny_cfg();
        let w = tiny_backbone(&cfg);
        let t = 6usize;
        let mut g = onnx::GraphBuilder::new("cosyvoice_lm_hidden_test");
        build_cosyvoice_lm_hidden_graph(&cfg, &w, t, Quant::F32, &mut g);

        let graph = g.graph();
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].name, "inputs_embeds");
        assert_eq!(graph.inputs[0].dims, vec![1, t as i64, cfg.qwen.d_model as i64]);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].name, "hidden");
        assert_eq!(graph.outputs[0].dims, vec![1, t as i64, cfg.qwen.d_model as i64]);

        let count = |op: &str| graph.nodes.iter().filter(|n| n.op_type == op).count();
        let nl = cfg.qwen.n_layers as usize;
        // 3 (qkv) + 1 (wo) + 3 (mlp) linear projections + 2 (q@k^T, probs@v)
        // attention matmuls per layer.
        assert_eq!(count("MatMul"), nl * 9, "unexpected MatMul count");
        assert_eq!(count("Softmax"), nl, "one softmax per layer");
        assert!(!g.finish().is_empty());
    }

    #[test]
    fn same_output_names_are_never_duplicated() {
        let cfg = tiny_cfg();
        let w = tiny_backbone(&cfg);
        let mut g = onnx::GraphBuilder::new("cosyvoice_lm_hidden_dupe_check");
        build_cosyvoice_lm_hidden_graph(&cfg, &w, 5, Quant::F32, &mut g);
        let mut seen = std::collections::HashSet::new();
        for n in &g.graph().nodes {
            for o in &n.outputs {
                assert!(seen.insert(o.clone()), "duplicate output tensor name: {o}");
            }
        }
    }

    #[test]
    fn int8_quant_builds_and_shrinks_the_weight_bytes() {
        let cfg = tiny_cfg();
        let w = tiny_backbone(&cfg);
        let t = 5usize;
        let mut gf = onnx::GraphBuilder::new("f32");
        build_cosyvoice_lm_hidden_graph(&cfg, &w, t, Quant::F32, &mut gf);
        let mut gi = onnx::GraphBuilder::new("int8");
        build_cosyvoice_lm_hidden_graph(&cfg, &w, t, Quant::Int8, &mut gi);
        assert!(gi.finish().len() < gf.finish().len(), "int8 weight-only graph should serialize smaller than fp32");
    }
}
