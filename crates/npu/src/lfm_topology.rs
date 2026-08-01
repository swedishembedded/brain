// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build the LFM2.5-Encoder as an ONNX graph (fixed sequence length `S`) for
//! whole-graph compilation on OpenVINO (NPU/GPU/CPU).
//!
//! The graph is the full encoder: in-graph token embedding (Gather over the
//! 65536×1024 table — external-data sidecar), the hybrid conv/attention stack
//! from `layer_types`, and the final `embedding_norm`; output is
//! `hidden:[1,S,D]`. The tied MLM head stays on host (a `[1,S,65536]` output
//! would dwarf the model at long S; fill-mask needs only a few probe rows).
//!
//! LFM2.5 specifics vs the qwen/chronos2 topologies it composes from:
//! - **Bidirectional** attention with the additive `kmask:[1,1,1,S]` graph
//!   input (chronos2 pattern — no T×T constant), scaled 1/√hd (unlike
//!   chronos2's unscaled), GQA 16Q/8KV expanded in-graph (qwen `expand_kv`).
//! - **Per-head QK-RMSNorm** on `[1,S,heads,64]` before RoPE (last-axis
//!   `TopoBase::rmsnorm` with the 64-wide gain).
//! - **Gated short-conv mixer**: in_proj → thirds, `B⊙X`, depthwise `Conv`
//!   (1-D, `group = d`, symmetric `pads=[1,1]`, weight `[d,1,k]` verbatim),
//!   `C⊙conv`, out_proj (codec_topology's conv precedent, non-causal pads).
//! - **Query-chunked attention** above `CHUNK_ABOVE`: statically unrolled
//!   Slice-q → scores → +kmask → softmax → ctx per chunk, Concat at the end —
//!   the in-graph twin of `model::block::chunked_bidir_fwd`, keeping the
//!   transient score tensor at `[1,heads,chunk,S]` so an 8192 bucket does not
//!   materialize ~2 GB per layer. Never Concat directly into a graph output
//!   (the mirror lesson): the concat lands in a temp, the final norm follows.

use lfm::config::{LayerType, LfmConfig};
use onnx::builder::GraphBuilder;
use onnx::graph::Node;

use crate::topo::{linear_quant, Quant, TopoBase};
use crate::topology::WeightSource;

/// Materialized attention up to this many query rows; above it, chunked.
const CHUNK_ABOVE: usize = 2048;

/// Assemble the LFM2.5-Encoder graph into `g` (fp32 weights).
pub fn build_lfm_graph(cfg: &LfmConfig, w: &dyn WeightSource, s: usize, g: &mut GraphBuilder) {
    build_lfm_graph_quant(cfg, w, s, g, Quant::F32)
}

/// As [`build_lfm_graph`] with a weight-quantization mode (per-output-channel
/// INT8/INT4 weight-only; norms, RoPE tables and the conv taps stay fp32).
pub fn build_lfm_graph_quant(cfg: &LfmConfig, w: &dyn WeightSource, s: usize, g: &mut GraphBuilder, quant: Quant) {
    let d = cfg.d_model as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let group = (cfg.group()) as usize;
    let ff = cfg.d_ff as usize;
    let k = cfg.conv_k as usize;
    let si = s as i64;
    let mut tp = Topo { b: TopoBase::new(g), quant };

    tp.g.input_i64("ids", &[1, si]);
    tp.g.input_f32("kmask", &[1, 1, 1, si]);
    tp.g.output_f32("hidden", &[1, si, d as i64]);

    tp.f32("c_eps", &[1], vec![cfg.norm_eps]);
    tp.f32("c_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);

    // Token embedding: Gather over the [vocab, d] table (external sidecar).
    tp.f32("tok.weight", &[cfg.vocab as i64, d as i64], w.get("tok.weight"));
    let mut x = tp.gather("tok.weight", "ids", 0, "emb"); // [1,S,d]

    // Half-split RoPE tables [1,S,1,hd] (full width: cos = cat(f,f)), theta
    // from the checkpoint config (1e6).
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; s * hd], vec![0f32; s * hd]);
    for p in 0..s {
        for j in 0..half {
            let ang = p as f32 / cfg.rope_theta.powf(j as f32 / half as f32);
            for c in [j, j + half] {
                cos[p * hd + c] = ang.cos();
                sin[p * hd + c] = ang.sin();
            }
        }
    }
    tp.f32("rope_cos", &[1, si, 1, hd as i64], cos);
    tp.f32("rope_sin", &[1, si, 1, hd as i64], sin);
    // rotate_half slice bounds (axis 3).
    tp.i64("rh_lo0", &[1], vec![0]);
    tp.i64("rh_hi0", &[1], vec![half as i64]);
    tp.i64("rh_lo1", &[1], vec![half as i64]);
    tp.i64("rh_hi1", &[1], vec![hd as i64]);
    tp.i64("rh_ax", &[1], vec![3]);

    // Reshape shapes.
    tp.i64("sh_q4", &[4], vec![1, si, nh as i64, hd as i64]);
    tp.i64("sh_kv4", &[4], vec![1, si, nkv as i64, hd as i64]);
    tp.i64("sh_q5", &[5], vec![1, nkv as i64, 1, si, hd as i64]);
    tp.i64("sh_exp", &[5], vec![1, nkv as i64, group as i64, si, hd as i64]);
    tp.i64("sh_nh", &[4], vec![1, nh as i64, si, hd as i64]);
    tp.i64("sh_flat", &[3], vec![1, si, (nh * hd) as i64]);
    // Conv-third slice bounds (axis 2 of [1,S,3d]).
    for (name, v) in [("t3_b0", 0i64), ("t3_b1", d as i64), ("t3_c1", 2 * d as i64), ("t3_x1", 3 * d as i64)] {
        tp.i64(name, &[1], vec![v]);
    }
    tp.i64("t3_ax", &[1], vec![2]);

    for (l, ty) in cfg.layer_types.iter().enumerate() {
        let p = |n: &str| format!("blocks.{l}.{n}");
        match ty {
            LayerType::Conv => {
                let h = tp.rmsnorm(&x, &p("ln1.weight"), w, d);
                let bcx = tp.linear(&h, &p("conv.in_proj.weight"), w, 3 * d, d); // [1,S,3d]
                let bg = tp.slice(&bcx, "t3_b0", "t3_b1", "t3_ax");
                let cg = tp.slice(&bcx, "t3_b1", "t3_c1", "t3_ax");
                let xg = tp.slice(&bcx, "t3_c1", "t3_x1", "t3_ax");
                let bx = tp.mul_t(&bg, &xg);
                let ncl = tp.transpose(&bx, &[0, 2, 1]); // [1,d,S]
                let cw = format!("{}.w", p("conv.conv.weight"));
                tp.f32(&cw, &[d as i64, 1, k as i64], w.get(&p("conv.conv.weight")));
                let conv = {
                    let o = tp.tmp("conv");
                    let pad = (k / 2) as i64;
                    tp.g.add(
                        Node::new("Conv", &[&ncl, &cw], &[&o])
                            .attr_ints("kernel_shape", &[k as i64])
                            .attr_ints("strides", &[1])
                            .attr_ints("pads", &[pad, pad])
                            .attr_ints("dilations", &[1])
                            .attr_int("group", d as i64),
                    );
                    o
                };
                let nlc = tp.transpose(&conv, &[0, 2, 1]); // [1,S,d]
                let gated = tp.mul_t(&cg, &nlc);
                let o = tp.linear(&gated, &p("conv.out_proj.weight"), w, d, d);
                x = tp.add_t(&x, &o);
            }
            LayerType::Attention => {
                let h = tp.rmsnorm(&x, &p("ln1.weight"), w, d);
                let q = tp.linear(&h, &p("attn.wq.weight"), w, nh * hd, d);
                let kk = tp.linear(&h, &p("attn.wk.weight"), w, nkv * hd, d);
                let v = tp.linear(&h, &p("attn.wv.weight"), w, nkv * hd, d);
                let q4 = tp.reshape(&q, "sh_q4"); // [1,S,nh,hd]
                let k4 = tp.reshape(&kk, "sh_kv4");
                let v4 = tp.reshape(&v, "sh_kv4");
                // Per-head QK-RMSNorm (last axis = hd), then RoPE.
                let q4 = tp.rmsnorm(&q4, &p("attn.q_norm.weight"), w, hd);
                let k4 = tp.rmsnorm(&k4, &p("attn.k_norm.weight"), w, hd);
                let q4 = tp.rope(&q4);
                let k4 = tp.rope(&k4);
                let qt = tp.transpose(&q4, &[0, 2, 1, 3]); // [1,nh,S,hd]
                let kt = tp.transpose(&k4, &[0, 2, 1, 3]); // [1,nkv,S,hd]
                let vt = tp.transpose(&v4, &[0, 2, 1, 3]);
                let ke = tp.expand_kv(&kt); // [1,nh,S,hd]
                let ve = tp.expand_kv(&vt);
                let ktt = tp.transpose(&ke, &[0, 1, 3, 2]); // [1,nh,hd,S]
                let ctx = tp.attention(&qt, &ktt, &ve, s);
                let ctx = tp.transpose(&ctx, &[0, 2, 1, 3]);
                let ctx = tp.reshape(&ctx, "sh_flat");
                let o = tp.linear(&ctx, &p("attn.wo.weight"), w, d, nh * hd);
                x = tp.add_t(&x, &o);
            }
        }
        // SwiGLU FFN.
        let h = tp.rmsnorm(&x, &p("ln2.weight"), w, d);
        let gate = tp.linear(&h, &p("mlp.gate.weight"), w, ff, d);
        let up = tp.linear(&h, &p("mlp.up.weight"), w, ff, d);
        let sg = tp.silu(&gate);
        let act = tp.mul_t(&sg, &up);
        let down = tp.linear(&act, &p("mlp.down.weight"), w, d, ff);
        x = tp.add_t(&x, &down);
    }

    // Final embedding_norm into the graph output (Mul writes `hidden` directly
    // — a norm, not a Concat, so the mirror Concat-into-output trap is avoided).
    let gain = format!("{}.g", "norm.weight");
    tp.b.rmsnorm_to(&x, &gain, w.get("norm.weight"), d, "c_eps", "hidden");
}

/// ONNX assembly helper (mirrors the qwen/chronos2 topologies' `Topo`).
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

    /// Bias-free linear via the shared weight-quantizing emitter.
    fn linear(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        let quant = self.quant;
        linear_quant(&mut self.b, x, name, &format!("{name}.wt"), w, out, inp, quant, &o);
        o
    }

    /// GQA head replication [1,nkv,S,hd] → [1,nh,S,hd] (repeat_kv layout).
    fn expand_kv(&mut self, x: &str) -> String {
        let r5 = self.reshape(x, "sh_q5");
        let e = self.tmp("exp");
        self.node("Expand", &[&r5, "sh_exp"], &e);
        self.reshape(&e, "sh_nh")
    }

    /// Half-split RoPE on [1,S,heads,hd]: `x*cos + rotate_half(x)*sin`.
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

    /// Scaled bidirectional attention `softmax(q·kᵀ/√hd + kmask)·v` over
    /// `[1,nh,S,hd]`, materialized whole at small S and statically query-chunked
    /// above [`CHUNK_ABOVE`] (transient scores stay `[1,nh,chunk,S]`).
    fn attention(&mut self, qt: &str, ktt: &str, ve: &str, s: usize) -> String {
        if s <= CHUNK_ABOVE {
            let scores = self.matmul(qt, ktt);
            let scaled = self.mul(&scores, "c_scale");
            let masked = self.add(&scaled, "kmask");
            let probs = self.softmax(&masked, -1);
            return self.matmul(&probs, ve);
        }
        // Chunk bounds along the query axis (axis 2 of [1,nh,S,hd]).
        self.i64("qc_ax", &[1], vec![2]);
        let mut parts: Vec<String> = Vec::new();
        let mut q0 = 0usize;
        while q0 < s {
            let qn = CHUNK_ABOVE.min(s - q0);
            let lo = format!("qc_lo_{q0}");
            let hi = format!("qc_hi_{}", q0 + qn);
            self.i64(&lo, &[1], vec![q0 as i64]);
            self.i64(&hi, &[1], vec![(q0 + qn) as i64]);
            let qc = self.slice(qt, &lo, &hi, "qc_ax"); // [1,nh,qn,hd]
            let scores = self.matmul(&qc, ktt); // [1,nh,qn,S]
            let scaled = self.mul(&scores, "c_scale");
            let masked = self.add(&scaled, "kmask");
            let probs = self.softmax(&masked, -1);
            parts.push(self.matmul(&probs, ve)); // [1,nh,qn,hd]
            q0 += qn;
        }
        // Fold the per-chunk contexts back together along the query axis.
        let mut acc = parts[0].clone();
        for part in &parts[1..] {
            acc = self.concat2(&acc, part, 2);
        }
        acc
    }
}
