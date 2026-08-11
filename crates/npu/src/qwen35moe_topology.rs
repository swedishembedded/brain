// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build Qwen3.5-35B-A3B (`qwen35moe`) as an ONNX graph (fixed sequence length
//! `T`) for a **best-effort** OpenVINO/NPU compile attempt. Fixed-`T`,
//! cache-free PREFILL only, text-only (no vision splice) — the same scope
//! every other `*_topology.rs` file in this crate documents for itself
//! (`qwen_topology.rs`'s own module doc), not a new limitation introduced
//! here. See `.agents/roadmap/qwen35.md`'s P14 entry for the task this
//! file closes and the exact boundary where this pass stops (recorded in this
//! module's doc below, and in the final report that shipped alongside it).
//!
//! Two sub-problems neither `qwen_topology.rs` (plain GQA) nor
//! `glm_topology.rs` (MoE, but small-`E` and no linear-attention layer) has an
//! existing precedent for:
//!
//! ## 1. A `gdn_chunk`-shaped emitter for Gated DeltaNet
//!
//! `model::gdn::gdn_chunk_fwd`'s own module doc lays out an 11-step chunked
//! recurrence (see that file for the authoritative derivation — this doc only
//! summarises how each step becomes ONNX). The WGSL engine needs a literal
//! chunk-major flat-buffer permute and a serial per-row forward-substitution
//! for the UT-transform because its own kernels can only address contiguous
//! batch ranges and cannot run a true parallel scan or a triangular solve in
//! one dispatch. ONNX has none of those constraints — real N-D tensors with
//! native broadcasting — so three steps get a strictly SIMPLER, still
//! bit-for-bit-equivalent translation instead of a literal port of the WGSL
//! kernel sequence:
//!
//! - **step 4 (per-chunk cumsum)**: instead of `CumSum` (opset-13 semantics
//!   are workable but the op's real-world ONNX-frontend support is
//!   inconsistent, and this repo has zero existing precedent for it), the
//!   cumulative sum over a chunk's `C` positions is computed as one `MatMul`
//!   against a constant upper-triangular-inclusive ones matrix
//!   (`out = in_row_vector @ M`, `M[j,i] = 1` iff `j<=i`) — exactly a cumsum,
//!   expressed with the one op (`MatMul`) every other node in this graph
//!   already relies on, so there is no new operator-support risk to chase;
//! - **step 7 (UT-transform, `(I-attn0)^-1` via forward substitution)**:
//!   `attn0` is strictly lower-triangular and therefore nilpotent
//!   (`attn0^C = 0` for a `C x C` matrix), so `(I-attn0)^-1` has the EXACT
//!   closed-form Neumann series `I + attn0 + attn0^2 + ... + attn0^(C-1)` —
//!   computed here as `C-1` statically unrolled `MatMul`+`Add` pairs (`C` =
//!   the GDN chunk size, fixed at export time by [`qwen35moe::model::gdn_chunk_size`]);
//! - **step 10 (the sequential across-chunk state recurrence)**: unrolled
//!   into `n_chunks` static blocks threading the recurrent `state` tensor
//!   forward, the same "unroll a known-small sequential loop at export time"
//!   precedent `crate::qwen_topology::build_mtp_fused_graph`'s 16-position
//!   unroll and `crate::lfm_topology`'s query-chunked attention already
//!   establish in this crate.
//!
//! `T`, the GDN chunk size, and `n_chunks` are all concrete at export time
//! (fixed-`T` prefill), so every shape below is static.
//!
//! ## 2. A sparse gather-based expert-dispatch emitter for 256 experts
//!
//! `glm_topology.rs::Topo::moe` (the only existing MoE emitter in this crate)
//! evaluates **every** expert densely over the whole row batch and combines
//! with a `TopK`+`ScatterElements`-built dense gate — correct, but
//! `.agents/roadmap/qwen35.md`'s own P14 note already flags that a literal
//! port of that at Qwen3.5's 256 experts is impractical (a dense per-expert
//! `swiglu` loop is 3 `MatMul`s/expert; 256 experts x 40 layers is tens of
//! thousands of `MatMul` nodes for weights that are 32x oversized relative to
//! what top-8 routing actually uses). GLM's own `n_routed_experts` is small
//! enough that the dense form was never meant to scale past it — this file
//! does not copy that approach.
//!
//! Instead, every layer's per-expert `gate`/`up`/`down` weights are stacked
//! ONCE into a single `[E, in, out]` ONNX initializer per projection
//! (`Gather`-indexable along axis 0). The router computes the **exact**
//! `router_gate.wgsl` math (`model::moe::router_fwd`'s `RouterKind::Softmax`,
//! the router this model actually uses per `qwen35moe::model::moe_sublayer`):
//! softmax over all `E` logits, keep the `top_k` largest **by softmax
//! probability** (equivalent to by logit, since softmax is monotonic
//! per-row — `TopK` is run directly on the softmax output, not the raw
//! logits, to make that equivalence explicit rather than relied upon
//! silently), then renormalise the kept weights to sum to 1
//! (`gate[e] = probs[e] / sum_selected(probs)`) — `router_gate.wgsl`'s own
//! passes 1-4 line for line. `Gather(stack, topk_idx, axis=0)` then pulls out
//! only the `top_k` selected experts' weights **per token**
//! (`[1,T,top_k,in,out]`, ONNX Gather's documented multi-dim-index output
//! shape), and a single broadcasting `MatMul` runs every selected expert's
//! FFN at once. Memory and FLOPs scale with `top_k`, not `E` — the actual
//! sparse property, not `glm_topology`'s "run everything, mask after" shape.
//!
//! ## Router math cross-check against `model::moe`
//!
//! `crates/model/src/moe.rs`'s `router_fwd`/`router_gate.wgsl` doc (read in
//! full before writing this file) is unambiguous about the 4-pass algorithm:
//! row-max-stabilised softmax, greedy top-k selection over the softmax
//! probabilities (not the raw logits, though the two orderings coincide),
//! then `gate[e] = probs[e] / sum_of_kept_probs` for kept `e`, else `0`. This
//! file's router block (`Topo::moe_layer`) is `Softmax` -> `TopK(top_k,
//! axis=-1, largest=1)` -> `ReduceSum` of the kept values -> `Div` — the same
//! four passes, with `TopK`'s own sort standing in for `router_gate.wgsl`'s
//! explicit greedy-max loop (both select the same top-k set; ONNX `TopK`
//! with `largest=1` is defined to return the `k` largest values, matching the
//! kernel's greedy selection exactly for a strict, no-tie ordering — ties are
//! the one place `TopK`'s internal tie-break and the kernel's first-found
//! greedy tie-break could in principle diverge, noted in this file's final
//! report as a low-probability, unverified edge case rather than silently
//! assumed identical).
//!
//! ## Scope boundary this file does NOT cross
//!
//! Fixed-`T`, cache-free PREFILL only (no `Qwen35::step`-shaped incremental
//! decode form — `model::gdn`'s own doc says the decode-step primitives
//! (`gdn_recurrent_step`) are a SEPARATE entry point this file does not use).
//! No vision splice (`qwen35moe::model` doesn't have one yet either). No
//! INT8/INT4 weight quantization for this model (a real follow-on — every
//! other topology file's `Quant::Int8` path is a mechanical
//! per-output-channel scale that composes cleanly with the emitters below,
//! but doubling the file's size to add it was judged out of scope for a
//! "compiles + best-effort compile attempt" pass). "Compiles to valid ONNX +
//! best-effort OpenVINO compile" is the explicit stopping point — see
//! `crates/npu/tests/qwen35moe_onnx.rs` for exactly the same
//! structural-always / `BRAIN_OV_PROBE`-gated split every other topology
//! file's own tests use, not a new verification mechanism.

use onnx::{GraphBuilder, Node};
use qwen35moe::config::{LayerType, Qwen35Config};

use crate::topo::{linear_quant, Quant, TopoBase};
use crate::topology::WeightSource;

/// Assemble the Qwen3.5-35B-A3B decoder graph into `g` (fp32 weights).
/// `input_ids:[1,T]` (i64) -> `logits:[1,T,vocab]` (f32).
pub fn build_qwen35_graph(cfg: &Qwen35Config, w: &dyn WeightSource, t: usize, g: &mut GraphBuilder) {
    let d = cfg.d_model as usize;
    let vocab = cfg.vocab as usize;
    let ti = t as i64;
    let mut tp = Topo { b: TopoBase::new(g) };

    tp.g.input_i64("input_ids", &[1, ti]);
    tp.g.output_f32("logits", &[1, ti, vocab as i64]);
    tp.f32("c_eps", &[1], vec![cfg.rms_eps]);

    tp.f32("tok.weight", &[vocab as i64, d as i64], w.get("tok.weight"));
    let mut res = tp.gather("tok.weight", "input_ids", 0, "emb"); // [1,T,d]

    let types = cfg.layer_types();
    for (l, ty) in types.iter().enumerate() {
        let xn1 = tp.rmsnorm(&res, &format!("blocks.{l}.ln1.weight"), w, d);
        let mix_out = match ty {
            LayerType::Linear => tp.gdn_layer(l, &xn1, w, cfg, t),
            LayerType::Full => tp.gqa_layer(l, &xn1, w, cfg, t),
        };
        let xmid = tp.add_t(&res, &mix_out);
        let moe_out = tp.moe_layer(l, &xmid, w, cfg, t);
        res = tp.add_t(&xmid, &moe_out);
    }

    let xf = tp.rmsnorm(&res, "norm.weight", w, d);
    let head = cfg.head_weight().to_string();
    tp.linear_to(&xf, &head, w, vocab, d, "logits");
}

/// ONNX assembly helper (mirrors every other `*_topology.rs`'s own `Topo`).
struct Topo<'a> {
    b: TopoBase<'a>,
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
    // ---- generic small helpers -------------------------------------------

    /// A `[1]`-shaped i64 constant holding `val`, registered idempotently and
    /// named by value — safe to reuse across every call site that needs the
    /// same scalar (an axis index, a slice bound, a top-k), since the name
    /// encodes the value itself, not the call site's intent.
    fn ci64(&mut self, val: i64) -> String {
        let n = format!("c_i64_{val}");
        self.i64(&n, &[1], vec![val]);
        n
    }

    fn rmsnorm(&mut self, x: &str, name: &str, w: &dyn WeightSource, dim: usize) -> String {
        let gain = format!("{name}.g");
        let data = w.get(name);
        self.b.rmsnorm(x, &gain, data, dim, "c_eps")
    }

    fn linear(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize) -> String {
        let o = self.tmp("lin");
        self.linear_to(x, name, w, out, inp, &o);
        o
    }

    fn linear_to(&mut self, x: &str, name: &str, w: &dyn WeightSource, out: usize, inp: usize, y: &str) {
        let winit = format!("{name}.wt");
        linear_quant(&mut self.b, x, name, &winit, w, out, inp, Quant::F32, y);
    }

    /// `ReduceSum(x, axis)`, opset-13-correct: `axes` is an INPUT tensor (not
    /// an attribute, unlike `ReduceMean` which keeps the attribute form until
    /// opset 18 — see `fincast_topology.rs::Topo::reduce_sum`'s own comment
    /// for the same fact, independently needed here).
    fn reduce_sum(&mut self, x: &str, axis: i64, keepdims: bool) -> String {
        let ax = self.ci64(axis);
        let o = self.tmp("rsum");
        self.g.add(Node::new("ReduceSum", &[x, &ax], &[&o]).attr_int("keepdims", keepdims as i64));
        o
    }

    /// Bare L2-normalize over the LAST axis (no learnable scale) — GDN's
    /// query/key norm (`model::gdn`'s caller uses `l2norm_scale.wgsl` with an
    /// all-ones scale buffer, i.e. plain L2 normalize):
    /// `y = x / sqrt(sum(x^2, -1) + eps)`. `eps=1e-6`, matching
    /// `qwen35moe::model::layer_gdn_fwd`'s own `L2NORM_SCALE` call.
    fn l2norm(&mut self, x: &str) -> String {
        let sq = self.mul_t(x, x);
        let ss = self.reduce_sum(&sq, -1, true);
        self.f32("gdn_l2eps", &[1], vec![1e-6]);
        let sse = self.add(&ss, "gdn_l2eps");
        let denom = self.unary("Sqrt", &sse);
        let o = self.tmp("l2n");
        self.node("Div", &[x, &denom], &o);
        o
    }

    /// `[1,rows,src_heads,dim] -> [1,rows,src_heads*group,dim]`, repeating
    /// each source head `group` times CONSECUTIVELY (`repeat_interleave`
    /// semantics — matches `model::block::kv_expand_fwd`'s `repeat_kv`
    /// convention that both GDN's own q/k head repeat (step 6,
    /// `linear_num_key_heads -> linear_num_value_heads`) and GQA's kv-expand
    /// need). `tag` scopes the generated shape-initializer names so distinct
    /// callers (GDN vs GQA, or different dims) don't collide; identical
    /// `(tag,rows,src_heads,group,dim)` calls safely share the same constants.
    fn repeat_heads(&mut self, x: &str, tag: &str, rows: usize, src_heads: usize, group: usize, dim: usize) -> String {
        let dst_heads = src_heads * group;
        let sh5 = format!("{tag}_r5_{rows}_{src_heads}_{group}_{dim}");
        let she = format!("{tag}_e5_{rows}_{src_heads}_{group}_{dim}");
        let shn = format!("{tag}_r4_{rows}_{dst_heads}_{dim}");
        self.i64(&sh5, &[5], vec![1, rows as i64, src_heads as i64, 1, dim as i64]);
        self.i64(&she, &[5], vec![1, rows as i64, src_heads as i64, group as i64, dim as i64]);
        self.i64(&shn, &[4], vec![1, rows as i64, dst_heads as i64, dim as i64]);
        let r5 = self.reshape(x, &sh5);
        let e = self.tmp("exp");
        self.node("Expand", &[&r5, &she], &e);
        self.reshape(&e, &shn)
    }

    /// Partial half-split RoPE on `[1,T,H,hd]`: only the first `rot` channels
    /// of each head rotate (`partial_rotary_factor`), the rest pass through
    /// unrotated. Qwen3.5's text-only degenerate M-RoPE (all three position
    /// axes equal for every token, since there is no vision splice in this
    /// export) collapses `qwenvl::mrope::mrope_tables` to plain 1-D RoPE —
    /// confirmed by `.agents/roadmap/qwen35.md`'s P11b note that this
    /// collapse is exact, not approximate — so a single `theta`-based table
    /// is exactly right here, no per-axis section bookkeeping needed.
    fn rope_partial(&mut self, x: &str, hd: usize, rot: usize, t: usize, theta: f32) -> String {
        if rot == 0 {
            return x.to_string();
        }
        let half = rot / 2;
        let ti = t as i64;
        let cos_name = format!("rope_cos_{rot}");
        let sin_name = format!("rope_sin_{rot}");
        if !self.has(&cos_name) {
            let (mut cos, mut sin) = (vec![0f32; t * rot], vec![0f32; t * rot]);
            for p in 0..t {
                for j in 0..rot {
                    let m = (j % half) as f32;
                    let ang = p as f32 * theta.powf(-2.0 * m / rot as f32);
                    cos[p * rot + j] = ang.cos();
                    sin[p * rot + j] = ang.sin();
                }
            }
            self.f32(&cos_name, &[1, ti, 1, rot as i64], cos);
            self.f32(&sin_name, &[1, ti, 1, rot as i64], sin);
        }
        let lo0 = self.ci64(0);
        let hi_rot = self.ci64(rot as i64);
        let half_rot = self.ci64(half as i64);
        let ax3 = self.ci64(3);

        let x_rot = self.slice(x, &lo0, &hi_rot, &ax3);
        let x2 = self.slice(&x_rot, &half_rot, &hi_rot, &ax3);
        let x1 = self.slice(&x_rot, &lo0, &half_rot, &ax3);
        let nx2 = self.unary("Neg", &x2);
        let rh = self.tmp("rp_rh");
        self.g.add(Node::new("Concat", &[&nx2, &x1], &[&rh]).attr_int("axis", 3));
        let a = self.mul(&x_rot, &cos_name);
        let b = self.mul(&rh, &sin_name);
        let rotated = self.add_t(&a, &b);
        if rot < hd {
            let hi_hd = self.ci64(hd as i64);
            let x_pass = self.slice(x, &hi_rot, &hi_hd, &ax3);
            self.concat2(&rotated, &x_pass, 3)
        } else {
            rotated
        }
    }

    /// Stack every expert's `[out,in]` weight (brain layout) into one
    /// `[E,in,out]` ONNX initializer (transposed once, like every other
    /// `linear` weight in this codebase) — the `Gather`-indexable form the
    /// sparse dispatch in [`Topo::moe_layer`] needs. Registered once per
    /// `name` (layer-scoped names, so always a fresh first call per layer).
    fn expert_stack(&mut self, name: &str, w: &dyn WeightSource, e: usize, out: usize, inp: usize, namer: impl Fn(usize) -> String) -> String {
        if !self.has(name) {
            let mut data = Vec::with_capacity(e * out * inp);
            for ei in 0..e {
                let raw = w.get(&namer(ei)); // [out,in]
                data.extend(transpose(&raw, out, inp)); // -> [in,out]
            }
            self.f32(name, &[e as i64, inp as i64, out as i64], data);
        }
        name.to_string()
    }

    /// `Slice` chunk `[lo,hi)` out of axis-1 of a 5-D chunk-major tensor, then
    /// `Reshape` away the resulting size-1 chunk axis to the plain 4-D
    /// per-chunk shape `sh4` names. Used everywhere in
    /// [`Topo::gdn_layer`]'s step-10 per-chunk loop.
    fn slice_chunk(&mut self, x: &str, lo: &str, hi: &str, ax1: &str, sh4: &str) -> String {
        let s5 = self.slice(x, lo, hi, ax1);
        self.reshape(&s5, sh4)
    }

    // ---- Gated DeltaNet (linear-attention) layer ---------------------------

    /// One Gated-DeltaNet mixer layer (`qwen35moe::model::layer_gdn_fwd`'s
    /// reference, steps enumerated in this module's own doc). Returns
    /// `mix_out:[1,T,d]`.
    fn gdn_layer(&mut self, l: usize, xn1: &str, w: &dyn WeightSource, c: &Qwen35Config, t: usize) -> String {
        let d = c.d_model as usize;
        let key_dim = c.linear_key_dim() as usize;
        let value_dim = c.linear_value_dim() as usize;
        let conv_dim = c.linear_conv_dim() as usize;
        let nkh = c.linear_num_key_heads as usize;
        let nvh = c.linear_num_value_heads as usize;
        let khd = c.linear_key_head_dim as usize;
        let vhd = c.linear_value_head_dim as usize;
        let group = c.linear_group() as usize;
        let kw = c.linear_conv_kernel_dim as usize;
        let chunk = qwen35moe::model::gdn_chunk_size(t as u32) as usize;
        let n_chunks = t / chunk;
        let ti = t as i64;
        let nci = n_chunks as i64;
        let ci_ = chunk as i64;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // ---- 1. mixed_qkv = in_proj_qkv(xn1) ----
        let in_proj_qkv_w = p("in_proj_qkv.weight");
        let mixed_qkv = self.linear(xn1, &in_proj_qkv_w, w, conv_dim, d);

        // ---- 2. causal depthwise conv1d (pads=[k-1,0]) + SiLU ----
        let ncl = self.transpose(&mixed_qkv, &[0, 2, 1]); // [1,conv_dim,T]
        let cw = format!("gdn_convw_{l}");
        let conv1d_w = w.get(&p("conv1d.weight"));
        self.f32(&cw, &[conv_dim as i64, 1, kw as i64], conv1d_w);
        let conv_out = self.tmp("gdn_conv");
        self.g.add(
            Node::new("Conv", &[&ncl, &cw], &[&conv_out])
                .attr_ints("kernel_shape", &[kw as i64])
                .attr_ints("strides", &[1])
                .attr_ints("pads", &[(kw - 1) as i64, 0])
                .attr_ints("dilations", &[1])
                .attr_int("group", conv_dim as i64),
        );
        let nlc = self.transpose(&conv_out, &[0, 2, 1]); // [1,T,conv_dim]
        let act = self.silu(&nlc);

        // ---- 3. split into query/key/value (whole-row contiguous) ----
        let ax2 = self.ci64(2);
        let s0 = self.ci64(0);
        let s1 = self.ci64(key_dim as i64);
        let s2 = self.ci64(2 * key_dim as i64);
        let s3 = self.ci64(conv_dim as i64);
        let query = self.slice(&act, &s0, &s1, &ax2); // [1,T,key_dim]
        let key = self.slice(&act, &s1, &s2, &ax2);
        let value = self.slice(&act, &s2, &s3, &ax2); // [1,T,value_dim]

        // ---- 4. L2-normalize query/key, per head (view as [1,T,nkh,khd]) ----
        let sh_headk4 = "gdn_sh_headk4".to_string();
        self.i64(&sh_headk4, &[4], vec![1, ti, nkh as i64, khd as i64]);
        let query4 = self.reshape(&query, &sh_headk4);
        let key4 = self.reshape(&key, &sh_headk4);
        let query_n = self.l2norm(&query4);
        let key_n = self.l2norm(&key4);

        // ---- 5. beta/g_decay/z ----
        let in_proj_b_w = p("in_proj_b.weight");
        let in_proj_a_w = p("in_proj_a.weight");
        let in_proj_z_w = p("in_proj_z.weight");
        let bproj = self.linear(xn1, &in_proj_b_w, w, nvh, d);
        let aproj = self.linear(xn1, &in_proj_a_w, w, nvh, d);
        let z = self.linear(xn1, &in_proj_z_w, w, value_dim, d);
        let beta = self.unary("Sigmoid", &bproj);
        let a_log_name = format!("gdn_alog_{l}");
        let a_log_data = w.get(&p("A_log"));
        self.f32(&a_log_name, &[1, 1, nvh as i64], a_log_data);
        let dt_bias_name = format!("gdn_dtbias_{l}");
        let dt_bias_data = w.get(&p("dt_bias"));
        self.f32(&dt_bias_name, &[1, 1, nvh as i64], dt_bias_data);
        let apb = self.add(&aproj, &dt_bias_name);
        let sp = self.unary("Softplus", &apb);
        let exp_a = self.unary("Exp", &a_log_name);
        let neg_exp_a = self.unary("Neg", &exp_a);
        let g_decay = self.mul_t(&sp, &neg_exp_a);

        // ---- 6. repeat query/key from nkh -> nvh heads ----
        let query_w = self.repeat_heads(&query_n, "gdn_rep_k", t, nkh, group, khd);
        let key_w = self.repeat_heads(&key_n, "gdn_rep_k", t, nkh, group, khd);

        // ---- 7. chunk-major reshape+transpose ----
        let sh_cm5k = "gdn_sh_cm5k".to_string();
        self.i64(&sh_cm5k, &[5], vec![1, nci, ci_, nvh as i64, khd as i64]);
        let sh_cm5v = "gdn_sh_cm5v".to_string();
        self.i64(&sh_cm5v, &[5], vec![1, nci, ci_, nvh as i64, vhd as i64]);
        let sh_cm4 = "gdn_sh_cm4".to_string();
        self.i64(&sh_cm4, &[4], vec![1, nci, ci_, nvh as i64]);
        let sh_cm4b = "gdn_sh_cm4b".to_string();
        self.i64(&sh_cm4b, &[4], vec![1, nci, nvh as i64, ci_]);

        let query_w5 = self.reshape(&query_w, &sh_cm5k);
        let query_cm = self.transpose(&query_w5, &[0, 1, 3, 2, 4]); // [1,nc,nvh,C,khd]
        let key_w5 = self.reshape(&key_w, &sh_cm5k);
        let key_cm = self.transpose(&key_w5, &[0, 1, 3, 2, 4]);
        let value5 = self.reshape(&value, &sh_cm5v);
        let value_cm = self.transpose(&value5, &[0, 1, 3, 2, 4]); // [1,nc,nvh,C,vhd]
        let g_decay4 = self.reshape(&g_decay, &sh_cm4);
        let g_cm = self.transpose(&g_decay4, &[0, 1, 3, 2]); // [1,nc,nvh,C]
        let beta4 = self.reshape(&beta, &sh_cm4);
        let beta_cm = self.transpose(&beta4, &[0, 1, 3, 2]);

        // ---- 8. v_beta / k_beta (row-broadcast by beta) ----
        let sh_bcast5 = "gdn_sh_bcast5".to_string();
        self.i64(&sh_bcast5, &[5], vec![1, nci, nvh as i64, ci_, 1]);
        let beta_u = self.reshape(&beta_cm, &sh_bcast5); // [1,nc,nvh,C,1]
        let v_beta = self.mul_t(&value_cm, &beta_u);
        let k_beta = self.mul_t(&key_cm, &beta_u);

        // ---- 9. g_cs = cumsum(g_cm) over C, via MatMul against an
        // upper-triangular-inclusive ones matrix (see module doc: exact,
        // avoids relying on CumSum's op-version-sensitive axis-as-0-D-tensor
        // input). ----
        let cummat = "gdn_cummat".to_string();
        if !self.has(&cummat) {
            let mut m = vec![0f32; chunk * chunk];
            for j in 0..chunk {
                for i2 in j..chunk {
                    m[j * chunk + i2] = 1.0;
                }
            }
            self.f32(&cummat, &[1, 1, 1, ci_, ci_], m);
        }
        let sh_cs5in = "gdn_sh_cs5in".to_string();
        self.i64(&sh_cs5in, &[5], vec![1, nci, nvh as i64, 1, ci_]);
        let g_cs_row_in = self.reshape(&g_cm, &sh_cs5in); // [1,nc,nvh,1,C]
        let g_cs5 = self.matmul(&g_cs_row_in, &cummat); // [1,nc,nvh,1,C]
        let g_cs = self.reshape(&g_cs5, &sh_cm4b); // [1,nc,nvh,C]

        // ---- 10. decay_mask[i,j] = exp(g_cs_i - g_cs_j) for j<=i ----
        let sh_gcscol5 = "gdn_sh_gcscol5".to_string();
        self.i64(&sh_gcscol5, &[5], vec![1, nci, nvh as i64, 1, ci_]);
        let g_cs_row = self.reshape(&g_cs, &sh_bcast5); // [1,nc,nvh,C,1]
        let g_cs_col = self.reshape(&g_cs, &sh_gcscol5); // [1,nc,nvh,1,C]
        let diff = self.sub_t(&g_cs_row, &g_cs_col); // [1,nc,nvh,C,C]
        let tril = "gdn_tril".to_string();
        let stril = "gdn_stril".to_string();
        let ident = "gdn_ident".to_string();
        if !self.has(&tril) {
            let (mut tr, mut st, mut id) = (vec![0f32; chunk * chunk], vec![0f32; chunk * chunk], vec![0f32; chunk * chunk]);
            for i2 in 0..chunk {
                for j in 0..chunk {
                    if j <= i2 {
                        tr[i2 * chunk + j] = 1.0;
                    }
                    if j < i2 {
                        st[i2 * chunk + j] = 1.0;
                    }
                }
                id[i2 * chunk + i2] = 1.0;
            }
            self.f32(&tril, &[1, 1, 1, ci_, ci_], tr);
            self.f32(&stril, &[1, 1, 1, ci_, ci_], st);
            self.f32(&ident, &[1, 1, 1, ci_, ci_], id);
        }
        let exp_diff = self.unary("Exp", &diff);
        let decay_mask = self.mul_t(&exp_diff, &tril);

        // ---- 11. attn0 = -(k_beta @ key^T) * decay_mask, strictly lower ----
        let kt = self.transpose(&key_cm, &[0, 1, 2, 4, 3]); // [1,nc,nvh,khd,C]
        let raw_attn0_pos = self.matmul(&k_beta, &kt); // [1,nc,nvh,C,C]
        let raw_attn0 = self.unary("Neg", &raw_attn0_pos);
        let attn0_masked = self.mul_t(&raw_attn0, &decay_mask);
        let attn0 = self.mul_t(&attn0_masked, &stril);

        // ---- 12. UT-transform via the exact Neumann series ----
        let mut p_name = ident.clone();
        let mut t_name = ident.clone();
        for _ in 1..chunk {
            p_name = self.matmul(&attn0, &p_name);
            t_name = self.add_t(&t_name, &p_name);
        }
        let t_mat = t_name;

        // ---- 13/14. u, w (k_cumdecay) ----
        let u = self.matmul(&t_mat, &v_beta); // [1,nc,nvh,C,vhd]
        let exp_g_cs = self.unary("Exp", &g_cs); // [1,nc,nvh,C]
        let exp_g_cs_u = self.reshape(&exp_g_cs, &sh_bcast5); // [1,nc,nvh,C,1]
        let k_beta_decay = self.mul_t(&k_beta, &exp_g_cs_u);
        let w_mat = self.matmul(&t_mat, &k_beta_decay); // [1,nc,nvh,C,khd]

        // ---- intra_scores (state-independent, precomputed once) ----
        self.f32("gdn_scale", &[1], vec![1.0 / (khd as f32).sqrt()]);
        let raw_intra_pre = self.matmul(&query_cm, &kt); // [1,nc,nvh,C,C]
        let raw_intra = self.mul(&raw_intra_pre, "gdn_scale");
        let intra_scores = self.mul_t(&raw_intra, &decay_mask);

        // ---- 16. sequential across-chunk state recurrence, unrolled ----
        let sh_chunkk4 = "gdn_sh_chunkk4".to_string();
        self.i64(&sh_chunkk4, &[4], vec![1, nvh as i64, ci_, khd as i64]);
        let sh_chunkv4 = "gdn_sh_chunkv4".to_string();
        self.i64(&sh_chunkv4, &[4], vec![1, nvh as i64, ci_, vhd as i64]);
        let sh_chunkcc4 = "gdn_sh_chunkcc4".to_string();
        self.i64(&sh_chunkcc4, &[4], vec![1, nvh as i64, ci_, ci_]);
        let sh_chunkg3 = "gdn_sh_chunkg3".to_string();
        self.i64(&sh_chunkg3, &[3], vec![1, nvh as i64, ci_]);
        let sh_bc4 = "gdn_sh_bc4".to_string();
        self.i64(&sh_bc4, &[4], vec![1, nvh as i64, ci_, 1]);
        let sh_statedecay4 = "gdn_sh_statedecay4".to_string();
        self.i64(&sh_statedecay4, &[4], vec![1, nvh as i64, 1, 1]);
        let sh_outci5 = "gdn_sh_outci5".to_string();
        self.i64(&sh_outci5, &[5], vec![1, 1, nvh as i64, ci_, vhd as i64]);
        let ax1 = self.ci64(1);
        let ax_last3 = self.ci64(2);
        let lo_last = self.ci64((chunk - 1) as i64);
        let hi_last = self.ci64(chunk as i64);

        let state0 = "gdn_state0".to_string();
        self.f32(&state0, &[1, nvh as i64, khd as i64, vhd as i64], vec![0.0; nvh * khd * vhd]);

        let mut state = state0;
        let mut out_parts: Vec<String> = Vec::with_capacity(n_chunks);
        for ci in 0..n_chunks {
            let lo = self.ci64(ci as i64);
            let hi = self.ci64((ci + 1) as i64);

            let w_ci = self.slice_chunk(&w_mat, &lo, &hi, &ax1, &sh_chunkk4);
            let u_ci = self.slice_chunk(&u, &lo, &hi, &ax1, &sh_chunkv4);
            let intra_ci = self.slice_chunk(&intra_scores, &lo, &hi, &ax1, &sh_chunkcc4);
            let query_ci = self.slice_chunk(&query_cm, &lo, &hi, &ax1, &sh_chunkk4);
            let key_ci = self.slice_chunk(&key_cm, &lo, &hi, &ax1, &sh_chunkk4);
            let g_cs_ci = self.slice_chunk(&g_cs, &lo, &hi, &ax1, &sh_chunkg3);
            let exp_g_cs_ci = self.slice_chunk(&exp_g_cs, &lo, &hi, &ax1, &sh_chunkg3);

            let g_cs_last = self.slice(&g_cs_ci, &lo_last, &hi_last, &ax_last3); // [1,nvh,1]
            let decay_sub = self.sub_t(&g_cs_last, &g_cs_ci);
            let decay_scale = self.unary("Exp", &decay_sub); // [1,nvh,C]
            let decay_scale_u = self.reshape(&decay_scale, &sh_bc4); // [1,nvh,C,1]
            let decayed_k = self.mul_t(&key_ci, &decay_scale_u); // [1,nvh,C,khd]

            let exp_g_cs_ci_u = self.reshape(&exp_g_cs_ci, &sh_bc4);
            let q_pre = self.mul_t(&query_ci, &exp_g_cs_ci_u);
            let q_scaled = self.mul(&q_pre, "gdn_scale"); // [1,nvh,C,khd]

            let attn_inter = self.matmul(&q_scaled, &state); // [1,nvh,C,vhd]
            let v_prime = self.matmul(&w_ci, &state);
            let v_new = self.sub_t(&u_ci, &v_prime);
            let core2 = self.matmul(&intra_ci, &v_new);
            let out_ci = self.add_t(&attn_inter, &core2); // [1,nvh,C,vhd]
            let out_ci5 = self.reshape(&out_ci, &sh_outci5);
            out_parts.push(out_ci5);

            let state_decay_exp = self.unary("Exp", &g_cs_last);
            let state_decay_scalar = self.reshape(&state_decay_exp, &sh_statedecay4); // [1,nvh,1,1]
            let state_decayed = self.mul_t(&state, &state_decay_scalar);
            let decayed_k_t = self.transpose(&decayed_k, &[0, 1, 3, 2]); // [1,nvh,khd,C]
            let delta_state = self.matmul(&decayed_k_t, &v_new); // [1,nvh,khd,vhd]
            state = self.add_t(&state_decayed, &delta_state);
        }

        let mut out_cm = out_parts[0].clone();
        for part in &out_parts[1..] {
            out_cm = self.concat2(&out_cm, part, 1);
        }
        // out_cm: [1,n_chunks,nvh,C,vhd]

        // ---- 17. permute back to token-major ----
        let out_perm = self.transpose(&out_cm, &[0, 1, 3, 2, 4]); // [1,nc,C,nvh,vhd]
        let sh_outtok4 = "gdn_sh_outtok4".to_string();
        self.i64(&sh_outtok4, &[4], vec![1, ti, nvh as i64, vhd as i64]);
        let out_tok4 = self.reshape(&out_perm, &sh_outtok4); // [1,T,nvh,vhd]

        // ---- 18. gated RMSNorm (per head, over vhd; NO "+1" -- see module doc) ----
        let norm_w = p("norm.weight");
        let normed4 = self.rmsnorm(&out_tok4, &norm_w, w, vhd);
        let sh_normedflat3 = "gdn_sh_normedflat3".to_string();
        self.i64(&sh_normedflat3, &[3], vec![1, ti, value_dim as i64]);
        let normed_flat = self.reshape(&normed4, &sh_normedflat3);
        let z_silu = self.silu(&z);
        let gated = self.mul_t(&normed_flat, &z_silu);

        // ---- 19. out_proj ----
        let out_proj_w = p("out_proj.weight");
        self.linear(&gated, &out_proj_w, w, d, value_dim)
    }

    // ---- GQA (full-attention) layer ---------------------------------------

    /// One GQA mixer layer (`qwen35moe::model::layer_gqa_fwd`'s reference):
    /// doubled `q_proj` (value+gate, per-head split), QK-RMSNorm, partial
    /// M-RoPE (degenerate to plain RoPE, text-only — see [`Self::rope_partial`]),
    /// GQA attention, sigmoid output gate, `o_proj`. Returns `mix_out:[1,T,d]`.
    fn gqa_layer(&mut self, l: usize, xn1: &str, w: &dyn WeightSource, c: &Qwen35Config, t: usize) -> String {
        let d = c.d_model as usize;
        let nh = c.n_heads as usize;
        let nkv = c.n_kv_heads as usize;
        let hd = c.head_dim as usize;
        let group = c.group() as usize;
        let (qpd, qd, kvd) = (c.q_proj_dim() as usize, c.q_dim() as usize, c.kv_dim() as usize);
        let rot = c.rotary_dim() as usize;
        let ti = t as i64;
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        let q_proj_w = p("q_proj.weight");
        let k_proj_w = p("k_proj.weight");
        let v_proj_w = p("v_proj.weight");
        let q_full = self.linear(xn1, &q_proj_w, w, qpd, d); // [1,T,qpd]
        let k = self.linear(xn1, &k_proj_w, w, kvd, d);
        let v = self.linear(xn1, &v_proj_w, w, kvd, d);

        let sh_q42h = "gqa_sh_q42h".to_string();
        self.i64(&sh_q42h, &[4], vec![1, ti, nh as i64, 2 * hd as i64]);
        let q4 = self.reshape(&q_full, &sh_q42h);
        let lo0 = self.ci64(0);
        let hi_hd = self.ci64(hd as i64);
        let hi_2hd = self.ci64(2 * hd as i64);
        let ax3 = self.ci64(3);
        let q_value = self.slice(&q4, &lo0, &hi_hd, &ax3); // [1,T,nh,hd]
        let q_gate = self.slice(&q4, &hi_hd, &hi_2hd, &ax3); // [1,T,nh,hd]

        let sh_kv4 = "gqa_sh_kv4".to_string();
        self.i64(&sh_kv4, &[4], vec![1, ti, nkv as i64, hd as i64]);
        let k4 = self.reshape(&k, &sh_kv4);
        let v4 = self.reshape(&v, &sh_kv4);

        let q_norm_w = p("q_norm.weight");
        let k_norm_w = p("k_norm.weight");
        let q_normed = self.rmsnorm(&q_value, &q_norm_w, w, hd);
        let k_normed = self.rmsnorm(&k4, &k_norm_w, w, hd);

        let q_rot = self.rope_partial(&q_normed, hd, rot, t, c.rope_theta);
        let k_rot = self.rope_partial(&k_normed, hd, rot, t, c.rope_theta);

        // Repeat kv heads BEFORE transposing to head-first (token-major
        // layout, matching `repeat_heads`'s own `[1,rows,heads,dim]` contract
        // — reused unchanged from GDN's own head-repeat, see that helper's doc).
        let k_exp = self.repeat_heads(&k_rot, "gqa_k", t, nkv, group, hd); // [1,T,nh,hd]
        let v_exp = self.repeat_heads(&v4, "gqa_v", t, nkv, group, hd);

        let qt = self.transpose(&q_rot, &[0, 2, 1, 3]); // [1,nh,T,hd]
        let ke = self.transpose(&k_exp, &[0, 2, 1, 3]);
        let ve = self.transpose(&v_exp, &[0, 2, 1, 3]);

        self.f32("gqa_scale", &[1], vec![1.0 / (hd as f32).sqrt()]);
        let mask_name = "gqa_causal_mask".to_string();
        if !self.has(&mask_name) {
            let mut mask = vec![0f32; t * t];
            for i2 in 0..t {
                for j in 0..t {
                    if j > i2 {
                        mask[i2 * t + j] = -1.0e9;
                    }
                }
            }
            self.f32(&mask_name, &[1, 1, ti, ti], mask);
        }

        let ktt = self.transpose(&ke, &[0, 1, 3, 2]); // [1,nh,hd,T]
        let scores_raw = self.matmul(&qt, &ktt);
        let scores_scaled = self.mul(&scores_raw, "gqa_scale");
        let scores = self.add(&scores_scaled, &mask_name);
        let probs = self.softmax(&scores, -1);
        let ctx = self.matmul(&probs, &ve); // [1,nh,T,hd]
        let ctx = self.transpose(&ctx, &[0, 2, 1, 3]); // [1,T,nh,hd]

        let sh_flat3 = "gqa_sh_flat3".to_string();
        self.i64(&sh_flat3, &[3], vec![1, ti, qd as i64]);
        let ctx_flat = self.reshape(&ctx, &sh_flat3);
        let gate_flat = self.reshape(&q_gate, &sh_flat3);
        let gate = self.unary("Sigmoid", &gate_flat);
        let ctx_gated = self.mul_t(&ctx_flat, &gate);

        let o_proj_w = p("o_proj.weight");
        self.linear(&ctx_gated, &o_proj_w, w, d, qd)
    }

    // ---- MoE sublayer, universal every layer -------------------------------

    /// Sparse-gather top-8-of-256 MoE + sigmoid-gated shared expert (see
    /// module doc for the router-math cross-check against `model::moe`).
    /// Returns `moe_out:[1,T,d]`.
    fn moe_layer(&mut self, l: usize, xmid: &str, w: &dyn WeightSource, c: &Qwen35Config, t: usize) -> String {
        let d = c.d_model as usize;
        let e = c.n_experts as usize;
        let ff = c.moe_intermediate_size as usize;
        let sff = c.shared_expert_intermediate_size as usize;
        let k = c.top_k as usize;
        let ti = t as i64;
        let p = |s: &str| format!("blocks.{l}.{s}");

        let ln2_w = p("ln2.weight");
        let xn2 = self.rmsnorm(xmid, &ln2_w, w, d);
        let router_w = p("mlp.router.weight");
        let logits = self.linear(&xn2, &router_w, w, e, d); // [1,T,E]

        // Router: EXACT `router_gate.wgsl` math -- softmax over all E experts,
        // top_k by softmax probability, renormalise the kept weights (see
        // module doc's router cross-check).
        let probs = self.softmax(&logits, -1);
        self.i64("moe_topk_k", &[1], vec![k as i64]);
        let vals = self.tmp("moe_tkv");
        let idx = self.tmp("moe_tki");
        self.g.add(
            Node::new("TopK", &[&probs, "moe_topk_k"], &[&vals, &idx])
                .attr_int("axis", 2)
                .attr_int("largest", 1)
                .attr_int("sorted", 1),
        );
        let denom = self.reduce_sum(&vals, 2, true); // [1,T,1]
        let gate_w = self.tmp("moe_gw");
        self.node("Div", &[&vals, &denom], &gate_w); // [1,T,k]

        // Stack every expert's weight into one [E,in,out] Gather-indexable
        // initializer (see module doc's sub-problem 2).
        let gate_stack_name = format!("moe_gs_{l}");
        let gate_stack = self.expert_stack(&gate_stack_name, w, e, ff, d, |ei| format!("blocks.{l}.mlp.experts.{ei}.gate.weight"));
        let up_stack_name = format!("moe_us_{l}");
        let up_stack = self.expert_stack(&up_stack_name, w, e, ff, d, |ei| format!("blocks.{l}.mlp.experts.{ei}.up.weight"));
        let down_stack_name = format!("moe_ds_{l}");
        let down_stack = self.expert_stack(&down_stack_name, w, e, d, ff, |ei| format!("blocks.{l}.mlp.experts.{ei}.down.weight"));

        let gk = self.gather(&gate_stack, &idx, 0, "moe_gk"); // [1,T,k,d,ff]
        let gu = self.gather(&up_stack, &idx, 0, "moe_gu");
        let gd = self.gather(&down_stack, &idx, 0, "moe_gd"); // [1,T,k,ff,d]

        let sh_x5 = "moe_sh_x5".to_string();
        self.i64(&sh_x5, &[5], vec![1, ti, 1, 1, d as i64]);
        let x5 = self.reshape(&xn2, &sh_x5);
        let gate_pre = self.matmul(&x5, &gk); // [1,T,k,1,ff]
        let up = self.matmul(&x5, &gu);
        let silu_g = self.silu(&gate_pre);
        let h = self.mul_t(&silu_g, &up); // [1,T,k,1,ff]
        let expert_out5 = self.matmul(&h, &gd); // [1,T,k,1,d]

        let sh_out4 = format!("moe_sh_out4_{k}");
        self.i64(&sh_out4, &[4], vec![1, ti, k as i64, d as i64]);
        let expert_out4 = self.reshape(&expert_out5, &sh_out4); // [1,T,k,d]
        let sh_gw4 = format!("moe_sh_gw4_{k}");
        self.i64(&sh_gw4, &[4], vec![1, ti, k as i64, 1]);
        let gate_w4 = self.reshape(&gate_w, &sh_gw4);
        let weighted = self.mul_t(&expert_out4, &gate_w4);
        let routed_sum = self.reduce_sum(&weighted, 2, false); // [1,T,d]

        // Sigmoid-gated shared expert (dense, always active) -- matches
        // `model::moe::shared_expert_fwd`'s `Some(shared_gate_w)` arm exactly:
        // `expert_output + sigmoid(shared_expert_gate(x)) * shared_expert(x)`.
        let sh_gate_w = p("mlp.shared_expert.gate.weight");
        let sh_up_w = p("mlp.shared_expert.up.weight");
        let sh_down_w = p("mlp.shared_expert.down.weight");
        let sh_gate_gate_w = p("mlp.shared_expert_gate.weight");
        let sh_gate = self.linear(&xn2, &sh_gate_w, w, sff, d);
        let sh_up = self.linear(&xn2, &sh_up_w, w, sff, d);
        let sh_silu = self.silu(&sh_gate);
        let sh_h = self.mul_t(&sh_silu, &sh_up);
        let sh_out = self.linear(&sh_h, &sh_down_w, w, d, sff);
        let sh_gate_logit = self.linear(&xn2, &sh_gate_gate_w, w, 1, d); // [1,T,1]
        let sh_gate_scalar = self.unary("Sigmoid", &sh_gate_logit);
        let sh_scaled = self.mul_t(&sh_out, &sh_gate_scalar);

        self.add_t(&routed_sum, &sh_scaled)
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]` (brain's
/// `[out,in]` weight layout -> ONNX `[in,out]`). Each topology file keeps its
/// own copy of this (see `qwen_topology.rs`/`glm_topology.rs`) rather than
/// sharing one — small enough that duplicating it is cheaper than a new
/// shared dependency edge.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
