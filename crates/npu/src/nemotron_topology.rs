// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Nemotron 3.5 ASR **FastConformer encoder** as an OpenVINO-compilable ONNX graph
//! (mel → pooler `[T', 640]`), built for a FIXED sequence length so it compiles to a
//! static NPU graph. The RNN-T decode stays on host (m=1 steps).
//!
//! This is brain's hardest NPU export: it lands incrementally, each stage
//! parity-gated against the dumped HF activations (`testdata/asr/golden/nemotron/`):
//!   1. depthwise-separable causal subsampling (×8) + linear     ← this pass
//!   2. macaron Conformer blocks (rel-pos attention + GLU conv)  (next)
//!   3. prompt + encoder projectors
//!
//! Design mirrors the host reference (`nemotron::reference`) and the device encoder
//! (`nemotron::encoder`) op-for-op so the same goldens gate all three. Weights arrive
//! through the shared `crate::topology::WeightSource` (a name→f32 map).

use onnx::{GraphBuilder, Node};

use crate::topology::WeightSource;

/// Config subset the encoder graph needs (mirrors `nemotron::NemotronConfig`).
#[derive(Clone, Copy, Debug)]
pub struct NemotronTopo {
    pub num_mel_bins: u32,
    pub hidden: u32,
    pub subsampling_channels: u32,
    pub subsampling_kernel: u32,
    pub subsampling_stride: u32,
    pub subsampling_stages: u32,
    // Conformer stack (stage 2) + projectors (stage 3).
    pub n_layers: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub conv_kernel: u32,
    /// chunked_limited band: left context (`sliding_window - 1`) and right lookahead.
    pub left_ctx: u32,
    pub right_ctx: u32,
    pub ln_eps: f32,
    pub num_prompts: u32,
    pub prompt_intermediate: u32,
    pub decoder_hidden: u32,
}

impl Default for NemotronTopo {
    fn default() -> Self {
        NemotronTopo {
            num_mel_bins: 128,
            hidden: 1024,
            subsampling_channels: 256,
            subsampling_kernel: 3,
            subsampling_stride: 2,
            subsampling_stages: 3,
            n_layers: 24,
            n_heads: 8,
            head_dim: 128,
            intermediate: 4096,
            conv_kernel: 9,
            left_ctx: 56, // sliding_window(57) - 1
            right_ctx: 3, // default_lookahead
            ln_eps: 1e-5,
            num_prompts: 128,
            prompt_intermediate: 2048,
            decoder_hidden: 640,
        }
    }
}

impl NemotronTopo {
    /// Length after one causal stride-2 stage: `(len + (k-1)+(s-1) - k)/s + 1`.
    fn stage_len(&self, len: u32) -> u32 {
        let (k, s) = (self.subsampling_kernel, self.subsampling_stride);
        (len + (k - 1) + (s - 1) - k) / s + 1
    }
    /// Subsampled time length after the full stack.
    pub fn subsampled_len(&self, mel_valid: u32) -> u32 {
        let mut l = mel_valid;
        for _ in 0..self.subsampling_stages {
            l = self.stage_len(l);
        }
        l
    }
    /// Output frequency bins after the stack.
    pub fn out_freq(&self) -> u32 {
        let mut f = self.num_mel_bins;
        for _ in 0..self.subsampling_stages {
            f = self.stage_len(f);
        }
        f
    }
}

/// A named tensor with its current spatial dims, threaded through the conv stack.
struct Feat {
    name: String,
    c: u32,
    t: u32,
    f: u32,
}

/// Emit a causal Conv2d (`(k-1,s-1)` asymmetric pad on both axes, matching NeMo) with
/// a per-channel bias. When `mask_relu` is set, also apply the time-mask (zero frames
/// `>= stage_len(valid)`) and a ReLU — the depthwise-separable stage masks+ReLUs only
/// AFTER the pointwise conv, so its depthwise call passes `mask_relu=false` (a raw
/// conv). Weight is `[cout, cin/groups, k, k]`, bias `[cout]`, verbatim.
#[allow(clippy::too_many_arguments)]
fn causal_conv(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, x: &Feat, cout: u32, wname: &str, bname: &str, groups: u32, valid: u32, mask_relu: bool, tag: &str) -> Feat {
    let (k, s) = (topo.subsampling_kernel, topo.subsampling_stride);
    let kw = w.get(wname);
    let bw = w.get(bname);
    let wn = format!("{tag}.w");
    let bn = format!("{tag}.b");
    g.init_f32(&wn, &[cout as i64, (x.c / groups) as i64, k as i64, k as i64], kw);
    g.init_f32(&bn, &[cout as i64], bw);
    let out = format!("{tag}.conv");
    // ONNX pads order for 2-D: [t_begin, f_begin, t_end, f_end] = causal (k-1) front, (s-1) back.
    let (pb, pe) = ((k - 1) as i64, (s - 1) as i64);
    g.add(
        Node::new("Conv", &[&x.name, &wn, &bn], &[&out])
            .name(&format!("{tag}.conv"))
            .attr_ints("kernel_shape", &[k as i64, k as i64])
            .attr_ints("strides", &[s as i64, s as i64])
            .attr_ints("pads", &[pb, pb, pe, pe])
            .attr_int("group", groups as i64),
    );
    let (to, fo) = (topo.stage_len(x.t), topo.stage_len(x.f));
    if !mask_relu {
        return Feat { name: out, c: cout, t: to, f: fo };
    }
    let vout = topo.stage_len(valid);
    let masked = mask_time(g, &out, cout, to, fo, vout, tag);
    let relu = format!("{tag}.relu");
    g.add(Node::new("Relu", &[&masked], &[&relu]).name(&format!("{tag}.relu")));
    Feat { name: relu, c: cout, t: to, f: fo }
}

/// Zero time frames `>= valid` in an NCHW `[1,C,T,F]` tensor (NeMo
/// `_mask_subsampled_frames`) by multiplying with a constant `[1,1,T,1]` 0/1 mask.
/// A no-op (all-ones) when `valid >= t`, so a full window costs nothing.
fn mask_time(g: &mut GraphBuilder, x: &str, _c: u32, t: u32, _f: u32, valid: u32, tag: &str) -> String {
    if valid >= t {
        return x.to_string();
    }
    let mask: Vec<f32> = (0..t).map(|i| if i < valid { 1.0 } else { 0.0 }).collect();
    let mn = format!("{tag}.tmask");
    g.init_f32(&mn, &[1, 1, t as i64, 1], mask);
    let out = format!("{tag}.masked");
    g.add(Node::new("Mul", &[x, &mn], &[&out]).name(&format!("{tag}.mask")));
    out
}

/// Build the **subsampling** stage: mel `[1,1,T,num_mel]` → `[T', hidden]` (named
/// `out_name`). `mel_t` is the input time length, `mel_valid` the real frames.
pub fn build_subsampling(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, mel_t: u32, mel_valid: u32, input_name: &str, out_name: &str) {
    let ch = topo.subsampling_channels;
    // stem: conv_in (1 -> ch)
    let x = Feat { name: input_name.to_string(), c: 1, t: mel_t, f: topo.num_mel_bins };
    let mut cur = causal_conv(g, topo, w, &x, ch, "encoder.subsampling.conv_in.weight", "encoder.subsampling.conv_in.bias", 1, mel_valid, true, "sub.stem");
    let mut vlen = topo.stage_len(mel_valid);

    // depthwise-separable stages
    for i in 0..topo.subsampling_stages - 1 {
        let dw = causal_conv(
            g,
            topo,
            w,
            &cur,
            ch,
            &format!("encoder.subsampling.layers.{i}.depthwise_conv.weight"),
            &format!("encoder.subsampling.layers.{i}.depthwise_conv.bias"),
            ch,
            vlen,
            false, // depthwise: no mask/relu — they come after the pointwise conv
            &format!("sub.dw{i}"),
        );
        // pointwise 1x1 (stride 1, no pad); the reference masks+relus after pointwise.
        let pw = pointwise(g, w, &dw, ch, &format!("encoder.subsampling.layers.{i}.pointwise_conv.weight"), &format!("encoder.subsampling.layers.{i}.pointwise_conv.bias"), topo.stage_len(vlen), &format!("sub.pw{i}"));
        vlen = topo.stage_len(vlen);
        cur = pw;
    }

    // reshape [1,C,T',F'] -> [T', C*F'] : transpose to [1,T',C,F'] then reshape.
    let (tt, ff) = (cur.t, cur.f);
    let flat = ch * ff;
    let tp = format!("sub.perm");
    g.add(Node::new("Transpose", &[&cur.name], &[&tp]).name("sub.perm").attr_ints("perm", &[0, 2, 1, 3]));
    let shp = "sub.flatshape";
    g.init_i64(shp, &[2], vec![tt as i64, flat as i64]);
    let flatn = "sub.flat";
    g.add(Node::new("Reshape", &[&tp, shp], &[flatn]).name("sub.reshape"));

    // linear [T', flat] @ W^T [flat, hidden] + bias -> [T', hidden]
    linear(g, w, flatn, "encoder.subsampling.linear.weight", "encoder.subsampling.linear.bias", topo.hidden, flat, out_name, "sub.lin");
}

/// 1×1 pointwise Conv2d (dense, stride 1) + bias, then mask+relu.
fn pointwise(g: &mut GraphBuilder, w: &dyn WeightSource, x: &Feat, cout: u32, wname: &str, bname: &str, valid: u32, tag: &str) -> Feat {
    let wn = format!("{tag}.w");
    let bn = format!("{tag}.b");
    g.init_f32(&wn, &[cout as i64, x.c as i64, 1, 1], w.get(wname));
    g.init_f32(&bn, &[cout as i64], w.get(bname));
    let out = format!("{tag}.conv");
    g.add(
        Node::new("Conv", &[&x.name, &wn, &bn], &[&out])
            .name(&format!("{tag}.conv"))
            .attr_ints("kernel_shape", &[1, 1])
            .attr_ints("strides", &[1, 1])
            .attr_ints("pads", &[0, 0, 0, 0])
            .attr_int("group", 1),
    );
    let masked = mask_time(g, &out, cout, x.t, x.f, valid, tag);
    let relu = format!("{tag}.relu");
    g.add(Node::new("Relu", &[&masked], &[&relu]).name(&format!("{tag}.relu")));
    Feat { name: relu, c: cout, t: x.t, f: x.f }
}

/// A linear `x[m, in] @ W^T[in, out] + b -> [m, out]` (weight stored `[out, in]`,
/// transposed into the graph as `[in, out]`).
fn linear(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, wname: &str, bname: &str, out: u32, inn: u32, out_name: &str, tag: &str) {
    let wt = transpose_2d(&w.get(wname), out as usize, inn as usize);
    let wn = format!("{tag}.wT");
    g.init_f32(&wn, &[inn as i64, out as i64], wt);
    let mm = format!("{tag}.mm");
    g.add(Node::new("MatMul", &[x, &wn], &[&mm]).name(&format!("{tag}.mm")));
    let bn = format!("{tag}.b");
    g.init_f32(&bn, &[out as i64], w.get(bname));
    g.add(Node::new("Add", &[&mm, &bn], &[out_name]).name(&format!("{tag}.bias")));
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose_2d(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

// ===================== stage 2: Conformer blocks =====================
//
// Op-for-op with `nemotron::reference::conformer_block`. Everything that is
// constant for a fixed `(t, valid)` — the relative-position ladder `pe`, the
// per-layer `rel_k = pe @ relative_k_proj^T`, and the chunked_limited+padding
// attention mask — is precomputed in Rust and baked as an initializer, so the
// graph carries no position/mask arithmetic.

const NEG_MASK: f32 = -1.0e9;

/// Relative-position rows `[positions, C]`: interleaved sin/cos,
/// `inv_freq[i] = 10000^(-2i/C)` — mirrors `nemotron::reference::rel_pos_rows`.
fn rel_pos_encoding_host(t: usize, c: usize) -> Vec<f32> {
    let half = c / 2;
    let inv: Vec<f32> = (0..half).map(|i| (10000f32).powf(-(2.0 * i as f32) / c as f32)).collect();
    // positions [t-1 .. -(t-1)], length 2t-1
    let l = 2 * t - 1;
    let mut pe = vec![0.0f32; l * c];
    for idx in 0..l {
        let pos = (t as i64 - 1 - idx as i64) as f32;
        for i in 0..half {
            let f = pos * inv[i];
            pe[idx * c + 2 * i] = f.sin();
            pe[idx * c + 2 * i + 1] = f.cos();
        }
    }
    pe
}

/// Host `a[m,k] @ w[n,k]^T -> [m,n]` (the `matmul_nt` convention: weight `[out,in]`).
fn matmul_nt_host(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for o in 0..n {
            let mut acc = 0.0f32;
            for d in 0..k {
                acc += a[i * k + d] * w[o * k + d];
            }
            out[i * n + o] = acc;
        }
    }
    out
}

/// chunked_limited validity (mirrors `reference::banded_ok`).
fn banded_ok(i: usize, j: usize, left: usize, right: usize) -> bool {
    let chunk = right + 1;
    let left_chunks = left / chunk;
    let (qc, kc) = (i / chunk, j / chunk);
    qc >= kc && qc - kc <= left_chunks
}

/// Additive attention mask `[t, t]`: 0 where key `j` is a valid, in-band key for
/// query `i`, else a large negative (so softmax zeroes it) — the padding mask
/// (`j < valid`) AND the chunked_limited band, both constant for `(t, valid)`.
fn attn_mask_host(topo: &NemotronTopo, t: usize, valid: usize) -> Vec<f32> {
    let (left, right) = (topo.left_ctx as usize, topo.right_ctx as usize);
    let mut m = vec![NEG_MASK; t * t];
    for i in 0..t {
        for j in 0..t {
            if j < valid && banded_ok(i, j, left, right) {
                m[i * t + j] = 0.0;
            }
        }
    }
    m
}

/// A linear `x @ W^T` with NO bias, returning the output tensor name. Weight
/// stored `[out, in]`, transposed into the graph as `[in, out]`.
fn linear_nb(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, wname: &str, out: u32, inn: u32, tag: &str) -> String {
    let wt = transpose_2d(&w.get(wname), out as usize, inn as usize);
    let wn = format!("{tag}.wT");
    g.init_f32(&wn, &[inn as i64, out as i64], wt);
    let mm = format!("{tag}.mm");
    g.add(Node::new("MatMul", &[x, &wn], &[&mm]).name(&format!("{tag}.mm")));
    mm
}

fn reshape(g: &mut GraphBuilder, x: &str, dims: &[i64], tag: &str) -> String {
    let shp = format!("{tag}.shape");
    g.init_i64(&shp, &[dims.len() as i64], dims.iter().copied().collect());
    let out = format!("{tag}.rs");
    g.add(Node::new("Reshape", &[x, &shp], &[&out]).name(&format!("{tag}.reshape")));
    out
}

fn transpose(g: &mut GraphBuilder, x: &str, perm: &[i64], tag: &str) -> String {
    let out = format!("{tag}.tp");
    g.add(Node::new("Transpose", &[x], &[&out]).name(&format!("{tag}.transpose")).attr_ints("perm", perm));
    out
}

fn add_t(g: &mut GraphBuilder, a: &str, b: &str, tag: &str) -> String {
    let out = format!("{tag}.add");
    g.add(Node::new("Add", &[a, b], &[&out]).name(&format!("{tag}.add")));
    out
}

/// LayerNorm over the last axis (opset-13 primitives), `(x-µ)·rsqrt(var+eps)·γ + β`.
fn layernorm_onnx(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, gname: &str, bname: &str, c: u32, eps: f32, tag: &str) -> String {
    let mean = format!("{tag}.mean");
    g.add(Node::new("ReduceMean", &[x], &[&mean]).name(&format!("{tag}.mean")).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
    let xc = format!("{tag}.xc");
    g.add(Node::new("Sub", &[x, &mean], &[&xc]).name(&format!("{tag}.sub")));
    let sq = format!("{tag}.sq");
    g.add(Node::new("Mul", &[&xc, &xc], &[&sq]).name(&format!("{tag}.sq")));
    let var = format!("{tag}.var");
    g.add(Node::new("ReduceMean", &[&sq], &[&var]).name(&format!("{tag}.var")).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
    let epsn = format!("{tag}.eps");
    g.init_f32(&epsn, &[1], vec![eps]);
    let vare = format!("{tag}.vare");
    g.add(Node::new("Add", &[&var, &epsn], &[&vare]).name(&format!("{tag}.vare")));
    let std = format!("{tag}.std");
    g.add(Node::new("Sqrt", &[&vare], &[&std]).name(&format!("{tag}.std")));
    let norm = format!("{tag}.norm");
    g.add(Node::new("Div", &[&xc, &std], &[&norm]).name(&format!("{tag}.div")));
    let gn = format!("{tag}.g");
    g.init_f32(&gn, &[c as i64], w.get(gname));
    let scaled = format!("{tag}.scaled");
    g.add(Node::new("Mul", &[&norm, &gn], &[&scaled]).name(&format!("{tag}.scale")));
    let bn = format!("{tag}.b");
    g.init_f32(&bn, &[c as i64], w.get(bname));
    let out = format!("{tag}.ln");
    g.add(Node::new("Add", &[&scaled, &bn], &[&out]).name(&format!("{tag}.beta")));
    out
}

/// Macaron feed-forward `Linear(c→ffn) → SiLU → Linear(ffn→c)`, no bias.
fn feed_forward_onnx(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, l1: &str, l2: &str, c: u32, ffn: u32, tag: &str) -> String {
    let h = linear_nb(g, w, x, l1, ffn, c, &format!("{tag}.l1"));
    let sig = format!("{tag}.sig");
    g.add(Node::new("Sigmoid", &[&h], &[&sig]).name(&format!("{tag}.sig")));
    let act = format!("{tag}.silu");
    g.add(Node::new("Mul", &[&h, &sig], &[&act]).name(&format!("{tag}.silu")));
    linear_nb(g, w, &act, l2, c, ffn, &format!("{tag}.l2"))
}

/// rel_shift on `[heads, t, l]` (Transformer-XL diagonal shift), then keep the
/// first `t` key columns → `[heads, t, t]`. Mirrors `kernels::rel_shift_ref`:
/// left-pad the last axis by 1, flatten, drop the first `t` per head, re-view.
fn rel_shift_onnx(g: &mut GraphBuilder, x: &str, heads: u32, t: u32, l: u32, tag: &str) -> String {
    // pad last axis left by 1 -> [heads, t, l+1]
    let padn = format!("{tag}.pads");
    g.init_i64(&padn, &[6], vec![0, 0, 1, 0, 0, 0]);
    let zero = format!("{tag}.zero");
    g.init_f32(&zero, &[1], vec![0.0]);
    let padded = format!("{tag}.padded");
    g.add(Node::new("Pad", &[x, &padn, &zero], &[&padded]).name(&format!("{tag}.pad")).attr_str("mode", "constant"));
    // flatten [heads, t*(l+1)]
    let flat = reshape(g, &padded, &[heads as i64, (t * (l + 1)) as i64], &format!("{tag}.flat"));
    // drop first t per head: slice axis1 [t : t + t*l]
    let starts = format!("{tag}.st");
    g.init_i64(&starts, &[1], vec![t as i64]);
    let ends = format!("{tag}.en");
    g.init_i64(&ends, &[1], vec![(t + t * l) as i64]);
    let axes = format!("{tag}.ax");
    g.init_i64(&axes, &[1], vec![1]);
    let sliced = format!("{tag}.sliced");
    g.add(Node::new("Slice", &[&flat, &starts, &ends, &axes], &[&sliced]).name(&format!("{tag}.slice")));
    // re-view [heads, t, l], then keep first t key columns -> [heads, t, t]
    let view = reshape(g, &sliced, &[heads as i64, t as i64, l as i64], &format!("{tag}.view"));
    let st2 = format!("{tag}.st2");
    g.init_i64(&st2, &[1], vec![0]);
    let en2 = format!("{tag}.en2");
    g.init_i64(&en2, &[1], vec![t as i64]);
    let ax2 = format!("{tag}.ax2");
    g.init_i64(&ax2, &[1], vec![2]);
    let out = format!("{tag}.bd");
    g.add(Node::new("Slice", &[&view, &st2, &en2, &ax2], &[&out]).name(&format!("{tag}.keepT")));
    out
}

/// Relative-position multi-head self-attention (Transformer-XL) under the
/// chunked_limited + padding mask. `pe` is the precomputed `[2t-1, C]` ladder;
/// `mask` the precomputed `[1,t,t]` additive mask name.
#[allow(clippy::too_many_arguments)]
fn rel_pos_attention_onnx(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, x: &str, pe: &[f32], mask: &str, prefix: &str, t: u32, tag: &str) -> String {
    let (c, heads, hd) = (topo.hidden, topo.n_heads, topo.head_dim);
    let l = 2 * t - 1;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let p = |n: &str| format!("{prefix}.{n}");

    // q,k,v = x @ {q,k,v}_proj^T  -> [t, C] -> [heads, t, hd]
    let mk_heads = |g: &mut GraphBuilder, name: &str, sub: &str| -> String {
        let r = reshape(g, name, &[t as i64, heads as i64, hd as i64], &format!("{tag}.{sub}.r"));
        transpose(g, &r, &[1, 0, 2], &format!("{tag}.{sub}.t"))
    };
    let q = linear_nb(g, w, x, &p("q_proj.weight"), c, c, &format!("{tag}.q"));
    let k = linear_nb(g, w, x, &p("k_proj.weight"), c, c, &format!("{tag}.k"));
    let v = linear_nb(g, w, x, &p("v_proj.weight"), c, c, &format!("{tag}.v"));
    let qh = mk_heads(g, &q, "qh");
    let kh = mk_heads(g, &k, "kh");
    let vh = mk_heads(g, &v, "vh");

    // bias_u/bias_v [heads*hd] -> [heads,1,hd]
    let bu = format!("{tag}.bu");
    g.init_f32(&bu, &[heads as i64, 1, hd as i64], w.get(&p("bias_u")));
    let bv = format!("{tag}.bv");
    g.init_f32(&bv, &[heads as i64, 1, hd as i64], w.get(&p("bias_v")));
    let q_bu = add_t(g, &qh, &bu, &format!("{tag}.qbu"));
    let q_bv = add_t(g, &qh, &bv, &format!("{tag}.qbv"));

    // scores_ac = q_bu @ k^T  -> [heads, t, t]
    let kt = transpose(g, &kh, &[0, 2, 1], &format!("{tag}.kt"));
    let ac = format!("{tag}.ac");
    g.add(Node::new("MatMul", &[&q_bu, &kt], &[&ac]).name(&format!("{tag}.ac")));

    // rel_k = pe @ relative_k_proj^T  (BUILD-TIME constant) -> [L, C] -> [heads, L, hd]
    let relk = matmul_nt_host(pe, &w.get(&p("relative_k_proj.weight")), l as usize, c as usize, c as usize);
    let relkn = format!("{tag}.relk");
    g.init_f32(&relkn, &[l as i64, c as i64], relk);
    let relk_h = {
        let r = reshape(g, &relkn, &[l as i64, heads as i64, hd as i64], &format!("{tag}.relk.r"));
        transpose(g, &r, &[1, 0, 2], &format!("{tag}.relk.t")) // [heads, L, hd]
    };
    let relk_t = transpose(g, &relk_h, &[0, 2, 1], &format!("{tag}.relk.tt")); // [heads, hd, L]
    // bd_raw = q_bv @ rel_k^T -> [heads, t, L]
    let bd_raw = format!("{tag}.bdraw");
    g.add(Node::new("MatMul", &[&q_bv, &relk_t], &[&bd_raw]).name(&format!("{tag}.bdraw")));
    let bd = rel_shift_onnx(g, &bd_raw, heads, t, l, &format!("{tag}.rs")); // [heads, t, t]

    // scores = (ac + bd) * scale + mask ; softmax
    let sum = add_t(g, &ac, &bd, &format!("{tag}.sum"));
    let scalen = format!("{tag}.scalek");
    g.init_f32(&scalen, &[1], vec![scale]);
    let scaled = format!("{tag}.scaled");
    g.add(Node::new("Mul", &[&sum, &scalen], &[&scaled]).name(&format!("{tag}.scaledscore")));
    let masked = add_t(g, &scaled, mask, &format!("{tag}.msk"));
    let probs = format!("{tag}.probs");
    g.add(Node::new("Softmax", &[&masked], &[&probs]).name(&format!("{tag}.softmax")).attr_int("axis", -1));

    // ctx = probs @ v -> [heads, t, hd] -> [t, C]
    let ctxh = format!("{tag}.ctxh");
    g.add(Node::new("MatMul", &[&probs, &vh], &[&ctxh]).name(&format!("{tag}.ctx")));
    let ctx_tp = transpose(g, &ctxh, &[1, 0, 2], &format!("{tag}.ctxtp")); // [t, heads, hd]
    let ctx = reshape(g, &ctx_tp, &[t as i64, c as i64], &format!("{tag}.ctxflat"));
    // o_proj (no bias)
    linear_nb(g, w, &ctx, &p("o_proj.weight"), c, c, &format!("{tag}.o"))
}

/// Conformer convolution module: pointwise_conv1(→2C) → GLU → causal depthwise
/// conv1d(k) → LayerNorm → SiLU → pointwise_conv2. `x` is pre-normalised `[t, C]`.
fn conv_module_onnx(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, x: &str, prefix: &str, t: u32, tag: &str) -> String {
    let (c, k) = (topo.hidden, topo.conv_kernel);
    let p = |n: &str| format!("{prefix}.{n}");
    // pointwise_conv1 [2C,C,1] == linear [2C,C]  -> [t, 2C]
    let pc1 = linear_nb(g, w, x, &p("pointwise_conv1.weight"), 2 * c, c, &format!("{tag}.pc1"));
    // GLU over channel: a = pc1[:, :C], b = pc1[:, C:] ; a * sigmoid(b)
    let a = format!("{tag}.glu.a");
    let (s0, e0, ax) = (format!("{tag}.glu.s0"), format!("{tag}.glu.e0"), format!("{tag}.glu.ax"));
    g.init_i64(&s0, &[1], vec![0]);
    g.init_i64(&e0, &[1], vec![c as i64]);
    g.init_i64(&ax, &[1], vec![1]);
    g.add(Node::new("Slice", &[&pc1, &s0, &e0, &ax], &[&a]).name(&format!("{tag}.glu.a")));
    let b = format!("{tag}.glu.b");
    let (s1, e1) = (format!("{tag}.glu.s1"), format!("{tag}.glu.e1"));
    g.init_i64(&s1, &[1], vec![c as i64]);
    g.init_i64(&e1, &[1], vec![2 * c as i64]);
    g.add(Node::new("Slice", &[&pc1, &s1, &e1, &ax], &[&b]).name(&format!("{tag}.glu.b")));
    let bsig = format!("{tag}.glu.bsig");
    g.add(Node::new("Sigmoid", &[&b], &[&bsig]).name(&format!("{tag}.glu.bsig")));
    let glu = format!("{tag}.glu");
    g.add(Node::new("Mul", &[&a, &bsig], &[&glu]).name(&format!("{tag}.glu.mul"))); // [t, C]

    // causal depthwise conv1d over time: view [t,C] -> [1,C,t], Conv group=C, left-pad k-1.
    let glu_ct = {
        let tp = transpose(g, &glu, &[1, 0], &format!("{tag}.dw.tp")); // [C, t]
        reshape(g, &tp, &[1, c as i64, t as i64], &format!("{tag}.dw.r")) // [1, C, t]
    };
    // depthwise weight [C,1,k] verbatim (Conv wants [Cout, Cin/group, k]).
    let dwn = format!("{tag}.dw.w");
    g.init_f32(&dwn, &[c as i64, 1, k as i64], w.get(&p("depthwise_conv.weight")));
    let conv = format!("{tag}.dw.conv");
    g.add(
        Node::new("Conv", &[&glu_ct, &dwn], &[&conv])
            .name(&format!("{tag}.dw.conv"))
            .attr_ints("kernel_shape", &[k as i64])
            .attr_ints("strides", &[1])
            .attr_ints("pads", &[(k - 1) as i64, 0]) // causal: left k-1, right 0
            .attr_int("group", c as i64),
    ); // [1, C, t]
    // back to [t, C]
    let conv_tc = {
        let r = reshape(g, &conv, &[c as i64, t as i64], &format!("{tag}.dw.back")); // [C, t]
        transpose(g, &r, &[1, 0], &format!("{tag}.dw.tc")) // [t, C]
    };
    // LayerNorm(norm) over channel, then SiLU
    let ln = layernorm_onnx(g, w, &conv_tc, &p("norm.weight"), &p("norm.bias"), c, topo.ln_eps, &format!("{tag}.ln"));
    let sig = format!("{tag}.act.sig");
    g.add(Node::new("Sigmoid", &[&ln], &[&sig]).name(&format!("{tag}.act.sig")));
    let act = format!("{tag}.act");
    g.add(Node::new("Mul", &[&ln, &sig], &[&act]).name(&format!("{tag}.act")));
    // pointwise_conv2 [C,C,1] == linear [C,C]
    linear_nb(g, w, &act, &p("pointwise_conv2.weight"), c, c, &format!("{tag}.pc2"))
}

/// One Conformer block (macaron, 5 LayerNorms) — op-for-op with the reference.
#[allow(clippy::too_many_arguments)]
fn conformer_block_onnx(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, pe: &[f32], mask: &str, x: &str, blk: u32, t: u32) -> String {
    let (c, ffn) = (topo.hidden, topo.intermediate);
    let tag = format!("blk{blk}");
    let pre = format!("encoder.layers.{blk}");
    let half = format!("{tag}.half");
    g.init_f32(&half, &[1], vec![0.5]);

    // 1) macaron FF1 (×0.5 residual)
    let n1 = layernorm_onnx(g, w, x, &format!("{pre}.norm_feed_forward1.weight"), &format!("{pre}.norm_feed_forward1.bias"), c, topo.ln_eps, &format!("{tag}.n1"));
    let ff1 = feed_forward_onnx(g, w, &n1, &format!("{pre}.feed_forward1.linear1.weight"), &format!("{pre}.feed_forward1.linear2.weight"), c, ffn, &format!("{tag}.ff1"));
    let ff1h = format!("{tag}.ff1h");
    g.add(Node::new("Mul", &[&ff1, &half], &[&ff1h]).name(&format!("{tag}.ff1h")));
    let h1 = add_t(g, x, &ff1h, &format!("{tag}.h1"));

    // 2) rel-pos self-attention
    let na = layernorm_onnx(g, w, &h1, &format!("{pre}.norm_self_att.weight"), &format!("{pre}.norm_self_att.bias"), c, topo.ln_eps, &format!("{tag}.na"));
    let att = rel_pos_attention_onnx(g, topo, w, &na, pe, mask, &format!("{pre}.self_attn"), t, &format!("{tag}.att"));
    let h2 = add_t(g, &h1, &att, &format!("{tag}.h2"));

    // 3) conv module
    let nc = layernorm_onnx(g, w, &h2, &format!("{pre}.norm_conv.weight"), &format!("{pre}.norm_conv.bias"), c, topo.ln_eps, &format!("{tag}.nc"));
    let cv = conv_module_onnx(g, topo, w, &nc, &format!("{pre}.conv"), t, &format!("{tag}.cv"));
    let h3 = add_t(g, &h2, &cv, &format!("{tag}.h3"));

    // 4) macaron FF2 (×0.5 residual)
    let n2 = layernorm_onnx(g, w, &h3, &format!("{pre}.norm_feed_forward2.weight"), &format!("{pre}.norm_feed_forward2.bias"), c, topo.ln_eps, &format!("{tag}.n2"));
    let ff2 = feed_forward_onnx(g, w, &n2, &format!("{pre}.feed_forward2.linear1.weight"), &format!("{pre}.feed_forward2.linear2.weight"), c, ffn, &format!("{tag}.ff2"));
    let ff2h = format!("{tag}.ff2h");
    g.add(Node::new("Mul", &[&ff2, &half], &[&ff2h]).name(&format!("{tag}.ff2h")));
    let h4 = add_t(g, &h3, &ff2h, &format!("{tag}.h4"));

    // 5) final LayerNorm
    layernorm_onnx(g, w, &h4, &format!("{pre}.norm_out.weight"), &format!("{pre}.norm_out.bias"), c, topo.ln_eps, &format!("{tag}.nout"))
}

/// Stage 3: prompt_projector(cat(hidden, one_hot(prompt_id))) → encoder_projector,
/// producing the pooler `[t, decoder_hidden]` (written to `out_name`).
fn projectors_onnx(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, x: &str, t: u32, prompt_id: u32, out_name: &str) {
    let (c, np, pi, dh) = (topo.hidden, topo.num_prompts, topo.prompt_intermediate, topo.decoder_hidden);
    // cat(hidden, one_hot(prompt_id)) -> [t, c+np]; the one-hot is a constant [t,np].
    let mut oh = vec![0.0f32; (t * np) as usize];
    for i in 0..t as usize {
        oh[i * np as usize + prompt_id as usize] = 1.0;
    }
    let ohn = "proj.onehot";
    g.init_f32(ohn, &[t as i64, np as i64], oh);
    let cat = "proj.cat";
    g.add(Node::new("Concat", &[x, ohn], &[cat]).name("proj.cat").attr_int("axis", 1)); // [t, c+np]
    // prompt_projector.linear_1: [pi, c+np] + bias, ReLU
    let f1 = linear_nb(g, w, cat, "prompt_projector.linear_1.weight", pi, c + np, "proj.l1");
    let b1 = "proj.b1";
    g.init_f32(b1, &[pi as i64], w.get("prompt_projector.linear_1.bias"));
    let f1b = "proj.f1b";
    g.add(Node::new("Add", &[&f1, b1], &[f1b]).name("proj.f1b"));
    let relu = "proj.relu";
    g.add(Node::new("Relu", &[f1b], &[relu]).name("proj.relu"));
    // prompt_projector.linear_2: [c, pi] + bias
    let f2 = linear_nb(g, w, relu, "prompt_projector.linear_2.weight", c, pi, "proj.l2");
    let b2 = "proj.b2";
    g.init_f32(b2, &[c as i64], w.get("prompt_projector.linear_2.bias"));
    let fused = "proj.fused";
    g.add(Node::new("Add", &[&f2, b2], &[fused]).name("proj.fused"));
    // encoder_projector: [dh, c] + bias -> pooler
    let po = linear_nb(g, w, fused, "encoder_projector.weight", dh, c, "proj.enc");
    let eb = "proj.eb";
    g.init_f32(eb, &[dh as i64], w.get("encoder_projector.bias"));
    g.add(Node::new("Add", &[&po, eb], &[out_name]).name("proj.pooler"));
}

/// Stages 2–3 only: the Conformer stack + projectors on a subsampled input
/// `[t, hidden]` (named `input_name`) → pooler `[t, decoder_hidden]` (`out_name`).
/// `valid` is the valid subsampled length. This is exactly
/// `nemotron::reference::encode_pooler`, so the parity gate feeds it a random
/// `sub` and diffs the two.
#[allow(clippy::too_many_arguments)]
pub fn build_nemotron_head(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, t: u32, valid: u32, prompt_id: u32, input_name: &str, out_name: &str) {
    // build-time constants shared by every block: pe ladder + additive band/pad mask.
    let pe = rel_pos_encoding_host(t as usize, topo.hidden as usize);
    let maskv = attn_mask_host(topo, t as usize, valid as usize);
    let maskn = "enc.mask";
    g.init_f32(maskn, &[1, t as i64, t as i64], maskv); // [1,t,t] broadcast over heads
    let mut cur = input_name.to_string();
    for b in 0..topo.n_layers {
        cur = conformer_block_onnx(g, topo, w, &pe, maskn, &cur, b, t);
    }
    projectors_onnx(g, topo, w, &cur, t, prompt_id, out_name);
}

/// Full FastConformer encoder: mel `[1,1,mel_t,num_mel]` → pooler `[t', decoder_hidden]`
/// (named `out_name`). `mel_valid` is the real (non-padded) frame count; `prompt_id`
/// selects the language one-hot. Fixed-shape (static NPU graph).
#[allow(clippy::too_many_arguments)]
pub fn build_nemotron_encoder(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, mel_t: u32, mel_valid: u32, prompt_id: u32, input_name: &str, out_name: &str) {
    let t = topo.subsampled_len(mel_t);
    let valid = topo.subsampled_len(mel_valid);
    // stage 1 (subsampling) → [t, hidden]
    build_subsampling(g, topo, w, mel_t, mel_valid, input_name, "sub.out");
    // stages 2–3 on the subsampled features.
    build_nemotron_head(g, topo, w, t, valid, prompt_id, "sub.out", out_name);
}
