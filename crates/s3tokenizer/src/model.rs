// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AudioEncoderV2` + FSQ head forward pass.
//!
//! Pure host math, not a device (WGSL) forward - the same call this repo
//! already made for `minimaxmusic3::condition_encoder` (see that module's
//! doc comment): this runs once per reference clip at a few hundred frames,
//! the whole tensor at real dims is a few MB, and a device round trip would
//! be pure overhead with nothing to parallelise across. It matters MORE here
//! than there: the gate is exact-integer FSQ token equality, not cosine, so
//! keeping the whole forward on one deterministic f32 code path (host
//! `matvec`/`layernorm_rows`/`erf` - `model::hostmath`, gradient-checked
//! against the WGSL kernels themselves) avoids a second source of fp
//! reassociation that a GPU dispatch sequence would add on top of it.
//!
//! Composed from `xingchensong/S3Tokenizer`'s `s3tokenizer/model_v2.py`,
//! read end to end (`AudioEncoderV2`, `FSMNMultiHeadAttention`,
//! `FSQCodebook`) - see [`crate::import`]'s module doc for how the ONNX
//! graph structurally confirms this reading (the FSQ head's `Round` is
//! wrapped in extra straight-through-estimator arithmetic that is a numeric
//! no-op, confirmed by reading its embedded `Constant` values out of the
//! graph directly).
//!
//! Single utterance, no padding (see [`crate::import`]'s masking note) - `x *
//! mask` and the attention mask bias are both omitted because they are
//! provably identity for that case, not because they were overlooked.

use audio::conv::{conv1d_ref, Conv1d};
use model::hostmath::{gelu_exact, layernorm_rows, linear_rows, rope_neox, softmax};
use onnx::walk::Tensors;

use crate::config::S3TokenizerConfig;

/// `3^i` for `i in 0..8` - the FSQ base-3 place values, `Σ h_i · 3^i`.
const POWERS: [f32; 8] = [1.0, 3.0, 9.0, 27.0, 81.0, 243.0, 729.0, 2187.0];

/// The per-dimension FSQ scale (`FSQCodebook.encode`, `s3tokenizer/
/// model_v2.py`): `h = round(tanh(z) * SCALE) + 1`. The literal `0.999`
/// parses to the identical f32 bit pattern as the reference's
/// `0.9990000128746033` (that decimal IS `0.999`'s nearest f32, printed back
/// out to full precision) - `clippy::excessive_precision` catches the
/// redundant digits, not a value change.
const FSQ_SCALE: f32 = 0.999;

/// One `ResidualAttentionBlock`'s weights, already canonicalised to
/// `model::hostmath`'s `[out, in]` linear layout (see
/// [`S3TokenizerWeights::from_tensors`]).
struct BlockWeights {
    attn_ln_w: Vec<f32>,
    attn_ln_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    /// `[n_audio_state, 1, 31]` depthwise, no bias.
    fsmn_w: Vec<f32>,
    mlp_ln_w: Vec<f32>,
    mlp_ln_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
}

/// The whole tokenizer's weights, ready to feed [`forward`].
pub struct S3TokenizerWeights {
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    blocks: Vec<BlockWeights>,
    project_down_w: Vec<f32>,
    project_down_b: Vec<f32>,
}

/// `[rows, cols] -> [cols, rows]`, row-major.
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

fn get<'t>(t: &'t Tensors, name: &str) -> &'t (Vec<usize>, Vec<f32>) {
    t.get(name).unwrap_or_else(|| panic!("s3tokenizer: missing imported tensor {name}"))
}

/// A linear weight AS THE ONNX GRAPH STORES IT (`[in, out]` - see
/// `config::S3TokenizerConfig::tensor_manifest`'s doc comment), transposed
/// once to `model::hostmath`'s `[out, in]`.
fn linear_weight(t: &Tensors, name: &str) -> Vec<f32> {
    let (shape, data) = get(t, name);
    assert_eq!(shape.len(), 2, "s3tokenizer: {name} has {} dims, expected 2", shape.len());
    transpose(data, shape[0], shape[1])
}

fn vector(t: &Tensors, name: &str) -> Vec<f32> {
    get(t, name).1.clone()
}

impl S3TokenizerWeights {
    /// Build from [`crate::import::import_s3tokenizer`]'s output.
    pub fn from_tensors(t: &Tensors, cfg: &S3TokenizerConfig) -> S3TokenizerWeights {
        let blocks = (0..cfg.n_audio_layer as usize)
            .map(|b| {
                let p = format!("blocks.{b}");
                BlockWeights {
                    attn_ln_w: vector(t, &format!("{p}.attn_ln.weight")),
                    attn_ln_b: vector(t, &format!("{p}.attn_ln.bias")),
                    q_w: linear_weight(t, &format!("{p}.attn.query.weight")),
                    q_b: vector(t, &format!("{p}.attn.query.bias")),
                    k_w: linear_weight(t, &format!("{p}.attn.key.weight")),
                    v_w: linear_weight(t, &format!("{p}.attn.value.weight")),
                    v_b: vector(t, &format!("{p}.attn.value.bias")),
                    out_w: linear_weight(t, &format!("{p}.attn.out.weight")),
                    out_b: vector(t, &format!("{p}.attn.out.bias")),
                    fsmn_w: vector(t, &format!("{p}.attn.fsmn_block.weight")),
                    mlp_ln_w: vector(t, &format!("{p}.mlp_ln.weight")),
                    mlp_ln_b: vector(t, &format!("{p}.mlp_ln.bias")),
                    fc1_w: linear_weight(t, &format!("{p}.mlp.fc1.weight")),
                    fc1_b: vector(t, &format!("{p}.mlp.fc1.bias")),
                    fc2_w: linear_weight(t, &format!("{p}.mlp.fc2.weight")),
                    fc2_b: vector(t, &format!("{p}.mlp.fc2.bias")),
                }
            })
            .collect();
        S3TokenizerWeights {
            conv1_w: vector(t, "conv1.weight"),
            conv1_b: vector(t, "conv1.bias"),
            conv2_w: vector(t, "conv2.weight"),
            conv2_b: vector(t, "conv2.bias"),
            blocks,
            project_down_w: linear_weight(t, "quantizer.project_down.weight"),
            project_down_b: vector(t, "quantizer.project_down.bias"),
        }
    }
}

/// Add a per-channel bias to an NCL `[c, l]` buffer, in place.
fn add_bias_ncl(x: &mut [f32], bias: &[f32], c: usize, l: usize) {
    for ci in 0..c {
        let b = bias[ci];
        for v in &mut x[ci * l..ci * l + l] {
            *v += b;
        }
    }
}

/// Add a per-column bias to a `[rows, c]` buffer, in place - the `Linear`
/// bias `linear_rows` itself does not apply.
fn add_bias_rows(x: &mut [f32], bias: &[f32], rows: usize, c: usize) {
    for r in 0..rows {
        for (v, &b) in x[r * c..r * c + c].iter_mut().zip(bias) {
            *v += b;
        }
    }
}

fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu_exact(*v);
    }
}

/// `FSMNMultiHeadAttention.forward_fsmn`: a depthwise `Conv1d(k=31, pad=15,
/// bias=False)` over `v` (BEFORE the attention's `out` projection), added
/// back to `v` as a residual. `v` is `[t, d]` row-major (`d = n_audio_state`,
/// already the flattened `[heads, head_dim]` view - the depthwise conv does
/// not care about the head split, it runs per-channel over all `d` of them).
fn forward_fsmn(v: &[f32], fsmn_w: &[f32], t: usize, d: usize) -> Vec<f32> {
    let v_ncl = transpose(v, t, d); // [d, t]
    let c = Conv1d { n: 1, cin: d as u32, l: t as u32, cout: d as u32, k: 31, stride: 1, pad: 15, dilation: 1, groups: d as u32, lo: t as u32 };
    let y_ncl = conv1d_ref(&c, &v_ncl, fsmn_w); // [d, t], no bias
    let mut y = transpose(&y_ncl, d, t); // [t, d]
    for (yi, &vi) in y.iter_mut().zip(v) {
        *yi += vi;
    }
    y
}

/// Standard scaled-dot-product multi-head self-attention (no mask - see the
/// module doc). `q`/`k`/`v` are `[t, heads*head_dim]` row-major, already
/// RoPE'd (q, k). Returns `[t, heads*head_dim]`.
///
/// `FSMNMultiHeadAttention.qkv_attention` scales BOTH `q` and `k` by
/// `head_dim**-0.25` before the dot product (`q@k` ends up scaled by
/// `head_dim**-0.5`, the usual `1/sqrt(d)` - just factored across both
/// operands rather than applied once to the scores).
fn attention(q: &[f32], k: &[f32], v: &[f32], t: usize, heads: usize, hd: usize) -> Vec<f32> {
    let scale = (hd as f32).powf(-0.25);
    let d = heads * hd;
    let mut out = vec![0.0f32; t * d];
    let mut scores = vec![0.0f32; t];
    for h in 0..heads {
        for ti in 0..t {
            for (tj, s) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for dd in 0..hd {
                    acc += q[ti * d + h * hd + dd] * scale * (k[tj * d + h * hd + dd] * scale);
                }
                *s = acc;
            }
            softmax(&mut scores);
            for dd in 0..hd {
                let mut acc = 0.0f32;
                for (tj, &s) in scores.iter().enumerate() {
                    acc += s * v[tj * d + h * hd + dd];
                }
                out[ti * d + h * hd + dd] = acc;
            }
        }
    }
    out
}

/// `AudioEncoderV2` + `FSQVectorQuantization.encode`: mel `[n_mels, t_in]`
/// (row-major NCL, batch folded away - single utterance) to FSQ token ids,
/// one per output frame (`t_in / 4` after the two stride-2 convs).
pub fn forward(cfg: &S3TokenizerConfig, w: &S3TokenizerWeights, mel: &[f32], t_in: usize) -> Vec<i32> {
    let n_mels = cfg.n_mels as usize;
    let d = cfg.n_audio_state as usize;
    let mlp_dim = cfg.mlp_dim() as usize;
    let heads = cfg.n_audio_head as usize;
    let hd = cfg.head_dim() as usize;
    assert_eq!(mel.len(), n_mels * t_in, "s3tokenizer: mel has {} elements, expected n_mels*t_in={}", mel.len(), n_mels * t_in);

    let t1 = Conv1d::out_len(t_in as u32, 3, 2, 1, 1, 1) as usize;
    let c1 = Conv1d { n: 1, cin: n_mels as u32, l: t_in as u32, cout: d as u32, k: 3, stride: 2, pad: 1, dilation: 1, groups: 1, lo: t1 as u32 };
    let mut x1 = conv1d_ref(&c1, mel, &w.conv1_w);
    add_bias_ncl(&mut x1, &w.conv1_b, d, t1);
    gelu_inplace(&mut x1);

    let t2 = Conv1d::out_len(t1 as u32, 3, 2, 1, 1, 1) as usize;
    let c2 = Conv1d { n: 1, cin: d as u32, l: t1 as u32, cout: d as u32, k: 3, stride: 2, pad: 1, dilation: 1, groups: 1, lo: t2 as u32 };
    let mut x2 = conv1d_ref(&c2, &x1, &w.conv2_w);
    add_bias_ncl(&mut x2, &w.conv2_b, d, t2);
    gelu_inplace(&mut x2);

    let t = t2;
    let mut x = transpose(&x2, d, t); // [d, t] -> [t, d] ("x.permute(0, 2, 1)")

    for blk in &w.blocks {
        let ln1 = layernorm_rows(&x, &blk.attn_ln_w, &blk.attn_ln_b, t, d, 1e-5);
        let mut q = linear_rows(&ln1, &blk.q_w, t, d, d);
        add_bias_rows(&mut q, &blk.q_b, t, d);
        let k = linear_rows(&ln1, &blk.k_w, t, d, d); // no bias
        let mut v = linear_rows(&ln1, &blk.v_w, t, d, d);
        add_bias_rows(&mut v, &blk.v_b, t, d);

        // `precompute_freqs_cis(64, 1024*2)` + `apply_rotary_emb`'s
        // rotate-half convention IS `rope_neox` at theta=10000 (verified
        // against `s3tokenizer/model_v2.py` term by term).
        let mut q_rot = q;
        let mut k_rot = k;
        rope_neox(&mut q_rot, t, heads, hd, 0, 10000.0);
        rope_neox(&mut k_rot, t, heads, hd, 0, 10000.0);

        let fsm = forward_fsmn(&v, &blk.fsmn_w, t, d);
        let wv = attention(&q_rot, &k_rot, &v, t, heads, hd);

        let mut attn_out = linear_rows(&wv, &blk.out_w, t, d, d);
        add_bias_rows(&mut attn_out, &blk.out_b, t, d);
        for (a, f) in attn_out.iter_mut().zip(&fsm) {
            *a += f;
        }
        for (xi, a) in x.iter_mut().zip(&attn_out) {
            *xi += a;
        }

        let ln2 = layernorm_rows(&x, &blk.mlp_ln_w, &blk.mlp_ln_b, t, d, 1e-5);
        let mut h1 = linear_rows(&ln2, &blk.fc1_w, t, d, mlp_dim);
        add_bias_rows(&mut h1, &blk.fc1_b, t, mlp_dim);
        gelu_inplace(&mut h1);
        let mut h2 = linear_rows(&h1, &blk.fc2_w, t, mlp_dim, d);
        add_bias_rows(&mut h2, &blk.fc2_b, t, d);
        for (xi, hh) in x.iter_mut().zip(&h2) {
            *xi += hh;
        }
    }

    // FSQCodebook.encode: no kernel, pure elementwise host arithmetic.
    let fsq = cfg.fsq_dims() as usize;
    let mut z = linear_rows(&x, &w.project_down_w, t, d, fsq);
    add_bias_rows(&mut z, &w.project_down_b, t, fsq);

    (0..t)
        .map(|row| {
            let mut idx = 0.0f32;
            for j in 0..fsq {
                let h = (z[row * fsq + j].tanh() * FSQ_SCALE).round_ties_even() + 1.0;
                idx += h * POWERS[j];
            }
            idx as i32
        })
        .collect()
}
