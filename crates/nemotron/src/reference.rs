// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CPU f32 reference for the FastConformer encoder blocks + projectors, validated
//! exactly against dumped HF activations. This is the oracle the device Step-graph
//! (and its backward) are checked against — the relative-position attention and the
//! cache-aware `chunked_limited` mask are subtle enough to warrant a from-scratch
//! reference (brain convention: a gradcheck/parity oracle may re-derive the math).
//!
//! Layout: activations are `[T, C]` row-major throughout; PyTorch `Linear.weight`
//! is `[out, in]` so a linear is `matmul_nt`.

use std::collections::HashMap;

use crate::config::NemotronConfig;

type W = HashMap<String, Vec<f32>>;

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `[m,k] · [n,k]ᵀ → [m,n]`  (y = x·Wᵀ, W row-major `[n,k]`).
fn matmul_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let (xr, wr) = (&x[i * k..i * k + k], &w[j * k..j * k + k]);
            for t in 0..k {
                acc += xr[t] * wr[t];
            }
            y[i * n + j] = acc;
        }
    }
    y
}

/// Row-wise LayerNorm over the last dim `c` (torch nn.LayerNorm), in place-returning.
pub(crate) fn layernorm(x: &[f32], g: &[f32], b: &[f32], t: usize, c: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; t * c];
    for i in 0..t {
        let row = &x[i * c..i * c + c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..c {
            y[i * c + j] = (row[j] - mean) * inv * g[j] + b[j];
        }
    }
    y
}

/// Macaron feed-forward: Linear(c→ffn) → SiLU → Linear(ffn→c), no bias.
fn feed_forward(x: &[f32], w1: &[f32], w2: &[f32], t: usize, c: usize, ffn: usize) -> Vec<f32> {
    let mut h = matmul_nt(x, w1, t, c, ffn);
    for v in &mut h {
        *v = silu(*v);
    }
    matmul_nt(&h, w2, t, ffn, c)
}

/// Relative positional encoding `[2T-1, C]`: interleaved sin/cos over positions
/// `[T-1 .. -(T-1)]`, `inv_freq[i] = 10000^(-2i/C)`.
pub(crate) fn rel_pos_encoding(t: usize, c: usize) -> Vec<f32> {
    let half = c / 2;
    let inv: Vec<f32> = (0..half).map(|i| (10000f32).powf(-(2.0 * i as f32) / c as f32)).collect();
    let l = 2 * t - 1;
    let mut pe = vec![0.0f32; l * c];
    for idx in 0..l {
        let pos = (t as i64 - 1 - idx as i64) as f32; // T-1 .. -(T-1)
        for i in 0..half {
            let f = pos * inv[i];
            pe[idx * c + 2 * i] = f.sin();
            pe[idx * c + 2 * i + 1] = f.cos();
        }
    }
    pe
}

/// `chunked_limited` validity: query `i` may attend key `j` iff
/// `0 <= i/chunk - j/chunk <= left_ctx_chunks`, `chunk = right+1`.
pub(crate) fn banded_ok(i: usize, j: usize, left: usize, right: usize) -> bool {
    let chunk = right + 1;
    let left_chunks = left / chunk;
    let (qc, kc) = (i / chunk, j / chunk);
    qc >= kc && qc - kc <= left_chunks
}

/// Relative-position multi-head self-attention (Transformer-XL) under the
/// `chunked_limited` mask. `hn` is the pre-normalised `[T, C]` input.
#[allow(clippy::too_many_arguments)]
fn rel_pos_attention(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize) -> Vec<f32> {
    let (c, heads, hd) = (cfg.hidden as usize, cfg.n_heads as usize, cfg.head_dim() as usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let left = (cfg.sliding_window - 1) as usize;
    let right = cfg.default_lookahead as usize;
    let p = |n: &str| &w[&format!("{prefix}.{n}")];

    let q = matmul_nt(hn, p("q_proj.weight"), t, c, c);
    let k = matmul_nt(hn, p("k_proj.weight"), t, c, c);
    let v = matmul_nt(hn, p("v_proj.weight"), t, c, c);
    let pe = rel_pos_encoding(t, c);
    let l = 2 * t - 1;
    let rel_k = matmul_nt(&pe, p("relative_k_proj.weight"), l, c, c); // [L, C]
    let bias_u = p("bias_u"); // [heads*hd]
    let bias_v = p("bias_v");

    let mut ctx = vec![0.0f32; t * c]; // [T, heads*hd]
    for h in 0..heads {
        // per-head slices: q[i, h*hd + d]
        let qh = |i: usize, d: usize| q[i * c + h * hd + d];
        let kh = |j: usize, d: usize| k[j * c + h * hd + d];
        let vh = |j: usize, d: usize| v[j * c + h * hd + d];
        let rkh = |pp: usize, d: usize| rel_k[pp * c + h * hd + d];
        let bu = &bias_u[h * hd..h * hd + hd];
        let bv = &bias_v[h * hd..h * hd + hd];

        // matrix_bd_raw[i, pp] = (q[i]+bias_v)·rel_k[pp]   → [T, L]
        let mut bd_raw = vec![0.0f32; t * l];
        for i in 0..t {
            for pp in 0..l {
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += (qh(i, d) + bv[d]) * rkh(pp, d);
                }
                bd_raw[i * l + pp] = acc;
            }
        }
        let bd = crate::kernels::rel_shift_ref(&bd_raw, 1, t, l); // [T, L]

        // scores[i,j] = (q[i]+bias_u)·k[j]*scale + bd[i, j]*scale, masked, softmax
        for i in 0..t {
            let mut scores = vec![f32::NEG_INFINITY; t];
            for j in 0..t {
                if j >= valid || !banded_ok(i, j, left, right) {
                    continue; // padding mask (invalid key) AND chunked_limited band
                }
                let mut ac = 0.0f32;
                for d in 0..hd {
                    ac += (qh(i, d) + bu[d]) * kh(j, d);
                }
                scores[j] = ac * scale + bd[i * l + j] * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut scores {
                *s = if s.is_finite() { (*s - mx).exp() } else { 0.0 };
                den += *s;
            }
            let inv_den = if den > 0.0 { 1.0 / den } else { 0.0 };
            for d in 0..hd {
                let mut acc = 0.0f32;
                for j in 0..t {
                    acc += scores[j] * vh(j, d);
                }
                ctx[i * c + h * hd + d] = acc * inv_den;
            }
        }
    }
    matmul_nt(&ctx, p("o_proj.weight"), t, c, c) // o_proj (no bias)
}

/// Transpose (backward) of `rel_shift_ref`: scatter `d_out[rows,q,p]` back to `d_x[rows,q,p]`.
fn rel_shift_backward(d_out: &[f32], rows: usize, q: usize, p: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * q * p];
    for r in 0..rows {
        let mut dxp = vec![0.0f32; q * (p + 1)];
        for idx2 in 0..q * p {
            dxp[q + idx2] = d_out[r * q * p + idx2];
        }
        for i in 0..q {
            for k in 1..=p {
                dx[r * q * p + i * p + (k - 1)] += dxp[i * (p + 1) + k];
            }
        }
    }
    dx
}

/// Backward of `rel_pos_attention` w.r.t. the input `hn` and `bias_v` (the novel
/// Transformer-XL positional bias). `d_out` is the loss grad of the attention
/// output `[T, C]`. Returns `(d_hn[T,C], d_bias_v[C])`.
#[allow(clippy::too_many_arguments)]
fn rel_pos_attention_backward(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (c, heads, hd) = (cfg.hidden as usize, cfg.n_heads as usize, cfg.head_dim() as usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let (left, right) = ((cfg.sliding_window - 1) as usize, cfg.default_lookahead as usize);
    let p = |n: &str| &w[&format!("{prefix}.{n}")];
    let (wq, wk, wv, wrel, wo) = (p("q_proj.weight"), p("k_proj.weight"), p("v_proj.weight"), p("relative_k_proj.weight"), p("o_proj.weight"));
    let (bu, bv) = (p("bias_u"), p("bias_v"));

    // forward recompute
    let q = matmul_nt(hn, wq, t, c, c);
    let k = matmul_nt(hn, wk, t, c, c);
    let v = matmul_nt(hn, wv, t, c, c);
    let l = 2 * t - 1;
    let pe = rel_pos_encoding(t, c);
    let rel_k = matmul_nt(&pe, wrel, l, c, c);

    // d_ctx = d_out · Wo  (out = ctx·Woᵀ)
    let mut d_ctx = vec![0.0f32; t * c];
    for i in 0..t {
        for j in 0..c {
            let mut acc = 0.0f32;
            for o in 0..c {
                acc += d_out[i * c + o] * wo[o * c + j];
            }
            d_ctx[i * c + j] = acc;
        }
    }

    let (mut dq, mut dk, mut dv) = (vec![0.0f32; t * c], vec![0.0f32; t * c], vec![0.0f32; t * c]);
    let mut d_bias_v = vec![0.0f32; c];
    for h in 0..heads {
        let (qh, kh, rkh) = (|i: usize, d: usize| q[i * c + h * hd + d], |j: usize, d: usize| k[j * c + h * hd + d], |pp: usize, d: usize| rel_k[pp * c + h * hd + d]);
        let (bus, bvs) = (&bu[h * hd..h * hd + hd], &bv[h * hd..h * hd + hd]);
        // recompute bd, scores, probs for this head
        let mut bd_raw = vec![0.0f32; t * l];
        for i in 0..t {
            for pp in 0..l {
                let mut a = 0.0f32;
                for d in 0..hd {
                    a += (qh(i, d) + bvs[d]) * rkh(pp, d);
                }
                bd_raw[i * l + pp] = a;
            }
        }
        let bd = crate::kernels::rel_shift_ref(&bd_raw, 1, t, l);
        let mut probs = vec![0.0f32; t * t];
        for i in 0..t {
            let mut sc = vec![f32::NEG_INFINITY; t];
            for j in 0..t {
                if j >= valid || !banded_ok(i, j, left, right) {
                    continue;
                }
                let mut ac = 0.0f32;
                for d in 0..hd {
                    ac += (qh(i, d) + bus[d]) * kh(j, d);
                }
                sc[j] = ac * scale + bd[i * l + j] * scale;
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            for s in &mut sc {
                *s = if s.is_finite() { (*s - mx).exp() } else { 0.0 };
                den += *s;
            }
            let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
            for j in 0..t {
                probs[i * t + j] = sc[j] * inv;
            }
        }
        // backward through ctx = probs·v_head
        // d_probs[i,j] = Σ_d d_ctx[i,h*hd+d]·v[j,d];  d_v[j,d] += Σ_i probs[i,j]·d_ctx[i,h*hd+d]
        let mut d_probs = vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                let mut a = 0.0f32;
                for d in 0..hd {
                    a += d_ctx[i * c + h * hd + d] * v[j * c + h * hd + d];
                }
                d_probs[i * t + j] = a;
            }
        }
        for j in 0..t {
            for d in 0..hd {
                let mut a = 0.0f32;
                for i in 0..t {
                    a += probs[i * t + j] * d_ctx[i * c + h * hd + d];
                }
                dv[j * c + h * hd + d] += a;
            }
        }
        // softmax backward per row: d_sc[i,j] = probs[i,j]·(d_probs[i,j] − Σ_j' probs·d_probs)
        let mut d_bd_raw = vec![0.0f32; t * l];
        for i in 0..t {
            let dot: f32 = (0..t).map(|j| probs[i * t + j] * d_probs[i * t + j]).sum();
            for j in 0..t {
                let d_sc = probs[i * t + j] * (d_probs[i * t + j] - dot);
                if d_sc == 0.0 {
                    continue;
                }
                let d_ac = d_sc * scale;
                let d_bd = d_sc * scale;
                d_bd_raw[i * l + j] += d_bd; // bd sliced [:T]; rel_shift_bwd handles the reindex
                for d in 0..hd {
                    // d_ac: ac = (q+bu)·k → d(q_u), d(k)
                    dq[i * c + h * hd + d] += d_ac * kh(j, d);
                    dk[j * c + h * hd + d] += d_ac * (qh(i, d) + bus[d]);
                }
            }
        }
        // bd path: d_bd_raw (already in [T,L] via slice) → transpose rel_shift → d(q_v),(rel_k)
        let d_bd_pre = rel_shift_backward(&d_bd_raw, 1, t, l);
        for i in 0..t {
            for pp in 0..l {
                let g = d_bd_pre[i * l + pp];
                if g == 0.0 {
                    continue;
                }
                for d in 0..hd {
                    // bd_raw = (q+bv)·rel_k → d(q_v) and d_bias_v; rel_k grad omitted (not checked here)
                    let dqv = g * rkh(pp, d);
                    dq[i * c + h * hd + d] += dqv;
                    d_bias_v[h * hd + d] += dqv;
                }
            }
        }
    }
    // d_hn = dq·Wq + dk·Wk + dv·Wv  (q = hn·Wqᵀ ⇒ d_hn[i,in] = Σ_o dq[i,o]·Wq[o,in])
    let mut d_hn = vec![0.0f32; t * c];
    for (dproj, wproj) in [(&dq, wq), (&dk, wk), (&dv, wv)] {
        for i in 0..t {
            for inp in 0..c {
                let mut acc = 0.0f32;
                for o in 0..c {
                    acc += dproj[i * c + o] * wproj[o * c + inp];
                }
                d_hn[i * c + inp] += acc;
            }
        }
    }
    (d_hn, d_bias_v)
}

/// Conformer convolution module: pointwise_conv1(→2C) → GLU → causal depthwise
/// conv1d(k) → LayerNorm → SiLU → pointwise_conv2. `hn` is pre-normalised `[T,C]`.
fn conv_module(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize) -> Vec<f32> {
    let (c, k) = (cfg.hidden as usize, cfg.conv_kernel as usize);
    let p = |n: &str| &w[&format!("{prefix}.{n}")];
    // pointwise_conv1: weight [2C, C, 1] == linear [2C, C]
    let pc1 = matmul_nt(hn, p("pointwise_conv1.weight"), t, c, 2 * c); // [T, 2C]
    // GLU over channel: a = pc1[:, :C], b = pc1[:, C:]; fully-masked frames zeroed
    // before the depthwise conv (all_masked_rows) so padding can't leak.
    let mut glu = vec![0.0f32; t * c];
    for i in 0..valid.min(t) {
        for j in 0..c {
            glu[i * c + j] = pc1[i * 2 * c + j] * sigmoid(pc1[i * 2 * c + c + j]);
        }
    }
    // causal depthwise conv1d over time, weight [C,1,k], left-pad k-1, no bias
    let dw = p("depthwise_conv.weight"); // [C, 1, k]
    let mut conv = vec![0.0f32; t * c];
    for ch in 0..c {
        for i in 0..t {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let src = i as i64 - (k as i64 - 1) + kk as i64;
                if src >= 0 {
                    acc += glu[src as usize * c + ch] * dw[ch * k + kk];
                }
            }
            conv[i * c + ch] = acc;
        }
    }
    // LayerNorm(norm) over channel, then SiLU
    let ln = layernorm(&conv, p("norm.weight"), p("norm.bias"), t, c, cfg.ln_eps);
    let mut act = ln;
    for v in &mut act {
        *v = silu(*v);
    }
    // pointwise_conv2: weight [C, C, 1] == linear [C, C]
    matmul_nt(&act, p("pointwise_conv2.weight"), t, c, c)
}

/// One Conformer block (macaron; five LayerNorms). `h` `[T, C]` in place-returning.
pub fn conformer_block(h: &[f32], w: &W, b: u32, cfg: &NemotronConfig, t: usize, valid: usize) -> Vec<f32> {
    let c = cfg.hidden as usize;
    let ffn = cfg.intermediate as usize;
    let pre = format!("encoder.layers.{b}");
    let g = |n: &str| &w[&format!("{pre}.{n}")];
    let mut h = h.to_vec();

    // 1) macaron FF1 (×0.5 residual)
    let n1 = layernorm(&h, g("norm_feed_forward1.weight"), g("norm_feed_forward1.bias"), t, c, cfg.ln_eps);
    let ff1 = feed_forward(&n1, g("feed_forward1.linear1.weight"), g("feed_forward1.linear2.weight"), t, c, ffn);
    for i in 0..t * c {
        h[i] += 0.5 * ff1[i];
    }
    // 2) rel-pos self-attention
    let na = layernorm(&h, g("norm_self_att.weight"), g("norm_self_att.bias"), t, c, cfg.ln_eps);
    let att = rel_pos_attention(&na, w, &format!("{pre}.self_attn"), cfg, t, valid);
    for i in 0..t * c {
        h[i] += att[i];
    }
    // 3) conv module
    let nc = layernorm(&h, g("norm_conv.weight"), g("norm_conv.bias"), t, c, cfg.ln_eps);
    let cv = conv_module(&nc, w, &format!("{pre}.conv"), cfg, t, valid);
    for i in 0..t * c {
        h[i] += cv[i];
    }
    // 4) macaron FF2 (×0.5 residual)
    let n2 = layernorm(&h, g("norm_feed_forward2.weight"), g("norm_feed_forward2.bias"), t, c, cfg.ln_eps);
    let ff2 = feed_forward(&n2, g("feed_forward2.linear1.weight"), g("feed_forward2.linear2.weight"), t, c, ffn);
    for i in 0..t * c {
        h[i] += 0.5 * ff2[i];
    }
    // 5) final LayerNorm
    layernorm(&h, g("norm_out.weight"), g("norm_out.bias"), t, c, cfg.ln_eps)
}

/// Full encoder head from the subsampling output: 24 Conformer blocks →
/// prompt_projector(cat(hidden, one_hot(prompt))) → encoder_projector → pooler
/// `[T, decoder_hidden]`. `valid` is the valid subsampled length.
pub fn encode_pooler(sub: &[f32], w: &W, cfg: &NemotronConfig, t: usize, valid: usize, prompt_id: usize) -> Vec<f32> {
    let c = cfg.hidden as usize;
    let mut h = sub.to_vec();
    for b in 0..cfg.n_layers {
        h = conformer_block(&h, w, b, cfg, t, valid);
    }
    // prompt_projector: Linear(c+num_prompts → prompt_intermediate) ReLU Linear(→ c)
    let np = cfg.num_prompts as usize;
    let pi = cfg.prompt_intermediate as usize;
    let mut cat = vec![0.0f32; t * (c + np)];
    for i in 0..t {
        cat[i * (c + np)..i * (c + np) + c].copy_from_slice(&h[i * c..i * c + c]);
        cat[i * (c + np) + c + prompt_id] = 1.0; // one-hot language prompt
    }
    let mut f1 = matmul_nt(&cat, &w["prompt_projector.linear_1.weight"], t, c + np, pi);
    let b1 = &w["prompt_projector.linear_1.bias"];
    for i in 0..t {
        for j in 0..pi {
            let v = f1[i * pi + j] + b1[j];
            f1[i * pi + j] = v.max(0.0); // ReLU
        }
    }
    let mut fused = matmul_nt(&f1, &w["prompt_projector.linear_2.weight"], t, pi, c);
    let b2 = &w["prompt_projector.linear_2.bias"];
    for i in 0..t {
        for j in 0..c {
            fused[i * c + j] += b2[j];
        }
    }
    // encoder_projector: Linear(c → decoder_hidden) + bias
    let dh = cfg.decoder_hidden as usize;
    let mut pooler = matmul_nt(&fused, &w["encoder_projector.weight"], t, c, dh);
    let eb = &w["encoder_projector.bias"];
    for i in 0..t {
        for j in 0..dh {
            pooler[i * dh + j] += eb[j];
        }
    }
    pooler
}


/// RNN-T joint-network backward. Given the loss grad `d_logits[vocab]` and the
/// forward inputs, returns `(d_enc[dh], d_dec[dh], d_head[vocab*dh], d_bias[vocab])`.
/// joint = head·relu(enc+dec) + bias, so d_enc = d_dec = relu'(enc+dec) ⊙ (headᵀ·d_logits).
pub fn joint_backward(enc: &[f32], dec: &[f32], d_logits: &[f32], w: &W, cfg: &NemotronConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (dh, vocab) = (cfg.decoder_hidden as usize, cfg.vocab as usize);
    let head = &w["joint.head.weight"]; // [vocab, dh]
    let sum: Vec<f32> = (0..dh).map(|j| (enc[j] + dec[j]).max(0.0)).collect(); // relu(enc+dec)
    let relu_mask: Vec<f32> = (0..dh).map(|j| if enc[j] + dec[j] > 0.0 { 1.0 } else { 0.0 }).collect();
    let d_bias = d_logits.to_vec();
    let mut d_head = vec![0.0f32; vocab * dh];
    for o in 0..vocab {
        for j in 0..dh {
            d_head[o * dh + j] = d_logits[o] * sum[j];
        }
    }
    // d_sum = headᵀ · d_logits, then through relu
    let mut d_ed = vec![0.0f32; dh];
    for j in 0..dh {
        let mut acc = 0.0f32;
        for o in 0..vocab {
            acc += d_logits[o] * head[o * dh + j];
        }
        d_ed[j] = acc * relu_mask[j];
    }
    (d_ed.clone(), d_ed, d_head, d_bias)
}

/// RNN-T LSTM prediction network state (2 layers).
pub struct LstmState {
    h: Vec<Vec<f32>>, // [layers][hidden]
    c: Vec<Vec<f32>>,
}

impl LstmState {
    pub fn new(layers: usize, hidden: usize) -> LstmState {
        LstmState { h: vec![vec![0.0; hidden]; layers], c: vec![vec![0.0; hidden]; layers] }
    }
}

/// One LSTM prediction step for `token`: embedding → 2-layer LSTM → decoder_projector.
/// Updates `st` in place; returns the projected decoder output `[decoder_hidden]`.
pub fn lstm_predict(token: u32, st: &mut LstmState, w: &W, cfg: &NemotronConfig) -> Vec<f32> {
    let dh = cfg.decoder_hidden as usize;
    let emb = &w["decoder.embedding.weight"][token as usize * dh..token as usize * dh + dh];
    let mut input = emb.to_vec();
    for layer in 0..cfg.num_decoder_layers as usize {
        let wih = &w[&format!("decoder.lstm.weight_ih_l{layer}")]; // [4dh, dh]
        let whh = &w[&format!("decoder.lstm.weight_hh_l{layer}")];
        let bih = &w[&format!("decoder.lstm.bias_ih_l{layer}")]; // [4dh]
        let bhh = &w[&format!("decoder.lstm.bias_hh_l{layer}")];
        // gates = W_ih·input + b_ih + W_hh·h + b_hh   (PyTorch gate order i,f,g,o)
        let gi = matmul_nt(&input, wih, 1, dh, 4 * dh);
        let gh = matmul_nt(&st.h[layer], whh, 1, dh, 4 * dh);
        let mut out = vec![0.0f32; dh];
        for j in 0..dh {
            let g = |o: usize| gi[o * dh + j] + bih[o * dh + j] + gh[o * dh + j] + bhh[o * dh + j];
            let ii = sigmoid(g(0));
            let ff = sigmoid(g(1));
            let gg = g(2).tanh();
            let oo = sigmoid(g(3));
            let ct = ff * st.c[layer][j] + ii * gg;
            st.c[layer][j] = ct;
            let ht = oo * ct.tanh();
            st.h[layer][j] = ht;
            out[j] = ht;
        }
        input = out;
    }
    // decoder_projector: Linear(dh→dh) + bias
    let mut dec = matmul_nt(&input, &w["decoder.decoder_projector.weight"], 1, dh, dh);
    let db = &w["decoder.decoder_projector.bias"];
    for j in 0..dh {
        dec[j] += db[j];
    }
    dec
}

/// RNN-T joint: `head(relu(enc_t + dec_u))` → `[vocab]` logits.
pub fn joint(enc_t: &[f32], dec_u: &[f32], w: &W, cfg: &NemotronConfig) -> Vec<f32> {
    let dh = cfg.decoder_hidden as usize;
    let sum: Vec<f32> = (0..dh).map(|j| (enc_t[j] + dec_u[j]).max(0.0)).collect(); // relu
    let mut logits = matmul_nt(&sum, &w["joint.head.weight"], 1, dh, cfg.vocab as usize);
    let jb = &w["joint.head.bias"];
    for j in 0..cfg.vocab as usize {
        logits[j] += jb[j];
    }
    logits
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i as u32;
        }
    }
    best
}

/// Greedy RNN-T transducer decode over encoder frames `pooler[T, decoder_hidden]`
/// (first `valid` frames). Returns the emitted non-blank token ids.
pub fn rnnt_greedy(pooler: &[f32], valid: usize, w: &W, cfg: &NemotronConfig) -> Vec<u32> {
    let dh = cfg.decoder_hidden as usize;
    let blank = cfg.blank_token_id;
    let mut st = LstmState::new(cfg.num_decoder_layers as usize, dh);
    let mut dec = lstm_predict(blank, &mut st, w, cfg); // decoder_start = blank
    let mut frame = 0usize;
    let mut symbols = 0u32;
    let mut emitted = Vec::new();
    while frame < valid {
        let logits = joint(&pooler[frame * dh..frame * dh + dh], &dec, w, cfg);
        let token = argmax(&logits);
        if token == blank || symbols >= cfg.max_symbols_per_step {
            frame += 1;
            symbols = 0; // blank: LSTM state / dec unchanged
        } else {
            emitted.push(token);
            symbols += 1;
            dec = lstm_predict(token, &mut st, w, cfg); // step LSTM for next frame's joint
        }
    }
    emitted
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::path::Path;

    const GOLD: &str = "/data/workspace/resources/asr/golden/nemotron";
    const CKPT: &str = "/data/workspace/resources/asr/nemotron/hf";

    fn read_f32(p: &str) -> Vec<f32> {
        let mut f = std::fs::File::open(p).unwrap_or_else(|_| panic!("missing {p}"));
        let mut b = Vec::new();
        f.read_to_end(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn rel_pos_attention_backward_matches_finite_diff() {
        use data::rng::Rng;
        // tiny synthetic conformer attention: c=8, heads=2, hd=4, T=6 (band covers all)
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.hidden = 8;
        cfg.n_heads = 2;
        let (c, t, valid) = (8usize, 6usize, 6usize);
        let mut rng = Rng::new(21);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.6).collect::<Vec<f32>>();
        let mut w: W = W::new();
        for leaf in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "relative_k_proj.weight", "o_proj.weight"] {
            w.insert(format!("attn.{leaf}"), r(c * c));
        }
        w.insert("attn.bias_u".into(), r(c));
        w.insert("attn.bias_v".into(), r(c));
        let hn = r(t * c);

        let d_out = vec![1.0f32; t * c]; // loss = Σ out
        let (d_hn, d_bias_v) = rel_pos_attention_backward(&hn, &w, "attn", &cfg, t, valid, &d_out);
        let loss = |hh: &[f32], ww: &W| -> f32 { rel_pos_attention(hh, ww, "attn", &cfg, t, valid).iter().sum() };
        let eps = 1e-3f32;
        let ok = |a: f32, n: f32| (a - n).abs() <= 5e-3 + 8e-2 * n.abs();
        for &i in &[0usize, 11, 23, 40] {
            let (mut hp, mut hm) = (hn.clone(), hn.clone());
            hp[i] += eps;
            hm[i] -= eps;
            let num = (loss(&hp, &w) - loss(&hm, &w)) / (2.0 * eps);
            assert!(ok(d_hn[i], num), "d_hn[{i}] {} vs {}", d_hn[i], num);
        }
        for &i in &[0usize, 3, 7] {
            let (mut wp, mut wm) = (w.clone(), w.clone());
            wp.get_mut("attn.bias_v").unwrap()[i] += eps;
            wm.get_mut("attn.bias_v").unwrap()[i] -= eps;
            let num = (loss(&hn, &wp) - loss(&hn, &wm)) / (2.0 * eps);
            assert!(ok(d_bias_v[i], num), "d_bias_v[{i}] {} vs {}", d_bias_v[i], num);
        }
    }

    #[test]
    fn joint_backward_matches_finite_diff() {
        // Tiny synthetic joint network (no checkpoint needed): gradcheck the
        // transducer joint backward (relu + head matmul) vs central differences.
        use data::rng::Rng;
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.decoder_hidden = 6;
        cfg.vocab = 5;
        let (dh, vocab) = (cfg.decoder_hidden as usize, cfg.vocab as usize);
        let mut rng = Rng::new(3);
        let mut r = |n: usize| (0..n).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>();
        let (enc, dec, head, bias) = (r(dh), r(dh), r(vocab * dh), r(vocab));
        let dlog: Vec<f32> = r(vocab); // arbitrary upstream grad (loss = Σ dlog·logits)
        let w: W = [("joint.head.weight".to_string(), head.clone()), ("joint.head.bias".to_string(), bias.clone())].into_iter().collect();

        let (d_enc, _d_dec, d_head, _d_bias) = joint_backward(&enc, &dec, &dlog, &w, &cfg);
        let loss = |e: &[f32], hd: &[f32]| -> f32 {
            let ww: W = [("joint.head.weight".to_string(), hd.to_vec()), ("joint.head.bias".to_string(), bias.clone())].into_iter().collect();
            joint(e, &dec, &ww, &cfg).iter().zip(&dlog).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-3f32;
        let ok = |a: f32, n: f32| (a - n).abs() <= 3e-3 + 6e-2 * n.abs();
        for &i in &[0usize, 3, 5] {
            let (mut ep, mut em) = (enc.clone(), enc.clone());
            ep[i] += eps;
            em[i] -= eps;
            let num = (loss(&ep, &head) - loss(&em, &head)) / (2.0 * eps);
            assert!(ok(d_enc[i], num), "d_enc[{i}] {} vs {}", d_enc[i], num);
        }
        for &i in &[0usize, 7, 20] {
            let (mut hp, mut hm) = (head.clone(), head.clone());
            hp[i] += eps;
            hm[i] -= eps;
            let num = (loss(&enc, &hp) - loss(&enc, &hm)) / (2.0 * eps);
            assert!(ok(d_head[i], num), "d_head[{i}] {} vs {}", d_head[i], num);
        }
    }

    #[test]
    fn conformer_block0_matches_reference() {
        if !Path::new(&format!("{GOLD}/block0.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let sub = read_f32(&format!("{GOLD}/subsampling.f32")); // block0 input [T, 1024]
        let ref_b0 = read_f32(&format!("{GOLD}/block0.f32"));
        let t = sub.len() / cfg.hidden as usize;
        let w = crate::import::load_tensors(Path::new(CKPT)).expect("load");
        let valid = cfg.subsampled_len(585) as usize;
        let out = conformer_block(&sub, &w, 0, &cfg, t, valid);
        // compare only valid frames; the invalid tail (frames >= valid) is garbage in
        // both and is dropped by the RNN-T (encoder_valid_lengths).
        let n = valid * cfg.hidden as usize;
        let d = out[..n].iter().zip(&ref_b0[..n]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("block0 (valid {valid}/{t}) maxdiff {d}");
        assert!(d < 2e-3, "block0 maxdiff {d}");
    }

    #[test]
    fn encoder_pooler_matches_reference() {
        if !Path::new(&format!("{GOLD}/pooler.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let sub = read_f32(&format!("{GOLD}/subsampling.f32"));
        let ref_pool = read_f32(&format!("{GOLD}/pooler.f32")); // [T, 640]
        let t = sub.len() / cfg.hidden as usize;
        let valid = cfg.subsampled_len(585) as usize;
        let w = crate::import::load_tensors(Path::new(CKPT)).expect("load");
        let pool = encode_pooler(&sub, &w, &cfg, t, valid, 0); // prompt_id 0 (en)
        let dh = cfg.decoder_hidden as usize;
        let n = valid * dh;
        let d = pool[..n].iter().zip(&ref_pool[..n]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("pooler (valid {valid}) maxdiff {d}");
        assert!(d < 5e-3, "pooler maxdiff {d}");
    }

    #[test]
    fn rnnt_greedy_matches_reference() {
        if !Path::new(&format!("{GOLD}/pooler.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let pooler = read_f32(&format!("{GOLD}/pooler.f32")); // [T, 640]
        let dh = cfg.decoder_hidden as usize;
        let t = pooler.len() / dh;
        let valid = cfg.subsampled_len(585) as usize;
        let w = crate::import::load_tensors(Path::new(CKPT)).expect("load");
        let emitted = rnnt_greedy(&pooler, valid.min(t), &w, &cfg);

        // golden output_ids include blanks + the decoder-start; the transcript is the
        // non-blank subsequence.
        let ref_ids: Vec<u32> = read_f32(&format!("{GOLD}/output_ids.f32")).iter().map(|&v| v as u32).collect();
        let ref_nonblank: Vec<u32> = ref_ids.into_iter().filter(|&x| x != cfg.blank_token_id).collect();
        eprintln!("brain emitted ({}): {:?}", emitted.len(), &emitted[..emitted.len().min(20)]);
        eprintln!("ref  nonblank ({}): {:?}", ref_nonblank.len(), &ref_nonblank[..ref_nonblank.len().min(20)]);
        assert_eq!(emitted, ref_nonblank, "RNN-T non-blank token sequence must match HF");
    }
}
