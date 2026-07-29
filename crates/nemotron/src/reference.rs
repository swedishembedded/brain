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

/// Backward of `feed_forward` w.r.t. `x` (loss grad `d_out[t,c]`). Returns `d_x[t,c]`.
fn feed_forward_backward(x: &[f32], w1: &[f32], w2: &[f32], t: usize, c: usize, ffn: usize, d_out: &[f32]) -> Vec<f32> {
    let h1 = matmul_nt(x, w1, t, c, ffn); // pre-silu
    let mut d_h1 = vec![0.0f32; t * ffn];
    for i in 0..t {
        for j in 0..ffn {
            let mut ds = 0.0f32; // d_s = d_out · w2
            for o in 0..c {
                ds += d_out[i * c + o] * w2[o * ffn + j];
            }
            let x0 = h1[i * ffn + j];
            let s = sigmoid(x0);
            d_h1[i * ffn + j] = ds * (s * (1.0 + x0 * (1.0 - s))); // × silu'
        }
    }
    let mut d_x = vec![0.0f32; t * c];
    for i in 0..t {
        for inp in 0..c {
            let mut acc = 0.0f32;
            for o in 0..ffn {
                acc += d_h1[i * ffn + o] * w1[o * c + inp];
            }
            d_x[i * c + inp] = acc;
        }
    }
    d_x
}

/// LayerNorm backward returning `(d_x, d_gamma, d_beta)`.
pub(crate) fn layernorm_grads(d_y: &[f32], x: &[f32], gamma: &[f32], t: usize, c: usize, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d_x = layernorm_backward(d_y, x, gamma, t, c, eps);
    let (mut d_g, mut d_b) = (vec![0.0f32; c], vec![0.0f32; c]);
    for i in 0..t {
        let row = &x[i * c..i * c + c];
        let mu = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..c {
            d_g[j] += d_y[i * c + j] * (row[j] - mu) * inv;
            d_b[j] += d_y[i * c + j];
        }
    }
    (d_x, d_g, d_b)
}

/// Feed-forward backward returning `(d_x, d_w1, d_w2)`.
fn feed_forward_grads(x: &[f32], w1: &[f32], w2: &[f32], t: usize, c: usize, ffn: usize, d_out: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h1 = matmul_nt(x, w1, t, c, ffn);
    let s: Vec<f32> = h1.iter().map(|&x0| silu(x0)).collect();
    let mut d_h1 = vec![0.0f32; t * ffn];
    for i in 0..t {
        for j in 0..ffn {
            let mut ds = 0.0f32;
            for o in 0..c {
                ds += d_out[i * c + o] * w2[o * ffn + j];
            }
            let (x0, sg) = (h1[i * ffn + j], sigmoid(h1[i * ffn + j]));
            d_h1[i * ffn + j] = ds * (sg * (1.0 + x0 * (1.0 - sg)));
        }
    }
    // d_w2 = d_outᵀ·s [c,ffn]; d_w1 = d_h1ᵀ·x [ffn,c]; d_x = d_h1·w1
    let mut d_w2 = vec![0.0f32; c * ffn];
    for o in 0..c {
        for j in 0..ffn {
            let mut a = 0.0f32;
            for i in 0..t {
                a += d_out[i * c + o] * s[i * ffn + j];
            }
            d_w2[o * ffn + j] = a;
        }
    }
    let mut d_w1 = vec![0.0f32; ffn * c];
    for o in 0..ffn {
        for j in 0..c {
            let mut a = 0.0f32;
            for i in 0..t {
                a += d_h1[i * ffn + o] * x[i * c + j];
            }
            d_w1[o * c + j] = a;
        }
    }
    let mut d_x = vec![0.0f32; t * c];
    for i in 0..t {
        for inp in 0..c {
            let mut a = 0.0f32;
            for o in 0..ffn {
                a += d_h1[i * ffn + o] * w1[o * c + inp];
            }
            d_x[i * c + inp] = a;
        }
    }
    (d_x, d_w1, d_w2)
}

/// LayerNorm backward w.r.t. the input `x` (per-row over `c`). Returns `d_x`.
pub(crate) fn layernorm_backward(d_y: &[f32], x: &[f32], gamma: &[f32], t: usize, c: usize, eps: f32) -> Vec<f32> {
    let mut d_x = vec![0.0f32; t * c];
    for i in 0..t {
        let row = &x[i * c..i * c + c];
        let mu = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / c as f32;
        let sig = (var + eps).sqrt();
        let dyh: Vec<f32> = (0..c).map(|j| d_y[i * c + j] * gamma[j]).collect();
        let mean_dyh = dyh.iter().sum::<f32>() / c as f32;
        let mean_dyh_xhat = (0..c).map(|j| dyh[j] * (row[j] - mu) / sig).sum::<f32>() / c as f32;
        for j in 0..c {
            let xhat = (row[j] - mu) / sig;
            d_x[i * c + j] = (dyh[j] - mean_dyh - xhat * mean_dyh_xhat) / sig;
        }
    }
    d_x
}

/// Full Conformer-block backward w.r.t. the block input `h0` (loss grad `d_out[T,C]`).
/// Composes the macaron structure: LN_out → FF2 → conv → attn → FF1, each residual
/// split + LayerNorm backward. Returns `d_h0[T,C]`. The single repeating unit; the
/// 24-layer encoder backward is this chained.
pub fn conformer_block_backward(h0: &[f32], w: &W, b: u32, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> Vec<f32> {
    let (c, ffn) = (cfg.hidden as usize, cfg.intermediate as usize);
    let eps = cfg.ln_eps;
    let pre = format!("encoder.layers.{b}");
    let g = |n: &str| &w[&format!("{pre}.{n}")];
    let add = |a: &mut [f32], x: &[f32], s: f32| a.iter_mut().zip(x).for_each(|(y, &v)| *y += s * v);

    // recompute the residual-stream checkpoints h0..h3 (inputs to attn/conv/FF2)
    let mut h1 = h0.to_vec();
    let n1 = layernorm(&h1, g("norm_feed_forward1.weight"), g("norm_feed_forward1.bias"), t, c, eps);
    add(&mut h1, &feed_forward(&n1, g("feed_forward1.linear1.weight"), g("feed_forward1.linear2.weight"), t, c, ffn), 0.5);
    let mut h2 = h1.clone();
    let na = layernorm(&h2, g("norm_self_att.weight"), g("norm_self_att.bias"), t, c, eps);
    add(&mut h2, &rel_pos_attention(&na, w, &format!("{pre}.self_attn"), cfg, t, valid), 1.0);
    let mut h3 = h2.clone();
    let nc = layernorm(&h3, g("norm_conv.weight"), g("norm_conv.bias"), t, c, eps);
    add(&mut h3, &conv_module(&nc, w, &format!("{pre}.conv"), cfg, t, valid), 1.0);
    let mut h4 = h3.clone();
    let n2 = layernorm(&h4, g("norm_feed_forward2.weight"), g("norm_feed_forward2.bias"), t, c, eps);
    add(&mut h4, &feed_forward(&n2, g("feed_forward2.linear1.weight"), g("feed_forward2.linear2.weight"), t, c, ffn), 0.5);

    // ---- backward ----
    // out = LN_out(h4)
    let mut d_h4 = layernorm_backward(d_out, &h4, g("norm_out.weight"), t, c, eps);
    // h4 = h3 + 0.5·FF2(LN(h3))
    let d_n2 = feed_forward_backward(&n2, g("feed_forward2.linear1.weight"), g("feed_forward2.linear2.weight"), t, c, ffn, &d_h4.iter().map(|v| 0.5 * v).collect::<Vec<_>>());
    let mut d_h3 = layernorm_backward(&d_n2, &h3, g("norm_feed_forward2.weight"), t, c, eps);
    for i in 0..t * c {
        d_h3[i] += d_h4[i];
    }
    // h3 = h2 + conv(LN(h2))
    let d_nc = conv_module_backward(&nc, w, &format!("{pre}.conv"), cfg, t, valid, &d_h3);
    let mut d_h2 = layernorm_backward(&d_nc, &h2, g("norm_conv.weight"), t, c, eps);
    for i in 0..t * c {
        d_h2[i] += d_h3[i];
    }
    // h2 = h1 + attn(LN(h1))
    let (d_na, _dbv) = rel_pos_attention_backward(&na, w, &format!("{pre}.self_attn"), cfg, t, valid, &d_h2);
    let mut d_h1 = layernorm_backward(&d_na, &h1, g("norm_self_att.weight"), t, c, eps);
    for i in 0..t * c {
        d_h1[i] += d_h2[i];
    }
    // h1 = h0 + 0.5·FF1(LN(h0))
    let d_n1 = feed_forward_backward(&n1, g("feed_forward1.linear1.weight"), g("feed_forward1.linear2.weight"), t, c, ffn, &d_h1.iter().map(|v| 0.5 * v).collect::<Vec<_>>());
    let mut d_h0 = layernorm_backward(&d_n1, h0, g("norm_feed_forward1.weight"), t, c, eps);
    for i in 0..t * c {
        d_h0[i] += d_h1[i];
    }
    let _ = &mut d_h4;
    d_h0
}

/// Relative positional rows `[positions.len(), C]`: interleaved sin/cos per
/// position value, `inv_freq[i] = 10000^(-2i/C)`. A row depends only on its
/// position *value*, so the offline `[2T-1]` ladder and the streaming band table
/// share this one implementation (bit-identical rows for equal positions).
pub(crate) fn rel_pos_rows(positions: &[f32], c: usize) -> Vec<f32> {
    let half = c / 2;
    let inv: Vec<f32> = (0..half).map(|i| (10000f32).powf(-(2.0 * i as f32) / c as f32)).collect();
    let mut pe = vec![0.0f32; positions.len() * c];
    for (idx, &pos) in positions.iter().enumerate() {
        for i in 0..half {
            let f = pos * inv[i];
            pe[idx * c + 2 * i] = f.sin();
            pe[idx * c + 2 * i + 1] = f.cos();
        }
    }
    pe
}

/// Relative positional encoding `[2T-1, C]` over positions `[T-1 .. -(T-1)]`.
pub(crate) fn rel_pos_encoding(t: usize, c: usize) -> Vec<f32> {
    let pos: Vec<f32> = (0..2 * t - 1).map(|idx| (t as i64 - 1 - idx as i64) as f32).collect();
    rel_pos_rows(&pos, c)
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

/// Backward of `rel_pos_attention` w.r.t. `hn` and `bias_v` (thin wrapper over
/// [`rel_pos_attention_grads`]).
#[allow(clippy::too_many_arguments)]
fn rel_pos_attention_backward(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (d_hn, g) = rel_pos_attention_grads(hn, w, prefix, cfg, t, valid, d_out);
    (d_hn, g[&format!("{prefix}.bias_v")].clone())
}

/// Full rel-pos attention backward: `(d_hn, weight_grads)` — grads for
/// `{prefix}.{q,k,v,o,relative_k}_proj.weight` and `{prefix}.bias_{u,v}`.
#[allow(clippy::too_many_arguments)]
fn rel_pos_attention_grads(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> (Vec<f32>, W) {
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
    let mut d_bias_u = vec![0.0f32; c];
    let mut d_rel_k = vec![0.0f32; (2 * t - 1) * c];
    let mut ctx = vec![0.0f32; t * c];
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
        // ctx = probs · v_head (recomputed for d_Wo)
        for i in 0..t {
            for d in 0..hd {
                let mut a = 0.0f32;
                for j in 0..t {
                    a += probs[i * t + j] * v[j * c + h * hd + d];
                }
                ctx[i * c + h * hd + d] = a;
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
                    // d_ac: ac = (q+bu)·k → d(q_u), d(k); bias_u grad
                    dq[i * c + h * hd + d] += d_ac * kh(j, d);
                    d_bias_u[h * hd + d] += d_ac * kh(j, d);
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
                    // bd_raw = (q+bv)·rel_k → d(q_v), d_bias_v, and d(rel_k)
                    let dqv = g * rkh(pp, d);
                    dq[i * c + h * hd + d] += dqv;
                    d_bias_v[h * hd + d] += dqv;
                    d_rel_k[pp * c + h * hd + d] += g * (qh(i, d) + bvs[d]);
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
    // weight grads d_W[o,in] = Σ_i dproj[i,o]·input[i,in]
    let dw = |dproj: &[f32], input: &[f32], m: usize, kk: usize, n: usize| -> Vec<f32> {
        let mut r = vec![0.0f32; n * kk];
        for o in 0..n {
            for j in 0..kk {
                let mut a = 0.0f32;
                for i in 0..m {
                    a += dproj[i * n + o] * input[i * kk + j];
                }
                r[o * kk + j] = a;
            }
        }
        r
    };
    let mut grads: W = W::new();
    grads.insert(format!("{prefix}.q_proj.weight"), dw(&dq, hn, t, c, c));
    grads.insert(format!("{prefix}.k_proj.weight"), dw(&dk, hn, t, c, c));
    grads.insert(format!("{prefix}.v_proj.weight"), dw(&dv, hn, t, c, c));
    grads.insert(format!("{prefix}.o_proj.weight"), dw(d_out, &ctx, t, c, c));
    grads.insert(format!("{prefix}.relative_k_proj.weight"), dw(&d_rel_k, &pe, 2 * t - 1, c, c));
    grads.insert(format!("{prefix}.bias_u"), d_bias_u);
    grads.insert(format!("{prefix}.bias_v"), d_bias_v);
    (d_hn, grads)
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

/// Backward of `conv_module` w.r.t. `hn` (thin wrapper over [`conv_module_grads`]).
fn conv_module_backward(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> Vec<f32> {
    conv_module_grads(hn, w, prefix, cfg, t, valid, d_out).0
}

/// Full conv-module backward: `(d_hn, weight_grads)` — grads for
/// `{prefix}.{pointwise_conv1,pointwise_conv2,depthwise_conv}.weight` and `{prefix}.norm.{weight,bias}`.
fn conv_module_grads(hn: &[f32], w: &W, prefix: &str, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> (Vec<f32>, W) {
    let (c, k) = (cfg.hidden as usize, cfg.conv_kernel as usize);
    let eps = cfg.ln_eps;
    let p = |n: &str| &w[&format!("{prefix}.{n}")];
    let (wpc1, dw, gnw, gnb, wpc2) = (p("pointwise_conv1.weight"), p("depthwise_conv.weight"), p("norm.weight"), p("norm.bias"), p("pointwise_conv2.weight"));

    // ---- forward recompute (need pc1, glu, conv, ln pre-silu) ----
    let pc1 = matmul_nt(hn, wpc1, t, c, 2 * c);
    let mut glu = vec![0.0f32; t * c];
    for i in 0..valid.min(t) {
        for j in 0..c {
            glu[i * c + j] = pc1[i * 2 * c + j] * sigmoid(pc1[i * 2 * c + c + j]);
        }
    }
    let mut conv = vec![0.0f32; t * c];
    for ch in 0..c {
        for i in 0..t {
            let mut a = 0.0f32;
            for kk in 0..k {
                let s = i as i64 - (k as i64 - 1) + kk as i64;
                if s >= 0 {
                    a += glu[s as usize * c + ch] * dw[ch * k + kk];
                }
            }
            conv[i * c + ch] = a;
        }
    }
    let ln = layernorm(&conv, gnw, gnb, t, c, eps); // pre-silu

    // ---- backward ----
    // d_out → d_act via pc2 (out = act·Wpc2ᵀ): d_act[i,j] = Σ_o d_out[i,o]·Wpc2[o,j]
    let mut d_act = vec![0.0f32; t * c];
    for i in 0..t {
        for j in 0..c {
            let mut acc = 0.0f32;
            for o in 0..c {
                acc += d_out[i * c + o] * wpc2[o * c + j];
            }
            d_act[i * c + j] = acc;
        }
    }
    // d_Wpc2 = d_outᵀ · act  (act = silu(ln))
    let act: Vec<f32> = ln.iter().map(|&x| silu(x)).collect();
    let mut d_wpc2 = vec![0.0f32; c * c];
    for o in 0..c {
        for j in 0..c {
            let mut a = 0.0f32;
            for i in 0..t {
                a += d_out[i * c + o] * act[i * c + j];
            }
            d_wpc2[o * c + j] = a;
        }
    }
    // through SiLU: act = silu(ln); silu'(x) = sig(x)(1 + x(1-sig(x)))
    let mut d_ln = vec![0.0f32; t * c];
    for idx in 0..t * c {
        let x = ln[idx];
        let s = sigmoid(x);
        d_ln[idx] = d_act[idx] * (s * (1.0 + x * (1.0 - s)));
    }
    // through LayerNorm (per row over C) + norm gamma/beta grads
    let mut d_conv = vec![0.0f32; t * c];
    let (mut d_gnw, mut d_gnb) = (vec![0.0f32; c], vec![0.0f32; c]);
    for i in 0..t {
        let row = &conv[i * c..i * c + c];
        let mu = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / c as f32;
        let sig = (var + eps).sqrt();
        let dyh: Vec<f32> = (0..c).map(|j| d_ln[i * c + j] * gnw[j]).collect();
        let mean_dyh = dyh.iter().sum::<f32>() / c as f32;
        let mean_dyh_xhat = (0..c).map(|j| dyh[j] * (row[j] - mu) / sig).sum::<f32>() / c as f32;
        for j in 0..c {
            let xhat = (row[j] - mu) / sig;
            d_conv[i * c + j] = (dyh[j] - mean_dyh - xhat * mean_dyh_xhat) / sig;
            d_gnw[j] += d_ln[i * c + j] * xhat;
            d_gnb[j] += d_ln[i * c + j];
        }
    }
    // through causal depthwise conv1d (transpose): d_glu[m,ch] = Σ_kk d_conv[m+8-kk,ch]·dw[ch,kk]
    let mut d_glu = vec![0.0f32; t * c];
    for ch in 0..c {
        for m in 0..t {
            let mut a = 0.0f32;
            for kk in 0..k {
                let i = m as i64 + (k as i64 - 1) - kk as i64;
                if i >= 0 && (i as usize) < t {
                    a += d_conv[i as usize * c + ch] * dw[ch * k + kk];
                }
            }
            d_glu[m * c + ch] = a;
        }
    }
    // through GLU → d_pc1 (both halves); masked rows contribute 0
    let mut d_pc1 = vec![0.0f32; t * 2 * c];
    for i in 0..valid.min(t) {
        for j in 0..c {
            let (a, b) = (pc1[i * 2 * c + j], pc1[i * 2 * c + c + j]);
            let s = sigmoid(b);
            let dg = d_glu[i * c + j];
            d_pc1[i * 2 * c + j] = dg * s;
            d_pc1[i * 2 * c + c + j] = dg * a * s * (1.0 - s);
        }
    }
    // depthwise weight grad: d_dw[ch,kk] = Σ_i d_conv[i,ch]·glu[i-(k-1)+kk, ch]
    let mut d_dw = vec![0.0f32; c * k];
    for ch in 0..c {
        for kk in 0..k {
            let mut a = 0.0f32;
            for i in 0..t {
                let s = i as i64 - (k as i64 - 1) + kk as i64;
                if s >= 0 {
                    a += d_conv[i * c + ch] * glu[s as usize * c + ch];
                }
            }
            d_dw[ch * k + kk] = a;
        }
    }
    // pc1 weight grad: d_Wpc1[o,in] = Σ_i d_pc1[i,o]·hn[i,in]  [2C, C]
    let mut d_wpc1 = vec![0.0f32; 2 * c * c];
    for o in 0..2 * c {
        for inp in 0..c {
            let mut a = 0.0f32;
            for i in 0..t {
                a += d_pc1[i * 2 * c + o] * hn[i * c + inp];
            }
            d_wpc1[o * c + inp] = a;
        }
    }
    // through pc1 (out=hn·Wpc1ᵀ): d_hn[i,in] = Σ_o d_pc1[i,o]·Wpc1[o,in]
    let mut d_hn = vec![0.0f32; t * c];
    for i in 0..t {
        for inp in 0..c {
            let mut acc = 0.0f32;
            for o in 0..2 * c {
                acc += d_pc1[i * 2 * c + o] * wpc1[o * c + inp];
            }
            d_hn[i * c + inp] = acc;
        }
    }
    let mut grads: W = W::new();
    grads.insert(format!("{prefix}.pointwise_conv1.weight"), d_wpc1);
    grads.insert(format!("{prefix}.pointwise_conv2.weight"), d_wpc2);
    grads.insert(format!("{prefix}.depthwise_conv.weight"), d_dw);
    grads.insert(format!("{prefix}.norm.weight"), d_gnw);
    grads.insert(format!("{prefix}.norm.bias"), d_gnb);
    (d_hn, grads)
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

/// BPTT backward of the LSTM prediction network over a token sequence. Given the
/// grad of each step's decoder output `d_dec[steps][dh]`, returns the gradient of
/// the embedding table rows used (`d_embed[vocab*dh]`). Exercises the full
/// recurrent training path (lstm gate backward + W_ih/W_hh matmul bwd, chained in
/// time). Validated by finite differences.
pub fn predictor_sequence_backward(tokens: &[u32], w: &W, cfg: &NemotronConfig, d_dec: &[Vec<f32>]) -> Vec<f32> {
    predictor_grads(tokens, w, cfg, d_dec).remove("decoder.embedding.weight").unwrap()
}

/// Full BPTT gradient map for the LSTM prediction net + decoder_projector: returns
/// grads for `decoder.embedding.weight`, `decoder.lstm.weight_{ih,hh}_l*`,
/// `decoder.lstm.bias_{ih,hh}_l*`, and `decoder.decoder_projector.{weight,bias}`.
pub fn predictor_grads(tokens: &[u32], w: &W, cfg: &NemotronConfig, d_dec: &[Vec<f32>]) -> W {
    let dh = cfg.decoder_hidden as usize;
    let layers = cfg.num_decoder_layers as usize;
    let emb = &w["decoder.embedding.weight"];
    let wih: Vec<&Vec<f32>> = (0..layers).map(|l| &w[&format!("decoder.lstm.weight_ih_l{l}")]).collect();
    let whh: Vec<&Vec<f32>> = (0..layers).map(|l| &w[&format!("decoder.lstm.weight_hh_l{l}")]).collect();
    let wproj = &w["decoder.decoder_projector.weight"];

    // forward, caching per (step, layer): input, h_prev, c_prev, pre, c_new
    let n = tokens.len();
    let mut ca_input = vec![vec![vec![0.0f32; dh]; layers]; n];
    let mut ca_hprev = vec![vec![vec![0.0f32; dh]; layers]; n];
    let mut ca_cprev = vec![vec![vec![0.0f32; dh]; layers]; n];
    let mut ca_pre = vec![vec![vec![0.0f32; 4 * dh]; layers]; n];
    let mut ca_cnew = vec![vec![vec![0.0f32; dh]; layers]; n];
    let mut ca_projin = vec![vec![0.0f32; dh]; n];
    let (mut hs, mut cs) = (vec![vec![0.0f32; dh]; layers], vec![vec![0.0f32; dh]; layers]);
    for (ti, &tok) in tokens.iter().enumerate() {
        let mut input = emb[tok as usize * dh..tok as usize * dh + dh].to_vec();
        for l in 0..layers {
            ca_input[ti][l] = input.clone();
            ca_hprev[ti][l] = hs[l].clone();
            ca_cprev[ti][l] = cs[l].clone();
            let gi = matmul_nt(&input, wih[l], 1, dh, 4 * dh);
            let gh = matmul_nt(&hs[l], whh[l], 1, dh, 4 * dh);
            let bih = &w[&format!("decoder.lstm.bias_ih_l{l}")];
            let bhh = &w[&format!("decoder.lstm.bias_hh_l{l}")];
            let mut out = vec![0.0f32; dh];
            for j in 0..dh {
                let pre = |o: usize| gi[o * dh + j] + bih[o * dh + j] + gh[o * dh + j] + bhh[o * dh + j];
                for o in 0..4 {
                    ca_pre[ti][l][o * dh + j] = pre(o);
                }
                let (ii, ff, gg, oo) = (sigmoid(pre(0)), sigmoid(pre(1)), pre(2).tanh(), sigmoid(pre(3)));
                let ct = ff * cs[l][j] + ii * gg;
                cs[l][j] = ct;
                out[j] = oo * ct.tanh();
            }
            ca_cnew[ti][l] = cs[l].clone();
            hs[l] = out.clone();
            input = out;
        }
        ca_projin[ti] = input; // h[last]
    }

    // backward (BPTT) — accumulate ALL predictor weight grads
    let mut d_embed = vec![0.0f32; emb.len()];
    let mut d_wih = vec![vec![0.0f32; 4 * dh * dh]; layers];
    let mut d_whh = vec![vec![0.0f32; 4 * dh * dh]; layers];
    let mut d_bih = vec![vec![0.0f32; 4 * dh]; layers];
    let mut d_bhh = vec![vec![0.0f32; 4 * dh]; layers];
    let mut d_wproj = vec![0.0f32; dh * dh];
    let mut d_bproj = vec![0.0f32; dh];
    let mut d_h = vec![vec![0.0f32; dh]; layers];
    let mut d_c = vec![vec![0.0f32; dh]; layers];
    for ti in (0..n).rev() {
        // decoder_projector: dec = projin·Wprojᵀ + b → d_projin = d_dec·Wproj; grads
        let dd = &d_dec[ti];
        let projin = &ca_projin[ti];
        let mut d_top = vec![0.0f32; dh];
        for o in 0..dh {
            d_bproj[o] += dd[o];
            for j in 0..dh {
                d_top[j] += dd[o] * wproj[o * dh + j];
                d_wproj[o * dh + j] += dd[o] * projin[j];
            }
        }
        for j in 0..dh {
            d_h[layers - 1][j] += d_top[j];
        }
        for l in (0..layers).rev() {
            let (pre, cprev, cnew) = (&ca_pre[ti][l], &ca_cprev[ti][l], &ca_cnew[ti][l]);
            let mut d_pre = vec![0.0f32; 4 * dh];
            for j in 0..dh {
                let (ii, ff, gg, oo) = (sigmoid(pre[j]), sigmoid(pre[dh + j]), pre[2 * dh + j].tanh(), sigmoid(pre[3 * dh + j]));
                let tc = cnew[j].tanh();
                let doo = d_h[l][j] * tc;
                let dc = d_c[l][j] + d_h[l][j] * oo * (1.0 - tc * tc);
                let (di, df, dg) = (dc * gg, dc * cprev[j], dc * ii);
                d_c[l][j] = dc * ff;
                d_pre[j] = di * ii * (1.0 - ii);
                d_pre[dh + j] = df * ff * (1.0 - ff);
                d_pre[2 * dh + j] = dg * (1.0 - gg * gg);
                d_pre[3 * dh + j] = doo * oo * (1.0 - oo);
            }
            // weight grads: d_W_ih += outer(d_pre, input); d_W_hh += outer(d_pre, h_prev); biases += d_pre
            let (inp, hprev) = (&ca_input[ti][l], &ca_hprev[ti][l]);
            for o in 0..4 * dh {
                d_bih[l][o] += d_pre[o];
                d_bhh[l][o] += d_pre[o];
                for j in 0..dh {
                    d_wih[l][o * dh + j] += d_pre[o] * inp[j];
                    d_whh[l][o * dh + j] += d_pre[o] * hprev[j];
                }
            }
            // d_input = W_ihᵀ·d_pre ; d_hprev = W_hhᵀ·d_pre
            let mut d_input = vec![0.0f32; dh];
            let mut d_hprev = vec![0.0f32; dh];
            for j in 0..dh {
                let mut ai = 0.0f32;
                let mut ah = 0.0f32;
                for o in 0..4 * dh {
                    ai += d_pre[o] * wih[l][o * dh + j];
                    ah += d_pre[o] * whh[l][o * dh + j];
                }
                d_input[j] = ai;
                d_hprev[j] = ah;
            }
            for j in 0..dh {
                d_h[l][j] = d_hprev[j];
            }
            if l > 0 {
                for j in 0..dh {
                    d_h[l - 1][j] += d_input[j];
                }
            } else {
                let tok = tokens[ti] as usize;
                for j in 0..dh {
                    d_embed[tok * dh + j] += d_input[j];
                }
            }
        }
    }
    let mut grads: W = W::new();
    grads.insert("decoder.embedding.weight".into(), d_embed);
    for l in 0..layers {
        grads.insert(format!("decoder.lstm.weight_ih_l{l}"), std::mem::take(&mut d_wih[l]));
        grads.insert(format!("decoder.lstm.weight_hh_l{l}"), std::mem::take(&mut d_whh[l]));
        grads.insert(format!("decoder.lstm.bias_ih_l{l}"), std::mem::take(&mut d_bih[l]));
        grads.insert(format!("decoder.lstm.bias_hh_l{l}"), std::mem::take(&mut d_bhh[l]));
    }
    grads.insert("decoder.decoder_projector.weight".into(), d_wproj);
    grads.insert("decoder.decoder_projector.bias".into(), d_bproj);
    grads
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

fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// RNN-T (transducer) loss over the `T×(U+1)` lattice and its gradient w.r.t. the
/// joint logits. `logits` is `[T, U+1, vocab]` (joint output for each encoder frame
/// t and predictor position u); `targets` are the `U` label ids. Returns
/// `(loss, d_logits)`. Standard Graves forward/backward (alpha/beta) with the
/// softmax backward folded in. This is the RNN-T training objective.
pub fn rnnt_loss(logits: &[f32], t_frames: usize, targets: &[u32], blank: usize, vocab: usize) -> (f32, Vec<f32>) {
    let u = targets.len();
    let up1 = u + 1;
    let at = |t: usize, uu: usize| (t * up1 + uu) * vocab;
    // log-softmax rows + the two transition log-probs
    let mut lsm = vec![0.0f32; t_frames * up1 * vocab]; // full log-softmax (for grad)
    let mut lpb = vec![f32::NEG_INFINITY; t_frames * up1]; // log P(blank | t,u)
    let mut lpl = vec![f32::NEG_INFINITY; t_frames * up1]; // log P(target_u | t,u)
    for t in 0..t_frames {
        for uu in 0..up1 {
            let row = &logits[at(t, uu)..at(t, uu) + vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + row.iter().map(|&v| (v - mx).exp()).sum::<f32>().ln();
            for v in 0..vocab {
                lsm[at(t, uu) + v] = row[v] - lse;
            }
            lpb[t * up1 + uu] = lsm[at(t, uu) + blank];
            if uu < u {
                lpl[t * up1 + uu] = lsm[at(t, uu) + targets[uu] as usize];
            }
        }
    }
    let idx = |t: usize, uu: usize| t * up1 + uu;
    // forward alpha
    let mut alpha = vec![f32::NEG_INFINITY; t_frames * up1];
    alpha[0] = 0.0;
    for t in 0..t_frames {
        for uu in 0..up1 {
            if t == 0 && uu == 0 {
                continue;
            }
            let mut a = f32::NEG_INFINITY;
            if t > 0 {
                a = logaddexp(a, alpha[idx(t - 1, uu)] + lpb[idx(t - 1, uu)]);
            }
            if uu > 0 {
                a = logaddexp(a, alpha[idx(t, uu - 1)] + lpl[idx(t, uu - 1)]);
            }
            alpha[idx(t, uu)] = a;
        }
    }
    let logz = alpha[idx(t_frames - 1, u)] + lpb[idx(t_frames - 1, u)];
    let loss = -logz;
    // backward beta
    let mut beta = vec![f32::NEG_INFINITY; t_frames * up1];
    beta[idx(t_frames - 1, u)] = lpb[idx(t_frames - 1, u)];
    for t in (0..t_frames).rev() {
        for uu in (0..up1).rev() {
            if t == t_frames - 1 && uu == u {
                continue;
            }
            let mut b = f32::NEG_INFINITY;
            if t < t_frames - 1 {
                b = logaddexp(b, beta[idx(t + 1, uu)] + lpb[idx(t, uu)]);
            }
            if uu < u {
                b = logaddexp(b, beta[idx(t, uu + 1)] + lpl[idx(t, uu)]);
            }
            beta[idx(t, uu)] = b;
        }
    }
    // grads w.r.t. the transition log-probs, then through log-softmax
    let mut d_logits = vec![0.0f32; t_frames * up1 * vocab];
    for t in 0..t_frames {
        for uu in 0..up1 {
            // d loss / d lp_blank[t,u]
            let d_lpb = if t < t_frames - 1 {
                -(alpha[idx(t, uu)] + lpb[idx(t, uu)] + beta[idx(t + 1, uu)] - logz).exp()
            } else if uu == u {
                -1.0 // terminal blank
            } else {
                0.0
            };
            let d_lpl = if uu < u {
                -(alpha[idx(t, uu)] + lpl[idx(t, uu)] + beta[idx(t, uu + 1)] - logz).exp()
            } else {
                0.0
            };
            // through log-softmax: d logit[v] = d_lp_k·(δ_{v,k} − softmax[v])
            let sm: Vec<f32> = (0..vocab).map(|v| lsm[at(t, uu) + v].exp()).collect();
            let dsum = d_lpb + d_lpl;
            for v in 0..vocab {
                let mut g = -sm[v] * dsum;
                if v == blank {
                    g += d_lpb;
                }
                if uu < u && v == targets[uu] as usize {
                    g += d_lpl;
                }
                d_logits[at(t, uu) + v] = g;
            }
        }
    }
    (loss, d_logits)
}

/// Full Conformer-block backward returning `(d_h0, block_weight_grads)` — every
/// parameter of the block keyed `encoder.layers.{b}.<leaf>`.
pub fn conformer_block_grads(h0: &[f32], w: &W, b: u32, cfg: &NemotronConfig, t: usize, valid: usize, d_out: &[f32]) -> (Vec<f32>, W) {
    let (c, ffn, eps) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.ln_eps);
    let pre = format!("encoder.layers.{b}");
    let g = |n: &str| &w[&format!("{pre}.{n}")];
    let mut gr: W = W::new();
    // recompute residual checkpoints + the normalized inputs
    let mut h1 = h0.to_vec();
    let n1 = layernorm(&h1, g("norm_feed_forward1.weight"), g("norm_feed_forward1.bias"), t, c, eps);
    let ff1 = feed_forward(&n1, g("feed_forward1.linear1.weight"), g("feed_forward1.linear2.weight"), t, c, ffn);
    for i in 0..t * c {
        h1[i] += 0.5 * ff1[i];
    }
    let mut h2 = h1.clone();
    let na = layernorm(&h2, g("norm_self_att.weight"), g("norm_self_att.bias"), t, c, eps);
    let att = rel_pos_attention(&na, w, &format!("{pre}.self_attn"), cfg, t, valid);
    for i in 0..t * c {
        h2[i] += att[i];
    }
    let mut h3 = h2.clone();
    let nc = layernorm(&h3, g("norm_conv.weight"), g("norm_conv.bias"), t, c, eps);
    let cv = conv_module(&nc, w, &format!("{pre}.conv"), cfg, t, valid);
    for i in 0..t * c {
        h3[i] += cv[i];
    }
    let mut h4 = h3.clone();
    let n2 = layernorm(&h4, g("norm_feed_forward2.weight"), g("norm_feed_forward2.bias"), t, c, eps);
    let ff2 = feed_forward(&n2, g("feed_forward2.linear1.weight"), g("feed_forward2.linear2.weight"), t, c, ffn);
    for i in 0..t * c {
        h4[i] += 0.5 * ff2[i];
    }

    let half = |d: &[f32]| -> Vec<f32> { d.iter().map(|v| 0.5 * v).collect() };
    // LN_out
    let (mut d_h4, dg, db) = layernorm_grads(d_out, &h4, g("norm_out.weight"), t, c, eps);
    gr.insert(format!("{pre}.norm_out.weight"), dg);
    gr.insert(format!("{pre}.norm_out.bias"), db);
    // FF2
    let (d_n2, dw1, dw2) = feed_forward_grads(&n2, g("feed_forward2.linear1.weight"), g("feed_forward2.linear2.weight"), t, c, ffn, &half(&d_h4));
    gr.insert(format!("{pre}.feed_forward2.linear1.weight"), dw1);
    gr.insert(format!("{pre}.feed_forward2.linear2.weight"), dw2);
    let (d_h3ln, dg, db) = layernorm_grads(&d_n2, &h3, g("norm_feed_forward2.weight"), t, c, eps);
    gr.insert(format!("{pre}.norm_feed_forward2.weight"), dg);
    gr.insert(format!("{pre}.norm_feed_forward2.bias"), db);
    let mut d_h3 = d_h3ln;
    for i in 0..t * c {
        d_h3[i] += d_h4[i];
    }
    // conv
    let (d_nc, cvg) = conv_module_grads(&nc, w, &format!("{pre}.conv"), cfg, t, valid, &d_h3);
    for (kk, vv) in cvg {
        gr.insert(kk, vv);
    }
    let (d_h2ln, dg, db) = layernorm_grads(&d_nc, &h2, g("norm_conv.weight"), t, c, eps);
    gr.insert(format!("{pre}.norm_conv.weight"), dg);
    gr.insert(format!("{pre}.norm_conv.bias"), db);
    let mut d_h2 = d_h2ln;
    for i in 0..t * c {
        d_h2[i] += d_h3[i];
    }
    // attention
    let (d_na, atg) = rel_pos_attention_grads(&na, w, &format!("{pre}.self_attn"), cfg, t, valid, &d_h2);
    for (kk, vv) in atg {
        gr.insert(kk, vv);
    }
    let (d_h1ln, dg, db) = layernorm_grads(&d_na, &h1, g("norm_self_att.weight"), t, c, eps);
    gr.insert(format!("{pre}.norm_self_att.weight"), dg);
    gr.insert(format!("{pre}.norm_self_att.bias"), db);
    let mut d_h1 = d_h1ln;
    for i in 0..t * c {
        d_h1[i] += d_h2[i];
    }
    // FF1
    let (d_n1, dw1, dw2) = feed_forward_grads(&n1, g("feed_forward1.linear1.weight"), g("feed_forward1.linear2.weight"), t, c, ffn, &half(&d_h1));
    gr.insert(format!("{pre}.feed_forward1.linear1.weight"), dw1);
    gr.insert(format!("{pre}.feed_forward1.linear2.weight"), dw2);
    let (d_h0ln, dg, db) = layernorm_grads(&d_n1, h0, g("norm_feed_forward1.weight"), t, c, eps);
    gr.insert(format!("{pre}.norm_feed_forward1.weight"), dg);
    gr.insert(format!("{pre}.norm_feed_forward1.bias"), db);
    let mut d_h0 = d_h0ln;
    for i in 0..t * c {
        d_h0[i] += d_h1[i];
    }
    let _ = &mut d_h4;
    (d_h0, gr)
}

/// Full encoder backward from the pooler grad `d_pooler[T, decoder_hidden]` to the
/// subsampling-output grad `d_sub[T, C]`: projector backward → 24 block backwards
/// (chained) → subsampling stack input. The model-level encoder training gradient.
pub fn encode_pooler_backward(sub: &[f32], w: &W, cfg: &NemotronConfig, t: usize, valid: usize, prompt_id: usize, d_pooler: &[f32]) -> Vec<f32> {
    let (c, np, pi, dh) = (cfg.hidden as usize, cfg.num_prompts as usize, cfg.prompt_intermediate as usize, cfg.decoder_hidden as usize);
    // ---- forward, caching each block's input ----
    let mut inputs = Vec::with_capacity(cfg.n_layers as usize + 1);
    inputs.push(sub.to_vec());
    for b in 0..cfg.n_layers {
        let out = conformer_block(inputs.last().unwrap(), w, b, cfg, t, valid);
        inputs.push(out);
    }
    let hidden = inputs.last().unwrap().clone();
    // prompt_projector forward (need cat + f1 pre-relu)
    let mut cat = vec![0.0f32; t * (c + np)];
    for i in 0..t {
        cat[i * (c + np)..i * (c + np) + c].copy_from_slice(&hidden[i * c..i * c + c]);
        cat[i * (c + np) + c + prompt_id] = 1.0;
    }
    let mut f1pre = matmul_nt(&cat, &w["prompt_projector.linear_1.weight"], t, c + np, pi);
    let b1 = &w["prompt_projector.linear_1.bias"];
    for i in 0..t {
        for j in 0..pi {
            f1pre[i * pi + j] += b1[j];
        }
    }

    // ---- backward ----
    // encoder_projector: pooler = fused·Wᵀ + b → d_fused = d_pooler·W
    let wep = &w["encoder_projector.weight"]; // [dh, c]
    let mut d_fused = vec![0.0f32; t * c];
    for i in 0..t {
        for j in 0..c {
            let mut a = 0.0f32;
            for o in 0..dh {
                a += d_pooler[i * dh + o] * wep[o * c + j];
            }
            d_fused[i * c + j] = a;
        }
    }
    // linear_2: fused = f1·W2ᵀ + b → d_f1 = d_fused·W2
    let w2 = &w["prompt_projector.linear_2.weight"]; // [c, pi]
    let mut d_f1 = vec![0.0f32; t * pi];
    for i in 0..t {
        for j in 0..pi {
            let mut a = 0.0f32;
            for o in 0..c {
                a += d_fused[i * c + o] * w2[o * pi + j];
            }
            d_f1[i * pi + j] = a * if f1pre[i * pi + j] > 0.0 { 1.0 } else { 0.0 }; // relu'
        }
    }
    // linear_1: cat = [hidden, onehot] → d_hidden = (d_f1·W1)[:, :c]
    let w1 = &w["prompt_projector.linear_1.weight"]; // [pi, c+np]
    let mut d_hidden = vec![0.0f32; t * c];
    for i in 0..t {
        for j in 0..c {
            let mut a = 0.0f32;
            for o in 0..pi {
                a += d_f1[i * pi + o] * w1[o * (c + np) + j];
            }
            d_hidden[i * c + j] = a;
        }
    }
    // chain block backwards (reverse), each consuming its cached input
    let mut d = d_hidden;
    for b in (0..cfg.n_layers).rev() {
        d = conformer_block_backward(&inputs[b as usize], w, b, cfg, t, valid, &d);
    }
    d // d_sub
}

/// Full encoder weight gradients: `(d_sub, encoder_grads)` — every encoder
/// parameter (all block params + prompt/encoder projectors) from the pooler grad.
pub fn encode_pooler_grads(sub: &[f32], w: &W, cfg: &NemotronConfig, t: usize, valid: usize, prompt_id: usize, d_pooler: &[f32]) -> (Vec<f32>, W) {
    let (c, np, pi, dh) = (cfg.hidden as usize, cfg.num_prompts as usize, cfg.prompt_intermediate as usize, cfg.decoder_hidden as usize);
    // forward cache: block inputs + projector intermediates
    let mut inputs = vec![sub.to_vec()];
    for b in 0..cfg.n_layers {
        let out = conformer_block(inputs.last().unwrap(), w, b, cfg, t, valid);
        inputs.push(out);
    }
    let hidden = inputs.last().unwrap().clone();
    let mut cat = vec![0.0f32; t * (c + np)];
    for i in 0..t {
        cat[i * (c + np)..i * (c + np) + c].copy_from_slice(&hidden[i * c..i * c + c]);
        cat[i * (c + np) + c + prompt_id] = 1.0;
    }
    let mut f1pre = matmul_nt(&cat, &w["prompt_projector.linear_1.weight"], t, c + np, pi);
    let b1 = &w["prompt_projector.linear_1.bias"];
    for i in 0..t {
        for j in 0..pi {
            f1pre[i * pi + j] += b1[j];
        }
    }
    let f1: Vec<f32> = f1pre.iter().map(|&x| x.max(0.0)).collect();
    let mut fused = matmul_nt(&f1, &w["prompt_projector.linear_2.weight"], t, pi, c);
    let b2 = &w["prompt_projector.linear_2.bias"];
    for i in 0..t {
        for j in 0..c {
            fused[i * c + j] += b2[j];
        }
    }

    let mut gr: W = W::new();
    let dwt = |dproj: &[f32], input: &[f32], m: usize, kk: usize, n: usize| {
        let mut r = vec![0.0f32; n * kk];
        for o in 0..n {
            for j in 0..kk {
                let mut a = 0.0f32;
                for i in 0..m {
                    a += dproj[i * n + o] * input[i * kk + j];
                }
                r[o * kk + j] = a;
            }
        }
        r
    };
    let dxt = |dproj: &[f32], wt: &[f32], m: usize, kk: usize, n: usize| {
        let mut r = vec![0.0f32; m * kk];
        for i in 0..m {
            for j in 0..kk {
                let mut a = 0.0f32;
                for o in 0..n {
                    a += dproj[i * n + o] * wt[o * kk + j];
                }
                r[i * kk + j] = a;
            }
        }
        r
    };
    // encoder_projector: pooler = fused·Wepᵀ + bep
    gr.insert("encoder_projector.weight".into(), dwt(d_pooler, &fused, t, c, dh));
    gr.insert("encoder_projector.bias".into(), (0..dh).map(|o| (0..t).map(|i| d_pooler[i * dh + o]).sum()).collect());
    let d_fused = dxt(d_pooler, &w["encoder_projector.weight"], t, c, dh);
    // prompt_projector linear_2
    gr.insert("prompt_projector.linear_2.weight".into(), dwt(&d_fused, &f1, t, pi, c));
    gr.insert("prompt_projector.linear_2.bias".into(), (0..c).map(|o| (0..t).map(|i| d_fused[i * c + o]).sum()).collect());
    let d_f1 = dxt(&d_fused, &w["prompt_projector.linear_2.weight"], t, pi, c);
    // relu' then linear_1
    let d_f1pre: Vec<f32> = (0..t * pi).map(|i| if f1pre[i] > 0.0 { d_f1[i] } else { 0.0 }).collect();
    gr.insert("prompt_projector.linear_1.weight".into(), dwt(&d_f1pre, &cat, t, c + np, pi));
    gr.insert("prompt_projector.linear_1.bias".into(), (0..pi).map(|o| (0..t).map(|i| d_f1pre[i * pi + o]).sum()).collect());
    let d_cat = dxt(&d_f1pre, &w["prompt_projector.linear_1.weight"], t, pi, c + np);
    // d_hidden = d_cat[:, :c]
    let mut d = vec![0.0f32; t * c];
    for i in 0..t {
        d[i * c..i * c + c].copy_from_slice(&d_cat[i * (c + np)..i * (c + np) + c]);
    }
    // chain block backwards (reverse), merging block grads
    for b in (0..cfg.n_layers).rev() {
        let (d_in, bg) = conformer_block_grads(&inputs[b as usize], w, b, cfg, t, valid, &d);
        for (k, v) in bg {
            gr.insert(k, v);
        }
        d = d_in;
    }
    (d, gr)
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // GOLD/CKPT test-path locals (see AGENTS.md: no absolute paths)
    use super::*;
    use std::io::Read;
    use std::path::Path;


    fn read_f32(p: &str) -> Vec<f32> {
        let mut f = std::fs::File::open(p).unwrap_or_else(|_| panic!("missing {p}"));
        let mut b = Vec::new();
        f.read_to_end(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn full_encoder_backward_matches_finite_diff() {
        use data::rng::Rng;
        // tiny 2-layer encoder + projectors; gradcheck d_sub end-to-end.
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.hidden = 8;
        cfg.n_heads = 2;
        cfg.intermediate = 16;
        cfg.conv_kernel = 3;
        cfg.n_layers = 2;
        cfg.num_prompts = 4;
        cfg.prompt_intermediate = 12;
        cfg.decoder_hidden = 6;
        let (c, ffn, k, np, pi, dh, t, valid) = (8usize, 16usize, 3usize, 4usize, 12usize, 6usize, 6usize, 6usize);
        let mut rng = Rng::new(51);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect::<Vec<f32>>();
        let mut w: W = W::new();
        for b in 0..2u32 {
            let pre = format!("encoder.layers.{b}");
            for nm in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] {
                w.insert(format!("{pre}.{nm}.weight"), r(c).iter().map(|v| 1.0 + v * 0.3).collect());
                w.insert(format!("{pre}.{nm}.bias"), r(c));
            }
            for ff in ["feed_forward1", "feed_forward2"] {
                w.insert(format!("{pre}.{ff}.linear1.weight"), r(ffn * c));
                w.insert(format!("{pre}.{ff}.linear2.weight"), r(c * ffn));
            }
            for leaf in ["q_proj", "k_proj", "v_proj", "relative_k_proj", "o_proj"] {
                w.insert(format!("{pre}.self_attn.{leaf}.weight"), r(c * c));
            }
            w.insert(format!("{pre}.self_attn.bias_u"), r(c));
            w.insert(format!("{pre}.self_attn.bias_v"), r(c));
            w.insert(format!("{pre}.conv.pointwise_conv1.weight"), r(2 * c * c));
            w.insert(format!("{pre}.conv.depthwise_conv.weight"), r(c * k));
            w.insert(format!("{pre}.conv.norm.weight"), r(c).iter().map(|v| 1.0 + v * 0.3).collect());
            w.insert(format!("{pre}.conv.norm.bias"), r(c));
            w.insert(format!("{pre}.conv.pointwise_conv2.weight"), r(c * c));
        }
        w.insert("prompt_projector.linear_1.weight".into(), r(pi * (c + np)));
        w.insert("prompt_projector.linear_1.bias".into(), r(pi));
        w.insert("prompt_projector.linear_2.weight".into(), r(c * pi));
        w.insert("prompt_projector.linear_2.bias".into(), r(c));
        w.insert("encoder_projector.weight".into(), r(dh * c));
        w.insert("encoder_projector.bias".into(), r(dh));
        let sub = r(t * c);

        let d_pool = vec![1.0f32; t * dh];
        let d_sub = encode_pooler_backward(&sub, &w, &cfg, t, valid, 0, &d_pool);
        let loss = |ss: &[f32]| -> f32 { encode_pooler(ss, &w, &cfg, t, valid, 0).iter().sum() };
        let eps = 1e-3f32;
        let ok = |a: f32, n: f32| (a - n).abs() <= 6e-3 + 1e-1 * n.abs();
        for &i in &[0usize, 7, 19, 33, 45] {
            let (mut sp, mut sm) = (sub.clone(), sub.clone());
            sp[i] += eps;
            sm[i] -= eps;
            let num = (loss(&sp) - loss(&sm)) / (2.0 * eps);
            assert!(ok(d_sub[i], num), "d_sub[{i}] {} vs {}", d_sub[i], num);
        }
        // encoder weight grads (blocks + projectors), end-to-end
        let lossw = |ww: &W| -> f32 { encode_pooler(&sub, ww, &cfg, t, valid, 0).iter().sum() };
        let (_ds, eg) = encode_pooler_grads(&sub, &w, &cfg, t, valid, 0, &vec![1.0f32; t * cfg.decoder_hidden as usize]);
        for (param, j) in [
            ("encoder_projector.weight", 3usize),
            ("prompt_projector.linear_1.weight", 5),
            ("encoder.layers.1.self_attn.o_proj.weight", 7),
            ("encoder.layers.0.conv.pointwise_conv2.weight", 10),
            ("encoder.layers.1.feed_forward2.linear2.weight", 4),
        ] {
            let (mut wp, mut wm) = (w.clone(), w.clone());
            wp.get_mut(param).unwrap()[j] += eps;
            wm.get_mut(param).unwrap()[j] -= eps;
            let num = (lossw(&wp) - lossw(&wm)) / (2.0 * eps);
            assert!(ok(eg[param][j], num), "d {param}[{j}] {} vs {}", eg[param][j], num);
        }
    }

    #[test]
    fn conformer_block_backward_matches_finite_diff() {
        use data::rng::Rng;
        // full tiny Conformer block: c=8, heads=2, ffn=16, k=3, T=6.
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.hidden = 8;
        cfg.n_heads = 2;
        cfg.intermediate = 16;
        cfg.conv_kernel = 3;
        let (c, ffn, t, valid) = (8usize, 16usize, 6usize, 6usize);
        let mut rng = Rng::new(41);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect::<Vec<f32>>();
        let mut w: W = W::new();
        let pre = "encoder.layers.0";
        for nm in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] {
            w.insert(format!("{pre}.{nm}.weight"), r(c).iter().map(|v| 1.0 + v * 0.3).collect());
            w.insert(format!("{pre}.{nm}.bias"), r(c));
        }
        for ff in ["feed_forward1", "feed_forward2"] {
            w.insert(format!("{pre}.{ff}.linear1.weight"), r(ffn * c));
            w.insert(format!("{pre}.{ff}.linear2.weight"), r(c * ffn));
        }
        for leaf in ["q_proj", "k_proj", "v_proj", "relative_k_proj", "o_proj"] {
            w.insert(format!("{pre}.self_attn.{leaf}.weight"), r(c * c));
        }
        w.insert(format!("{pre}.self_attn.bias_u"), r(c));
        w.insert(format!("{pre}.self_attn.bias_v"), r(c));
        w.insert(format!("{pre}.conv.pointwise_conv1.weight"), r(2 * c * c));
        w.insert(format!("{pre}.conv.depthwise_conv.weight"), r(c * cfg.conv_kernel as usize));
        w.insert(format!("{pre}.conv.norm.weight"), r(c).iter().map(|v| 1.0 + v * 0.3).collect());
        w.insert(format!("{pre}.conv.norm.bias"), r(c));
        w.insert(format!("{pre}.conv.pointwise_conv2.weight"), r(c * c));
        let h0 = r(t * c);

        let d_out = vec![1.0f32; t * c];
        let d_h0 = conformer_block_backward(&h0, &w, 0, &cfg, t, valid, &d_out);
        let loss = |hh: &[f32]| -> f32 { conformer_block(hh, &w, 0, &cfg, t, valid).iter().sum() };
        let eps = 1e-3f32;
        let ok = |a: f32, n: f32| (a - n).abs() <= 6e-3 + 8e-2 * n.abs();
        for &i in &[0usize, 5, 13, 22, 31, 44] {
            let (mut hp, mut hm) = (h0.clone(), h0.clone());
            hp[i] += eps;
            hm[i] -= eps;
            let num = (loss(&hp) - loss(&hm)) / (2.0 * eps);
            assert!(ok(d_h0[i], num), "d_h0[{i}] {} vs {}", d_h0[i], num);
        }
        // block weight grads across all module types
        let lossw = |ww: &W| -> f32 { conformer_block(&h0, ww, 0, &cfg, t, valid).iter().sum() };
        let (_dh, bg) = conformer_block_grads(&h0, &w, 0, &cfg, t, valid, &d_out);
        for (param, j) in [
            ("encoder.layers.0.feed_forward1.linear1.weight", 3usize),
            ("encoder.layers.0.self_attn.q_proj.weight", 11),
            ("encoder.layers.0.self_attn.bias_u", 2),
            ("encoder.layers.0.conv.pointwise_conv1.weight", 20),
            ("encoder.layers.0.conv.depthwise_conv.weight", 5),
            ("encoder.layers.0.norm_out.weight", 4),
            ("encoder.layers.0.norm_conv.bias", 1),
        ] {
            let (mut wp, mut wm) = (w.clone(), w.clone());
            wp.get_mut(param).unwrap()[j] += eps;
            wm.get_mut(param).unwrap()[j] -= eps;
            let num = (lossw(&wp) - lossw(&wm)) / (2.0 * eps);
            assert!(ok(bg[param][j], num), "d {param}[{j}] {} vs {}", bg[param][j], num);
        }
    }

    #[test]
    fn conv_module_backward_matches_finite_diff() {
        use data::rng::Rng;
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.hidden = 8;
        cfg.conv_kernel = 3;
        let (c, k, t, valid) = (8usize, 3usize, 6usize, 6usize);
        let mut rng = Rng::new(31);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.6).collect::<Vec<f32>>();
        let mut w: W = W::new();
        w.insert("conv.pointwise_conv1.weight".into(), r(2 * c * c));
        w.insert("conv.depthwise_conv.weight".into(), r(c * k));
        w.insert("conv.norm.weight".into(), r(c).iter().map(|v| 1.0 + v * 0.3).collect());
        w.insert("conv.norm.bias".into(), r(c));
        w.insert("conv.pointwise_conv2.weight".into(), r(c * c));
        let hn = r(t * c);

        let d_out = vec![1.0f32; t * c];
        let d_hn = conv_module_backward(&hn, &w, "conv", &cfg, t, valid, &d_out);
        let loss = |hh: &[f32]| -> f32 { conv_module(hh, &w, "conv", &cfg, t, valid).iter().sum() };
        let eps = 1e-3f32;
        let ok = |a: f32, n: f32| (a - n).abs() <= 5e-3 + 8e-2 * n.abs();
        for &i in &[0usize, 5, 17, 30, 40] {
            let (mut hp, mut hm) = (hn.clone(), hn.clone());
            hp[i] += eps;
            hm[i] -= eps;
            let num = (loss(&hp) - loss(&hm)) / (2.0 * eps);
            assert!(ok(d_hn[i], num), "d_hn[{i}] {} vs {}", d_hn[i], num);
        }
        // conv-module weight grads
        let lossw = |ww: &W| -> f32 { conv_module(&hn, ww, "conv", &cfg, t, valid).iter().sum() };
        let (_dh, wg) = conv_module_grads(&hn, &w, "conv", &cfg, t, valid, &d_out);
        for (param, idxs) in [
            ("conv.pointwise_conv1.weight", [0usize, 20, 60]),
            ("conv.pointwise_conv2.weight", [1, 22, 50]),
            ("conv.depthwise_conv.weight", [0, 5, 10]),
            ("conv.norm.weight", [0, 3, 7]),
        ] {
            for &j in &idxs {
                let (mut wp, mut wm) = (w.clone(), w.clone());
                wp.get_mut(param).unwrap()[j] += eps;
                wm.get_mut(param).unwrap()[j] -= eps;
                let num = (lossw(&wp) - lossw(&wm)) / (2.0 * eps);
                assert!(ok(wg[param][j], num), "d {param}[{j}] {} vs {}", wg[param][j], num);
            }
        }
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
        // NEW: all attention weight grads
        let (_dh2, wg) = rel_pos_attention_grads(&hn, &w, "attn", &cfg, t, valid, &d_out);
        for (param, idxs) in [
            ("attn.q_proj.weight", [0usize, 11, 40]),
            ("attn.o_proj.weight", [1, 22, 50]),
            ("attn.relative_k_proj.weight", [2, 30, 60]),
            ("attn.bias_u", [0, 3, 7]),
        ] {
            for &j in &idxs {
                let (mut wp, mut wm) = (w.clone(), w.clone());
                wp.get_mut(param).unwrap()[j] += eps;
                wm.get_mut(param).unwrap()[j] -= eps;
                let num = (loss(&hn, &wp) - loss(&hn, &wm)) / (2.0 * eps);
                assert!(ok(wg[param][j], num), "d {param}[{j}] {} vs {}", wg[param][j], num);
            }
        }
    }

    #[test]
    fn predictor_bptt_matches_finite_diff() {
        use data::rng::Rng;
        // tiny 2-layer LSTM predictor: dh=4, 3 tokens, vocab=6
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.decoder_hidden = 4;
        cfg.num_decoder_layers = 2;
        cfg.vocab = 6;
        let (dh, vocab) = (4usize, 6usize);
        let tokens = vec![1u32, 3u32, 5u32];
        let mut rng = Rng::new(71);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect::<Vec<f32>>();
        let mut w: W = W::new();
        w.insert("decoder.embedding.weight".into(), r(vocab * dh));
        for l in 0..2 {
            w.insert(format!("decoder.lstm.weight_ih_l{l}"), r(4 * dh * dh));
            w.insert(format!("decoder.lstm.weight_hh_l{l}"), r(4 * dh * dh));
            w.insert(format!("decoder.lstm.bias_ih_l{l}"), r(4 * dh));
            w.insert(format!("decoder.lstm.bias_hh_l{l}"), r(4 * dh));
        }
        w.insert("decoder.decoder_projector.weight".into(), r(dh * dh));
        w.insert("decoder.decoder_projector.bias".into(), r(dh));

        // loss = Σ_steps Σ dec → d_dec = ones
        let d_dec: Vec<Vec<f32>> = (0..tokens.len()).map(|_| vec![1.0f32; dh]).collect();
        let d_embed = predictor_sequence_backward(&tokens, &w, &cfg, &d_dec);
        let loss = |ww: &W| -> f32 {
            let mut st = LstmState::new(2, dh);
            tokens.iter().map(|&t| lstm_predict(t, &mut st, ww, &cfg).iter().sum::<f32>()).sum()
        };
        let eps = 1e-3f32;
        // check embedding rows actually used by the tokens
        for &i in &[1 * dh, 1 * dh + 2, 3 * dh + 1, 5 * dh, 5 * dh + 3] {
            let (mut wp, mut wm) = (w.clone(), w.clone());
            wp.get_mut("decoder.embedding.weight").unwrap()[i] += eps;
            wm.get_mut("decoder.embedding.weight").unwrap()[i] -= eps;
            let num = (loss(&wp) - loss(&wm)) / (2.0 * eps);
            assert!((d_embed[i] - num).abs() <= 3e-3 + 6e-2 * num.abs(), "d_embed[{i}] {} vs {}", d_embed[i], num);
        }
    }

    #[test]
    fn rnnt_loss_gradient_matches_finite_diff() {
        use data::rng::Rng;
        // tiny lattice: T=4 frames, U=2 labels, V=5 (blank=4)
        let (t_frames, targets, vocab, blank) = (4usize, vec![1u32, 3u32], 5usize, 4usize);
        let up1 = targets.len() + 1;
        let n = t_frames * up1 * vocab;
        let mut rng = Rng::new(61);
        let logits: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 2.0).collect();

        let (loss, d) = rnnt_loss(&logits, t_frames, &targets, blank, vocab);
        assert!(loss.is_finite() && loss > 0.0, "loss {loss}");
        let eps = 1e-3f32;
        for &i in &[0usize, 7, 19, 33, 41, 55] {
            let (mut lp, mut lm) = (logits.clone(), logits.clone());
            lp[i] += eps;
            lm[i] -= eps;
            let num = (rnnt_loss(&lp, t_frames, &targets, blank, vocab).0 - rnnt_loss(&lm, t_frames, &targets, blank, vocab).0) / (2.0 * eps);
            assert!((d[i] - num).abs() <= 3e-3 + 6e-2 * num.abs(), "d_logits[{i}] {} vs {}", d[i], num);
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
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/block0.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let sub = read_f32(&format!("{GOLD}/subsampling.f32")); // block0 input [T, 1024]
        let ref_b0 = read_f32(&format!("{GOLD}/block0.f32"));
        let t = sub.len() / cfg.hidden as usize;
        let w = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
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
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/pooler.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let sub = read_f32(&format!("{GOLD}/subsampling.f32"));
        let ref_pool = read_f32(&format!("{GOLD}/pooler.f32")); // [T, 640]
        let t = sub.len() / cfg.hidden as usize;
        let valid = cfg.subsampled_len(585) as usize;
        let w = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
        let pool = encode_pooler(&sub, &w, &cfg, t, valid, 0); // prompt_id 0 (en)
        let dh = cfg.decoder_hidden as usize;
        let n = valid * dh;
        let d = pool[..n].iter().zip(&ref_pool[..n]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("pooler (valid {valid}) maxdiff {d}");
        assert!(d < 5e-3, "pooler maxdiff {d}");
    }

    #[test]
    fn rnnt_greedy_matches_reference() {
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/pooler.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let pooler = read_f32(&format!("{GOLD}/pooler.f32")); // [T, 640]
        let dh = cfg.decoder_hidden as usize;
        let t = pooler.len() / dh;
        let valid = cfg.subsampled_len(585) as usize;
        let w = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
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
