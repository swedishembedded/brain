// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Differentiable Chronos-2 — host (CPU) forward+backward, gradcheck-gated.
//!
//! Chronos-2's inference path ([`crate::model`]) is a device-op forward with no
//! backward, and its attention uses the **unscaled** `_FULL` kernels that have no
//! GPU backward yet. Rather than author new gradient kernels, the trainable twin
//! is built **host-side**: a plain-`Vec<f32>` forward that records its
//! intermediates and an analytic backward, validated by finite differences on a
//! tiny config (the same discipline as the Kronos decoder's gradcheck). Training
//! is CPU-bound but correct; a GPU-speed Step-tape version can follow once the
//! unscaled-attention backward kernels exist.
//!
//! Built milestone by milestone, each gradcheck-gated:
//! - **M2 (this file, now):** the terminal path — final RMSNorm → the quantile
//!   head (a biased `ResidualBlock`) → rearrange → mean pinball loss. Exercises
//!   rmsnorm/matmul/bias/relu/residual + [`forecast::metrics::mean_pinball_grad`]
//!   backward, and the gradient w.r.t. the encoder output (`d_emb`) that the
//!   block stack (M3) will consume.
//! - M3: the encoder blocks (time attention + group degeneration + FFN).
//! - M4: full backbone + a gated fine-tune entry (LoRA / promotion gate).

use crate::config::Chronos2Config;
// Single implementation of the elementwise/normalisation math.
use model::hostmath;
use std::collections::HashMap;

// ---- host math primitives (fwd + bwd), row-major, weights stored `[out, in]` ---

/// `out[m,n] = sum_k x[m,k] * w[n,k]` (weight is `[n, k]`, i.e. `[out, in]`).
fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[c * k + i];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

/// Backward of [`matmul`]: `dout[m,n]` → `dx[m,k]` and accumulate into `dw[n,k]`.
fn matmul_bwd(x: &[f32], w: &[f32], dout: &[f32], m: usize, k: usize, n: usize, dw: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; m * k];
    for r in 0..m {
        for c in 0..n {
            let g = dout[r * n + c];
            for i in 0..k {
                dx[r * k + i] += g * w[c * k + i];
                dw[c * k + i] += g * x[r * k + i];
            }
        }
    }
    dx
}

/// Add a bias `b[n]` to each of `m` rows, in place.
fn bias_add(y: &mut [f32], b: &[f32], m: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            y[r * n + c] += b[c];
        }
    }
}

/// Backward of [`bias_add`]: accumulate `db[c] += sum_r dout[r,c]`.
fn bias_bwd(dout: &[f32], m: usize, n: usize, db: &mut [f32]) {
    for r in 0..m {
        for c in 0..n {
            db[c] += dout[r * n + c];
        }
    }
}

/// Backward of [`rmsnorm`]: `dy` → `dx`, accumulate `dg`. `inv` from the forward.
fn rmsnorm_bwd(x: &[f32], g: &[f32], inv: &[f32], dy: &[f32], rows: usize, d: usize, dg: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * d];
    for r in 0..rows {
        let iv = inv[r];
        // s = Σ_i dy_i * g_i * x_i  (the coupling term through the shared norm)
        let mut s = 0.0f32;
        for i in 0..d {
            let xi = x[r * d + i];
            s += dy[r * d + i] * g[i] * xi;
            dg[i] += dy[r * d + i] * xi * iv;
        }
        let iv3 = iv * iv * iv / d as f32;
        for i in 0..d {
            let xi = x[r * d + i];
            dx[r * d + i] = iv * g[i] * dy[r * d + i] - iv3 * xi * s;
        }
    }
    dx
}

// ---- the biased ResidualBlock: output(relu(hidden(x))) + residual(x) ----------

/// Intermediates a [`residual_block`] backward needs.
struct ResidualCache {
    x: Vec<f32>,
    hid: Vec<f32>,      // pre-ReLU
    hid_relu: Vec<f32>, // post-ReLU
}

/// Names of the three linear layers' `weight`/`bias` under `prefix`.
fn rb_names(prefix: &str) -> [String; 6] {
    [
        format!("{prefix}.hidden_layer.weight"),
        format!("{prefix}.hidden_layer.bias"),
        format!("{prefix}.output_layer.weight"),
        format!("{prefix}.output_layer.bias"),
        format!("{prefix}.residual_layer.weight"),
        format!("{prefix}.residual_layer.bias"),
    ]
}

/// Forward: `[rows, in_dim] -> [rows, out_dim]`, mirroring `model::residual_block`.
fn residual_block(
    w: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x: &[f32],
    rows: usize,
    in_dim: usize,
    h: usize,
    out_dim: usize,
) -> (Vec<f32>, ResidualCache) {
    let n = rb_names(prefix);
    let mut hid = matmul(x, &w[&n[0]], rows, in_dim, h);
    bias_add(&mut hid, &w[&n[1]], rows, h);
    let hid_relu: Vec<f32> = hid.iter().map(|&v| v.max(0.0)).collect();

    let mut o1 = matmul(&hid_relu, &w[&n[2]], rows, h, out_dim);
    bias_add(&mut o1, &w[&n[3]], rows, out_dim);

    let mut res = matmul(x, &w[&n[4]], rows, in_dim, out_dim);
    bias_add(&mut res, &w[&n[5]], rows, out_dim);

    for i in 0..rows * out_dim {
        res[i] += o1[i]; // y = o1 + res
    }
    (res, ResidualCache { x: x.to_vec(), hid, hid_relu })
}

/// Backward of [`residual_block`]: `dy` → `dx`, accumulate weight/bias grads.
#[allow(clippy::too_many_arguments)]
fn residual_block_bwd(
    w: &HashMap<String, Vec<f32>>,
    g: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    cache: &ResidualCache,
    dy: &[f32],
    rows: usize,
    in_dim: usize,
    h: usize,
    out_dim: usize,
) -> Vec<f32> {
    let n = rb_names(prefix);
    // y = o1 + res  →  do1 = dres = dy
    bias_bwd(dy, rows, out_dim, g.get_mut(&n[5]).unwrap());
    let mut dx = matmul_bwd(&cache.x, &w[&n[4]], dy, rows, in_dim, out_dim, g.get_mut(&n[4]).unwrap());

    bias_bwd(dy, rows, out_dim, g.get_mut(&n[3]).unwrap());
    let d_hid_relu = matmul_bwd(&cache.hid_relu, &w[&n[2]], dy, rows, h, out_dim, g.get_mut(&n[2]).unwrap());

    // ReLU
    let mut d_hid = vec![0.0f32; rows * h];
    for i in 0..rows * h {
        d_hid[i] = if cache.hid[i] > 0.0 { d_hid_relu[i] } else { 0.0 };
    }
    bias_bwd(&d_hid, rows, h, g.get_mut(&n[1]).unwrap());
    let dx_hidden = matmul_bwd(&cache.x, &w[&n[0]], &d_hid, rows, in_dim, h, g.get_mut(&n[0]).unwrap());
    for i in 0..rows * in_dim {
        dx[i] += dx_hidden[i];
    }
    dx
}

// ---- NeoX RoPE (half-split) fwd + bwd -----------------------------------------

/// Apply half-split NeoX RoPE in place to `buf` `[s, heads*hd]`: for each head,
/// rotate the pair `(x[j], x[j+hd/2])` by `angle = t * theta^(-2j/hd)`.
fn rope_neox(buf: &mut [f32], s: usize, heads: usize, hd: usize, theta: f32) {
    let inner = heads * hd;
    let half = hd / 2;
    for t in 0..s {
        for h in 0..heads {
            let base = t * inner + h * hd;
            for j in 0..half {
                let ang = t as f32 * theta.powf(-2.0 * j as f32 / hd as f32);
                let (sn, cs) = ang.sin_cos();
                let a = buf[base + j];
                let b = buf[base + j + half];
                buf[base + j] = a * cs - b * sn;
                buf[base + j + half] = b * cs + a * sn;
            }
        }
    }
}

/// Backward of [`rope_neox`]: the rotation is orthogonal, so the gradient is the
/// same rotation by the negated angle — applied in place to `dbuf`.
fn rope_neox_bwd(dbuf: &mut [f32], s: usize, heads: usize, hd: usize, theta: f32) {
    let inner = heads * hd;
    let half = hd / 2;
    for t in 0..s {
        for h in 0..heads {
            let base = t * inner + h * hd;
            for j in 0..half {
                let ang = t as f32 * theta.powf(-2.0 * j as f32 / hd as f32);
                let (sn, cs) = ang.sin_cos();
                let dj = dbuf[base + j];
                let dj2 = dbuf[base + j + half];
                dbuf[base + j] = dj * cs + dj2 * sn;
                dbuf[base + j + half] = -dj * sn + dj2 * cs;
            }
        }
    }
}

// ---- unscaled bidirectional multi-head attention fwd + bwd ---------------------

/// `q,k,v` are `[s, heads*hd]`; `mask[s]` is additive per key. UNSCALED (no
/// `1/sqrt(hd)`), matching Chronos-2's `attn_scores_full`. Returns
/// `(ctx[s, heads*hd], probs[heads*s*s])`.
fn attention(q: &[f32], k: &[f32], v: &[f32], mask: &[f32], s: usize, heads: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let inner = heads * hd;
    let mut probs = vec![0.0f32; heads * s * s];
    let mut ctx = vec![0.0f32; s * inner];
    for h in 0..heads {
        for i in 0..s {
            let mut sc = vec![0.0f32; s];
            for (j, scj) in sc.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for dd in 0..hd {
                    dot += q[i * inner + h * hd + dd] * k[j * inner + h * hd + dd];
                }
                *scj = dot + mask[j];
            }
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0f32;
            for scj in sc.iter_mut() {
                *scj = (*scj - mx).exp();
                sum += *scj;
            }
            for (j, &e) in sc.iter().enumerate() {
                let p = e / sum;
                probs[h * s * s + i * s + j] = p;
                for dd in 0..hd {
                    ctx[i * inner + h * hd + dd] += p * v[j * inner + h * hd + dd];
                }
            }
        }
    }
    (ctx, probs)
}

/// Backward of [`attention`]: `d_ctx` + saved `probs` → `(dq, dk, dv)`.
fn attention_bwd(
    d_ctx: &[f32],
    probs: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    heads: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let inner = heads * hd;
    let mut dq = vec![0.0f32; s * inner];
    let mut dk = vec![0.0f32; s * inner];
    let mut dv = vec![0.0f32; s * inner];
    for h in 0..heads {
        for i in 0..s {
            let mut dprob = vec![0.0f32; s];
            for (j, dpj) in dprob.iter_mut().enumerate() {
                let p = probs[h * s * s + i * s + j];
                let mut dp = 0.0f32;
                for dd in 0..hd {
                    let g = d_ctx[i * inner + h * hd + dd];
                    dp += g * v[j * inner + h * hd + dd];
                    dv[j * inner + h * hd + dd] += p * g;
                }
                *dpj = dp;
            }
            // softmax jacobian: d_score_j = p_j (dprob_j - Σ_k p_k dprob_k)
            let sdot: f32 = (0..s).map(|j| probs[h * s * s + i * s + j] * dprob[j]).sum();
            for j in 0..s {
                let dscore = probs[h * s * s + i * s + j] * (dprob[j] - sdot);
                for dd in 0..hd {
                    dq[i * inner + h * hd + dd] += dscore * k[j * inner + h * hd + dd];
                    dk[j * inner + h * hd + dd] += dscore * q[i * inner + h * hd + dd];
                }
            }
        }
    }
    (dq, dk, dv)
}

// ---- one encoder block (M3): time attention + group degeneration + FFN ---------

/// Intermediates one [`Chronos2Train::block_forward`] backward needs.
pub struct BlockCache {
    emb_in: Vec<f32>,
    xn0: Vec<f32>,
    inv0: Vec<f32>,
    q: Vec<f32>, // roped
    k: Vec<f32>, // roped
    v: Vec<f32>,
    probs: Vec<f32>,
    ctx: Vec<f32>,
    emb1: Vec<f32>,
    xn1: Vec<f32>,
    inv1: Vec<f32>,
    vg: Vec<f32>,
    emb2: Vec<f32>,
    xn2: Vec<f32>,
    inv2: Vec<f32>,
    hid: Vec<f32>, // pre-ReLU
    hid_relu: Vec<f32>,
    s: usize,
}

/// Intermediates for the full-core backward (M4): every block's cache + the head's.
pub struct FullCache {
    blocks: Vec<BlockCache>,
    head: HeadCache,
}

// ---- the differentiable head path (M2) ----------------------------------------

/// The trainable Chronos-2 (host). Holds a config and a name→weights map; the
/// grad map mirrors the weight names. M2 exposes the terminal head path.
pub struct Chronos2Train {
    pub cfg: Chronos2Config,
    pub w: HashMap<String, Vec<f32>>,
}

/// Everything the head backward needs from the forward.
pub struct HeadCache {
    emb: Vec<f32>,
    inv: Vec<f32>,     // final-norm reciprocal norms
    rb: ResidualCache, // the head ResidualBlock's cache
    quantiles: Vec<f32>, // [H, Q]
    s: usize,
    n_out: usize,
}

impl Chronos2Train {
    /// Zeroed grad accumulator with an entry per weight.
    pub fn zero_grads(&self) -> HashMap<String, Vec<f32>> {
        self.w.iter().map(|(k, v)| (k.clone(), vec![0.0f32; v.len()])).collect()
    }

    /// Forward of the head path: encoder output `emb` `[s, d]` → final RMSNorm →
    /// quantile head on the trailing `n_out` patch tokens → rearranged
    /// `quantiles[H, Q]` (`H = n_out*patch`, quantile-minor) → mean pinball loss
    /// vs the standardized `target[H]` at `levels[Q]`. Returns `(loss, cache)`.
    pub fn head_forward(&self, emb: &[f32], n_out: usize, target: &[f32], levels: &[f32]) -> (f32, HeadCache) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let head_out = cfg.head_out_dim();
        let s = emb.len() / d;
        let eps = cfg.layer_norm_epsilon;

        let (normed, inv) = hostmath::rmsnorm_rows_with_inv(emb, &self.w["encoder.final_layer_norm.weight"], s, d, eps);
        let head_in = normed[(s - n_out) * d..].to_vec();
        let (qp, rb) = residual_block(&self.w, "output_patch_embedding", &head_in, n_out, d, cfg.d_ff, head_out);

        // rearrange [n_out, head_out=(q*patch)] -> quantiles[H=n_out*patch, Q]
        let hlen = n_out * patch;
        let mut quantiles = vec![0.0f32; hlen * q];
        for nn in 0..n_out {
            for qi in 0..q {
                for pp in 0..patch {
                    quantiles[(nn * patch + pp) * q + qi] = qp[nn * head_out + qi * patch + pp];
                }
            }
        }
        let loss = forecast::metrics::mean_pinball(&quantiles, levels, target);
        (loss, HeadCache { emb: emb.to_vec(), inv, rb, quantiles, s, n_out })
    }

    /// Backward of [`head_forward`]: accumulate head + final-norm grads into `g`
    /// and return `d_emb` `[s, d]` (the gradient the block stack consumes in M3).
    pub fn head_backward(&self, cache: &HeadCache, target: &[f32], levels: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let head_out = cfg.head_out_dim();
        let (s, n_out) = (cache.s, cache.n_out);

        let d_quant = forecast::metrics::mean_pinball_grad(&cache.quantiles, levels, target);
        // rearrange the gradient back to d_qp[n_out, head_out]
        let mut d_qp = vec![0.0f32; n_out * head_out];
        for nn in 0..n_out {
            for qi in 0..q {
                for pp in 0..patch {
                    d_qp[nn * head_out + qi * patch + pp] = d_quant[(nn * patch + pp) * q + qi];
                }
            }
        }
        let d_head_in =
            residual_block_bwd(&self.w, g, "output_patch_embedding", &cache.rb, &d_qp, n_out, d, cfg.d_ff, head_out);

        // scatter d_head_in into the trailing rows of d_normed[s, d]
        let mut d_normed = vec![0.0f32; s * d];
        d_normed[(s - n_out) * d..].copy_from_slice(&d_head_in);

        rmsnorm_bwd(
            &cache.emb,
            &self.w["encoder.final_layer_norm.weight"],
            &cache.inv,
            &d_normed,
            s,
            d,
            g.get_mut("encoder.final_layer_norm.weight").unwrap(),
        )
    }

    /// Forward of encoder block `b` on `emb_in` `[s, d]` with additive key `mask`:
    /// time attention (pre-norm bidirectional MHA + NeoX RoPE, residual) → group
    /// degeneration (B=1: `o(v(rmsnorm))`, residual) → FFN (pre-norm ReLU MLP,
    /// residual). Matches [`crate::model::Chronos2::block`]. Returns `(emb_out, cache)`.
    pub fn block_forward(&self, b: usize, emb_in: &[f32], mask: &[f32], s: usize) -> (Vec<f32>, BlockCache) {
        let cfg = &self.cfg;
        let (d, inner, heads, hd, f, eps, theta) =
            (cfg.d_model, cfg.inner_dim(), cfg.num_heads, cfg.d_kv, cfg.d_ff, cfg.layer_norm_epsilon, cfg.rope_theta);
        let p = format!("encoder.block.{b}");
        let ww = |n: &str| &self.w[n];

        // -- time attention --
        let (xn0, inv0) = hostmath::rmsnorm_rows_with_inv(emb_in, ww(&format!("{p}.layer.0.layer_norm.weight")), s, d, eps);
        let mut q = matmul(&xn0, ww(&format!("{p}.layer.0.self_attention.q.weight")), s, d, inner);
        let mut k = matmul(&xn0, ww(&format!("{p}.layer.0.self_attention.k.weight")), s, d, inner);
        let v = matmul(&xn0, ww(&format!("{p}.layer.0.self_attention.v.weight")), s, d, inner);
        rope_neox(&mut q, s, heads, hd, theta);
        rope_neox(&mut k, s, heads, hd, theta);
        let (ctx, probs) = attention(&q, &k, &v, mask, s, heads, hd);
        let o = matmul(&ctx, ww(&format!("{p}.layer.0.self_attention.o.weight")), s, inner, d);
        let emb1: Vec<f32> = (0..s * d).map(|i| emb_in[i] + o[i]).collect();

        // -- group degeneration --
        let (xn1, inv1) = hostmath::rmsnorm_rows_with_inv(&emb1, ww(&format!("{p}.layer.1.layer_norm.weight")), s, d, eps);
        let vg = matmul(&xn1, ww(&format!("{p}.layer.1.self_attention.v.weight")), s, d, inner);
        let og = matmul(&vg, ww(&format!("{p}.layer.1.self_attention.o.weight")), s, inner, d);
        let emb2: Vec<f32> = (0..s * d).map(|i| emb1[i] + og[i]).collect();

        // -- FFN --
        let (xn2, inv2) = hostmath::rmsnorm_rows_with_inv(&emb2, ww(&format!("{p}.layer.2.layer_norm.weight")), s, d, eps);
        let hid = matmul(&xn2, ww(&format!("{p}.layer.2.mlp.wi.weight")), s, d, f);
        let hid_relu: Vec<f32> = hid.iter().map(|&x| x.max(0.0)).collect();
        let ff = matmul(&hid_relu, ww(&format!("{p}.layer.2.mlp.wo.weight")), s, f, d);
        let emb3: Vec<f32> = (0..s * d).map(|i| emb2[i] + ff[i]).collect();

        (emb3, BlockCache { emb_in: emb_in.to_vec(), xn0, inv0, q, k, v, probs, ctx, emb1, xn1, inv1, vg, emb2, xn2, inv2, hid, hid_relu, s })
    }

    /// Backward of [`block_forward`]: `d_emb_out` → `d_emb_in`, accumulating the
    /// block's weight grads into `g`. Note: q/k/v/o attention projections and the
    /// group/FFN linears are bias-free (matmul only), matching the forward.
    pub fn block_backward(&self, b: usize, cache: &BlockCache, d_emb_out: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, inner, heads, hd, f, theta) =
            (cfg.d_model, cfg.inner_dim(), cfg.num_heads, cfg.d_kv, cfg.d_ff, cfg.rope_theta);
        let s = cache.s;
        let p = format!("encoder.block.{b}");
        let ww = |n: &str| &self.w[n];

        // -- FFN backward (emb3 = emb2 + ff; residual passes d_emb_out to d_emb2) --
        let d_hid_relu = matmul_bwd(&cache.hid_relu, ww(&format!("{p}.layer.2.mlp.wo.weight")), d_emb_out, s, f, d, g.get_mut(&format!("{p}.layer.2.mlp.wo.weight")).unwrap());
        let d_hid: Vec<f32> = (0..s * f).map(|i| if cache.hid[i] > 0.0 { d_hid_relu[i] } else { 0.0 }).collect();
        let d_xn2 = matmul_bwd(&cache.xn2, ww(&format!("{p}.layer.2.mlp.wi.weight")), &d_hid, s, d, f, g.get_mut(&format!("{p}.layer.2.mlp.wi.weight")).unwrap());
        let d_emb2_a = rmsnorm_bwd(&cache.emb2, ww(&format!("{p}.layer.2.layer_norm.weight")), &cache.inv2, &d_xn2, s, d, g.get_mut(&format!("{p}.layer.2.layer_norm.weight")).unwrap());
        let d_emb2: Vec<f32> = (0..s * d).map(|i| d_emb_out[i] + d_emb2_a[i]).collect();

        // -- group backward (emb2 = emb1 + og) --
        let d_vg = matmul_bwd(&cache.vg, ww(&format!("{p}.layer.1.self_attention.o.weight")), &d_emb2, s, inner, d, g.get_mut(&format!("{p}.layer.1.self_attention.o.weight")).unwrap());
        let d_xn1 = matmul_bwd(&cache.xn1, ww(&format!("{p}.layer.1.self_attention.v.weight")), &d_vg, s, d, inner, g.get_mut(&format!("{p}.layer.1.self_attention.v.weight")).unwrap());
        let d_emb1_a = rmsnorm_bwd(&cache.emb1, ww(&format!("{p}.layer.1.layer_norm.weight")), &cache.inv1, &d_xn1, s, d, g.get_mut(&format!("{p}.layer.1.layer_norm.weight")).unwrap());
        let d_emb1: Vec<f32> = (0..s * d).map(|i| d_emb2[i] + d_emb1_a[i]).collect();

        // -- time-attention backward (emb1 = emb_in + o) --
        let d_ctx = matmul_bwd(&cache.ctx, ww(&format!("{p}.layer.0.self_attention.o.weight")), &d_emb1, s, inner, d, g.get_mut(&format!("{p}.layer.0.self_attention.o.weight")).unwrap());
        let (mut d_q, mut d_k, d_v) = attention_bwd(&d_ctx, &cache.probs, &cache.q, &cache.k, &cache.v, s, heads, hd);
        rope_neox_bwd(&mut d_q, s, heads, hd, theta);
        rope_neox_bwd(&mut d_k, s, heads, hd, theta);
        // three projections share the same input xn0; accumulate d_xn0 from each
        // (sequential calls, each borrowing its own distinct grad tensor).
        let d_xn0_q = matmul_bwd(&cache.xn0, ww(&format!("{p}.layer.0.self_attention.q.weight")), &d_q, s, d, inner, g.get_mut(&format!("{p}.layer.0.self_attention.q.weight")).unwrap());
        let d_xn0_k = matmul_bwd(&cache.xn0, ww(&format!("{p}.layer.0.self_attention.k.weight")), &d_k, s, d, inner, g.get_mut(&format!("{p}.layer.0.self_attention.k.weight")).unwrap());
        let d_xn0_v = matmul_bwd(&cache.xn0, ww(&format!("{p}.layer.0.self_attention.v.weight")), &d_v, s, d, inner, g.get_mut(&format!("{p}.layer.0.self_attention.v.weight")).unwrap());
        let d_xn0: Vec<f32> = (0..s * d).map(|i| d_xn0_q[i] + d_xn0_k[i] + d_xn0_v[i]).collect();
        let d_emb_in_a = rmsnorm_bwd(&cache.emb_in, ww(&format!("{p}.layer.0.layer_norm.weight")), &cache.inv0, &d_xn0, s, d, g.get_mut(&format!("{p}.layer.0.layer_norm.weight")).unwrap());

        (0..s * d).map(|i| d_emb1[i] + d_emb_in_a[i]).collect()
    }

    /// Full differentiable core (M4): `num_layers` blocks → head path. `emb_in`
    /// `[s, d]` is the assembled token sequence (host scaler/patch/embed, kept
    /// out of the trained graph — the scaler is data-derived, not learned).
    /// Returns `(loss, cache)`.
    pub fn full_forward(&self, emb_in: &[f32], mask: &[f32], n_out: usize, target: &[f32], levels: &[f32]) -> (f32, FullCache) {
        let s = mask.len();
        let mut emb = emb_in.to_vec();
        let mut blocks = Vec::with_capacity(self.cfg.num_layers);
        for b in 0..self.cfg.num_layers {
            let (out, c) = self.block_forward(b, &emb, mask, s);
            emb = out;
            blocks.push(c);
        }
        let (loss, head) = self.head_forward(&emb, n_out, target, levels);
        (loss, FullCache { blocks, head })
    }

    /// Backward of [`full_forward`]: head then blocks in reverse, accumulating all
    /// grads into `g`; returns `d_emb_in`.
    pub fn full_backward(&self, cache: &FullCache, target: &[f32], levels: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let mut d_emb = self.head_backward(&cache.head, target, levels, g);
        for b in (0..self.cfg.num_layers).rev() {
            d_emb = self.block_backward(b, &cache.blocks[b], &d_emb, g);
        }
        d_emb
    }

    /// One plain-SGD step on a single example (accumulate grads, then `w -= lr*g`).
    /// Returns the pre-step loss. Used by the from-scratch learning test; the real
    /// fine-tune entry (AdamW + LoRA + promotion gate) is milestone 5.
    pub fn sgd_step(&mut self, emb_in: &[f32], mask: &[f32], n_out: usize, target: &[f32], levels: &[f32], lr: f32) -> f32 {
        let (loss, cache) = self.full_forward(emb_in, mask, n_out, target, levels);
        let mut g = self.zero_grads();
        self.full_backward(&cache, target, levels, &mut g);
        for (name, grad) in &g {
            let w = self.w.get_mut(name).unwrap();
            for (wi, gi) in w.iter_mut().zip(grad) {
                *wi -= lr * gi;
            }
        }
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small-magnitude weights/inputs for a gradcheck.
    fn rng(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }

    fn head_params(cfg: &Chronos2Config, seed: &mut u64) -> HashMap<String, Vec<f32>> {
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let ho = cfg.head_out_dim();
        let mut w = HashMap::new();
        let fill = |name: &str, len: usize, w: &mut HashMap<String, Vec<f32>>, seed: &mut u64| {
            w.insert(name.to_string(), (0..len).map(|_| rng(seed)).collect());
        };
        fill("encoder.final_layer_norm.weight", d, &mut w, seed);
        // final-norm gain near 1 (its default), plus small jitter.
        for v in w.get_mut("encoder.final_layer_norm.weight").unwrap() {
            *v += 1.0;
        }
        for (nm, len) in [
            ("output_patch_embedding.hidden_layer.weight", ff * d),
            ("output_patch_embedding.hidden_layer.bias", ff),
            ("output_patch_embedding.output_layer.weight", ho * ff),
            ("output_patch_embedding.output_layer.bias", ho),
            ("output_patch_embedding.residual_layer.weight", ho * d),
            ("output_patch_embedding.residual_layer.bias", ho),
        ] {
            fill(nm, len, &mut w, seed);
        }
        w
    }

    fn block_params(cfg: &Chronos2Config, b: usize, seed: &mut u64) -> HashMap<String, Vec<f32>> {
        let (d, inner, f) = (cfg.d_model, cfg.inner_dim(), cfg.d_ff);
        let p = format!("encoder.block.{b}");
        let mut w = HashMap::new();
        let fill = |name: String, len: usize, gain1: bool, w: &mut HashMap<String, Vec<f32>>, seed: &mut u64| {
            let base = if gain1 { 1.0 } else { 0.0 };
            w.insert(name, (0..len).map(|_| base + rng(seed)).collect());
        };
        for (nm, len, g1) in [
            (format!("{p}.layer.0.layer_norm.weight"), d, true),
            (format!("{p}.layer.0.self_attention.q.weight"), inner * d, false),
            (format!("{p}.layer.0.self_attention.k.weight"), inner * d, false),
            (format!("{p}.layer.0.self_attention.v.weight"), inner * d, false),
            (format!("{p}.layer.0.self_attention.o.weight"), d * inner, false),
            (format!("{p}.layer.1.layer_norm.weight"), d, true),
            (format!("{p}.layer.1.self_attention.v.weight"), inner * d, false),
            (format!("{p}.layer.1.self_attention.o.weight"), d * inner, false),
            (format!("{p}.layer.2.layer_norm.weight"), d, true),
            (format!("{p}.layer.2.mlp.wi.weight"), f * d, false),
            (format!("{p}.layer.2.mlp.wo.weight"), d * f, false),
        ] {
            fill(nm, len, g1, &mut w, seed);
        }
        w
    }

    /// Full param set: every block + the head path, for a `num_layers`-deep model.
    fn full_params(cfg: &Chronos2Config, seed: &mut u64) -> HashMap<String, Vec<f32>> {
        let mut w = head_params(cfg, seed);
        for b in 0..cfg.num_layers {
            w.extend(block_params(cfg, b, seed));
        }
        w
    }

    #[test]
    fn full_backbone_gradcheck() {
        let cfg = Chronos2Config::tiny(); // 2 layers
        let mut seed = 0xFACE_u64;
        let model = Chronos2Train { cfg: cfg.clone(), w: full_params(&cfg, &mut seed) };

        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let (s, n_out) = (5usize, 2usize);
        let h = n_out * patch;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let mask = vec![0.0f32; s];
        let target: Vec<f32> = (0..h).map(|_| rng(&mut seed)).collect();
        let levels: Vec<f32> = (0..q).map(|i| (i as f32 + 0.5) / q as f32).collect();

        let (_l, cache) = model.full_forward(&emb, &mask, n_out, &target, &levels);
        let mut g = model.zero_grads();
        let d_emb = model.full_backward(&cache, &target, &levels, &mut g);

        let loss = |m: &Chronos2Train, e: &[f32]| m.full_forward(e, &mask, n_out, &target, &levels).0;
        let eps = 4e-3f32;
        let tol = |a: f32, n: f32| (a - n).abs() <= 4e-3 + 8e-2 * a.abs().max(n.abs());
        let mut checked = 0usize;
        // sample across head + both blocks (weights end-to-end through the stack).
        for name in model.w.keys() {
            let len = model.w[name].len();
            for &idx in &[0usize, len / 2, len - 1] {
                let mut mp = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mp.w.get_mut(name).unwrap()[idx] += eps;
                let mut mm = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mm.w.get_mut(name).unwrap()[idx] -= eps;
                let numeric = (loss(&mp, &emb) - loss(&mm, &emb)) / (2.0 * eps);
                assert!(tol(g[name][idx], numeric), "grad {name}[{idx}]: {} vs {numeric}", g[name][idx]);
                checked += 1;
            }
        }
        for &idx in &[0usize, s * d - 1] {
            let mut ep = emb.clone();
            ep[idx] += eps;
            let mut em = emb.clone();
            em[idx] -= eps;
            let numeric = (loss(&model, &ep) - loss(&model, &em)) / (2.0 * eps);
            assert!(tol(d_emb[idx], numeric), "d_emb[{idx}]: {} vs {numeric}", d_emb[idx]);
            checked += 1;
        }
        assert!(checked > 60, "expected a broad end-to-end gradcheck, only {checked}");
    }

    #[test]
    fn full_backbone_learns_a_fixed_example() {
        // The whole differentiable core must actually reduce the pinball loss on a
        // fixed (input, target) under plain SGD — end-to-end proof the forward,
        // backward, and update compose correctly (not just locally consistent).
        let cfg = Chronos2Config::tiny();
        let mut seed = 0x5EED_u64;
        let mut model = Chronos2Train { cfg: cfg.clone(), w: full_params(&cfg, &mut seed) };

        let (s, n_out) = (5usize, 2usize);
        let d = cfg.d_model;
        let h = n_out * cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let mask = vec![0.0f32; s];
        let target: Vec<f32> = (0..h).map(|_| rng(&mut seed) * 3.0).collect();
        let levels: Vec<f32> = (0..q).map(|i| (i as f32 + 0.5) / q as f32).collect();

        let l0 = model.full_forward(&emb, &mask, n_out, &target, &levels).0;
        let mut last = l0;
        for _ in 0..300 {
            last = model.sgd_step(&emb, &mask, n_out, &target, &levels, 0.05);
        }
        assert!(last < 0.5 * l0, "loss must fall substantially: {l0} -> {last}");
    }

    #[test]
    fn block_gradcheck() {
        let cfg = Chronos2Config::tiny();
        let mut seed = 0xB10C_u64;
        let w = block_params(&cfg, 0, &mut seed);
        let model = Chronos2Train { cfg: cfg.clone(), w };

        let d = cfg.d_model;
        let s = 4usize;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let mask = vec![0.0f32; s]; // all-attend
        let proj: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect(); // random downstream grad

        // scalar loss = <emb_out, proj>, so d_emb_out = proj.
        let loss = |m: &Chronos2Train, e: &[f32]| -> f32 {
            let (out, _) = m.block_forward(0, e, &mask, s);
            out.iter().zip(&proj).map(|(a, b)| a * b).sum()
        };

        let (_out, cache) = model.block_forward(0, &emb, &mask, s);
        let mut g = model.zero_grads();
        let d_emb_in = model.block_backward(0, &cache, &proj, &mut g);

        let eps = 4e-3f32;
        let tol = |a: f32, n: f32| (a - n).abs() <= 4e-3 + 8e-2 * a.abs().max(n.abs());
        let mut checked = 0usize;
        for name in model.w.keys() {
            let len = model.w[name].len();
            for &idx in &[0usize, len / 3, len / 2, len - 1] {
                let mut mp = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mp.w.get_mut(name).unwrap()[idx] += eps;
                let lp = loss(&mp, &emb);
                let mut mm = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mm.w.get_mut(name).unwrap()[idx] -= eps;
                let lm = loss(&mm, &emb);
                let numeric = (lp - lm) / (2.0 * eps);
                assert!(tol(g[name][idx], numeric), "grad {name}[{idx}]: {} vs {numeric}", g[name][idx]);
                checked += 1;
            }
        }
        for &idx in &[0usize, s * d / 2, s * d - 1] {
            let mut ep = emb.clone();
            ep[idx] += eps;
            let mut em = emb.clone();
            em[idx] -= eps;
            let numeric = (loss(&model, &ep) - loss(&model, &em)) / (2.0 * eps);
            assert!(tol(d_emb_in[idx], numeric), "d_emb_in[{idx}]: {} vs {numeric}", d_emb_in[idx]);
            checked += 1;
        }
        assert!(checked > 40, "expected a broad gradcheck, only {checked}");
    }

    #[test]
    fn head_path_gradcheck() {
        let cfg = Chronos2Config::tiny();
        let mut seed = 0x51ED_u64;
        let w = head_params(&cfg, &mut seed);
        let model = Chronos2Train { cfg: cfg.clone(), w };

        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let (s, n_out) = (5usize, 2usize);
        let h = n_out * patch;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let target: Vec<f32> = (0..h).map(|_| rng(&mut seed)).collect();
        let levels: Vec<f32> = (0..q).map(|i| (i as f32 + 0.5) / q as f32).collect();

        // analytic grads
        let (_l0, cache) = model.head_forward(&emb, n_out, &target, &levels);
        let mut g = model.zero_grads();
        let d_emb = model.head_backward(&cache, &target, &levels, &mut g);

        let loss = |m: &Chronos2Train, e: &[f32]| m.head_forward(e, n_out, &target, &levels).0;
        let eps = 4e-3f32;
        let mut checked = 0usize;

        // (a) weight grads: sample a few entries per tensor via directional FD.
        for name in model.w.keys() {
            let len = model.w[name].len();
            for &idx in &[0usize, len / 3, len / 2, len - 1] {
                let mut mp = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mp.w.get_mut(name).unwrap()[idx] += eps;
                let lp = loss(&mp, &emb);
                let mut mm = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mm.w.get_mut(name).unwrap()[idx] -= eps;
                let lm = loss(&mm, &emb);
                let numeric = (lp - lm) / (2.0 * eps);
                let analytic = g[name][idx];
                assert!(
                    (numeric - analytic).abs() <= 4e-3 + 8e-2 * analytic.abs().max(numeric.abs()),
                    "grad mismatch {name}[{idx}]: analytic {analytic} vs numeric {numeric}"
                );
                checked += 1;
            }
        }

        // (b) input grad d_emb: a few entries.
        for &idx in &[0usize, s * d / 2, s * d - 1] {
            let mut ep = emb.clone();
            ep[idx] += eps;
            let mut em = emb.clone();
            em[idx] -= eps;
            let numeric = (loss(&model, &ep) - loss(&model, &em)) / (2.0 * eps);
            let analytic = d_emb[idx];
            assert!(
                (numeric - analytic).abs() <= 4e-3 + 8e-2 * analytic.abs().max(numeric.abs()),
                "d_emb[{idx}]: analytic {analytic} vs numeric {numeric}"
            );
            checked += 1;
        }
        assert!(checked > 20, "expected a broad gradcheck, only {checked} params");
    }
}
