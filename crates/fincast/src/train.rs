// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Differentiable FinCast — host (CPU) forward+backward, gradcheck-gated.
//!
//! FinCast's inference path ([`crate::model`]) is a device-op forward with no
//! backward. The trainable twin is built **host-side** (plain `Vec<f32>` forward
//! recording intermediates + an analytic backward), validated by finite
//! differences on a tiny config — the same discipline as the Chronos-2/Kronos
//! decoders. Training is CPU-bound but correct.
//!
//! The trained graph is the transformer core (`num_layers` decoder blocks) + the
//! horizon head, over the assembled patch embeddings `emb[s,d]` (the
//! preprocessing/patch-embed is data-derived and kept out of the trained graph).
//! Loss on the last patch: `mean_pinball(9 quantiles) + MSE(mean)` — a
//! probabilistic-quantile (PQ) loss. **Deterministic top-2 MoE routing**: the
//! gradcheck uses configs where the selected experts are stable under the FD
//! perturbation (the top-2 boundary is measure-zero).

use crate::config::{FincastConfig, QUANTILES};
// Single implementation of the elementwise/normalisation math.
use model::hostmath;
use std::collections::HashMap;

// ---- host math primitives (fwd + bwd), weights stored `[out, in]` -------------

/// `out[m,n] = Σ_k x[m,k]·w[n,k]` (weight `[out,in]`).
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

/// Backward of [`matmul`]: `dout[m,n]` → `dx[m,k]`, accumulate `dw[n,k]`.
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

fn bias_add(y: &mut [f32], b: &[f32], m: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            y[r * n + c] += b[c];
        }
    }
}
fn bias_bwd(dout: &[f32], m: usize, n: usize, db: &mut [f32]) {
    for r in 0..m {
        for c in 0..n {
            db[c] += dout[r * n + c];
        }
    }
}

/// RMSNorm per row with gain `g[d]` (eps matches the kernel's 1e-6). Returns
/// `(y, inv_rms[rows])`.
fn rmsnorm_bwd(x: &[f32], g: &[f32], inv: &[f32], dy: &[f32], rows: usize, d: usize, dg: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * d];
    for r in 0..rows {
        let iv = inv[r];
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

/// LayerNorm per row with gain/bias (eps 1e-6). Returns `(y, mean[rows], inv[rows])`.
#[allow(clippy::too_many_arguments)]
fn layernorm_bwd(x: &[f32], g: &[f32], mean: &[f32], inv: &[f32], dy: &[f32], rows: usize, d: usize, dg: &mut [f32], db: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * d];
    let df = d as f32;
    for r in 0..rows {
        let (mu, iv) = (mean[r], inv[r]);
        // normalized values
        let mut xhat = vec![0.0f32; d];
        for i in 0..d {
            xhat[i] = (x[r * d + i] - mu) * iv;
            dg[i] += dy[r * d + i] * xhat[i];
            db[i] += dy[r * d + i];
        }
        // dxhat = dy * g
        let mut sum_dxhat = 0.0f32;
        let mut sum_dxhat_xhat = 0.0f32;
        for i in 0..d {
            let dxh = dy[r * d + i] * g[i];
            sum_dxhat += dxh;
            sum_dxhat_xhat += dxh * xhat[i];
        }
        for i in 0..d {
            let dxh = dy[r * d + i] * g[i];
            dx[r * d + i] = iv * (dxh - sum_dxhat / df - xhat[i] * sum_dxhat_xhat / df);
        }
    }
    dx
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ---- causal MHA with per-dim softplus q-scaling (fwd + bwd) --------------------

struct AttnCache {
    q: Vec<f32>, // pre-scale q
    k: Vec<f32>,
    v: Vec<f32>,
    qscaled: Vec<f32>,
    qscale: Vec<f32>, // [hd]
    probs: Vec<f32>,  // [heads*s*s]
    s: usize,
}

/// `qkv` fused `[s, 3*inner]` (num_kv_heads==num_heads). Causal, keys j<=i.
/// Returns `(ctx[s,inner], cache)`.
fn attention(qkv: &[f32], scaling: &[f32], s: usize, heads: usize, hd: usize) -> (Vec<f32>, AttnCache) {
    let inner = heads * hd;
    let qkvd = 3 * inner;
    let base = 1.442_695_f32 / (hd as f32).sqrt();
    let qscale: Vec<f32> = (0..hd).map(|dd| base * softplus(scaling[dd])).collect();
    let mut q = vec![0.0f32; s * inner];
    let mut k = vec![0.0f32; s * inner];
    let mut v = vec![0.0f32; s * inner];
    let mut qscaled = vec![0.0f32; s * inner];
    for t in 0..s {
        for h in 0..heads {
            for dd in 0..hd {
                let qi = qkv[t * qkvd + h * hd + dd];
                q[t * inner + h * hd + dd] = qi;
                qscaled[t * inner + h * hd + dd] = qi * qscale[dd];
                k[t * inner + h * hd + dd] = qkv[t * qkvd + inner + h * hd + dd];
                v[t * inner + h * hd + dd] = qkv[t * qkvd + 2 * inner + h * hd + dd];
            }
        }
    }
    let mut probs = vec![0.0f32; heads * s * s];
    let mut ctx = vec![0.0f32; s * inner];
    for h in 0..heads {
        for i in 0..s {
            let mut sc = vec![0.0f32; i + 1];
            let mut mx = f32::MIN;
            for (j, scj) in sc.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for dd in 0..hd {
                    dot += qscaled[i * inner + h * hd + dd] * k[j * inner + h * hd + dd];
                }
                *scj = dot;
                mx = mx.max(dot);
            }
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
    (ctx, AttnCache { q, k, v, qscaled, qscale, probs, s })
}

/// Backward of [`attention`]: `d_ctx` → `d_qkv[s,3*inner]` and `d_scaling[hd]`.
fn attention_bwd(d_ctx: &[f32], c: &AttnCache, scaling: &[f32], heads: usize, hd: usize, d_scaling: &mut [f32]) -> Vec<f32> {
    let s = c.s;
    let inner = heads * hd;
    let qkvd = 3 * inner;
    let mut d_qscaled = vec![0.0f32; s * inner];
    let mut d_k = vec![0.0f32; s * inner];
    let mut d_v = vec![0.0f32; s * inner];
    for h in 0..heads {
        for i in 0..s {
            let mut dprob = vec![0.0f32; i + 1];
            for (j, dpj) in dprob.iter_mut().enumerate() {
                let p = c.probs[h * s * s + i * s + j];
                let mut dp = 0.0f32;
                for dd in 0..hd {
                    let g = d_ctx[i * inner + h * hd + dd];
                    dp += g * c.v[j * inner + h * hd + dd];
                    d_v[j * inner + h * hd + dd] += p * g;
                }
                *dpj = dp;
            }
            let sdot: f32 = (0..=i).map(|j| c.probs[h * s * s + i * s + j] * dprob[j]).sum();
            for j in 0..=i {
                let dscore = c.probs[h * s * s + i * s + j] * (dprob[j] - sdot);
                for dd in 0..hd {
                    d_qscaled[i * inner + h * hd + dd] += dscore * c.k[j * inner + h * hd + dd];
                    d_k[j * inner + h * hd + dd] += dscore * c.qscaled[i * inner + h * hd + dd];
                }
            }
        }
    }
    // unscale: qscaled = q * qscale[dd]; d_q = d_qscaled * qscale; d_qscale accum
    let base = 1.442_695_f32 / (hd as f32).sqrt();
    let mut d_qkv = vec![0.0f32; s * qkvd];
    for t in 0..s {
        for h in 0..heads {
            for dd in 0..hd {
                let dqs = d_qscaled[t * inner + h * hd + dd];
                d_qkv[t * qkvd + h * hd + dd] = dqs * c.qscale[dd];
                // d qscale[dd] += dqs * q ; qscale = base*softplus(scaling) -> d scaling = base*sigmoid(scaling)
                d_scaling[dd] += dqs * c.q[t * inner + h * hd + dd] * base * sigmoid(scaling[dd]);
                d_qkv[t * qkvd + inner + h * hd + dd] = d_k[t * inner + h * hd + dd];
                d_qkv[t * qkvd + 2 * inner + h * hd + dd] = d_v[t * inner + h * hd + dd];
            }
        }
    }
    d_qkv
}

// ---- caches -------------------------------------------------------------------

struct ExpertCache {
    ln: Vec<f32>,     // layernorm output
    ln_mean: Vec<f32>,
    ln_inv: Vec<f32>,
    g_pre: Vec<f32>,  // pre-relu gate
    g: Vec<f32>,      // post-relu
    idx: usize,       // expert index
}

struct BlockCache {
    emb_in: Vec<f32>,
    xn: Vec<f32>,
    xn_inv: Vec<f32>,
    attn: AttnCache,
    ctx: Vec<f32>,
    emb_attn: Vec<f32>, // after attention residual (= MoE input x_in)
    p: Vec<f32>,        // moe_prenorm output
    p_inv: Vec<f32>,
    probs_full: Vec<f32>, // softmax over E, [s,E]
    sel: Vec<Vec<usize>>, // top-k indices per token
    w: Vec<f32>,          // combine weights [s,E]
    experts: Vec<ExpertCache>, // computed experts (one per (token-set) actually per unique idx); we store per-expert-idx caches computed on p
    s: usize,
}

pub struct FullCache {
    blocks: Vec<BlockCache>,
    head: HeadCache,
}

struct HeadCache {
    emb: Vec<f32>,
    hid_pre: Vec<f32>,  // pre-silu
    hid: Vec<f32>,      // post-silu
    head_out: Vec<f32>, // [s, head_out]
    s: usize,
}

/// The trainable FinCast (host). Holds a config and a name→weights map.
pub struct FincastTrain {
    pub cfg: FincastConfig,
    pub w: HashMap<String, Vec<f32>>,
}

impl FincastTrain {
    pub fn zero_grads(&self) -> HashMap<String, Vec<f32>> {
        self.w.iter().map(|(k, v)| (k.clone(), vec![0.0f32; v.len()])).collect()
    }

    fn softmax_top2(&self, glog: &[f32], s: usize, e: usize) -> (Vec<f32>, Vec<Vec<usize>>, Vec<f32>) {
        let top = self.cfg.gating_top_n.min(e);
        let mut probs = vec![0.0f32; s * e];
        let mut sel = Vec::with_capacity(s);
        let mut w = vec![0.0f32; s * e];
        for t in 0..s {
            let row = &glog[t * e..t * e + e];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - mx).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for c in 0..e {
                probs[t * e + c] = exps[c] / denom;
            }
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| probs[t * e + b].partial_cmp(&probs[t * e + a]).unwrap());
            let s2: Vec<usize> = idx[..top].to_vec();
            let ssum: f32 = s2.iter().map(|&i| probs[t * e + i]).sum::<f32>().max(1e-12);
            for &i in &s2 {
                w[t * e + i] = probs[t * e + i] / ssum;
            }
            sel.push(s2);
        }
        (probs, sel, w)
    }

    /// Forward of one decoder block on `emb_in[s,d]`. Returns `(emb_out, cache)`.
    fn block_forward(&self, b: usize, emb_in: &[f32]) -> (Vec<f32>, BlockCache) {
        let cfg = &self.cfg;
        let (d, inner, heads, hd, eps) =
            (cfg.hidden_size, cfg.inner_dim(), cfg.num_heads, cfg.head_dim, cfg.rms_norm_eps);
        let s = emb_in.len() / d;
        let p = format!("stacked_transformer.layers.{b}");
        let ww = |n: &str| &self.w[n];

        // attention
        let (xn, xn_inv) = hostmath::rmsnorm_rows_with_inv(emb_in, ww(&format!("{p}.input_layernorm.weight")), s, d, eps);
        let mut qkv = matmul(&xn, ww(&format!("{p}.self_attn.qkv_proj.weight")), s, d, cfg.qkv_dim());
        bias_add(&mut qkv, ww(&format!("{p}.self_attn.qkv_proj.bias")), s, cfg.qkv_dim());
        let scaling = ww(&format!("{p}.self_attn.scaling")).clone();
        let (ctx, attn) = attention(&qkv, &scaling, s, heads, hd);
        let mut o = matmul(&ctx, ww(&format!("{p}.self_attn.o_proj.weight")), s, inner, d);
        bias_add(&mut o, ww(&format!("{p}.self_attn.o_proj.bias")), s, d);
        let emb_attn: Vec<f32> = (0..s * d).map(|i| emb_in[i] + o[i]).collect();

        // MoE
        let (pp, p_inv) = hostmath::rmsnorm_rows_with_inv(&emb_attn, ww(&format!("{p}.moe.moe_prenorm.gamma")), s, d, eps);
        let e = cfg.num_experts;
        let glog = matmul(&pp, ww(&format!("{p}.moe.moe.gate.to_gates.weight")), s, d, e);
        let (probs_full, sel, w) = self.softmax_top2(&glog, s, e);
        // compute each expert that is selected by any token
        let mut needed: Vec<usize> = Vec::new();
        for row in &sel {
            for &i in row {
                if !needed.contains(&i) {
                    needed.push(i);
                }
            }
        }
        let mut moe_out = pp.clone();
        let mut experts = Vec::new();
        for &ei in &needed {
            let ep = format!("{p}.moe.moe.experts.experts.{ei}");
            let (ln, ln_mean, ln_inv) = hostmath::layernorm_rows_with_stats(&pp, ww(&format!("{ep}.layer_norm.weight")), ww(&format!("{ep}.layer_norm.bias")), s, d, 1e-6);
            let mut g_pre = matmul(&ln, ww(&format!("{ep}.gate_proj.weight")), s, d, d);
            bias_add(&mut g_pre, ww(&format!("{ep}.gate_proj.bias")), s, d);
            let g: Vec<f32> = g_pre.iter().map(|&v| v.max(0.0)).collect();
            let mut mlp = matmul(&g, ww(&format!("{ep}.down_proj.weight")), s, d, d);
            bias_add(&mut mlp, ww(&format!("{ep}.down_proj.bias")), s, d);
            for t in 0..s {
                let wt = w[t * e + ei];
                if wt != 0.0 {
                    for c in 0..d {
                        moe_out[t * d + c] += wt * mlp[t * d + c];
                    }
                }
            }
            experts.push(ExpertCache { ln, ln_mean, ln_inv, g_pre, g, idx: ei });
        }
        let emb_out: Vec<f32> = (0..s * d).map(|i| moe_out[i] + emb_attn[i]).collect();
        let _ = &qkv;
        (emb_out, BlockCache { emb_in: emb_in.to_vec(), xn, xn_inv, attn, ctx, emb_attn, p: pp, p_inv, probs_full, sel, w, experts, s })
    }

    fn block_backward(&self, b: usize, c: &BlockCache, d_emb_out: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, inner, heads, hd) = (cfg.hidden_size, cfg.inner_dim(), cfg.num_heads, cfg.head_dim);
        let e = cfg.num_experts;
        let s = c.s;
        let pre = format!("stacked_transformer.layers.{b}");
        let ww = |n: &str| &self.w[n];

        // emb_out = moe_out + emb_attn
        let d_moe_out = d_emb_out.to_vec();
        let mut d_emb_attn: Vec<f32> = d_emb_out.to_vec();
        // moe_out = p + Σ w * mlp ; so d_p starts from d_moe_out
        let mut d_p = d_moe_out.clone();
        let mut d_w = vec![0.0f32; s * e];
        // backprop each expert
        for ec in &c.experts {
            let ei = ec.idx;
            let ep = format!("{pre}.moe.moe.experts.experts.{ei}");
            // d_mlp[t] = w[t,ei] * d_moe_out[t]; d_w[t,ei] = <d_moe_out[t], mlp[t]>
            // recompute mlp[t] = down(g)+bias -> need it for d_w; but we have g cache.
            let mlp = {
                let mut m = matmul(&ec.g, ww(&format!("{ep}.down_proj.weight")), s, d, d);
                bias_add(&mut m, ww(&format!("{ep}.down_proj.bias")), s, d);
                m
            };
            let mut d_mlp = vec![0.0f32; s * d];
            for t in 0..s {
                let wt = c.w[t * e + ei];
                let mut acc = 0.0f32;
                for cc in 0..d {
                    d_mlp[t * d + cc] = wt * d_moe_out[t * d + cc];
                    acc += d_moe_out[t * d + cc] * mlp_get(&mlp, t, cc, d);
                }
                d_w[t * e + ei] = acc;
            }
            // down_proj bwd
            bias_bwd(&d_mlp, s, d, g.get_mut(&format!("{ep}.down_proj.bias")).unwrap());
            let d_g = matmul_bwd(&ec.g, ww(&format!("{ep}.down_proj.weight")), &d_mlp, s, d, d, g.get_mut(&format!("{ep}.down_proj.weight")).unwrap());
            // relu bwd
            let d_gpre: Vec<f32> = (0..s * d).map(|i| if ec.g_pre[i] > 0.0 { d_g[i] } else { 0.0 }).collect();
            // gate_proj bwd
            bias_bwd(&d_gpre, s, d, g.get_mut(&format!("{ep}.gate_proj.bias")).unwrap());
            let d_ln = matmul_bwd(&ec.ln, ww(&format!("{ep}.gate_proj.weight")), &d_gpre, s, d, d, g.get_mut(&format!("{ep}.gate_proj.weight")).unwrap());
            // layernorm bwd -> accumulate into d_p (temp grads to avoid a double borrow)
            let mut dgln = vec![0.0f32; d];
            let mut dbln = vec![0.0f32; d];
            let d_pp = layernorm_bwd(&c.p, ww(&format!("{ep}.layer_norm.weight")), &ec.ln_mean, &ec.ln_inv, &d_ln, s, d, &mut dgln, &mut dbln);
            add_into(g.get_mut(&format!("{ep}.layer_norm.weight")).unwrap(), &dgln);
            add_into(g.get_mut(&format!("{ep}.layer_norm.bias")).unwrap(), &dbln);
            for i in 0..s * d {
                d_p[i] += d_pp[i];
            }
        }
        // d_w -> d_glog via top-2 renorm + softmax
        let mut d_glog = vec![0.0f32; s * e];
        for t in 0..s {
            let sel = &c.sel[t];
            let ssum: f32 = sel.iter().map(|&i| c.probs_full[t * e + i]).sum::<f32>().max(1e-12);
            // d_s[j] for j in sel = (d_w[j]*S - Σ_i d_w[i] s_i)/S^2
            let sum_dw_s: f32 = sel.iter().map(|&i| d_w[t * e + i] * c.probs_full[t * e + i]).sum();
            let mut d_s = vec![0.0f32; e];
            for &j in sel {
                d_s[j] = (d_w[t * e + j] * ssum - sum_dw_s) / (ssum * ssum);
            }
            // softmax bwd over full E
            let sdot: f32 = (0..e).map(|m| c.probs_full[t * e + m] * d_s[m]).sum();
            for kk in 0..e {
                d_glog[t * e + kk] = c.probs_full[t * e + kk] * (d_s[kk] - sdot);
            }
        }
        let d_pp2 = matmul_bwd(&c.p, ww(&format!("{pre}.moe.moe.gate.to_gates.weight")), &d_glog, s, d, e, g.get_mut(&format!("{pre}.moe.moe.gate.to_gates.weight")).unwrap());
        for i in 0..s * d {
            d_p[i] += d_pp2[i];
        }
        // moe_prenorm rmsnorm bwd: d_p -> d_emb_attn
        let d_emb_attn_a = rmsnorm_bwd(&c.emb_attn, ww(&format!("{pre}.moe.moe_prenorm.gamma")), &c.p_inv, &d_p, s, d, g.get_mut(&format!("{pre}.moe.moe_prenorm.gamma")).unwrap());
        for i in 0..s * d {
            d_emb_attn[i] += d_emb_attn_a[i];
        }

        // attention block: emb_attn = emb_in + o
        let mut d_emb_in = d_emb_attn.clone();
        let d_ctx = matmul_bwd(&c.ctx, ww(&format!("{pre}.self_attn.o_proj.weight")), &d_emb_attn, s, inner, d, g.get_mut(&format!("{pre}.self_attn.o_proj.weight")).unwrap());
        bias_bwd(&d_emb_attn, s, d, g.get_mut(&format!("{pre}.self_attn.o_proj.bias")).unwrap());
        let scaling = ww(&format!("{pre}.self_attn.scaling")).clone();
        let d_qkv = attention_bwd(&d_ctx, &c.attn, &scaling, heads, hd, g.get_mut(&format!("{pre}.self_attn.scaling")).unwrap());
        bias_bwd(&d_qkv, s, cfg.qkv_dim(), g.get_mut(&format!("{pre}.self_attn.qkv_proj.bias")).unwrap());
        let d_xn = matmul_bwd(&c.xn, ww(&format!("{pre}.self_attn.qkv_proj.weight")), &d_qkv, s, d, cfg.qkv_dim(), g.get_mut(&format!("{pre}.self_attn.qkv_proj.weight")).unwrap());
        let d_emb_in_a = rmsnorm_bwd(&c.emb_in, ww(&format!("{pre}.input_layernorm.weight")), &c.xn_inv, &d_xn, s, d, g.get_mut(&format!("{pre}.input_layernorm.weight")).unwrap());
        for i in 0..s * d {
            d_emb_in[i] += d_emb_in_a[i];
        }
        d_emb_in
    }

    // -- head: horizon_ff residual block + PQ loss on the last patch row --------

    fn head_forward(&self, emb: &[f32], target: &[f32]) -> (f32, HeadCache) {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let ho = cfg.head_out_dim();
        let s = emb.len() / d;
        let ww = |n: &str| &self.w[n];
        // residual block with SiLU hidden
        let mut hid_pre = matmul(emb, ww("horizon_ff_layer.hidden_layer.0.weight"), s, d, f);
        bias_add(&mut hid_pre, ww("horizon_ff_layer.hidden_layer.0.bias"), s, f);
        let hid: Vec<f32> = hid_pre.iter().map(|&v| v * sigmoid(v)).collect();
        let mut o1 = matmul(&hid, ww("horizon_ff_layer.output_layer.weight"), s, f, ho);
        bias_add(&mut o1, ww("horizon_ff_layer.output_layer.bias"), s, ho);
        let mut res = matmul(emb, ww("horizon_ff_layer.residual_layer.weight"), s, d, ho);
        bias_add(&mut res, ww("horizon_ff_layer.residual_layer.bias"), s, ho);
        let head_out: Vec<f32> = (0..s * ho).map(|i| o1[i] + res[i]).collect();
        let loss = self.head_loss(&head_out, s, target);
        (loss, HeadCache { emb: emb.to_vec(), hid_pre, hid, head_out, s })
    }

    /// Loss on the last patch row: `mean_pinball(9 quantiles) + MSE(mean)`.
    fn head_loss(&self, head_out: &[f32], s: usize, target: &[f32]) -> f32 {
        let cfg = &self.cfg;
        let ho = cfg.head_out_dim();
        let no = cfg.num_outputs();
        let hlen = cfg.horizon_len;
        let last = &head_out[(s - 1) * ho..s * ho]; // [hlen, no]
        // quantiles [hlen, 9] step-major
        let mut quant = vec![0.0f32; hlen * cfg.num_quantiles];
        let mut mse = 0.0f32;
        for t in 0..hlen {
            mse += (last[t * no] - target[t]) * (last[t * no] - target[t]);
            for qi in 0..cfg.num_quantiles {
                quant[t * cfg.num_quantiles + qi] = last[t * no + 1 + qi];
            }
        }
        mse /= hlen as f32;
        let pin = forecast::metrics::mean_pinball(&quant, &QUANTILES, target);
        pin + mse
    }

    fn head_backward(&self, c: &HeadCache, target: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let ho = cfg.head_out_dim();
        let no = cfg.num_outputs();
        let hlen = cfg.horizon_len;
        let s = c.s;
        let ww = |n: &str| &self.w[n];

        // d head_out from loss (only last row nonzero)
        let mut d_head = vec![0.0f32; s * ho];
        let last = &c.head_out[(s - 1) * ho..s * ho];
        let mut quant = vec![0.0f32; hlen * cfg.num_quantiles];
        for t in 0..hlen {
            for qi in 0..cfg.num_quantiles {
                quant[t * cfg.num_quantiles + qi] = last[t * no + 1 + qi];
            }
        }
        let d_quant = forecast::metrics::mean_pinball_grad(&quant, &QUANTILES, target);
        let base = (s - 1) * ho;
        for t in 0..hlen {
            // MSE on mean: d = 2*(pred-y)/hlen
            d_head[base + t * no] = 2.0 * (last[t * no] - target[t]) / hlen as f32;
            for qi in 0..cfg.num_quantiles {
                d_head[base + t * no + 1 + qi] = d_quant[t * cfg.num_quantiles + qi];
            }
        }

        // residual block bwd: head_out = o1 + res
        bias_bwd(&d_head, s, ho, g.get_mut("horizon_ff_layer.output_layer.bias").unwrap());
        let d_hid = matmul_bwd(&c.hid, ww("horizon_ff_layer.output_layer.weight"), &d_head, s, f, ho, g.get_mut("horizon_ff_layer.output_layer.weight").unwrap());
        bias_bwd(&d_head, s, ho, g.get_mut("horizon_ff_layer.residual_layer.bias").unwrap());
        let d_emb_res = matmul_bwd(&c.emb, ww("horizon_ff_layer.residual_layer.weight"), &d_head, s, d, ho, g.get_mut("horizon_ff_layer.residual_layer.weight").unwrap());
        // silu bwd
        let d_hidpre: Vec<f32> = (0..s * f).map(|i| {
            let x = c.hid_pre[i];
            let sg = sigmoid(x);
            d_hid[i] * (sg + x * sg * (1.0 - sg))
        }).collect();
        bias_bwd(&d_hidpre, s, f, g.get_mut("horizon_ff_layer.hidden_layer.0.bias").unwrap());
        let d_emb_hid = matmul_bwd(&c.emb, ww("horizon_ff_layer.hidden_layer.0.weight"), &d_hidpre, s, d, f, g.get_mut("horizon_ff_layer.hidden_layer.0.weight").unwrap());
        (0..s * d).map(|i| d_emb_res[i] + d_emb_hid[i]).collect()
    }

    /// Full differentiable core: blocks → head, PQ loss on the last patch.
    pub fn full_forward(&self, emb_in: &[f32], target: &[f32]) -> (f32, FullCache) {
        let mut emb = emb_in.to_vec();
        let mut blocks = Vec::with_capacity(self.cfg.num_layers);
        for b in 0..self.cfg.num_layers {
            let (out, c) = self.block_forward(b, &emb);
            emb = out;
            blocks.push(c);
        }
        let (loss, head) = self.head_forward(&emb, target);
        (loss, FullCache { blocks, head })
    }

    pub fn full_backward(&self, cache: &FullCache, target: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let mut d_emb = self.head_backward(&cache.head, target, g);
        for b in (0..self.cfg.num_layers).rev() {
            d_emb = self.block_backward(b, &cache.blocks[b], &d_emb, g);
        }
        d_emb
    }

    /// One plain-SGD step on a single example. Returns the pre-step loss.
    pub fn sgd_step(&mut self, emb_in: &[f32], target: &[f32], lr: f32) -> f32 {
        let (loss, cache) = self.full_forward(emb_in, target);
        let mut g = self.zero_grads();
        self.full_backward(&cache, target, &mut g);
        for (name, grad) in &g {
            let w = self.w.get_mut(name).unwrap();
            for (wi, gi) in w.iter_mut().zip(grad) {
                *wi -= lr * gi;
            }
        }
        loss
    }
}

#[inline]
fn mlp_get(m: &[f32], t: usize, c: usize, d: usize) -> f32 {
    m[t * d + c]
}

#[inline]
fn add_into(dst: &mut [f32], src: &[f32]) {
    for (a, b) in dst.iter_mut().zip(src) {
        *a += b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }

    fn params(cfg: &FincastConfig, seed: &mut u64) -> HashMap<String, Vec<f32>> {
        let mut w = HashMap::new();
        for (name, shape) in cfg.param_list() {
            let n: usize = shape.iter().product();
            // gains (layernorm/rmsnorm .weight/.gamma) near 1, others small.
            let is_gain = name.ends_with("layer_norm.weight")
                || name.ends_with("input_layernorm.weight")
                || name.ends_with("moe_prenorm.gamma");
            let is_scaling = name.ends_with("self_attn.scaling");
            let data: Vec<f32> = (0..n).map(|_| {
                let base = if is_gain { 1.0 } else if is_scaling { 0.5 } else { 0.0 };
                base + rng(seed)
            }).collect();
            w.insert(name, data);
        }
        w
    }

    #[test]
    fn full_core_gradcheck() {
        let cfg = FincastConfig::tiny();
        let mut seed = 0xF1C_u64;
        let model = FincastTrain { cfg: cfg.clone(), w: params(&cfg, &mut seed) };
        let d = cfg.hidden_size;
        let s = 5usize;
        let hlen = cfg.horizon_len;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let target: Vec<f32> = (0..hlen).map(|_| rng(&mut seed)).collect();

        let (_l, cache) = model.full_forward(&emb, &target);
        let mut g = model.zero_grads();
        let d_emb = model.full_backward(&cache, &target, &mut g);

        let loss = |m: &FincastTrain, e: &[f32]| m.full_forward(e, &target).0;
        let eps = 5e-3f32;
        let tol = |a: f32, n: f32| (a - n).abs() <= 4e-3 + 8e-2 * a.abs().max(n.abs());
        let mut checked = 0usize;
        for name in model.w.keys() {
            let len = model.w[name].len();
            for &idx in &[0usize, len / 2, len - 1] {
                let mut mp = FincastTrain { cfg: cfg.clone(), w: model.w.clone() };
                mp.w.get_mut(name).unwrap()[idx] += eps;
                let mut mm = FincastTrain { cfg: cfg.clone(), w: model.w.clone() };
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
        assert!(checked > 60, "expected a broad gradcheck, only {checked}");
    }

    #[test]
    fn core_learns_a_fixed_example() {
        let cfg = FincastConfig::tiny();
        let mut seed = 0x5EED_u64;
        let mut model = FincastTrain { cfg: cfg.clone(), w: params(&cfg, &mut seed) };
        let d = cfg.hidden_size;
        let s = 5usize;
        let hlen = cfg.horizon_len;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let target: Vec<f32> = (0..hlen).map(|_| rng(&mut seed) * 3.0).collect();

        let l0 = model.full_forward(&emb, &target).0;
        let mut last = l0;
        for _ in 0..200 {
            last = model.sgd_step(&emb, &target, 0.03);
        }
        assert!(last < 0.6 * l0, "loss must fall substantially: {l0} -> {last}");
    }
}
