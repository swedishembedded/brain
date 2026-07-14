// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DSA indexer distillation (host-side, fp32). The indexer is detached from the
//! LM loss; it is trained by a **separate** objective — match the dense MLA
//! attention distribution over keys (the DeepSeek-V3.2 / IndexCache recipe). We
//! compute the indexer forward + the distillation cross-entropy + the analytic
//! gradient w.r.t. the indexer params entirely on the host (the indexer is small,
//! and for a from-scratch tiny GLM this is simple and gradient-checkable). Grads
//! flow only into the `idx.*` params — never into the frozen backbone.
//!
//! Per query s and causal key t≤s:
//!   q[s,h,:] = q_resid[s] · Wq_bᵀ ;  k[t,:] = LayerNorm(x[t]·Wkᵀ) ;  w[s,h] = x[s]·Wprojᵀ
//!   (RoPE the rope slice of q,k) ;  score[s,t] = Σ_h (w[s,h]·H^-½)·relu((q[s,h]·k[t])·D^-½)
//!   qdist = softmax_t(score) ;  target = mean_h dense_probs[h,s,t]
//!   loss  = Σ_s −Σ_t target·log(qdist)          (cross-entropy → d score = qdist − target)

/// Indexer params for one `Full` layer (brain names under `blocks.{l}.idx.`).
pub struct IdxWeights {
    pub wq_b: Vec<f32>,        // [H*D, ql]
    pub wk: Vec<f32>,          // [D, d]
    pub k_norm_w: Vec<f32>,    // [D]
    pub k_norm_b: Vec<f32>,    // [D]
    pub weights_proj: Vec<f32>, // [H, d]
}

/// Analytic gradients (same shapes as [`IdxWeights`]).
#[derive(Clone)]
pub struct IdxGrads {
    pub wq_b: Vec<f32>,
    pub wk: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub k_norm_b: Vec<f32>,
    pub weights_proj: Vec<f32>,
}

/// Geometry for one indexer layer.
pub struct IdxDims {
    pub b: usize,
    pub t: usize,
    pub h: usize,   // index_n_heads
    pub d: usize,   // index_head_dim
    pub rope: usize, // index rope slice (= qk_rope_head_dim)
    pub ql: usize,  // q_lora_rank
    pub dm: usize,  // d_model
    pub mla_heads: usize, // number of MLA heads (for averaging dense_probs)
}

const LN_EPS: f32 = 1e-5;

fn rope_angle(pos: usize, pair: usize, rope: usize) -> f32 {
    pos as f32 * 10000f32.powf(-(2.0 * pair as f32) / rope as f32)
}

/// Interleaved RoPE on the first `rope` channels of a `d`-wide head (in place).
fn rope_fwd(v: &mut [f32], pos: usize, rope: usize) {
    for p in 0..rope / 2 {
        let (c, s) = {
            let a = rope_angle(pos, p, rope);
            (a.cos(), a.sin())
        };
        let (e, o) = (v[2 * p], v[2 * p + 1]);
        v[2 * p] = e * c - o * s;
        v[2 * p + 1] = o * c + e * s;
    }
}

/// Backward of [`rope_fwd`] (rotate the grad by −angle).
fn rope_bwd(g: &mut [f32], pos: usize, rope: usize) {
    for p in 0..rope / 2 {
        let (c, s) = {
            let a = rope_angle(pos, p, rope);
            (a.cos(), a.sin())
        };
        let (g0, g1) = (g[2 * p], g[2 * p + 1]);
        g[2 * p] = g0 * c + g1 * s;
        g[2 * p + 1] = -g0 * s + g1 * c;
    }
}

/// Compute the distillation loss + analytic idx grads for one layer.
/// `xn1` = `[n,dm]` (input_ln output), `q_resid` = `[n,ql]` (q_a_layernorm output),
/// `dense_probs` = `[b, mla_heads, t, t]` (the MLA attention probs).
pub fn layer_distill(dm_dims: &IdxDims, xn1: &[f32], q_resid: &[f32], dense_probs: &[f32], w: &IdxWeights) -> (f32, IdxGrads) {
    let IdxDims { b, t, h, d, rope, ql, dm, mla_heads } = *dm_dims;
    let n = b * t;
    let qscale = 1.0 / (d as f32).sqrt();
    let wscale = 1.0 / (h as f32).sqrt();

    let mut g = IdxGrads {
        wq_b: vec![0.0; h * d * ql],
        wk: vec![0.0; d * dm],
        k_norm_w: vec![0.0; d],
        k_norm_b: vec![0.0; d],
        weights_proj: vec![0.0; h * dm],
    };
    let mut loss = 0.0f32;

    // ---- forward: per-row projections (cached for backward) ----
    // q_idx[row, h, d], roped ; k[row, d] = rope(LayerNorm(x·Wkᵀ)) ; weights[row, h]
    let mut q_idx = vec![0.0f32; n * h * d];
    let mut k = vec![0.0f32; n * d];
    let mut k_pre = vec![0.0f32; n * d]; // pre-LayerNorm (for LN backward)
    let mut k_mean = vec![0.0f32; n];
    let mut k_inv = vec![0.0f32; n]; // 1/sqrt(var+eps)
    let mut weights = vec![0.0f32; n * h];
    for r in 0..n {
        let pos = r % t;
        // q_idx
        for hh in 0..h {
            for dd in 0..d {
                let mut acc = 0.0;
                for j in 0..ql {
                    acc += q_resid[r * ql + j] * w.wq_b[(hh * d + dd) * ql + j];
                }
                q_idx[(r * h + hh) * d + dd] = acc;
            }
            rope_fwd(&mut q_idx[(r * h + hh) * d..(r * h + hh) * d + d], pos, rope);
        }
        // k_pre = x·Wkᵀ
        for dd in 0..d {
            let mut acc = 0.0;
            for j in 0..dm {
                acc += xn1[r * dm + j] * w.wk[dd * dm + j];
            }
            k_pre[r * d + dd] = acc;
        }
        // LayerNorm over D
        let mean = k_pre[r * d..r * d + d].iter().sum::<f32>() / d as f32;
        let var = k_pre[r * d..r * d + d].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        k_mean[r] = mean;
        k_inv[r] = inv;
        for dd in 0..d {
            k[r * d + dd] = (k_pre[r * d + dd] - mean) * inv * w.k_norm_w[dd] + w.k_norm_b[dd];
        }
        rope_fwd(&mut k[r * d..r * d + d], pos, rope);
        // weights = x·Wprojᵀ
        for hh in 0..h {
            let mut acc = 0.0;
            for j in 0..dm {
                acc += xn1[r * dm + j] * w.weights_proj[hh * dm + j];
            }
            weights[r * h + hh] = acc;
        }
    }

    // grad accumulators for the roped q/k (filled in the score backward, then
    // un-roped and pushed into the weight grads).
    let mut d_q_idx = vec![0.0f32; n * h * d];
    let mut d_k = vec![0.0f32; n * d];

    // ---- per (b, query i): scores, softmax, CE loss, backward to q/k/weights ----
    for bi in 0..b {
        for i in 0..t {
            let rq = bi * t + i;
            // scores[j] over j<=i, plus per-(h,j) relu-hit for backward
            let mut score = vec![0.0f32; i + 1];
            let mut hit = vec![false; (i + 1) * h]; // relu active mask
            for j in 0..=i {
                let rk = bi * t + j;
                let mut sc = 0.0;
                for hh in 0..h {
                    let mut dot = 0.0;
                    for dd in 0..d {
                        dot += q_idx[(rq * h + hh) * d + dd] * k[rk * d + dd];
                    }
                    let pre = dot * qscale;
                    if pre > 0.0 {
                        hit[j * h + hh] = true;
                        sc += weights[rq * h + hh] * wscale * pre;
                    }
                }
                score[j] = sc;
            }
            // softmax over j<=i
            let mx = score.iter().cloned().fold(f32::MIN, f32::max);
            let mut z = 0.0;
            let mut qd = vec![0.0f32; i + 1];
            for j in 0..=i {
                qd[j] = (score[j] - mx).exp();
                z += qd[j];
            }
            for j in 0..=i {
                qd[j] /= z;
            }
            // target = mean_h dense_probs[bi,h,i,j]  (causal)
            let mut tgt = vec![0.0f32; i + 1];
            for j in 0..=i {
                let mut acc = 0.0;
                for hh in 0..mla_heads {
                    acc += dense_probs[((bi * mla_heads + hh) * t + i) * t + j];
                }
                tgt[j] = acc / mla_heads as f32;
            }
            // CE loss + d score = qdist - target
            for j in 0..=i {
                loss += -tgt[j] * (qd[j] + 1e-20).ln();
            }
            for j in 0..=i {
                let d_sc = qd[j] - tgt[j];
                let rk = bi * t + j;
                for hh in 0..h {
                    if !hit[j * h + hh] {
                        continue;
                    }
                    // score = Σ w·wscale·(dot·qscale) ; d w[h] += d_sc·wscale·pre
                    let mut dot = 0.0;
                    for dd in 0..d {
                        dot += q_idx[(rq * h + hh) * d + dd] * k[rk * d + dd];
                    }
                    let pre = dot * qscale;
                    // d weights
                    let idx_w = rq * h + hh;
                    // accumulate into d_weights (stored per-row below via g.weights_proj path)
                    // d_pre -> d_dot -> d_q/d_k
                    let d_w = d_sc * wscale * pre;
                    let d_dot = d_sc * weights[idx_w] * wscale * qscale; // relu' = 1 (hit)
                    // stash d_weights into a temporary via weights_proj grad later; do it inline:
                    // d weights_proj[h,:] += d_w * xn1[rq,:]
                    for jj in 0..dm {
                        g.weights_proj[hh * dm + jj] += d_w * xn1[rq * dm + jj];
                    }
                    for dd in 0..d {
                        d_q_idx[(rq * h + hh) * d + dd] += d_dot * k[rk * d + dd];
                        d_k[rk * d + dd] += d_dot * q_idx[(rq * h + hh) * d + dd];
                    }
                }
            }
        }
    }

    // ---- un-rope the q/k grads, then push through the projections ----
    for r in 0..n {
        let pos = r % t;
        for hh in 0..h {
            rope_bwd(&mut d_q_idx[(r * h + hh) * d..(r * h + hh) * d + d], pos, rope);
            // d wq_b[h*D+dd, :] += d_q_idx[r,h,dd] * q_resid[r,:]
            for dd in 0..d {
                let dg = d_q_idx[(r * h + hh) * d + dd];
                for j in 0..ql {
                    g.wq_b[(hh * d + dd) * ql + j] += dg * q_resid[r * ql + j];
                }
            }
        }
        rope_bwd(&mut d_k[r * d..r * d + d], pos, rope);
        // LayerNorm backward: d_k (post-LN, pre-rope now un-roped) -> d_k_pre
        // y_dd = (x_dd - mean)*inv*gw_dd + gb_dd
        let inv = k_inv[r];
        let mean = k_mean[r];
        // gamma/beta grads + normalized value
        let mut dnorm = vec![0.0f32; d]; // d_k * gamma  (grad wrt normalized xhat)
        for dd in 0..d {
            let xhat = (k_pre[r * d + dd] - mean) * inv;
            g.k_norm_w[dd] += d_k[r * d + dd] * xhat;
            g.k_norm_b[dd] += d_k[r * d + dd];
            dnorm[dd] = d_k[r * d + dd] * w.k_norm_w[dd];
        }
        // xhat = (x-mean)*inv ; standard LN input grad
        let sum_dn: f32 = dnorm.iter().sum();
        let sum_dn_xhat: f32 = (0..d).map(|dd| dnorm[dd] * (k_pre[r * d + dd] - mean) * inv).sum();
        for dd in 0..d {
            let xhat = (k_pre[r * d + dd] - mean) * inv;
            let d_kpre = inv / d as f32 * (d as f32 * dnorm[dd] - sum_dn - xhat * sum_dn_xhat);
            // d wk[dd,:] += d_kpre * xn1[r,:]
            for j in 0..dm {
                g.wk[dd * dm + j] += d_kpre * xn1[r * dm + j];
            }
        }
    }

    (loss, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        // xorshift -> [-0.5,0.5]
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// Finite-difference check of the analytic idx grads against the distillation
    /// loss — proves the host indexer backward (matmuls, LayerNorm, interleaved
    /// RoPE, relu-weighted scores, softmax-CE) is correct.
    #[test]
    fn distill_grads_match_finite_differences() {
        let dims = IdxDims { b: 2, t: 4, h: 2, d: 6, rope: 4, ql: 5, dm: 7, mla_heads: 3 };
        let n = dims.b * dims.t;
        let mut s = 12345u64;
        let mk = |len: usize, s: &mut u64| (0..len).map(|_| rng(s)).collect::<Vec<f32>>();
        let xn1 = mk(n * dims.dm, &mut s);
        let q_resid = mk(n * dims.ql, &mut s);
        // dense_probs: random then softmax over causal keys per (b,h,i)
        let mut dp = vec![0.0f32; dims.b * dims.mla_heads * dims.t * dims.t];
        for bi in 0..dims.b {
            for hh in 0..dims.mla_heads {
                for i in 0..dims.t {
                    let base = ((bi * dims.mla_heads + hh) * dims.t + i) * dims.t;
                    let mut z = 0.0;
                    for j in 0..=i {
                        let v = (rng(&mut s) + 0.6).exp();
                        dp[base + j] = v;
                        z += v;
                    }
                    for j in 0..=i {
                        dp[base + j] /= z;
                    }
                }
            }
        }
        let mut w = IdxWeights {
            wq_b: mk(dims.h * dims.d * dims.ql, &mut s),
            wk: mk(dims.d * dims.dm, &mut s),
            k_norm_w: (0..dims.d).map(|_| 1.0 + 0.1 * rng(&mut s)).collect(),
            k_norm_b: mk(dims.d, &mut s),
            weights_proj: mk(dims.h * dims.dm, &mut s),
        };

        let (_l0, g) = layer_distill(&dims, &xn1, &q_resid, &dp, &w);
        let loss = |w: &IdxWeights| layer_distill(&dims, &xn1, &q_resid, &dp, w).0;
        let eps = 1e-3f32;
        let mut worst = 0.0f32;
        // Finite-difference a sample of entries from each param tensor.
        macro_rules! fd {
            ($field:ident, $ana:expr) => {{
                let len = w.$field.len();
                let step = (len / 4).max(1);
                let mut i = 0;
                while i < len {
                    let orig = w.$field[i];
                    w.$field[i] = orig + eps;
                    let lp = loss(&w);
                    w.$field[i] = orig - eps;
                    let lm = loss(&w);
                    w.$field[i] = orig;
                    let num = (lp - lm) / (2.0 * eps);
                    let err = (num - $ana[i]).abs() / (num.abs().max($ana[i].abs()).max(1e-2));
                    worst = worst.max(err);
                    i += step;
                }
            }};
        }
        fd!(wq_b, g.wq_b);
        fd!(wk, g.wk);
        fd!(k_norm_w, g.k_norm_w);
        fd!(k_norm_b, g.k_norm_b);
        fd!(weights_proj, g.weights_proj);
        // 8e-2 rel (brain's standard gradcheck tolerance): the analytic backward is
        // exact, but central-difference FD is noisy near the relu kinks (a q·k dot
        // crossing 0 makes the difference see a jump the analytic hit-mask elides).
        assert!(worst < 8e-2, "indexer distillation grad vs FD too far: {worst}");
    }

    /// RMS-normalized gradient descent (the same update rule `Glm::distill_step`
    /// uses) drives the distillation cross-entropy down toward a controlled,
    /// strongly-peaked target — validating that the indexer actually *learns* to
    /// track a non-uniform attention distribution (decoupled from a tiny model's
    /// near-uniform attention, which has almost nothing to distill).
    #[test]
    fn distill_training_converges() {
        let dims = IdxDims { b: 1, t: 6, h: 3, d: 8, rope: 4, ql: 6, dm: 8, mla_heads: 2 };
        let n = dims.b * dims.t;
        let mut s = 999u64;
        let mk = |len: usize, s: &mut u64| (0..len).map(|_| rng(s)).collect::<Vec<f32>>();
        let xn1 = mk(n * dims.dm, &mut s);
        let q_resid = mk(n * dims.ql, &mut s);
        // Peaked target: each query i attends mostly to key 0 (prob 0.85), the rest
        // split the remainder — clearly non-uniform, so a real gap to close.
        let mut dp = vec![0.0f32; dims.mla_heads * dims.t * dims.t];
        for hh in 0..dims.mla_heads {
            for i in 0..dims.t {
                let base = (hh * dims.t + i) * dims.t;
                let rest = if i > 0 { 0.15 / i as f32 } else { 0.0 };
                for j in 0..=i {
                    dp[base + j] = if j == 0 { if i == 0 { 1.0 } else { 0.85 } } else { rest };
                }
            }
        }
        // Small init, like the real indexer.
        let scale_small = |v: Vec<f32>| v.into_iter().map(|x| x * 0.04).collect::<Vec<f32>>();
        let mut w = IdxWeights {
            wq_b: scale_small(mk(dims.h * dims.d * dims.ql, &mut s)),
            wk: scale_small(mk(dims.d * dims.dm, &mut s)),
            k_norm_w: vec![1.0; dims.d],
            k_norm_b: vec![0.0; dims.d],
            weights_proj: scale_small(mk(dims.h * dims.dm, &mut s)),
        };
        let before = layer_distill(&dims, &xn1, &q_resid, &dp, &w).0;
        let lr = 0.05f32;
        let rms_step = |cur: &mut [f32], gr: &[f32]| {
            let ms = gr.iter().map(|g| g * g).sum::<f32>() / gr.len().max(1) as f32;
            let sc = lr / (ms.sqrt() + 1e-8);
            for (x, &g) in cur.iter_mut().zip(gr) {
                *x -= sc * g;
            }
        };
        for _ in 0..600 {
            let (_l, g) = layer_distill(&dims, &xn1, &q_resid, &dp, &w);
            rms_step(&mut w.wq_b, &g.wq_b);
            rms_step(&mut w.wk, &g.wk);
            rms_step(&mut w.k_norm_w, &g.k_norm_w);
            rms_step(&mut w.k_norm_b, &g.k_norm_b);
            rms_step(&mut w.weights_proj, &g.weights_proj);
        }
        let after = layer_distill(&dims, &xn1, &q_resid, &dp, &w).0;
        assert!(after < before * 0.7, "indexer distillation did not converge: {before} -> {after}");
    }
}
