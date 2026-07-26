// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host reference forward + analytic backward for ONE S³-DiT block, the unit the
//! whole DiT training step is built from. This is the correctness anchor: a
//! finite-difference gradcheck (`tests/block_grad.rs`) gates the analytic
//! gradients, exactly as brain gates every hand-written backward.
//!
//! The forward mirrors [`crate::block::build_block_steps`] op-for-op (bias-free
//! linears; double-RMSNorm sandwich with adaLN scale/gate folded into the norm
//! weights; QK-RMSNorm; interleaved multi-axis RoPE; bidirectional attention;
//! SwiGLU) so validating this validates the device path's math. The backward
//! then routes gradient into every trainable tensor — including, novelly, back
//! through the adaLN fold into the timestep-conditioning vector `c` (`dc`),
//! which is what couples the blocks to the shared `t_embedder` during training.
//!
//! Pure host f64, no device — a training step drives the same op sequence on the
//! GPU via the existing `matmul_dx_reg`/`matmul_dw_reg`/`rmsnorm_dx`/`attn_bwd_*`
//! /`swiglu_bwd` kernels; this module is the ground truth those must reproduce.

/// Shape of the block being differentiated.
#[derive(Clone, Copy)]
pub struct Dims {
    pub t: usize,
    pub dim: usize,
    pub nh: usize,
    pub hd: usize,
    pub cdim: usize,
    pub hidden: usize,
}

impl Dims {
    pub fn new(t: usize, dim: usize, nh: usize) -> Dims {
        Dims { t, dim, nh, hd: dim / nh, cdim: dim.min(256), hidden: dim * 8 / 3 }
    }
    pub fn half(&self) -> usize {
        self.hd / 2
    }
}

/// One block's trainable weights (host, row-major; linears `[out,in]`, bias-free).
#[derive(Clone)]
pub struct Weights {
    pub wq: Vec<f64>,
    pub wk: Vec<f64>,
    pub wv: Vec<f64>,
    pub wo: Vec<f64>,
    pub w1: Vec<f64>,
    pub w2: Vec<f64>,
    pub w3: Vec<f64>,
    pub nq: Vec<f64>, // QK-norm weights [hd]
    pub nk: Vec<f64>,
    pub an1: Vec<f64>, // raw norm weights [dim]
    pub an2: Vec<f64>,
    pub fn1: Vec<f64>,
    pub fn2: Vec<f64>,
    pub adaln_w: Vec<f64>, // [4*dim, cdim]
    pub adaln_b: Vec<f64>, // [4*dim]
}

/// Gradients w.r.t. every [`Weights`] field, plus `dx` (to the previous block)
/// and `dc` (to the timestep conditioning). Same layout as [`Weights`].
#[derive(Clone)]
pub struct Grads {
    pub wq: Vec<f64>,
    pub wk: Vec<f64>,
    pub wv: Vec<f64>,
    pub wo: Vec<f64>,
    pub w1: Vec<f64>,
    pub w2: Vec<f64>,
    pub w3: Vec<f64>,
    pub nq: Vec<f64>,
    pub nk: Vec<f64>,
    pub an1: Vec<f64>,
    pub an2: Vec<f64>,
    pub fn1: Vec<f64>,
    pub fn2: Vec<f64>,
    pub adaln_w: Vec<f64>,
    pub adaln_b: Vec<f64>,
    pub dx: Vec<f64>,
    pub dc: Vec<f64>,
}

const EPS: f64 = 1e-5;

// ---- primitive fwd/bwd (host) ----

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `y = x @ w^T`, `x:[rows,inp]`, `w:[out,inp]` → `y:[rows,out]`.
fn linear(x: &[f64], rows: usize, inp: usize, w: &[f64], out: usize) -> Vec<f64> {
    let mut y = vec![0f64; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut a = 0.0;
            for i in 0..inp {
                a += x[r * inp + i] * w[o * inp + i];
            }
            y[r * out + o] = a;
        }
    }
    y
}

/// Linear backward: returns `(dx, dw)`. `dx = dy @ w`, `dw = dy^T @ x`.
fn linear_bwd(x: &[f64], rows: usize, inp: usize, w: &[f64], out: usize, dy: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut dx = vec![0f64; rows * inp];
    let mut dw = vec![0f64; out * inp];
    for r in 0..rows {
        for o in 0..out {
            let g = dy[r * out + o];
            for i in 0..inp {
                dx[r * inp + i] += g * w[o * inp + i];
                dw[o * inp + i] += g * x[r * inp + i];
            }
        }
    }
    (dx, dw)
}

/// RMSNorm over the last `d` of `[rows,d]`: `y = w ⊙ x·inv`, `inv=1/√(mean(x²)+eps)`.
/// Returns `(y, inv[rows])`.
fn rmsnorm(x: &[f64], rows: usize, d: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0f64; rows * d];
    let mut inv = vec![0f64; rows];
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let ss = xr.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let iv = 1.0 / (ss + EPS).sqrt();
        inv[r] = iv;
        for c in 0..d {
            y[r * d + c] = w[c] * xr[c] * iv;
        }
    }
    (y, inv)
}

/// RMSNorm backward. Accumulates weight grad into `dw` (len `d`), returns `dx`.
/// `dx[c] = inv·g[c] − x[c]·inv³·(1/d)·Σ_k g[k]·x[k]`, `g[c]=w[c]·dy[c]`.
fn rmsnorm_bwd(x: &[f64], rows: usize, d: usize, w: &[f64], inv: &[f64], dy: &[f64], dw: &mut [f64]) -> Vec<f64> {
    let mut dx = vec![0f64; rows * d];
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let iv = inv[r];
        let mut dot = 0.0; // Σ g[k]·x[k]
        for c in 0..d {
            let g = w[c] * dy[r * d + c];
            dot += g * xr[c];
            dw[c] += dy[r * d + c] * xr[c] * iv;
        }
        let coef = iv * iv * iv / d as f64 * dot;
        for c in 0..d {
            let g = w[c] * dy[r * d + c];
            dx[r * d + c] = iv * g - xr[c] * coef;
        }
    }
    dx
}

/// Interleaved RoPE on `[t, nh*hd]`: pair `(2j,2j+1)` rotated by table `(cos,sin)`
/// row `t`. Same table for every head. `cos/sin:[t,half]`.
fn rope(x: &[f64], t: usize, nh: usize, hd: usize, cos: &[f64], sin: &[f64]) -> Vec<f64> {
    let half = hd / 2;
    let mut y = x.to_vec();
    for ti in 0..t {
        for h in 0..nh {
            for j in 0..half {
                let base = (ti * nh + h) * hd + 2 * j;
                let (c, s) = (cos[ti * half + j], sin[ti * half + j]);
                let (e, o) = (x[base], x[base + 1]);
                y[base] = e * c - o * s;
                y[base + 1] = e * s + o * c;
            }
        }
    }
    y
}

/// RoPE backward (rotate the grad by −angle).
fn rope_bwd(dy: &[f64], t: usize, nh: usize, hd: usize, cos: &[f64], sin: &[f64]) -> Vec<f64> {
    let half = hd / 2;
    let mut dx = dy.to_vec();
    for ti in 0..t {
        for h in 0..nh {
            for j in 0..half {
                let base = (ti * nh + h) * hd + 2 * j;
                let (c, s) = (cos[ti * half + j], sin[ti * half + j]);
                let (de, dobits) = (dy[base], dy[base + 1]);
                dx[base] = de * c + dobits * s;
                dx[base + 1] = -de * s + dobits * c;
            }
        }
    }
    dx
}

// ---- forward with cached intermediates ----

/// Everything the backward needs from the forward pass.
pub struct Cache {
    modulation: bool,
    x: Vec<f64>,
    c: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    // adaLN
    scale_msa: Vec<f64>,
    gate_msa: Vec<f64>,
    scale_mlp: Vec<f64>,
    gate_mlp: Vec<f64>,
    an1f: Vec<f64>,
    an2f: Vec<f64>,
    fn1f: Vec<f64>,
    fn2f: Vec<f64>,
    // attention
    n1: Vec<f64>,
    inv_n1: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    inv_qn: Vec<f64>,
    inv_kn: Vec<f64>,
    qr: Vec<f64>,
    kr: Vec<f64>,
    probs: Vec<f64>,
    ctx: Vec<f64>,
    attn_out: Vec<f64>,
    inv_n2: Vec<f64>,
    x1: Vec<f64>,
    // mlp
    f1: Vec<f64>,
    inv_f1: Vec<f64>,
    g: Vec<f64>,
    u: Vec<f64>,
    hsw: Vec<f64>,
    ff: Vec<f64>,
    inv_f2: Vec<f64>,
}

/// One block forward (modulated — the default). Returns `(out[t·dim], cache)`.
pub fn forward(d: Dims, w: &Weights, x: &[f64], c: &[f64], cos: &[f64], sin: &[f64]) -> (Vec<f64>, Cache) {
    forward_m(d, w, x, c, cos, sin, true)
}

/// One block forward with an explicit `modulation` flag. When `false` (Z-Image's
/// `context_refiner`), adaLN is skipped: the four norm weights pass through raw
/// and `c` is unused, so the backward produces no `dc`/adaLN grads.
pub fn forward_m(d: Dims, w: &Weights, x: &[f64], c: &[f64], cos: &[f64], sin: &[f64], modulation: bool) -> (Vec<f64>, Cache) {
    let (t, dim, nh, hd) = (d.t, d.dim, d.nh, d.hd);
    // adaLN modulation: mod = adaln_w @ c + adaln_b, split 4×dim (zero when off).
    let m = if modulation {
        let mut m = w.adaln_b.clone();
        for i in 0..4 * dim {
            let mut a = m[i];
            for (j, &cj) in c.iter().enumerate() {
                a += w.adaln_w[i * d.cdim + j] * cj;
            }
            m[i] = a;
        }
        m
    } else {
        vec![0.0; 4 * dim]
    };
    let scale_msa = m[0..dim].to_vec();
    let gate_msa = m[dim..2 * dim].to_vec();
    let scale_mlp = m[2 * dim..3 * dim].to_vec();
    let gate_mlp = m[3 * dim..4 * dim].to_vec();
    // Modulated: an=raw·(1+scale) / raw·tanh(gate). Unmodulated: an=raw.
    let (an1f, an2f, fn1f, fn2f): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = if modulation {
        (
            w.an1.iter().zip(&scale_msa).map(|(&r, &s)| r * (1.0 + s)).collect(),
            w.an2.iter().zip(&gate_msa).map(|(&r, &g)| r * g.tanh()).collect(),
            w.fn1.iter().zip(&scale_mlp).map(|(&r, &s)| r * (1.0 + s)).collect(),
            w.fn2.iter().zip(&gate_mlp).map(|(&r, &g)| r * g.tanh()).collect(),
        )
    } else {
        (w.an1.clone(), w.an2.clone(), w.fn1.clone(), w.fn2.clone())
    };

    // attention
    let (n1, inv_n1) = rmsnorm(x, t, dim, &an1f);
    let q = linear(&n1, t, dim, &w.wq, dim);
    let k = linear(&n1, t, dim, &w.wk, dim);
    let v = linear(&n1, t, dim, &w.wv, dim);
    let (qn, inv_qn) = rmsnorm(&q, t * nh, hd, &w.nq);
    let (kn, inv_kn) = rmsnorm(&k, t * nh, hd, &w.nk);
    let qr = rope(&qn, t, nh, hd, cos, sin);
    let kr = rope(&kn, t, nh, hd, cos, sin);
    // bidirectional attention per head
    let scale = 1.0 / (hd as f64).sqrt();
    let mut probs = vec![0f64; nh * t * t];
    let mut ctx = vec![0f64; t * dim];
    for h in 0..nh {
        for i in 0..t {
            let mut row = vec![0f64; t];
            let mut mx = f64::NEG_INFINITY;
            for j in 0..t {
                let mut s = 0.0;
                for dd in 0..hd {
                    s += qr[(i * nh + h) * hd + dd] * kr[(j * nh + h) * hd + dd];
                }
                row[j] = s * scale;
                mx = mx.max(row[j]);
            }
            let mut den = 0.0;
            for j in 0..t {
                row[j] = (row[j] - mx).exp();
                den += row[j];
            }
            for j in 0..t {
                let p = row[j] / den;
                probs[(h * t + i) * t + j] = p;
                for dd in 0..hd {
                    ctx[(i * nh + h) * hd + dd] += p * v[(j * nh + h) * hd + dd];
                }
            }
        }
    }
    let attn_out = linear(&ctx, t, dim, &w.wo, dim);
    let (n2, inv_n2) = rmsnorm(&attn_out, t, dim, &an2f);
    let x1: Vec<f64> = x.iter().zip(&n2).map(|(&a, &b)| a + b).collect();

    // mlp
    let (f1, inv_f1) = rmsnorm(&x1, t, dim, &fn1f);
    let g = linear(&f1, t, dim, &w.w1, d.hidden);
    let u = linear(&f1, t, dim, &w.w3, d.hidden);
    let hsw: Vec<f64> = (0..t * d.hidden).map(|i| (g[i] * sigmoid(g[i])) * u[i]).collect();
    let ff = linear(&hsw, t, d.hidden, &w.w2, dim);
    let (f2, inv_f2) = rmsnorm(&ff, t, dim, &fn2f);
    let out: Vec<f64> = x1.iter().zip(&f2).map(|(&a, &b)| a + b).collect();

    let cache = Cache {
        modulation,
        x: x.to_vec(), c: c.to_vec(), cos: cos.to_vec(), sin: sin.to_vec(),
        scale_msa, gate_msa, scale_mlp, gate_mlp, an1f, an2f, fn1f, fn2f,
        n1, inv_n1, q, k, v, inv_qn, inv_kn, qr, kr, probs, ctx, attn_out, inv_n2, x1,
        f1, inv_f1, g, u, hsw, ff, inv_f2,
    };
    (out, cache)
}

/// One block backward. `dout[t·dim]` → grads for all params + `dx` + `dc`.
pub fn backward(d: Dims, w: &Weights, cache: &Cache, dout: &[f64]) -> Grads {
    let (t, dim, nh, hd) = (d.t, d.dim, d.nh, d.hd);
    let mut g = Grads {
        wq: vec![0.0; dim * dim], wk: vec![0.0; dim * dim], wv: vec![0.0; dim * dim], wo: vec![0.0; dim * dim],
        w1: vec![0.0; d.hidden * dim], w2: vec![0.0; dim * d.hidden], w3: vec![0.0; d.hidden * dim],
        nq: vec![0.0; hd], nk: vec![0.0; hd],
        an1: vec![0.0; dim], an2: vec![0.0; dim], fn1: vec![0.0; dim], fn2: vec![0.0; dim],
        adaln_w: vec![0.0; 4 * dim * d.cdim], adaln_b: vec![0.0; 4 * dim],
        dx: vec![0.0; t * dim], dc: vec![0.0; d.cdim],
    };
    // folded-norm-weight grads, later routed through the adaLN fold.
    let (mut d_an1f, mut d_an2f, mut d_fn1f, mut d_fn2f) = (vec![0f64; dim], vec![0f64; dim], vec![0f64; dim], vec![0f64; dim]);

    // out = x1 + f2
    let mut dx1 = dout.to_vec();
    let df2 = dout;
    // f2 = rmsnorm(ff, fn2f)
    let dff = rmsnorm_bwd(&cache.ff, t, dim, &cache.fn2f, &cache.inv_f2, df2, &mut d_fn2f);
    // ff = hsw @ w2^T
    let (dhsw, dw2) = linear_bwd(&cache.hsw, t, d.hidden, &w.w2, dim, &dff);
    g.w2 = dw2;
    // hsw = silu(g) ⊙ u
    let mut dg = vec![0f64; t * d.hidden];
    let mut du = vec![0f64; t * d.hidden];
    for i in 0..t * d.hidden {
        let gi = cache.g[i];
        let sg = sigmoid(gi);
        let silu = gi * sg;
        du[i] = silu * dhsw[i];
        let dsilu = dhsw[i] * cache.u[i];
        dg[i] = dsilu * (sg + gi * sg * (1.0 - sg)); // silu'(g)
    }
    // g = f1@w1^T, u = f1@w3^T
    let (df1a, dw1) = linear_bwd(&cache.f1, t, dim, &w.w1, d.hidden, &dg);
    let (df1b, dw3) = linear_bwd(&cache.f1, t, dim, &w.w3, d.hidden, &du);
    g.w1 = dw1;
    g.w3 = dw3;
    let df1: Vec<f64> = df1a.iter().zip(&df1b).map(|(&a, &b)| a + b).collect();
    // f1 = rmsnorm(x1, fn1f)
    let dx1_mlp = rmsnorm_bwd(&cache.x1, t, dim, &cache.fn1f, &cache.inv_f1, &df1, &mut d_fn1f);
    for i in 0..t * dim {
        dx1[i] += dx1_mlp[i];
    }

    // x1 = x + n2  →  dx += dx1, dn2 = dx1
    let mut dx = dx1.clone();
    let dn2 = &dx1;
    // n2 = rmsnorm(attn_out, an2f)
    let dattn_out = rmsnorm_bwd(&cache.attn_out, t, dim, &cache.an2f, &cache.inv_n2, dn2, &mut d_an2f);
    // attn_out = ctx @ wo^T
    let (dctx, dwo) = linear_bwd(&cache.ctx, t, dim, &w.wo, dim, &dattn_out);
    g.wo = dwo;

    // attention backward
    let scale = 1.0 / (hd as f64).sqrt();
    let mut dqr = vec![0f64; t * dim];
    let mut dkr = vec![0f64; t * dim];
    let mut dv = vec![0f64; t * dim];
    for h in 0..nh {
        for i in 0..t {
            // dprobs[j] = Σ_d dctx[i,h,d]·v[j,h,d] ; dv += probs·dctx
            let mut dprobs = vec![0f64; t];
            for j in 0..t {
                let p = cache.probs[(h * t + i) * t + j];
                let mut dp = 0.0;
                for dd in 0..hd {
                    let gc = dctx[(i * nh + h) * hd + dd];
                    dp += gc * cache.v[(j * nh + h) * hd + dd];
                    dv[(j * nh + h) * hd + dd] += p * gc;
                }
                dprobs[j] = dp;
            }
            // softmax jacobian: dscore[j] = p[j]·(dprobs[j] − Σ_j' p[j']·dprobs[j'])
            let mut sdot = 0.0;
            for j in 0..t {
                sdot += cache.probs[(h * t + i) * t + j] * dprobs[j];
            }
            for j in 0..t {
                let p = cache.probs[(h * t + i) * t + j];
                let dscore = p * (dprobs[j] - sdot) * scale;
                for dd in 0..hd {
                    dqr[(i * nh + h) * hd + dd] += dscore * cache.kr[(j * nh + h) * hd + dd];
                    dkr[(j * nh + h) * hd + dd] += dscore * cache.qr[(i * nh + h) * hd + dd];
                }
            }
        }
    }
    // rope backward → dqn, dkn
    let dqn = rope_bwd(&dqr, t, nh, hd, &cache.cos, &cache.sin);
    let dkn = rope_bwd(&dkr, t, nh, hd, &cache.cos, &cache.sin);
    // qk-norm backward
    let dq = rmsnorm_bwd(&cache.q, t * nh, hd, &w.nq, &cache.inv_qn, &dqn, &mut g.nq);
    let dk = rmsnorm_bwd(&cache.k, t * nh, hd, &w.nk, &cache.inv_kn, &dkn, &mut g.nk);
    // q,k,v = n1 @ {wq,wk,wv}^T
    let (dn1q, dwq) = linear_bwd(&cache.n1, t, dim, &w.wq, dim, &dq);
    let (dn1k, dwk) = linear_bwd(&cache.n1, t, dim, &w.wk, dim, &dk);
    let (dn1v, dwv) = linear_bwd(&cache.n1, t, dim, &w.wv, dim, &dv);
    g.wq = dwq;
    g.wk = dwk;
    g.wv = dwv;
    let dn1: Vec<f64> = (0..t * dim).map(|i| dn1q[i] + dn1k[i] + dn1v[i]).collect();
    // n1 = rmsnorm(x, an1f)
    let dx_attn = rmsnorm_bwd(&cache.x, t, dim, &cache.an1f, &cache.inv_n1, &dn1, &mut d_an1f);
    for i in 0..t * dim {
        dx[i] += dx_attn[i];
    }
    g.dx = dx;

    // ---- adaLN fold backward: fold folded-weight grads → raw norm weights,
    //      scale/gate, then through the modulation linear into raw weights + dc.
    // Unmodulated (context_refiner): folded == raw, so the raw-norm grads are the
    // folded grads directly and there is no adaLN/dc contribution.
    if !cache.modulation {
        g.an1 = d_an1f;
        g.an2 = d_an2f;
        g.fn1 = d_fn1f;
        g.fn2 = d_fn2f;
        return g;
    }
    let mut dmod = vec![0f64; 4 * dim];
    for c in 0..dim {
        // an1f = an1·(1+scale_msa)
        g.an1[c] += d_an1f[c] * (1.0 + cache.scale_msa[c]);
        dmod[c] += d_an1f[c] * w.an1[c];
        // an2f = an2·tanh(gate_msa)
        let tg = cache.gate_msa[c].tanh();
        g.an2[c] += d_an2f[c] * tg;
        dmod[dim + c] += d_an2f[c] * w.an2[c] * (1.0 - tg * tg);
        // fn1f = fn1·(1+scale_mlp)
        g.fn1[c] += d_fn1f[c] * (1.0 + cache.scale_mlp[c]);
        dmod[2 * dim + c] += d_fn1f[c] * w.fn1[c];
        // fn2f = fn2·tanh(gate_mlp)
        let tgm = cache.gate_mlp[c].tanh();
        g.fn2[c] += d_fn2f[c] * tgm;
        dmod[3 * dim + c] += d_fn2f[c] * w.fn2[c] * (1.0 - tgm * tgm);
    }
    // mod = adaln_w @ c + adaln_b
    for i in 0..4 * dim {
        g.adaln_b[i] += dmod[i];
        for j in 0..d.cdim {
            g.adaln_w[i * d.cdim + j] += dmod[i] * cache.c[j];
            g.dc[j] += dmod[i] * w.adaln_w[i * d.cdim + j];
        }
    }
    g
}
